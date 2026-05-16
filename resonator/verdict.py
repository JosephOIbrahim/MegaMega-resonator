"""
The four-way verdict — deterministic mapping from results to next action.

Precedence (Round 5 locked spec):

  0. INVALID            sparsity-guard trigger rate > 30% across B/C.
                        The model ignored the sparse-delta instruction;
                        we measured prompt-adherence failure, not cavity
                        gain. Metrics void. Fix the prompt, rerun.

  1. DEAD               R_oracle <= B. Even spoon-fed the exact failing
                        constraints, iterative conditioning cannot beat
                        best-of-N variance. G_t = 0. Kill the thesis.

  -- (R_oracle > B from here: a conditional gain mechanism exists) --

  2. TRUE_LASING        R_blind > B AND >=60% of blind tasks show
                        monotonic score growth. Verifier-free gain is
                        real. Build Phase 2 (real USD cavity).

  3. PARASITIC_LASING   >=60% of blind tasks show the parasitic
                        signature: state volume grows monotonically but
                        score flatlines or degrades after round 1. The
                        cavity faithfully amplifies coherent noise.
                        Stop; research mode-selectivity first.

  4. ORACLE_DEPENDENT   R_oracle > B but R_blind <= B and not clearly
                        parasitic. The gain mechanism is real but blind;
                        Phase 2 must include a programmatic verifier.

The order matters: a parasitic run can also have R_blind <= B, so the
parasitic signature is checked before falling through to oracle-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from resonator.arms import RoundRecord, TaskResult
from resonator.config import ExperimentConfig


class Verdict(str, Enum):
    INVALID = "INVALID"
    DEAD = "DEAD"
    TRUE_LASING = "TRUE_LASING"
    PARASITIC_LASING = "PARASITIC_LASING"
    ORACLE_DEPENDENT = "ORACLE_DEPENDENT"


ACTION = {
    Verdict.INVALID: "Test instrumentation failed. Fix the sparse-delta "
    "prompt and rerun. Do NOT trust these numbers.",
    Verdict.DEAD: "Kill the resonator thesis. Revert to the interferometer "
    "track or stop.",
    Verdict.TRUE_LASING: "BUILD PHASE 2 — the real USD/LSA cavity. "
    "Verifier-free conditional gain confirmed.",
    Verdict.PARASITIC_LASING: "STOP building. Research mode-selectivity "
    "(pre-composition injection filter or ground-truth anchor density) "
    "before any USD code.",
    Verdict.ORACLE_DEPENDENT: "PIVOT PHASE 2 — the gain mechanism is real "
    "but blind. Deployment must include a fast programmatic verifier in "
    "the loop as the pump.",
}


def _is_monotonic_score_growth(traj: list[RoundRecord]) -> bool:
    """Strictly non-decreasing with a net increase across the trajectory."""
    scores = [r.score for r in traj]
    if len(scores) < 2:
        return False
    non_decreasing = all(b >= a for a, b in zip(scores, scores[1:]))
    net_gain = scores[-1] > scores[0]
    return non_decreasing and net_gain


def _is_parasitic(traj: list[RoundRecord]) -> bool:
    """
    Parasitic signature: state volume climbs monotonically (model keeps
    confidently emitting accepted deltas) while score flatlines or
    degrades after round 1.
    """
    if len(traj) < 2:
        return False
    vols = [r.state_volume_tokens for r in traj]
    scores = [r.score for r in traj]
    volume_grows = all(b >= a for a, b in zip(vols, vols[1:])) and vols[-1] > vols[0]
    score_stalls = scores[-1] <= scores[0] or all(
        b <= a + 1e-9 for a, b in zip(scores[1:], scores[2:])
    )
    return volume_grows and score_stalls


@dataclass
class VerdictReport:
    verdict: Verdict
    action: str
    baseline_mean: float
    oracle_mean: float
    blind_mean: float
    monotonic_blind_frac: float
    parasitic_blind_frac: float
    sparsity_trigger_rate: float
    n_tasks: int


def decide(
    results: list[TaskResult],
    sparsity_trigger_rate: float,
    cfg: ExperimentConfig,
) -> VerdictReport:
    n = len(results)
    B = sum(r.baseline_score for r in results) / n
    R_oracle = sum(r.oracle_final for r in results) / n
    R_blind = sum(r.blind_final for r in results) / n

    mono_frac = (
        sum(1 for r in results if _is_monotonic_score_growth(r.blind_trajectory))
        / n
    )
    para_frac = (
        sum(1 for r in results if _is_parasitic(r.blind_trajectory)) / n
    )

    # 0. Invalidation gate — checked first, overrides everything.
    if sparsity_trigger_rate > cfg.invalidation_trigger_rate:
        v = Verdict.INVALID
    # 1. Dead — gain mechanism absent even with the oracle.
    elif R_oracle <= B:
        v = Verdict.DEAD
    # 2. True lasing — verifier-free gain, monotonic.
    elif R_blind > B and mono_frac >= cfg.monotonic_score_threshold:
        v = Verdict.TRUE_LASING
    # 3. Parasitic — volume up, score flat (checked before oracle-dependent).
    elif para_frac >= cfg.parasitic_signature_threshold:
        v = Verdict.PARASITIC_LASING
    # 4. Oracle-dependent — gain real but blind.
    else:
        v = Verdict.ORACLE_DEPENDENT

    return VerdictReport(
        verdict=v,
        action=ACTION[v],
        baseline_mean=B,
        oracle_mean=R_oracle,
        blind_mean=R_blind,
        monotonic_blind_frac=mono_frac,
        parasitic_blind_frac=para_frac,
        sparsity_trigger_rate=sparsity_trigger_rate,
        n_tasks=n,
    )
