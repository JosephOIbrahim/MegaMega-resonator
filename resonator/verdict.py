"""
The four-way verdict — deterministic mapping from results to next action.

Precedence (Round 5 locked spec, Mile 5 instrumentation):

  0. INVALID            Any instrumentation gate is exceeded:
                        - sparsity-guard trigger rate > 30% (the model
                          rewrites the whole file instead of emitting a
                          sparse delta),
                        - parse failure rate > 10% (no recoverable JSON),
                        - max_tokens stop rate > 10% (truncated answers).
                        Metrics void. Fix the prompt or response budget.

  1. DEAD               R_oracle <= B. Even spoon-fed the exact failing
                        constraints, iterative conditioning cannot beat
                        best-of-N variance. G_t = 0. Kill the thesis.

  -- (R_oracle > B from here: a conditional gain mechanism exists) --

  2. TRUE_LASING        R_blind > B, blind gain >= min_blind_gain, and the
                        paired bootstrap 95% CI of (blind - baseline)
                        excludes zero. Verifier-free gain is real. Build
                        Phase 2 (real USD cavity). Monotonic growth is
                        reported as a diagnostic, not required: a blind arm
                        that is perfect on round 1 has no growth to show.

  3. PARASITIC_LASING   >=60% of blind tasks show the parasitic signature:
                        state volume grows monotonically but score flatlines
                        or degrades after round 1. The cavity faithfully
                        amplifies coherent noise. Stop; research
                        mode-selectivity first.

  4. ORACLE_DEPENDENT   R_oracle > B but the blind arm does not clear the
                        TRUE_LASING gate and is not clearly parasitic. The
                        gain mechanism is real but blind; Phase 2 must
                        include a programmatic verifier.

The order matters: a parasitic run can also have R_blind <= B, so the
parasitic signature is checked before falling through to oracle-dependent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
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
    Verdict.INVALID: "Test instrumentation failed. Read "
    "invalidation_reasons: the model ignored the sparse-delta instruction, "
    "returned unparseable output, or hit max_tokens. Fix the prompt or "
    "response budget and rerun. Do NOT trust these numbers.",
    Verdict.DEAD: "Kill the resonator thesis. Revert to the interferometer "
    "track or stop.",
    Verdict.TRUE_LASING: "BUILD PHASE 2 - the real USD/LSA cavity. "
    "Verifier-free conditional gain confirmed.",
    Verdict.PARASITIC_LASING: "STOP building. Research mode-selectivity "
    "(pre-composition injection filter or ground-truth anchor density) "
    "before any USD code.",
    Verdict.ORACLE_DEPENDENT: "PIVOT PHASE 2 - the gain mechanism is real "
    "but blind. Deployment must include a fast programmatic verifier in "
    "the loop as the pump.",
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def paired_bootstrap_ci(
    differences: list[float],
    *,
    resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Percentile bootstrap confidence interval for the mean paired difference.

    Deterministic for a fixed seed, so a given run's verdict is replayable.
    The interval is on the *mean* gain across tasks, which is the quantity
    the verdict compares to baseline.
    """
    if not differences:
        return 0.0, 0.0
    if resamples <= 0:
        mean = _mean(differences)
        return mean, mean

    rng = random.Random(seed)
    n = len(differences)
    means: list[float] = []
    for _ in range(resamples):
        sample_total = sum(
            differences[rng.randrange(n)] for _ in range(n)
        )
        means.append(sample_total / n)
    means.sort()

    alpha = (1.0 - confidence) / 2.0
    lo_index = max(0, int(alpha * resamples))
    hi_index = min(resamples - 1, int((1.0 - alpha) * resamples) - 1)
    return means[lo_index], means[hi_index]


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
    parse_failure_rate: float = 0.0
    max_tokens_rate: float = 0.0
    invalidation_reasons: list[str] = field(default_factory=list)
    oracle_gain: float = 0.0
    blind_gain: float = 0.0
    blind_gain_ci_low: float = 0.0
    blind_gain_ci_high: float = 0.0


def decide(
    results: list[TaskResult],
    sparsity_trigger_rate: float,
    cfg: ExperimentConfig,
    *,
    parse_failure_rate: float = 0.0,
    max_tokens_rate: float = 0.0,
) -> VerdictReport:
    if not results:
        raise ValueError("decide() requires at least one task result")

    n = len(results)
    B = sum(r.baseline_score for r in results) / n
    R_oracle = sum(r.oracle_final for r in results) / n
    R_blind = sum(r.blind_final for r in results) / n

    diffs = [r.blind_final - r.baseline_score for r in results]
    blind_gain = _mean(diffs)
    ci_low, ci_high = paired_bootstrap_ci(
        diffs,
        resamples=cfg.bootstrap_resamples,
        seed=cfg.bootstrap_seed,
    )
    oracle_gain = R_oracle - B

    mono_frac = (
        sum(1 for r in results if _is_monotonic_score_growth(r.blind_trajectory))
        / n
    )
    para_frac = (
        sum(1 for r in results if _is_parasitic(r.blind_trajectory))
        / n
    )

    invalidation: list[str] = []
    if sparsity_trigger_rate > cfg.invalidation_trigger_rate:
        invalidation.append(
            f"sparsity trigger rate {sparsity_trigger_rate:.0%} > "
            f"{cfg.invalidation_trigger_rate:.0%}"
        )
    if parse_failure_rate > cfg.parse_failure_threshold:
        invalidation.append(
            f"parse failure rate {parse_failure_rate:.0%} > "
            f"{cfg.parse_failure_threshold:.0%}"
        )
    if max_tokens_rate > cfg.max_tokens_threshold:
        invalidation.append(
            f"max_tokens stop rate {max_tokens_rate:.0%} > "
            f"{cfg.max_tokens_threshold:.0%}"
        )

    # 0. Invalidation gates — checked first, overrides everything.
    if invalidation:
        v = Verdict.INVALID
    # 1. Dead — gain mechanism absent even with the oracle.
    elif R_oracle <= B:
        v = Verdict.DEAD
    # 2. True lasing — absolute blind gain, statistically clean. Monotonic
    #    growth is diagnostic only; a perfect round 1 has no growth.
    elif (
        R_blind > B
        and blind_gain >= cfg.min_blind_gain
        and ci_low > 0.0
    ):
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
        parse_failure_rate=parse_failure_rate,
        max_tokens_rate=max_tokens_rate,
        invalidation_reasons=invalidation,
        oracle_gain=oracle_gain,
        blind_gain=blind_gain,
        blind_gain_ci_low=ci_low,
        blind_gain_ci_high=ci_high,
        n_tasks=n,
    )
