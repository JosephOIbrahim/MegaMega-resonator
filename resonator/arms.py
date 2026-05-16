"""
The three experimental arms.

Track A — Baseline (interferometer): N independent zero-shot generations,
          score = max over the N. Records B = mean over tasks.

Track B — Oracle resonator: T rounds, each fed the DSL diff of failing
          constraints. Tests whether conditional gain G_t > 0 exists at
          all (the model as a mechanic handed a perfect diagnostic).

Track C — Blind resonator: T rounds, fed only the accumulated state, no
          failure report. Tests the deployment thesis (verifier-free
          self-localization).

Every iterative round records (score, state_volume_tokens) so the verdict
layer can detect parasitic lasing — volume climbing while score does not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from megamega.bench import LogicTask
from resonator.cavity import (
    CavityState,
    SparsityGuard,
    compose,
    parse_response,
)
from resonator.config import ExperimentConfig, approx_tokens
from resonator.llm import LLMClient
from resonator.task_adapter import (
    blind_feedback_prompt,
    oracle_feedback_prompt,
    render_task_prompt,
    score,
)


@dataclass
class RoundRecord:
    round_index: int
    score: float
    state_volume_tokens: int
    dropped_opinions: int
    sparsity_rejected: bool


@dataclass
class TaskResult:
    seed: int
    baseline_score: float
    oracle_trajectory: list[RoundRecord] = field(default_factory=list)
    blind_trajectory: list[RoundRecord] = field(default_factory=list)

    @property
    def oracle_final(self) -> float:
        return self.oracle_trajectory[-1].score if self.oracle_trajectory else 0.0

    @property
    def blind_final(self) -> float:
        return self.blind_trajectory[-1].score if self.blind_trajectory else 0.0


def run_baseline(
    task: LogicTask, client: LLMClient, cfg: ExperimentConfig
) -> float:
    """Track A: best-of-N zero-shot. Returns max score over N samples."""
    prompt = render_task_prompt(task)
    best = 0.0
    for _ in range(cfg.baseline_samples):
        raw = client.complete(prompt, cfg.temperature_baseline)
        st = parse_response(raw)
        best = max(best, score(task, st))
    return best


def _run_resonator(
    task: LogicTask,
    client: LLMClient,
    cfg: ExperimentConfig,
    guard: SparsityGuard,
    *,
    oracle: bool,
) -> list[RoundRecord]:
    """Shared T-round loop for the oracle / blind arms."""
    state: CavityState = {}
    traj: list[RoundRecord] = []
    task_triggered = False

    for t in range(1, cfg.n_rounds + 1):
        if t == 1:
            prompt = render_task_prompt(task)
        elif oracle:
            prompt = oracle_feedback_prompt(task, state)
        else:
            prompt = blind_feedback_prompt(task, state)

        raw = client.complete(prompt, cfg.temperature_iterative)

        rejected = False
        if t > 1:
            admitted = guard.admit(raw, state)
            if not admitted:
                rejected = True
                task_triggered = True
                # Freeze state for this round (Full-File Cheat defense).
                traj.append(
                    RoundRecord(
                        round_index=t,
                        score=score(task, state),
                        state_volume_tokens=approx_tokens(json.dumps(state)),
                        dropped_opinions=0,
                        sparsity_rejected=True,
                    )
                )
                continue

        delta = parse_response(raw)
        state, dropped = compose(state, delta)
        traj.append(
            RoundRecord(
                round_index=t,
                score=score(task, state),
                state_volume_tokens=approx_tokens(json.dumps(state)),
                dropped_opinions=dropped,
                sparsity_rejected=rejected,
            )
        )

    guard.per_task_triggered.append(task_triggered)
    return traj


def run_oracle(
    task: LogicTask,
    client: LLMClient,
    cfg: ExperimentConfig,
    guard: SparsityGuard,
) -> list[RoundRecord]:
    return _run_resonator(task, client, cfg, guard, oracle=True)


def run_blind(
    task: LogicTask,
    client: LLMClient,
    cfg: ExperimentConfig,
    guard: SparsityGuard,
) -> list[RoundRecord]:
    return _run_resonator(task, client, cfg, guard, oracle=False)
