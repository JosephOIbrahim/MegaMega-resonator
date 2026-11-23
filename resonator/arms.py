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

Every round records (score, state_volume_tokens, usage, stop_reason,
parse_failed) so the verdict layer can detect parasitic lasing — volume
climbing while score does not — and distinguish a model failure from an
instrumentation failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from megamega.bench import LogicTask
from resonator.cavity import (
    CavityState,
    SparsityGuard,
    compose,
    parse_response_checked,
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
    parse_failed: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""


@dataclass
class BaselineResult:
    """Best-of-N score plus the usage that produced it."""

    score: float
    input_tokens: int = 0
    output_tokens: int = 0
    parse_failures: int = 0
    stop_reasons: list[str] = field(default_factory=list)


@dataclass
class TaskResult:
    seed: int
    baseline_score: float
    baseline_input_tokens: int = 0
    baseline_output_tokens: int = 0
    baseline_parse_failures: int = 0
    baseline_stop_reasons: list[str] = field(default_factory=list)
    oracle_trajectory: list[RoundRecord] = field(default_factory=list)
    blind_trajectory: list[RoundRecord] = field(default_factory=list)

    @property
    def oracle_final(self) -> float:
        return self.oracle_trajectory[-1].score if self.oracle_trajectory else 0.0

    @property
    def blind_final(self) -> float:
        return self.blind_trajectory[-1].score if self.blind_trajectory else 0.0


def run_baseline(
    task: LogicTask,
    client: LLMClient,
    cfg: ExperimentConfig,
    guard: SparsityGuard,
) -> BaselineResult:
    """Track A: best-of-N zero-shot. Returns max score plus usage."""
    prompt = render_task_prompt(task)
    best = 0.0
    input_tokens = 0
    output_tokens = 0
    parse_failures = 0
    stop_reasons: list[str] = []
    for _ in range(cfg.baseline_samples):
        completion = client.complete(prompt)
        input_tokens += completion.input_tokens
        output_tokens += completion.output_tokens
        stop_reasons.append(completion.stop_reason)

        state, parse_failed = parse_response_checked(completion.text)
        guard.record_parse(parse_failed)
        if parse_failed:
            parse_failures += 1
        best = max(best, score(task, state))

    return BaselineResult(
        score=best,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        parse_failures=parse_failures,
        stop_reasons=stop_reasons,
    )


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

        completion = client.complete(prompt)
        admission = guard.admit(completion.text, state, round_index=t)

        if not admission.admitted:
            if not admission.parse_failed:
                task_triggered = True  # sparsity rejection, not a parse failure
            # Freeze state for this round. A parse failure and a full-file
            # rewrite are different instrumentation events.
            traj.append(
                RoundRecord(
                    round_index=t,
                    score=score(task, state),
                    state_volume_tokens=approx_tokens(json.dumps(state)),
                    dropped_opinions=0,
                    sparsity_rejected=not admission.parse_failed,
                    parse_failed=admission.parse_failed,
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                    stop_reason=completion.stop_reason,
                )
            )
            continue

        state, dropped = compose(state, admission.delta)
        traj.append(
            RoundRecord(
                round_index=t,
                score=score(task, state),
                state_volume_tokens=approx_tokens(json.dumps(state)),
                dropped_opinions=dropped,
                sparsity_rejected=False,
                parse_failed=False,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                stop_reason=completion.stop_reason,
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
