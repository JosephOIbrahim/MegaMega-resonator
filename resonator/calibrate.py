"""
Calibration pre-check — measure task difficulty before a full run.

Runs K tasks x M independent zero-shot samples on the same strict-additive
task class the resonator uses, and reports:

- first-shot mean score (the first sample of each task)
- best-of-N mean score (B, the experiment's baseline)
- first-shot success rate (fraction of tasks solved on the first sample)
- best-of-N success rate (fraction of tasks solved by any sample)
- parse failures and truncation stops

The pre-check aborts (non-zero exit) when B is above the target. B is the
number the resonator must beat, so a B near 1.0 means there is no headroom
and a full run would spend money to learn nothing.

With RESONATOR_PROVIDER unset it runs the offline stub: a wiring check
only, NOT a difficulty estimate. The stub is deliberately weak.

    python -m resonator.calibrate --tasks 10 --samples 5 --target-b 0.50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field, replace

from megamega.bench import generate_task
from resonator.cavity import parse_response_checked
from resonator.config import ExperimentConfig
from resonator.llm import (
    BudgetExceeded,
    BudgetGuard,
    LLMClient,
    UsageLedger,
    make_client,
)
from resonator.task_adapter import render_task_prompt, score


@dataclass
class TaskCalibration:
    seed: int
    first_shot: float
    best_of_n: float
    parse_failures: int
    scores: list[float] = field(default_factory=list)


@dataclass
class CalibrationResult:
    provider: str
    model: str
    n_tasks: int
    samples_per_task: int
    n_entities: int
    m_constraints: int
    constraints_per_slot: int
    seed_base: int
    first_shot_mean: float
    best_of_n_mean: float
    first_shot_success_rate: float
    best_of_n_success_rate: float
    parse_failures: int
    parse_checks: int
    parse_failure_rate: float
    target_b: float
    target_first_shot_success: float
    parse_failure_threshold: float
    passed_b: bool
    passed_first_shot: bool
    passed_parse: bool
    passed: bool
    meaningful: bool
    recommendation: str
    elapsed_sec: float
    total_calls: int
    usage: dict[str, object]
    per_task: list[TaskCalibration]
    artifact_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _success_rate(values: list[float]) -> float:
    """Fraction of tasks whose score is a perfect 1.0."""
    if not values:
        return 0.0
    return sum(1 for v in values if v >= 1.0 - 1e-9) / len(values)


def _recommendation(
    *,
    meaningful: bool,
    passed_parse: bool,
    passed_b: bool,
    passed_first_shot: bool,
    best_of_n_mean: float,
    target_b: float,
    first_shot_success_rate: float,
    target_first_shot_success: float,
    parse_failure_rate: float,
    parse_failure_threshold: float,
) -> str:
    if not meaningful:
        return (
            "Wiring check only: the stub is deliberately weak. Re-run with a "
            "real provider to estimate difficulty."
        )
    if not passed_parse:
        return (
            f"Calibration invalid: parse failure rate "
            f"{parse_failure_rate:.0%} > {parse_failure_threshold:.0%}. "
            "Fix the response format first."
        )
    if not passed_b:
        return (
            f"Too easy: B={best_of_n_mean:.2f} > target {target_b:.2f}. "
            "Harden the task class (Mile 6) before spending on a full run."
        )
    if not passed_first_shot:
        return (
            f"B is within target, but first-shot success "
            f"{first_shot_success_rate:.0%} > "
            f"{target_first_shot_success:.0%}. The resonator has less "
            "headroom than the spec wants; consider a harder task class."
        )
    return (
        f"Calibration passes: B={best_of_n_mean:.2f} <= {target_b:.2f}, "
        f"first-shot success {first_shot_success_rate:.0%} <= "
        f"{target_first_shot_success:.0%}. The task class has headroom."
    )


def run_calibration(
    cfg: ExperimentConfig | None = None,
    *,
    n_tasks: int = 10,
    samples_per_task: int = 5,
    target_b: float = 0.50,
    target_first_shot_success: float = 0.20,
    parse_failure_threshold: float = 0.10,
    n_entities: int | None = None,
    m_constraints: int | None = None,
    constraints_per_slot: int = 1,
    seed_base: int | None = None,
    client: LLMClient | None = None,
    meaningful: bool | None = None,
    write_artifact: bool = True,
    artifacts_dir: str | None = None,
) -> CalibrationResult:
    """
    Measure first-shot and best-of-N difficulty on K tasks.

    Args:
        cfg: Experiment config. Provider/model/budget defaults come from it.
        n_tasks: K, number of tasks to sample.
        samples_per_task: M, independent zero-shot samples per task.
        target_b: Maximum acceptable best-of-N mean (B). B above this fails.
        target_first_shot_success: Maximum acceptable fraction of tasks
            solved on the first sample.
        parse_failure_threshold: Maximum acceptable parse failure rate.
        n_entities: Override cfg.n_entities for a difficulty candidate.
        m_constraints: Override cfg.m_constraints for a difficulty candidate.
        constraints_per_slot: 1 keeps the original equality task; >1
            enables the Mile 6 non-pinning single-slot constraint class.
        seed_base: Override cfg.seed_base.
        client: Inject a client (tests). The client's ledger is used if it
            has one; otherwise usage is reported as zero.
        meaningful: Override the "real model vs stub" flag. None means
            meaningful = (provider != "stub").
        write_artifact: Write a calibration JSON artifact.
        artifacts_dir: Override cfg.artifacts_dir.

    Raises:
        ValueError: invalid counts or thresholds, or a task-generation error.
        RuntimeError: no model or no usable client configuration.
    """
    if n_tasks < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
    if samples_per_task < 1:
        raise ValueError(
            f"samples_per_task must be >= 1, got {samples_per_task}"
        )
    if constraints_per_slot < 1:
        raise ValueError(
            f"constraints_per_slot must be >= 1, got {constraints_per_slot}"
        )
    for name, value in (
        ("target_b", target_b),
        ("target_first_shot_success", target_first_shot_success),
        ("parse_failure_threshold", parse_failure_threshold),
    ):
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")

    cfg = cfg or ExperimentConfig()
    n_entities = cfg.n_entities if n_entities is None else n_entities
    m_constraints = cfg.m_constraints if m_constraints is None else m_constraints
    seed_base = cfg.seed_base if seed_base is None else seed_base
    model = cfg.require_model()
    if meaningful is None:
        meaningful = cfg.provider != "stub"

    seeds = list(range(seed_base, seed_base + n_tasks))
    ledger = UsageLedger()
    if client is None:
        guard = BudgetGuard(max_calls=n_tasks * samples_per_task)
        client = make_client(cfg, guard, ledger)
    else:
        # Injected clients (tests) may already carry a ledger.
        ledger = getattr(client, "ledger", ledger)

    per_task: list[TaskCalibration] = []
    first_scores: list[float] = []
    best_scores: list[float] = []
    parse_failures = 0
    parse_checks = 0
    actual_m_constraints = m_constraints
    t0 = time.time()

    try:
        for task_index, seed in enumerate(seeds, 1):
            task = generate_task(
                seed=seed,
                mu=1.0,
                n_entities=n_entities,
                m_constraints=m_constraints,
                strict_additive=True,
                constraints_per_slot=constraints_per_slot,
            )
            actual_m_constraints = len(task.constraints)
            prompt = render_task_prompt(task)
            scores: list[float] = []
            task_parse_failures = 0
            for _ in range(samples_per_task):
                completion = client.complete(prompt)
                state, parse_failed = parse_response_checked(completion.text)
                parse_checks += 1
                if parse_failed:
                    task_parse_failures += 1
                    parse_failures += 1
                scores.append(score(task, state))

            first = scores[0]
            best = max(scores)
            first_scores.append(first)
            best_scores.append(best)
            per_task.append(
                TaskCalibration(
                    seed=seed,
                    first_shot=first,
                    best_of_n=best,
                    parse_failures=task_parse_failures,
                    scores=scores,
                )
            )
            print(
                f"[calibration task {task_index}/{n_tasks}] seed={seed} "
                f"first={first:.3f} best={best:.3f} "
                f"parse_failures={task_parse_failures} "
                f"calls={task_index * samples_per_task}",
                flush=True,
            )
    except BudgetExceeded as e:
        print(f"\n[calibrate] BUDGET HALT: {e}", flush=True)

    if not per_task:
        raise RuntimeError("Calibration completed no tasks; nothing to report.")

    first_shot_mean = _mean(first_scores)
    best_of_n_mean = _mean(best_scores)
    first_shot_success_rate = _success_rate(first_scores)
    best_of_n_success_rate = _success_rate(best_scores)
    parse_failure_rate = (
        parse_failures / parse_checks if parse_checks else 0.0
    )
    passed_b = best_of_n_mean <= target_b
    passed_first_shot = first_shot_success_rate <= target_first_shot_success
    passed_parse = parse_failure_rate <= parse_failure_threshold
    passed = passed_b and passed_first_shot and passed_parse
    recommendation = _recommendation(
        meaningful=meaningful,
        passed_parse=passed_parse,
        passed_b=passed_b,
        passed_first_shot=passed_first_shot,
        best_of_n_mean=best_of_n_mean,
        target_b=target_b,
        first_shot_success_rate=first_shot_success_rate,
        target_first_shot_success=target_first_shot_success,
        parse_failure_rate=parse_failure_rate,
        parse_failure_threshold=parse_failure_threshold,
    )

    result = CalibrationResult(
        provider=cfg.provider,
        model=model,
        n_tasks=len(per_task),
        samples_per_task=samples_per_task,
        n_entities=n_entities,
        m_constraints=actual_m_constraints,
        constraints_per_slot=constraints_per_slot,
        seed_base=seed_base,
        first_shot_mean=first_shot_mean,
        best_of_n_mean=best_of_n_mean,
        first_shot_success_rate=first_shot_success_rate,
        best_of_n_success_rate=best_of_n_success_rate,
        parse_failures=parse_failures,
        parse_checks=parse_checks,
        parse_failure_rate=parse_failure_rate,
        target_b=target_b,
        target_first_shot_success=target_first_shot_success,
        parse_failure_threshold=parse_failure_threshold,
        passed_b=passed_b,
        passed_first_shot=passed_first_shot,
        passed_parse=passed_parse,
        passed=passed,
        meaningful=meaningful,
        recommendation=recommendation,
        elapsed_sec=round(time.time() - t0, 1),
        total_calls=len(per_task) * samples_per_task,
        usage=ledger.summary(),
        per_task=per_task,
    )

    if write_artifact:
        directory = artifacts_dir or cfg.artifacts_dir
        os.makedirs(directory, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        uid = time.time_ns() % 1_000_000
        path = os.path.join(directory, f"calibration-{stamp}-{uid:06d}.json")
        result.artifact_path = path
        artifact = {
            "config": asdict(cfg),
            "result": result.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)

    return result


def _status(result: CalibrationResult) -> str:
    if not result.meaningful:
        return "WIRING ONLY"
    if not result.passed_parse:
        return "INVALID (parse failures)"
    if not result.passed_b:
        return "FAIL (too easy)"
    if not result.passed_first_shot:
        return "FAIL (first-shot headroom)"
    return "PASS"


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    base = ExperimentConfig()
    return replace(
        base,
        provider=args.provider or base.provider,
        model=args.model or base.model,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure task difficulty before a full resonator run."
    )
    parser.add_argument("--tasks", type=int, default=10, help="K tasks")
    parser.add_argument(
        "--samples", type=int, default=5, help="M samples per task"
    )
    parser.add_argument(
        "--target-b", type=float, default=0.50,
        help="Maximum acceptable best-of-N mean (B)",
    )
    parser.add_argument(
        "--target-first-shot-success", type=float, default=0.20,
        help="Maximum acceptable first-shot success rate",
    )
    parser.add_argument(
        "--parse-failure-threshold", type=float, default=0.10,
        help="Maximum acceptable parse failure rate",
    )
    parser.add_argument("--n-entities", type=int, default=None)
    parser.add_argument("--m-constraints", type=int, default=None)
    parser.add_argument(
        "--constraints-per-slot", type=int, default=1,
        help="1 = equality-only task; >1 = non-pinning Mile 6 task class",
    )
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--no-artifact", action="store_true",
        help="Do not write a calibration JSON artifact",
    )
    args = parser.parse_args(argv)

    cfg = _build_config(args)
    try:
        model = cfg.require_model()
    except RuntimeError as e:
        print(f"[calibrate] {e}", flush=True)
        return 1

    print(
        f"[calibrate] provider={cfg.provider} model={model} "
        f"tasks={args.tasks} samples={args.samples} "
        f"calls={args.tasks * args.samples}",
        flush=True,
    )
    n_entities = args.n_entities or cfg.n_entities
    if args.constraints_per_slot > 1:
        # The per-slot generator ignores the legacy m_constraints argument
        # and emits one set of constraints for every (node, attr) slot.
        effective_m_constraints = (
            n_entities * 3 * args.constraints_per_slot
        )
    else:
        effective_m_constraints = (
            args.m_constraints or cfg.m_constraints
        )
    print(
        f"[calibrate] task class: n_entities={n_entities}, "
        f"m_constraints={effective_m_constraints}, "
        f"constraints_per_slot={args.constraints_per_slot}, "
        "strict_additive=True",
        flush=True,
    )
    print(
        f"[calibrate] targets: B <= {args.target_b:.2f}, "
        f"first-shot success <= {args.target_first_shot_success:.0%}, "
        f"parse failures <= {args.parse_failure_threshold:.0%}",
        flush=True,
    )
    if cfg.provider == "stub":
        print(
            "[calibrate] STUB mode - wiring check only, NOT a difficulty "
            "estimate. The stub is deliberately weak.",
            flush=True,
        )

    try:
        result = run_calibration(
            cfg,
            n_tasks=args.tasks,
            samples_per_task=args.samples,
            target_b=args.target_b,
            target_first_shot_success=args.target_first_shot_success,
            parse_failure_threshold=args.parse_failure_threshold,
            n_entities=args.n_entities,
            m_constraints=args.m_constraints,
            constraints_per_slot=args.constraints_per_slot,
            seed_base=args.seed_base,
            write_artifact=not args.no_artifact,
        )
    except (RuntimeError, ValueError) as e:
        print(f"[calibrate] {e}", flush=True)
        return 1

    usage = result.usage
    cost = usage.get("estimated_cost_usd")
    cost_text = f"${cost:.4f}" if isinstance(cost, float) else "n/a"
    print("\n" + "=" * 64, flush=True)
    print(f"  CALIBRATION: {_status(result)}", flush=True)
    print("=" * 64, flush=True)
    print(f"  First-shot mean            {result.first_shot_mean:.3f}")
    print(
        f"  Best-of-{result.samples_per_task} mean (B)       "
        f"{result.best_of_n_mean:.3f}  (target <= {result.target_b:.2f})"
    )
    print(
        f"  First-shot success rate    "
        f"{result.first_shot_success_rate:.0%}  "
        f"(target <= {result.target_first_shot_success:.0%})"
    )
    print(
        f"  Best-of-N success rate     "
        f"{result.best_of_n_success_rate:.0%}"
    )
    print(
        f"  Parse failures             {result.parse_failures} / "
        f"{result.parse_checks}  "
        f"(invalid if > {result.parse_failure_threshold:.0%})"
    )
    print(
        f"  Calls {result.total_calls} | tokens in/out "
        f"{usage.get('input_tokens', 0)}/{usage.get('output_tokens', 0)} | "
        f"est. cost {cost_text}"
    )
    print("-" * 64)
    print(f"  ACTION: {result.recommendation}")
    print("=" * 64)
    if result.artifact_path:
        print(f"\n[calibrate] artifact: {result.artifact_path}", flush=True)

    if not result.meaningful:
        return 0  # wiring check passed
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
