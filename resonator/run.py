"""
Orchestrator — run the full 3-arm resonator validation end to end.

    python -m resonator.run

Provider/model/budget come from the environment (see config.py). With
RESONATOR_PROVIDER unset it runs the deterministic StubClient: zero cost,
zero network, exercises the entire pipeline so you can verify wiring
before spending a cent.

Progress is printed inline at every task — long runs never go silent.
The artifact records actual token usage and stop reasons, so a truncated
or refused response is visible instead of looking like a model that stopped
improving.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict

from megamega.bench import generate_task
from resonator.arms import (
    TaskResult,
    run_baseline,
    run_blind,
    run_oracle,
)
from resonator.cavity import SparsityGuard
from resonator.config import ExperimentConfig
from resonator.llm import (
    BudgetExceeded,
    BudgetGuard,
    UsageLedger,
    make_client,
)
from resonator.verdict import decide


def run(cfg: ExperimentConfig | None = None) -> dict:
    cfg = cfg or ExperimentConfig()

    try:
        model = cfg.require_model()
    except RuntimeError as e:
        print(f"[resonator] {e}", flush=True)
        return {}

    seeds = cfg.task_seeds()

    print(
        f"[resonator] provider={cfg.provider} model={model} "
        f"tasks={cfg.n_tasks} rounds={cfg.n_rounds} "
        f"call_cap={cfg.max_api_calls}",
        flush=True,
    )
    if cfg.provider == "stub":
        print(
            "[resonator] STUB mode - no network, no cost. Pipeline check "
            "only; the stub is deliberately weak and will NOT produce a "
            "meaningful scientific verdict.",
            flush=True,
        )

    budget = BudgetGuard(cfg.max_api_calls)
    ledger = UsageLedger()
    try:
        client = make_client(cfg, budget, ledger)
    except RuntimeError as e:
        print(f"[resonator] {e}", flush=True)
        return {}
    # One shared sparsity guard across B+C arms (invalidation is computed
    # over both, per the spec).
    guard = SparsityGuard(ratio=cfg.sparsity_ratio)

    results: list[TaskResult] = []
    t0 = time.time()

    try:
        for i, seed in enumerate(seeds, 1):
            task = generate_task(
                seed=seed,
                mu=1.0,
                n_entities=cfg.n_entities,
                m_constraints=cfg.m_constraints,
                strict_additive=True,
            )
            b = run_baseline(task, client, cfg, guard)
            o = run_oracle(task, client, cfg, guard)
            c = run_blind(task, client, cfg, guard)
            results.append(
                TaskResult(
                    seed=seed,
                    baseline_score=b.score,
                    baseline_input_tokens=b.input_tokens,
                    baseline_output_tokens=b.output_tokens,
                    baseline_parse_failures=b.parse_failures,
                    baseline_stop_reasons=b.stop_reasons,
                    oracle_trajectory=o,
                    blind_trajectory=c,
                )
            )
            print(
                f"[task {i}/{cfg.n_tasks}] seed={seed} "
                f"B={b.score:.3f} O={o[-1].score:.3f} C={c[-1].score:.3f} "
                f"calls={budget.calls}",
                flush=True,
            )
    except BudgetExceeded as e:
        print(f"\n[resonator] BUDGET HALT: {e}", flush=True)
        print(
            f"[resonator] Completed {len(results)}/{cfg.n_tasks} tasks "
            f"before halt. Verdict computed on partial data - treat as "
            f"indicative only.",
            flush=True,
        )

    if not results:
        print("[resonator] No completed tasks. Nothing to decide.", flush=True)
        return {}

    elapsed = time.time() - t0
    usage = ledger.summary()
    max_tokens_rate = ledger.max_tokens_rate
    report = decide(
        results,
        guard.trigger_rate,
        cfg,
        parse_failure_rate=guard.parse_failure_rate,
        max_tokens_rate=max_tokens_rate,
    )

    # --- Persist artifact ------------------------------------------------
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    uid = time.time_ns() % 1_000_000
    artifact_path = os.path.join(
        cfg.artifacts_dir, f"resonator-{stamp}-{uid:06d}.json"
    )
    artifact = {
        "config": asdict(cfg),
        "model": model,
        "elapsed_sec": round(elapsed, 1),
        "total_calls": budget.calls,
        "usage": usage,
        "report": {
            "verdict": report.verdict.value,
            "action": report.action,
            "baseline_mean": report.baseline_mean,
            "oracle_mean": report.oracle_mean,
            "blind_mean": report.blind_mean,
            "monotonic_blind_frac": report.monotonic_blind_frac,
            "parasitic_blind_frac": report.parasitic_blind_frac,
            "sparsity_trigger_rate": report.sparsity_trigger_rate,
            "parse_failure_rate": report.parse_failure_rate,
            "max_tokens_rate": report.max_tokens_rate,
            "invalidation_reasons": report.invalidation_reasons,
            "oracle_gain": report.oracle_gain,
            "blind_gain": report.blind_gain,
            "blind_gain_ci_low": report.blind_gain_ci_low,
            "blind_gain_ci_high": report.blind_gain_ci_high,
            "n_tasks": report.n_tasks,
        },
        "per_task": [
            {
                "seed": r.seed,
                "baseline": r.baseline_score,
                "baseline_usage": {
                    "input_tokens": r.baseline_input_tokens,
                    "output_tokens": r.baseline_output_tokens,
                    "parse_failures": r.baseline_parse_failures,
                    "stop_reasons": r.baseline_stop_reasons,
                },
                "oracle": [asdict(x) for x in r.oracle_trajectory],
                "blind": [asdict(x) for x in r.blind_trajectory],
            }
            for r in results
        ],
    }
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)

    # --- Human-readable verdict -----------------------------------------
    cost = usage["estimated_cost_usd"]
    cost_text = f"${cost:.4f}" if isinstance(cost, float) else "n/a"
    print("\n" + "=" * 64, flush=True)
    print(f"  VERDICT: {report.verdict.value}", flush=True)
    print("=" * 64, flush=True)
    print(f"  Baseline (best-of-{cfg.baseline_samples})  B = {report.baseline_mean:.3f}")
    print(f"  Oracle resonator           R_oracle = {report.oracle_mean:.3f}")
    print(f"  Blind  resonator           R_blind  = {report.blind_mean:.3f}")
    print(
        f"  Blind gain over baseline   {report.blind_gain:+.3f} "
        f"(95% CI {report.blind_gain_ci_low:+.3f} .. "
        f"{report.blind_gain_ci_high:+.3f}; gate: gain >= "
        f"{cfg.min_blind_gain:.2f}, CI > 0)"
    )
    print(f"  Monotonic blind growth     {report.monotonic_blind_frac:.0%} of tasks")
    print(f"  Parasitic signature        {report.parasitic_blind_frac:.0%} of tasks")
    print(f"  Sparsity trigger rate      {report.sparsity_trigger_rate:.0%} "
          f"(invalid if > {cfg.invalidation_trigger_rate:.0%})")
    print(f"  Parse failure rate         {report.parse_failure_rate:.0%} "
          f"(invalid if > {cfg.parse_failure_threshold:.0%})")
    print(f"  max_tokens stop rate       {report.max_tokens_rate:.0%} "
          f"(invalid if > {cfg.max_tokens_threshold:.0%})")
    if report.invalidation_reasons:
        print("  INVALID because:")
        for reason in report.invalidation_reasons:
            print(f"    - {reason}")
    print(
        f"  Calls {usage['calls']} | tokens in/out "
        f"{usage['input_tokens']}/{usage['output_tokens']} | "
        f"est. cost {cost_text}"
    )
    unusual = {
        reason: count
        for reason, count in usage["stop_reasons"].items()
        if reason != "end_turn"
    }
    if unusual:
        print(f"  WARNING stop reasons       {unusual}")
    print("-" * 64)
    print(f"  ACTION: {report.action}")
    print("=" * 64)
    print(f"\n[resonator] artifact: {artifact_path}", flush=True)
    return artifact


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
