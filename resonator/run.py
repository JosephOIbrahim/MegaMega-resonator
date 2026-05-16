"""
Orchestrator — run the full 3-arm resonator validation end to end.

    python -m resonator.run

Provider/model/budget come from the environment (see config.py). With
RESONATOR_PROVIDER unset it runs the deterministic StubClient: zero cost,
zero network, exercises the entire pipeline so you can verify wiring
before spending a cent.

Progress is printed inline at every task — long runs never go silent.
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
from resonator.llm import BudgetExceeded, BudgetGuard, make_client
from resonator.verdict import decide


def run(cfg: ExperimentConfig | None = None) -> dict:
    cfg = cfg or ExperimentConfig()
    seeds = cfg.task_seeds()

    print(
        f"[resonator] provider={cfg.provider} model={cfg.default_model()} "
        f"tasks={cfg.n_tasks} rounds={cfg.n_rounds} "
        f"call_cap={cfg.max_api_calls}",
        flush=True,
    )
    if cfg.provider == "stub":
        print(
            "[resonator] STUB mode — no network, no cost. Pipeline check "
            "only; the stub is deliberately weak and will NOT produce a "
            "meaningful scientific verdict.",
            flush=True,
        )

    budget = BudgetGuard(cfg.max_api_calls)
    client = make_client(cfg, budget)
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
            b = run_baseline(task, client, cfg)
            o = run_oracle(task, client, cfg, guard)
            c = run_blind(task, client, cfg, guard)
            results.append(
                TaskResult(
                    seed=seed,
                    baseline_score=b,
                    oracle_trajectory=o,
                    blind_trajectory=c,
                )
            )
            print(
                f"[task {i}/{cfg.n_tasks}] seed={seed} "
                f"B={b:.3f} O={o[-1].score:.3f} C={c[-1].score:.3f} "
                f"calls={budget.calls}",
                flush=True,
            )
    except BudgetExceeded as e:
        print(f"\n[resonator] BUDGET HALT: {e}", flush=True)
        print(
            f"[resonator] Completed {len(results)}/{cfg.n_tasks} tasks "
            f"before halt. Verdict computed on partial data — treat as "
            f"indicative only.",
            flush=True,
        )

    if not results:
        print("[resonator] No completed tasks. Nothing to decide.", flush=True)
        return {}

    report = decide(results, guard.trigger_rate, cfg)
    elapsed = time.time() - t0

    # --- Persist artifact ------------------------------------------------
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    artifact_path = os.path.join(
        cfg.artifacts_dir, f"resonator-{stamp}.json"
    )
    artifact = {
        "config": {
            k: v
            for k, v in vars(cfg).items()
            # never persist anything that could carry a secret
            if k not in ("provider",) or True
        },
        "model": cfg.default_model(),
        "elapsed_sec": round(elapsed, 1),
        "total_calls": budget.calls,
        "report": {
            "verdict": report.verdict.value,
            "action": report.action,
            "baseline_mean": report.baseline_mean,
            "oracle_mean": report.oracle_mean,
            "blind_mean": report.blind_mean,
            "monotonic_blind_frac": report.monotonic_blind_frac,
            "parasitic_blind_frac": report.parasitic_blind_frac,
            "sparsity_trigger_rate": report.sparsity_trigger_rate,
            "n_tasks": report.n_tasks,
        },
        "per_task": [
            {
                "seed": r.seed,
                "baseline": r.baseline_score,
                "oracle": [asdict(x) for x in r.oracle_trajectory],
                "blind": [asdict(x) for x in r.blind_trajectory],
            }
            for r in results
        ],
    }
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)

    # --- Human-readable verdict -----------------------------------------
    print("\n" + "=" * 64, flush=True)
    print(f"  VERDICT: {report.verdict.value}", flush=True)
    print("=" * 64, flush=True)
    print(f"  Baseline (best-of-{cfg.baseline_samples})  B = {report.baseline_mean:.3f}")
    print(f"  Oracle resonator           R_oracle = {report.oracle_mean:.3f}")
    print(f"  Blind  resonator           R_blind  = {report.blind_mean:.3f}")
    print(f"  Monotonic blind growth     {report.monotonic_blind_frac:.0%} of tasks")
    print(f"  Parasitic signature        {report.parasitic_blind_frac:.0%} of tasks")
    print(f"  Sparsity trigger rate      {report.sparsity_trigger_rate:.0%} "
          f"(invalid if > {cfg.invalidation_trigger_rate:.0%})")
    print("-" * 64)
    print(f"  ACTION: {report.action}")
    print("=" * 64)
    print(f"\n[resonator] artifact: {artifact_path}", flush=True)
    return artifact


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
