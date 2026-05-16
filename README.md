# MegaMega

**Track A** experimental harness — measuring USD LIVRPS composition as a
text-space substrate for cloud LLM signal multiplication on procedural
combinatorial tasks.

**Mile 1 of ~7 — benchmark generator.**

The thesis under test: composing N cloud LLM responses via Pixar USD's
LIVRPS priority resolution produces measurably higher constraint
satisfaction than naive Best-of-N, Self-Consistency, or flat dictionary
merge — on tasks where the schema can express the constraints.

The Bifurcation Theorem (Round 2) bounds the achievable gain: text-space
composition is strictly filtering (DPI bound), so the question is whether
schema-mediated filtering produces a measurable lift over flat baselines,
and at what modularity coefficient μ the lift is maximized.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Phase plan

```
Phase 1 (~Day 1-2): Substrate
  ├─ megamega/bench/constraints.py  ✓
  ├─ megamega/bench/generator.py    ✓
  ├─ tests/test_generator.py        ✓
  └─ schemas/usd_base.usda          ← next

Phase 2 (~Day 3-4): Pipeline
  ├─ megamega/cloud/orchestrator.py   16-variant prompt schedule, parallel async
  ├─ megamega/cloud/json_to_usda.py   Pydantic → Sdf.Layer with sparsity enforcement
  └─ megamega/compose/livrps_runner.py  Sublayer stacking, judge-score ordering, registry eviction

Phase 3 (~Day 5): Scoring + baselines
  ├─ megamega/score/                 Proportional constraint satisfaction
  └─ megamega/baselines/             best_of_n, self_consistency, flat_merge (null-safe)

Phase 4 (~Day 6-7): Sweep + analysis
  ├─ megamega/experiment/runner.py   5 μ × 5 conditions × 100 tasks
  └─ megamega/experiment/analysis.py paired Wilcoxon + Benjamini-Hochberg FDR
```

## Design invariants

- **Ground-truth-first construction.** Every task is guaranteed satisfiable
  because constraints are derived from a sampled GT. The generator validates
  this invariant at construction time.
- **Deterministic replay.** A (seed, mu, N, M) tuple produces a bit-identical
  task. Required for cross-condition pairing.
- **Missing-or-None fails.** A constraint over a missing attribute fails,
  matching USD's "opinion absence" semantics. Sparse responses lose points.
- **No LLM-as-judge for final scoring.** Constraint satisfaction is
  deterministic Python; the LLM judge is used only for sublayer ordering.

## Critical risks designed-against

Per the engineering blueprint (Round 3):

- **Dense-Output Degeneration (85%):** the Pydantic translator will enforce
  structural masking per prompt variant, not merely type validation.
  Region Specialist responses get filtered to region-only, etc.
- **Sdf.LayerRegistry OOM (Phase 2):** explicit `Sdf.LayerRegistry().Erase(id)`
  in the runner loop after every task evaluation.
- **Oracle Sublayer Trap:** sublayer ordering must use the same LLM-judge
  scores generated for the Best-of-N condition, computed first.
- **Null-Destruction Confound:** the flat-merge baseline must ignore
  None/null to be mathematically isomorphic to USD's occlusive fall-through.
