# MegaMega

**Track A** experimental harness — testing whether composing cloud LLM
responses through a lossless cavity produces measurable constraint-
satisfaction gain over naive best-of-N on procedural combinatorial tasks.

**Mile 6 — task redesign.** The substrate (Mile 1–2), the 3-arm resonator
(Mile 3), the November 2026 compatibility pass (Mile 4), and the
instrument-validity pass (Mile 5) are built. The non-pinning constraint
kinds and the calibration pre-check are in place; the next step is tuning
`constraints_per_slot` to bring B below 0.5. 501 tests pass. The May 2026
live run found the benchmark too easy for Haiku 4.5 (B = 1.000 on the
first four tasks); the task class is the open scientific blocker.

---

## What is being tested

The thesis: composing N cloud LLM responses through Pixar USD's LIVRPS
priority resolution produces measurably higher constraint satisfaction than
naive Best-of-N, Self-Consistency, or flat dictionary merge — on tasks where
the schema can express the constraints.

The Bifurcation Theorem (Round 2) bounds the achievable gain: text-space
composition is strictly filtering (DPI bound), so the question is whether
schema-mediated filtering and iterative feedback produce a measurable lift
over flat baselines, and at what modularity coefficient μ the lift is
maximized.

The cheap test is a **resonator**, not an interferometer: signal bounces
inside a lossless proxy cavity across feedback rounds and ratchets upward.
USD is the lossless mirror, not the amplifier — the gain is the LLM's
conditional generation across rounds. Real USD enters only if gain is proven.

---

## Current state

```
Mile 1–2  DONE    substrate: generator, constraint algebra, scoring
Mile 3    BUILT   3-arm resonator test: baseline / oracle / blind
Mile 4    DONE    Nov 2026 compatibility: API 2026-10-21, Haiku 5.5,
                  no sampling params, usage/stop_reason capture
Mile 5    DONE    instrument validity: parse-first guard, parse/truncation
                  invalidation, TRUE_LASING CI gate
Mile 6    IN PROGRESS  non-pinning constraint kinds + calibration pre-check
                       next: tune constraints_per_slot to B < 0.5
Mile 7    QUEUED  dry run → calibration → real runs → verdict
```

**The May 2026 finding:** a live run on Claude Haiku 4.5 showed best-of-5
baseline **B = 1.000** on the first 4 tasks. Zero headroom, so the run was
stopped after about 2 cents. That is a calibration finding, not a thesis
result.

**Why it happened:** 8 entities × 3 attributes = 24 slots. The generator
pins every slot with a `Node.attr == value` constraint, so the prompt
contains the answer and the model only has to reformat it as JSON. At
`n_entities=8`, `m_constraints=24` is the full grid; raising it to 30 raises
`ValueError: Constraint domain exhausted`. Difficulty hardening (Mile 6)
changes the constraint class, not the count.

**Mile 6 task class:** `generate_task(..., constraints_per_slot=N)` emits
four single-slot kinds: equality, inequality, set membership, and capacity
range. `N=1` keeps the original equality task; `N>1` uses the non-pinning
mix. No constraint couples two slots, so the proxy cavity's add-only
guarantee still holds. Calibration chooses N.

---

## Setup (PowerShell, in the repo folder)

```powershell
cd C:\Users\User\MegaMega-resonator
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,cloud]"
pytest -q
```

You want **501 passed**. The `cloud` extra installs `anthropic>=2.0.0`,
which sends API version `2026-10-21` by default. The `openai` extra is
optional and requires an explicit `RESONATOR_MODEL`.

For a free dry run (no key, no network):

```powershell
python -m resonator.run
```

It should print `VERDICT: INVALID` with a 100% sparsity trigger rate. That
is correct: the stub deliberately ignores the sparse-delta instruction, and
the invalidation gate catches it.

For a real run, see `RUN.md`.

---

## Design invariants

- **Ground-truth-first construction.** Every task is guaranteed satisfiable
  because constraints are derived from a sampled GT. The generator validates
  this invariant at construction time.
- **Deterministic task generation.** A (seed, mu, N, M) tuple produces the
  same task. Model sampling is the provider default, so model outputs are
  not deterministic; replicate runs for variance.
- **Missing-or-None fails.** A constraint over a missing attribute fails,
  matching USD's "opinion absence" semantics. Sparse responses lose points.
- **No LLM-as-judge for final scoring.** Constraint satisfaction is
  deterministic Python; any LLM judge is used only for sublayer ordering.
- **Append-only proxy cavity.** The cheap test never overwrites an existing
  (node, attribute) slot. This is first-writer-wins, not LIVRPS; model
  strength ordering belongs to Phase 2.
- **Instrumentation gates.** A run is INVALID if sparsity rejections exceed
  30%, parse failures exceed 10%, or `max_tokens` stops exceed 10%.
  TRUE_LASING requires a blind gain of at least 5 points and a paired
  bootstrap 95% CI that excludes zero; monotonic growth is diagnostic only.
- **Environment-only secrets, hard call cap.** API keys are read from the
  environment and never serialized. A `BudgetGuard` stops the run at the
  call ceiling.

---

## Layout

```
megamega/bench/            constraint algebra + procedural task generator
resonator/config.py        every experimental knob; model registry; pricing
resonator/llm.py           Anthropic / OpenAI / stub clients, budget + usage
resonator/cavity.py        append-only proxy cavity + sparsity guard
resonator/arms.py          baseline / oracle / blind trajectories
resonator/task_adapter.py  task prompt, oracle feedback, scoring
resonator/verdict.py       four-way decision tree
resonator/run.py           orchestrator + artifact writer
resonator/calibrate.py     K tasks × M samples difficulty pre-check
tests/                     generator and resonator test suites
runs/                      run artifacts (gitignored)
docs/archive/              historical interferometer plan
```

---

## History

The original interferometer / USD plan is archived in
`docs/archive/interferometer-plan.md`. It is kept for the thesis context,
not as the current build order.
