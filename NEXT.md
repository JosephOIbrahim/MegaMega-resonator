# NEXT — resume here

**Status (2026-11-23):** Mile 6 prep is in progress. The calibration
pre-check is done; the constraint redesign remains the open scientific
blocker.

---

## What changed in Mile 6 prep (2026-11-23)

- Added `resonator/calibrate.py`: runs K tasks × M independent samples,
  reports first-shot and best-of-N mean and success rates, counts parse
  failures, and exits non-zero when B is above the target.
- Writes `runs/calibration-*.json` with per-task scores and usage.
- `--n-entities` / `--m-constraints` test a candidate difficulty without
  editing the config.
- 6 new tests; 466 total.

Remaining prep: per-task JSONL checkpointing, CLI arguments for the full
run, and the oracle-repair diagnostic.

## What changed in Mile 5 (2026-11-23)

- **Parse-first sparsity guard.** The guard parses the response, then
  compares the parsed delta to the prior state. Prose and code fences around
  a small delta no longer trigger a false "full-file rewrite."
- **Parse-failure invalidation.** `parse_response_checked` separates a
  malformed response from an empty one. Parse failures are counted and void
  the run above 10%.
- **max_tokens invalidation.** The `UsageLedger` stop-reason counts feed a
  second gate: more than 10% truncated calls and the run is INVALID.
- **TRUE_LASING gate fixed.** It now requires `R_blind > B`, a blind gain of
  at least 5 points, and a paired bootstrap 95% CI that excludes zero.
  Monotonic growth is reported but no longer required, so a perfect blind
  first round is not mislabelled as ORACLE_DEPENDENT.
- Artifact records all three rates, `invalidation_reasons`, blind gain and
  its CI, and per-task baseline parse failures.
- 466 tests pass.

## What changed in Mile 4 (2026-11-23)

- Default model is `claude-haiku-5-5` (current cheap Haiku, 10× cheaper
  than Haiku 4.5).
- `anthropic>=2.0.0`; API version `2026-10-21`.
- Removed `temperature` / `top_p` / `top_k`: the current API rejects them on
  every model. The harness uses the provider default and records usage so
  replicate runs can be compared.
- `max_tokens` 1024 → 4096; effort `low` + thinking disabled on Anthropic.
- Every call records input/output tokens and `stop_reason` in a
  `UsageLedger`; the artifact carries `usage` and per-round usage.
- README is the current index; the old interferometer plan is archived in
  `docs/archive/interferometer-plan.md`.

---

## History: the May 2026 finding

First live run on Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) showed
best-of-5 baseline **B = 1.000** on the first 4 tasks. The benchmark is too
easy for a current frontier-cheap model — zero headroom for the resonator.
Run was stopped after 4 tasks (~2 cents). This is a calibration finding,
not a thesis result. The instrumentation caught it immediately.

**Why:** 8 entities × 3 attributes = 24 slots, and `m_constraints=24` pins
every slot with `Node.attr == value`. The prompt contains the answer; the
model only reformats it. At `n_entities=8`, 24 is the full grid and 25
raises `Constraint domain exhausted`. The old RUN.md advice to raise it to
30 was wrong and has been removed.

---

## Next actions, in order

1. **Mile 6 — task redesign (open blocker).** Add local non-pinning
   constraints (inequality, set membership, range) that stay single-slot
   and add-only. Calibrate to single-shot < 20% / B < 0.5. If the task class
   cannot reach it without truncation, report that and escalate.
2. **Mile 6 prep — remaining instrumentation.** Add per-task JSONL
   checkpointing so an interrupted run keeps data, and CLI arguments for
   tasks/provider/model. Add an oracle-repair diagnostic before trusting a
   DEAD verdict: the proxy drops corrections to already-filled slots, so
   the oracle arm currently measures "did the model guess wrong early" as
   well as "can it use feedback." These are not blockers for the redesign.
3. **Mile 7 — run.** Dry run → calibration → three real runs → verdict with
   CIs → update this file.

---

## Decisions already made (2026-11-23)

- **Model:** Haiku 5.5, with Haiku 4.5 kept for replication.
- **Sampling:** delete temperature; use provider default; three replicate
  runs and CIs.
- **Cavity:** keep first-writer-wins for the cheap test; document it
  honestly and add `dropped_opinions` + an oracle-repair diagnostic. Model
  strength ordering belongs to Phase 2.
- **Difficulty:** local non-pinning constraints + calibration (Mile 6).
- **Instrumentation:** parse-first guard; INVALID above 10% parse failures
  or 10% max_tokens stops; TRUE_LASING needs ≥5-point blind gain and a CI
  excluding zero.
- **OpenAI:** no hardcoded model; `RESONATOR_MODEL` required.

**Everything else is done:** substrate (Mile 1–2), resonator (Mile 3),
466 tests, environment-only key handling. Rotate the key if it was ever
exposed.
