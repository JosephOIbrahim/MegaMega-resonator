# Resonator Validation — Execution Plan

**Project:** MegaMega-resonator
**Goal:** Run the sub-$5 test that returns a four-way verdict on the resonator
thesis — true lasing / oracle-dependent gain / parasitic lasing / dead.
**Status (2026-11-23):** Mile 5 core instrumentation is done. Mile 6 (task
redesign) is next and remains the open scientific blocker.

---

## Where we are

```
Mile 1-2  DONE    generator + constraints + scoring
Mile 3    BUILT   3-arm resonator test (460 tests, stub-verified)
Mile 4    DONE    Nov 2026 compatibility: API 2026-10-21, Haiku 5.5,
                  no sampling params, usage/stop_reason capture
Mile 5    DONE    instrument validity: parse-first guard, parse/truncation
                  invalidation, TRUE_LASING CI gate
Mile 6    NEXT    task redesign: local non-pinning constraints + calibration
Mile 7    QUEUED  dry run -> calibration -> real runs -> verdict
```

The interferometer build was abandoned. The thesis is now a **resonator**:
signal bounces inside a lossless cavity across feedback rounds and ratchets
upward. USD is the lossless mirror, not the amplifier — the gain is the LLM's
conditional generation across rounds. The cheap test uses a Python
strict-additive proxy cavity; real USD enters only if gain is proven.

**November 2026 note:** API version `2026-10-21` removed
`temperature` / `top_p` / `top_k` on every model. The harness uses the
provider default and records usage/`stop_reason`. Haiku 5.5 is 10× cheaper
than Haiku 4.5, so budget is no longer the constraint; validity is.

---

## Fixed (spec-independent)

- **Substrate:** Mile 1/2 — generator, constraint algebra, scoring DSL. DONE.
- **Proxy cavity:** Python strict-additive merge (append-only, never
  overwrite an existing key). First-writer-wins by construction — not
  LIVRPS; model strength ordering belongs to Phase 2. No `pxr.Usd` for the
  cheap test.
- **Model:** Claude Haiku 5.5 by default (`claude-haiku-5-5`), under $0.25
  per 50-task run. `claude-haiku-4-5` is available for replication.
- **Secrets:** API key local-only via environment variable. Never in the
  repo, never in chat, never committed.
- **Execution host:** Runs on the local Windows machine (needs live API
  key), not in any sandbox.
- **Sparsity guard:** `len(r_t) << len(S_{t-1})` enforced per round to
  defeat the Full-File Cheat false positive. Exact threshold set by Round 5.

## Locked by Round 5

- [x] Arm structure: 3-arm (baseline / oracle / blind)
- [x] Four-way outcome taxonomy + observables
- [x] Feedback content per arm (oracle diff vs state-only)
- [x] Parasitic-lasing diagnostic signature
- [x] Task count (50), round count (5), sparsity threshold (0.5)
- [x] Invalidation observables (sparsity > 30%, parse failures > 10%, max_tokens stops > 10%)

## Mile 5 — instrument validity (done)

1. [x] Parse first, then measure the parsed delta for the sparsity guard.
2. [x] Count parse failures and `max_tokens` stop reasons; add them to the
   invalidation gate (10% each).
3. [x] Record `parse_failed` per round and baseline parse failures; surface
   all rates and `invalidation_reasons` in the verdict and artifact.
4. [x] Add paired bootstrap CIs. TRUE_LASING requires `R_blind > B`, a
   blind gain of at least 5 points, and a CI excluding zero. Monotonic
   growth is diagnostic only.

**Remaining instrumentation (not a blocker for the redesign):**
`resonator/calibrate.py`, per-task JSONL checkpointing, and CLI arguments.
Queued with Mile 6.

## Mile 6 — task redesign (next, open blocker)

Add local non-pinning constraint kinds (inequality, set membership, range).
They stay single-slot and add-only, so the strict-additive property survives,
but the prompt stops listing the answer. Generate them from GT and calibrate
to single-shot below 20% / B below 0.5.

## Mile 7 — run

Dry run → calibration → three real runs → verdict with CIs → NEXT.md. At
Haiku 5.5 prices the whole sequence is under $1.

## Decision routing (what each outcome triggers)

| Outcome                 | Meaning                                  | Next move                       |
|-------------------------|------------------------------------------|---------------------------------|
| True lasing             | Cavity gain real, verifier-free          | Build Phase 2 — real USD cavity |
| Oracle-dependent gain   | Gain real but needs a verifier in loop   | Different Phase 2 — verifier arc |
| Parasitic lasing        | Cavity works, gain not truth-selective   | Research problem surfaced — $5  |
| Dead                    | No conditional gain                      | Thesis dead — stop, archive     |

---

## Risks designed-against (carried from Rounds 3–4)

- **Full-File Cheat (false positive):** model regenerates whole state instead
  of a sparse delta → not testing a cavity. Guard: programmatic length check.
- **Parasitic lasing:** lossless cavity amplifies a structurally-anchored
  hallucination with perfect fidelity. The four-way taxonomy exists to
  *detect* this, not just risk it.
- **Oracle over-interpretation:** feeding the ground-truth diff back proves
  the gain mechanism, not verifier-free deployment. Track C isolates this.
- **First-writer-wins bias:** the proxy cannot revise an early wrong
  opinion; the oracle's correction is dropped. `dropped_opinions` is
  recorded per round; an oracle-repair diagnostic remains open before a
  DEAD verdict is trusted.
- **Task-class ceiling:** all-local equality constraints are transcription;
  at 8 entities the grid is 24 slots. Mile 6 changes the constraint class.

## Historical build order (superseded)

The original build order (`difficulty.py`, `cloud.py`, `proxy_cavity.py`,
arm runners, `verdict.py`) was executed under different module names. The
current layout is in `README.md`.
