# Resonator Validation — Execution Plan

**Project:** MegaMega-resonator
**Goal:** Run the sub-$5 test that returns a four-way verdict on the resonator
thesis — true lasing / oracle-dependent gain / parasitic lasing / dead.
**Status:** Repo scaffolded. Substrate (Mile 1/2) in place. Round 5 spec pending.

---

## Where we are

```
Mile 1-2  ████████████████████  DONE   generator + constraints + scoring (414 tests)
Mile 3    ░░░░░░░░░░░░░░░░░░░░  PENDING cheap validating test (blocked on Round 5 spec)
```

The interferometer build was abandoned. The thesis is now a **resonator**:
signal bounces inside a lossless cavity across feedback rounds and ratchets
upward. USD is the lossless mirror, not the amplifier — the gain is the LLM's
conditional generation across rounds. The cheap test uses a Python
strict-additive proxy cavity; real USD enters only if gain is proven.

---

## Fixed (spec-independent — true regardless of Round 5)

- **Substrate:** Mile 1/2 — generator, constraint algebra, scoring DSL. DONE.
- **Proxy cavity:** Python strict-additive merge (append-only, never overwrite
  an existing key). Lossless by construction. No `pxr.Usd` for the cheap test.
- **Model:** Haiku-class, ~$5 hard budget cap.
- **Secrets:** API key local-only via environment variable. Never in the repo,
  never in chat, never committed.
- **Execution host:** Runs on the local Windows machine (needs live API key),
  not in any sandbox.
- **Sparsity guard:** `len(r_t) << len(S_{t-1})` enforced per round to defeat
  the Full-File Cheat false positive. Exact threshold set by Round 5.

## Locked by Round 5 (pending — do not build until filled)

- [ ] Arm structure: 2-arm or 3-arm (Challenge 2 — oracle vs verifier-free)
- [ ] Four-way outcome taxonomy + the exact observable for each outcome
- [ ] Exact feedback content per arm (diff vs composed-state-only)
- [ ] Parasitic-lasing diagnostic signature (trajectory divergence test)
- [ ] Task count, round count, sparsity-guard threshold
- [ ] The single observable that means the *test itself* is invalid

## Build order (only after the spec lands)

1. `difficulty.py`     — tune generator so single-shot success < 20%
2. `cloud.py`          — minimal API loop, budget cap, key from env
3. `proxy_cavity.py`   — strict-additive merge, lossless by construction
4. arm runners         — Track A baseline + Track B/(C) feedback loops
5. `verdict.py`        — four-way taxonomy + parasitic-lasing diagnostic

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
  the gain mechanism, not verifier-free deployment. Track C (if locked)
  isolates this.
