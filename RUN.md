# RUN.md — Running the resonator test

Mile 5 is built and stub-verified (460 tests pass). A full 50-task run on
Claude Haiku 5.5 costs roughly **$0.10–$0.20** at November 2026 prices, not
$0.50.
The harness uses API version `2026-10-21` through `anthropic>=2.0.0`; that
version rejects `temperature` / `top_p` / `top_k` on every model, so the
harness sends none and uses the provider default.

---

## The whole job in one picture

```
1. install        pip install -e ".[dev,cloud]"
2. set the key    one environment variable (never in the repo)
3. dry run        prove wiring works, $0, no network
4. real run       ~$0.08, prints the verdict and writes usage to the artifact
```

---

## Step 1 — Install (PowerShell, in the repo folder)

```powershell
cd C:\Users\User\MegaMega-resonator
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,cloud]"
```

Confirm it is healthy:

```powershell
pytest -q
```

You want: **460 passed**.

For the optional OpenAI path, install `.[dev,openai]` and set
`RESONATOR_MODEL`; there is no hardcoded 2026 default.

---

## Step 2 — Set your API key (this session only, never saved to disk)

Claude:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-...your key..."
$env:RESONATOR_PROVIDER = "anthropic"
```

Default model: `claude-haiku-5-5` ($0.10 input / $0.50 output per 1M
tokens, up to 100K-token prompts). To replicate the May 2026 run instead:

```powershell
$env:RESONATOR_MODEL = "claude-haiku-4-5"
```

GPT (optional, unverified in 2026):

```powershell
$env:OPENAI_API_KEY = "sk-...your key..."
$env:RESONATOR_PROVIDER = "openai"
$env:RESONATOR_MODEL = "your-current-model-id"
```

The key lives only in this PowerShell window. Close the window, it is gone.
It is never written to the repo, the artifact, or any log.

---

## Step 3 — Dry run (free, proves wiring)

Skip the key and provider. With nothing set it uses the offline stub:

```powershell
python -m resonator.run
```

Expect **VERDICT: INVALID** with a 100% sparsity trigger rate. That is
correct: the stub is a dumb hash with no instruction-following, so the
invalidation gate fires exactly as designed. It proves a non-compliant
model cannot produce a false positive.

---

## Step 4 — The real run

With the key + provider set from Step 2:

```powershell
python -m resonator.run
```

It prints progress every task (`[task 7/50] ...`) so it never looks frozen.
At the end it prints the verdict, calls, tokens in/out, estimated cost, and
any non-`end_turn` stop reasons. The JSON artifact in `runs\` carries the
same accounting per task and per round.

Hard budget cap is 900 calls. At Haiku 5.5 prices a full 50-task run is
under $0.25; the cap is a loop-bug guard, not a realistic spend ceiling.
The artifact's `usage.estimated_cost_usd` is the real number.

---

## Reading the result

| Verdict | What it means | Next move |
|---|---|---|
| **TRUE_LASING** | Verifier-free gain is real | Build Phase 2 — real USD cavity |
| **ORACLE_DEPENDENT** | Gain real but needs a verifier | Pivot Phase 2 — verifier in loop |
| **PARASITIC_LASING** | Cavity amplifies coherent noise | Stop. Research mode-selectivity |
| **DEAD** | No conditional gain at all | Kill thesis, revert to interferometer |
| **INVALID** | An instrumentation gate tripped | Fix prompt or budget, rerun. Numbers void |

A run is INVALID if **any** instrumentation gate is exceeded:

- **Sparsity trigger rate > 30%** — the model rewrites the whole file
  instead of emitting a sparse delta. The guard measures the parsed delta,
  so prose and code fences around a small delta do not trigger it.
- **Parse failure rate > 10%** — no recoverable JSON object.
- **max_tokens stop rate > 10%** — responses were truncated.

The verdict block prints all three and names every reason that fired.

**The TRUE_LASING gate:** R_blind must beat B, the blind gain must be at
least 5 points, and the paired bootstrap 95% CI for (blind - baseline) must
exclude zero. Monotonic growth is reported but no longer required, so a
blind arm that solves the task on round 1 is not mislabelled.

---

## Tuning before you spend

The spec wants single-shot success below 20% so the resonator has headroom.
The old advice to raise `m_constraints` from 24 to 30 is wrong: 8 entities ×
3 attributes = 24 slots, so 24 is the full grid and 25 raises
`Constraint domain exhausted`. Difficulty hardening is Mile 6 (local
non-pinning constraints + calibration). Until then, a real run is a pipeline
test, not a thesis result.

A quick, free check of the task size:

```powershell
python -c "from megamega.bench import generate_task; t=generate_task(seed=1000,mu=1.0,strict_additive=True,n_entities=8,m_constraints=24); print('slots:', t.n_entities*3, 'constraints:', len(t.constraints))"
```
