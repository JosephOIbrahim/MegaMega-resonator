# RUN.md — Running the resonator test

Mile 5 is built, and the Mile 6 constraint kinds and calibration pre-check
are in (500 tests pass). A full 50-task run on
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

You want: **500 passed**.

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

## Calibration pre-check (Mile 6)

Measure difficulty before a full run. The pre-check runs K tasks × M
independent samples and reports first-shot and best-of-N scores. It exits
non-zero when B is above the target, so a mis-tuned run stops before the
full experiment spends money.

```powershell
# Stub: wiring check only, no network, no cost.
python -m resonator.calibrate --tasks 4 --samples 2

# Original equality task (Mile 5 behavior):
python -m resonator.calibrate --tasks 10 --samples 5 --target-b 0.50

# Non-pinning Mile 6 task class, 2 constraints per slot:
python -m resonator.calibrate --tasks 10 --samples 5 --target-b 0.50 --constraints-per-slot 2
```

It prints the first-shot mean and success rate, the best-of-N mean (B) and
success rate, parse failures, and a PASS / FAIL / WIRING ONLY status with
the next action. It writes `runs\calibration-*.json`.

`--constraints-per-slot 1` keeps the original equality task. Values above 1
emit the non-pinning mix (inequality, set membership, capacity range); the
total is `n_entities × 3 × constraints_per_slot`, so `m_constraints` is
ignored on that path. Start at 2, then try 3 or 4 if B is still too high.

`--n-entities` and `--m-constraints` test a candidate difficulty on the
equality path. At `n_entities=8`, `m_constraints` above 24 raises
`Constraint domain exhausted`; the non-pinning path does not use that
ceiling.

The spec wants single-shot success below 20% so the resonator has headroom.
Until the Mile 6 class is tuned to B < 0.5, a real run is a pipeline test,
not a thesis result.
