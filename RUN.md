# RUN.md — Running the resonator test

Mile 3 is built and stub-verified (441 tests pass). To get the real
four-way verdict you run it once against a live model. ~$0.50, ~15 min.

---

## The whole job in one picture

```
1. install        pip install -e ".[dev]" + one provider package
2. set the key    one environment variable (never in the repo)
3. dry run        prove wiring works, $0, no network
4. real run       ~$0.50, prints the verdict
```

---

## Step 1 — Install (PowerShell, in the repo folder)

```powershell
cd C:\Users\User\MegaMega-resonator
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Then ONE provider package — pick the model you have a key for:

```powershell
# If using Claude:
pip install anthropic

# If using GPT (Gemini's spec'd model gpt-4o-mini):
pip install openai
```

Confirm it's healthy:

```powershell
pytest -q
```

You want: **441 passed**.

---

## Step 2 — Set your API key (this session only, never saved to disk)

Claude:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-...your key..."
$env:RESONATOR_PROVIDER = "anthropic"
```

GPT:

```powershell
$env:OPENAI_API_KEY = "sk-...your key..."
$env:RESONATOR_PROVIDER = "openai"
```

The key lives only in this PowerShell window. Close the window, it's gone.
It is never written to the repo, the artifact, or any log.

---

## Step 3 — Dry run (free, proves wiring)

Skip setting the key and the provider. With nothing set it uses the
offline stub:

```powershell
python -m resonator.run
```

Expect **VERDICT: INVALID** with a 100% sparsity trigger rate. That is
CORRECT — the stub is a dumb hash with no instruction-following, so the
invalidation gate fires exactly as designed. It proves the safety
mechanism works: a non-compliant model cannot produce a false positive.

---

## Step 4 — The real run

With the key + provider set from Step 2:

```powershell
python -m resonator.run
```

It prints progress every task (`[task 7/50] ...`) so it never looks
frozen. At the end it prints the verdict block and writes a JSON
artifact to `runs\`.

Hard budget cap is 900 calls (~$0.60 ceiling). It physically halts
before exceeding that — a prompt-loop bug cannot run up a bill.

---

## Reading the result

| Verdict | What it means | Next move |
|---|---|---|
| **TRUE_LASING** | Verifier-free gain is real | Build Phase 2 — real USD cavity |
| **ORACLE_DEPENDENT** | Gain real but needs a verifier | Pivot Phase 2 — verifier in loop |
| **PARASITIC_LASING** | Cavity amplifies coherent noise | Stop. Research mode-selectivity |
| **DEAD** | No conditional gain at all | Kill thesis, revert to interferometer |
| **INVALID** | Model ignored sparse-delta instruction | Fix prompt, rerun. Numbers void |

The single observable that means the *test* failed, not the thesis:
**sparsity trigger rate > 30%** → INVALID. The model is rewriting the
whole file each round instead of emitting deltas. Tighten the prompt
wording and rerun before trusting anything.

---

## Tuning before you spend (optional but recommended)

The spec wants single-shot success **< 20%** so there's headroom for
the resonator to climb. Quick calibration — run a tiny free check of
how hard the tasks are at the default size, then adjust:

```powershell
python -c "from megamega.bench import generate_task; from resonator.task_adapter import score; import statistics; ts=[generate_task(seed=s,mu=1.0,strict_additive=True,n_entities=8,m_constraints=24) for s in range(20)]; print('grid slots per task:', ts[0].m_constraints)"
```

If your first real run shows baseline B already > 0.20, the tasks are
too easy — raise `m_constraints` (env: not exposed yet; edit
`resonator/config.py` `m_constraints`, e.g. 24 → 30) and rerun. Higher
= harder = more headroom.
