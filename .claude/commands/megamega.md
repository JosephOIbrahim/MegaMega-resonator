Run the MegaMega resonator harness: status, tests, dry run, calibration, or full run.

Arguments: $ARGUMENTS

## Dispatch

Parse the first word of the arguments. If there are no arguments, use `status`.

Run every command from the repo root, `/c/Users/User/MegaMega-resonator`
(`C:\Users\User\MegaMega-resonator`), using the project venv at
`./.venv/Scripts/python.exe`.

**status** (default)
- Read `NEXT.md`.
- Run `git status --short --branch` and `git log -3 --oneline`.
- Report the current branch, last commit, the next action from NEXT.md, and
  whether the working tree is clean. Do not start a run.

**test**
- Run `./.venv/Scripts/python.exe -m pytest -q`.
- Report the passed count and the first failure, if any.

**dry**
- Run `./.venv/Scripts/python.exe -m resonator.run`.
- Report the verdict block. `INVALID` with a 100% sparsity trigger rate is
  expected in stub mode; say so.

**calibrate [args...]**
- Run `./.venv/Scripts/python.exe -m resonator.calibrate <args>`.
- If there are no extra args, use `--tasks 10 --samples 5 --target-b 0.50`.
- Report first-shot mean, best-of-N mean (B), success rates, parse failures,
  status, and the artifact path.
- If the exit code is non-zero, name the gate that failed.

**run [args...]**
- Run `./.venv/Scripts/python.exe -m resonator.run`.
- Report the verdict block and the artifact path.
- If the client reports a missing key or model, stop and show the fix from
  RUN.md. Do not retry.

**help** or anything else
- Show this usage table and stop.

| Command | What it does |
|---|---|
| `/megamega` | Project status: branch, last commit, next action |
| `/megamega test` | Run the full pytest suite |
| `/megamega dry` | Free offline stub run |
| `/megamega calibrate --tasks 10 --samples 5` | Measure difficulty, gate on B |
| `/megamega run` | Full experiment (needs key + provider) |

## Rules

- Use the project venv; do not install packages.
- Never echo or print API keys. They come from the environment only.
- Do not commit, push, merge, or delete branches from this command unless
  explicitly asked.
- Keep the reply in the operator register: status, numbers, artifact path,
  next action.
- Use the Bash tool for commands; use PowerShell only when a command needs
  Windows-specific behavior.
