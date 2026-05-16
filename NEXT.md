# NEXT — resume here

**Status:** Build complete and on GitHub. One open thread before a real verdict.

**The finding (2026-05-16):** First live run on Claude Haiku 4.5
(`claude-haiku-4-5-20251001`) showed best-of-5 baseline **B = 1.000** on
the first 4 tasks. The benchmark is too easy for a current frontier-cheap
model — zero headroom for the resonator to demonstrate gain. Run was
stopped after 4 tasks (~2 cents). This is a calibration finding, not a
thesis result. The instrumentation worked: the problem was caught
immediately via the live progress markers.

**The open thread:** Harden the benchmark so single-shot success drops
below ~20% for Haiku 4.5, WITHOUT breaking the strict-additive property
that the proxy-cavity validity depends on (Gemini Round 5, Challenge 1).
This is a design judgment, not a one-liner — likely worth a short Gemini
handoff: more entities + more constraints is the obvious lever, but it
must preserve "solvable purely by adding correct triples, never by
revision." Also add a free calibration pre-check so a mis-tuned run can
never silently waste budget again.

**Concrete next actions, in order:**
1. Decide difficulty-hardening approach (Gemini handoff recommended —
   the constraint is preserving strict-additivity while raising
   difficulty for a strong model).
2. Patch `resonator/config.py` difficulty knobs + add calibration
   pre-check (free, estimates single-shot success before spending).
3. Re-run dry (stub) → confirm pipeline → real run with new key.
4. Read the four-way verdict.

**Everything else is done:** substrate (Mile 1-2), resonator (Mile 3),
441 tests, repo live on GitHub (private), model string fixed to
`claude-haiku-4-5-20251001`. Key handling is environment-only; rotate
if ever exposed.
