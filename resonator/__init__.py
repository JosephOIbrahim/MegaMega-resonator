"""
Resonator validation harness.

Tests whether iterative feedback composition (cloud LLM ↔ lossless cavity)
escapes the single-shot DPI bound on strict-additive tasks. Returns one of
four verdicts: TRUE_LASING / ORACLE_DEPENDENT / PARASITIC_LASING / DEAD
(or INVALID if instrumentation failed).

Built against the locked Gemini Round 5 specification.
"""

__version__ = "0.1.0"
