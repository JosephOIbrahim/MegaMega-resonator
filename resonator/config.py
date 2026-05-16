"""
Central configuration for the resonator validation test.

Every experimental knob lives here. Defaults are the locked Gemini Round 5
spec values. Anything sensitive (API keys) is read from the environment and
never stored, logged, or serialized.

Locked spec (Round 5):
- 3-arm randomized block: baseline / oracle / blind
- 50 strict-additive tasks tuned to <20% zero-shot success
- 5 rounds per iterative arm
- Sparsity guard: reject r_t if approx_tokens(r_t) >= 0.5 * approx_tokens(S_{t-1})
- Invalidation: sparsity-guard trigger rate > 30% across B/C => metrics void
- Budget: ~750 calls, hard cap enforced
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentConfig:
    # --- Task generation -------------------------------------------------
    n_tasks: int = 50
    n_entities: int = 8
    # 24 = 8 nodes x 3 attrs. Full grid. Difficulty is tuned via
    # m_constraints and n_entities so single-shot success < 20%.
    m_constraints: int = 24
    seed_base: int = 1000  # task seeds are seed_base .. seed_base + n_tasks-1

    # --- Resonator dynamics ----------------------------------------------
    n_rounds: int = 5
    baseline_samples: int = 5  # Track A: best-of-N

    # --- Sparsity guard --------------------------------------------------
    # For t > 1: reject r_t if approx_tokens(r_t) >= ratio * approx_tokens(S_{t-1})
    sparsity_ratio: float = 0.5
    # If trigger rate across B/C exceeds this, the run is INVALID.
    invalidation_trigger_rate: float = 0.30

    # --- Verdict thresholds ----------------------------------------------
    # Fraction of Track-C tasks that must show monotonic SCORE growth for
    # a TRUE_LASING verdict.
    monotonic_score_threshold: float = 0.60
    # Fraction of Track-C tasks that must show the parasitic signature
    # (volume up, score flat/down) for a PARASITIC_LASING verdict.
    parasitic_signature_threshold: float = 0.60

    # --- Model / provider ------------------------------------------------
    # provider: "anthropic" | "openai" | "stub"
    provider: str = field(
        default_factory=lambda: os.environ.get("RESONATOR_PROVIDER", "stub")
    )
    # Model string is provider-specific. Sensible cheap defaults; override
    # via RESONATOR_MODEL.
    model: str = field(
        default_factory=lambda: os.environ.get("RESONATOR_MODEL", "")
    )
    temperature_iterative: float = 0.0  # oracle + blind: determinism
    temperature_baseline: float = 0.7  # baseline: variance for best-of-N
    max_tokens: int = 1024

    # --- Budget guard ----------------------------------------------------
    # Hard cap on total API calls. Spec expects ~750
    # (50 tasks * [5 baseline + 5 oracle + 5 blind]). Cap with headroom.
    max_api_calls: int = 900

    # --- Output ----------------------------------------------------------
    artifacts_dir: str = "runs"

    def default_model(self) -> str:
        """Resolve the model string if not explicitly set."""
        if self.model:
            return self.model
        return {
            "anthropic": "claude-3-5-haiku-20241022",
            "openai": "gpt-4o-mini",
            "stub": "stub-deterministic",
        }.get(self.provider, "stub-deterministic")

    def task_seeds(self) -> list[int]:
        return list(range(self.seed_base, self.seed_base + self.n_tasks))


# Rough, dependency-free token estimate. The sparsity guard is RATIO-based,
# so the exact constant cancels out — only relative size matters. ~4 chars
# per token is the standard heuristic.
def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)
