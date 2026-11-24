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
- Invalidation: sparsity trigger rate > 30%, parse failure rate > 10%, or
  max_tokens stop rate > 10% => metrics void
- TRUE_LASING gate: blind gain >= min_blind_gain and the paired bootstrap
  95% CI excludes zero. Monotonic growth is reported, not required.
- Budget: ~750 calls, hard cap enforced

Mile 6 task class:
- constraints_per_slot = 1 keeps the original equality-only task.
- constraints_per_slot > 1 enables non-pinning single-slot constraints
  (inequality, set membership, capacity range). The tuned value comes from
  the calibration pre-check, not from this default.

November 2026 compatibility:
- API version 2026-10-21 rejects temperature / top_p / top_k on every model.
  The harness sends none of them and uses the provider's default sampling.
- Default model is claude-haiku-5-5 (the current cheap Haiku). Override with
  RESONATOR_MODEL. claude-haiku-4-5 stays available for replication.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

# Estimated per-1M-token prices in USD. Used only to annotate a run artifact;
# never used for billing decisions. A model missing here reports cost as None.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
}


@dataclass(frozen=True)
class ExperimentConfig:
    # --- Task generation -------------------------------------------------
    n_tasks: int = 50
    n_entities: int = 8
    # 8 nodes x 3 attrs = 24 slots. At n_entities=8 this is the FULL grid:
    # every local constraint pins one slot, so m_constraints cannot exceed
    # 24 without raising "Constraint domain exhausted". Difficulty hardening
    # (Mile 6) changes the constraint class, not this number.
    m_constraints: int = 24
    # Mile 6: 1 = original equality-only task. Values > 1 generate
    # non-pinning single-slot constraints per slot. The tuned default comes
    # from calibration; 1 preserves the pre-Mile-6 behavior.
    constraints_per_slot: int = 1
    seed_base: int = 1000  # task seeds are seed_base .. seed_base + n_tasks-1

    # --- Resonator dynamics ----------------------------------------------
    n_rounds: int = 5
    baseline_samples: int = 5  # Track A: best-of-N

    # --- Sparsity guard --------------------------------------------------
    # For t > 1: reject r_t if approx_tokens(parsed delta) >=
    # ratio * approx_tokens(S_{t-1}). The check runs on the parsed delta,
    # not the raw response, so prose and code fences do not look like a
    # full-file rewrite.
    sparsity_ratio: float = 0.5

    # --- Invalidation gates ----------------------------------------------
    # Exceeding any one voids the run. Sparsity is the original Round 5
    # signal; parse failures and max_tokens stops are instrumentation
    # failures added in Mile 5.
    invalidation_trigger_rate: float = 0.30
    parse_failure_threshold: float = 0.10
    max_tokens_threshold: float = 0.10

    # --- Verdict thresholds ----------------------------------------------
    # Monotonic blind growth is reported as a diagnostic, not a gate: a
    # blind arm that is perfect on round 1 has no growth to show.
    # Fraction of Track-C tasks that show the parasitic signature
    # (volume up, score flat/down) for a PARASITIC_LASING verdict.
    parasitic_signature_threshold: float = 0.60
    # Minimum blind gain over baseline, alongside a paired bootstrap 95%
    # CI that excludes zero. A tiny but statistically clean gain is not a
    # strong enough result to justify Phase 2.
    min_blind_gain: float = 0.05
    bootstrap_resamples: int = 5000
    bootstrap_seed: int = 0

    # --- Model / provider ------------------------------------------------
    # provider: "anthropic" | "openai" | "stub"
    provider: str = field(
        default_factory=lambda: os.environ.get("RESONATOR_PROVIDER", "stub")
    )
    # Model string is provider-specific. Anthropic has a current default;
    # OpenAI must be set explicitly because model names move faster than
    # this harness. Override either via RESONATOR_MODEL.
    model: str = field(
        default_factory=lambda: os.environ.get("RESONATOR_MODEL", "")
    )

    # Anthropic request controls. Haiku 5.5 runs adaptive thinking by
    # default and only accepts the default sampling. For a JSON-output task
    # we want the token budget on the answer itself, so default to low
    # effort with thinking off. (Thinking cannot be disabled above "high".)
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    disable_thinking: bool = True
    max_tokens: int = 4096

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
            "anthropic": "claude-haiku-5-5",
            # No hardcoded OpenAI default in 2026. Require RESONATOR_MODEL.
            "openai": "",
            "stub": "stub-deterministic",
        }.get(self.provider, "stub-deterministic")

    def require_model(self) -> str:
        """Return a non-empty model ID, or raise with an operator-facing fix."""
        model = self.default_model()
        if not model:
            raise RuntimeError(
                f"No model configured for provider={self.provider!r}. "
                "Set RESONATOR_MODEL to a current model ID (for example "
                "claude-haiku-5-5 for Anthropic), then rerun."
            )
        return model

    def task_seeds(self) -> list[int]:
        return list(range(self.seed_base, self.seed_base + self.n_tasks))


# Rough, dependency-free token estimate. The sparsity guard is RATIO-based,
# so the exact constant cancels out — only relative size matters. ~4 chars
# per token is the standard heuristic.
def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)
