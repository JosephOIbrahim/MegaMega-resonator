"""
LLM client abstraction with a hard budget guard.

Three providers:
- AnthropicClient   (claude-3-5-haiku class)
- OpenAIClient      (gpt-4o-mini)
- StubClient        (deterministic, no network, no cost — for tests/dry-runs)

Design rules:
- API keys are read from the environment ONLY. Never passed in code, never
  logged, never serialized into artifacts.
- Every client shares a BudgetGuard. The guard hard-stops the run if the
  call cap is exceeded, so a prompt-loop bug cannot run up a bill.
- Clients return plain strings. JSON parsing happens downstream so a
  malformed response degrades gracefully instead of crashing.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod

from resonator.config import ExperimentConfig


class BudgetExceeded(RuntimeError):
    """Raised when the API call cap is hit. Halts the run by design."""


class BudgetGuard:
    """Shared call counter with a hard ceiling."""

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.calls = 0

    def charge(self) -> None:
        self.calls += 1
        if self.calls > self.max_calls:
            raise BudgetExceeded(
                f"API call cap reached ({self.max_calls}). Run halted to "
                f"protect budget. Increase max_api_calls only if intended."
            )


class LLMClient(ABC):
    def __init__(self, cfg: ExperimentConfig, guard: BudgetGuard) -> None:
        self.cfg = cfg
        self.guard = guard
        self.model = cfg.default_model()

    @abstractmethod
    def _complete(self, prompt: str, temperature: float) -> str: ...

    def complete(self, prompt: str, temperature: float) -> str:
        """Charge the budget, then complete. Order matters: charge first."""
        self.guard.charge()
        return self._complete(prompt, temperature)


# ---------------------------------------------------------------------------
# Stub — deterministic, offline, free. Used by tests and dry-runs.
# ---------------------------------------------------------------------------


class StubClient(LLMClient):
    """
    Deterministic pseudo-model. Given a prompt, returns a stable JSON
    object derived from a hash of the prompt. It is intentionally NOT
    good at the task — it emits a small, partially-correct grid so the
    pipeline (cavity, scoring, guard, verdict) can be exercised end to
    end without network or cost. Real signal requires a real provider.
    """

    def _complete(self, prompt: str, temperature: float) -> str:
        h = hashlib.sha256(prompt.encode()).hexdigest()
        # Emit 2-3 nodes with hash-derived attribute guesses.
        regions = ["us", "eu", "ap"]
        roles = ["web", "db", "cache"]
        caps = [2, 4, 8, 16]
        out: dict[str, dict] = {}
        for i in range(3):
            seg = h[i * 6 : i * 6 + 6]
            n = int(seg[:2], 16)
            out[f"Node_{n % self.cfg.n_entities}"] = {
                "region": regions[int(seg[2], 16) % 3],
                "role": roles[int(seg[3], 16) % 3],
                "capacity": caps[int(seg[4], 16) % 4],
            }
        return json.dumps(out)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class AnthropicClient(LLMClient):
    def __init__(self, cfg: ExperimentConfig, guard: BudgetGuard) -> None:
        super().__init__(cfg, guard)
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it in your shell; never "
                "put it in code or the repo."
            )
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from e
        self._client = anthropic.Anthropic(api_key=key)

    def _complete(self, prompt: str, temperature: float) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.cfg.max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Concatenate text blocks defensively (response may be multi-block).
        return "".join(
            b.text for b in msg.content if getattr(b, "type", None) == "text"
        )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIClient(LLMClient):
    def __init__(self, cfg: ExperimentConfig, guard: BudgetGuard) -> None:
        super().__init__(cfg, guard)
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Export it in your shell; never "
                "put it in code or the repo."
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            ) from e
        self._client = OpenAI(api_key=key)

    def _complete(self, prompt: str, temperature: float) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=self.cfg.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


def make_client(cfg: ExperimentConfig, guard: BudgetGuard) -> LLMClient:
    return {
        "anthropic": AnthropicClient,
        "openai": OpenAIClient,
        "stub": StubClient,
    }.get(cfg.provider, StubClient)(cfg, guard)
