"""
LLM client abstraction with a hard budget guard and usage accounting.

Three providers:
- AnthropicClient   (Claude Haiku 5.5 by default)
- OpenAIClient      (model must be set via RESONATOR_MODEL)
- StubClient        (deterministic, no network, no cost — for tests/dry-runs)

Design rules:
- API keys are read from the environment ONLY. Never passed in code, never
  logged, never serialized into artifacts.
- Every client shares a BudgetGuard. The guard hard-stops the run if the
  call cap is exceeded, so a prompt-loop bug cannot run up a bill.
- Every call records token usage and stop_reason in a shared UsageLedger.
- Sampling parameters are NOT sent. API version 2026-10-21 rejects
  temperature / top_p / top_k on every model, so the provider default
  applies. A non-end_turn stop reason (max_tokens, refusal) is recorded,
  not hidden: it means the answer may be incomplete.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from resonator.config import MODEL_PRICING, ExperimentConfig, approx_tokens


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


@dataclass(frozen=True)
class Completion:
    """One model response plus the accounting the run needs."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""
    model: str = ""


@dataclass
class UsageLedger:
    """Aggregate token usage and stop reasons across a run."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reasons: dict[str, int] = field(default_factory=dict)
    by_model: dict[str, int] = field(default_factory=dict)

    def record(self, completion: Completion) -> None:
        self.calls += 1
        self.input_tokens += completion.input_tokens
        self.output_tokens += completion.output_tokens
        reason = completion.stop_reason or "unknown"
        self.stop_reasons[reason] = self.stop_reasons.get(reason, 0) + 1
        if completion.model:
            self.by_model[completion.model] = (
                self.by_model.get(completion.model, 0) + 1
            )

    @property
    def max_tokens_rate(self) -> float:
        """Fraction of calls that hit the output cap."""
        return (
            self.stop_reasons.get("max_tokens", 0) / self.calls
            if self.calls
            else 0.0
        )

    def estimated_cost_usd(self) -> float | None:
        """Estimate cost only when one known model served the whole run."""
        if len(self.by_model) != 1:
            return None
        model = next(iter(self.by_model))
        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            # Tolerate dated snapshots such as claude-haiku-5-5-20260101.
            for known, known_pricing in MODEL_PRICING.items():
                if model.startswith(known):
                    pricing = known_pricing
                    break
        if pricing is None:
            return None
        input_per_m, output_per_m = pricing
        return (
            self.input_tokens * input_per_m
            + self.output_tokens * output_per_m
        ) / 1_000_000

    def summary(self) -> dict[str, object]:
        cost = self.estimated_cost_usd()
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "stop_reasons": dict(self.stop_reasons),
            "by_model": dict(self.by_model),
            "estimated_cost_usd": None if cost is None else round(cost, 6),
        }


class LLMClient(ABC):
    def __init__(
        self,
        cfg: ExperimentConfig,
        guard: BudgetGuard,
        ledger: UsageLedger | None = None,
    ) -> None:
        self.cfg = cfg
        self.guard = guard
        self.ledger = ledger if ledger is not None else UsageLedger()
        self.model = cfg.require_model()

    @abstractmethod
    def _complete(self, prompt: str) -> Completion: ...

    def complete(self, prompt: str) -> Completion:
        """Charge the budget, complete once, record usage. Charge first."""
        self.guard.charge()
        result = self._complete(prompt)
        self.ledger.record(result)
        return result


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

    def _complete(self, prompt: str) -> Completion:
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
        text = json.dumps(out)
        return Completion(
            text=text,
            input_tokens=approx_tokens(prompt),
            output_tokens=approx_tokens(text),
            stop_reason="end_turn",
            model=self.model,
        )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class AnthropicClient(LLMClient):
    def __init__(
        self,
        cfg: ExperimentConfig,
        guard: BudgetGuard,
        ledger: UsageLedger | None = None,
    ) -> None:
        super().__init__(cfg, guard, ledger)
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
                "anthropic package not installed. Run: "
                'pip install -e ".[cloud]"'
            ) from e
        self._client = anthropic.Anthropic(api_key=key)

    def _complete(self, prompt: str) -> Completion:
        request: dict[str, object] = {
            "model": self.model,
            "max_tokens": self.cfg.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            # Effort is the current quality/cost control; there is no
            # sampling-temperature knob on API 2026-10-21.
            "output_config": {"effort": self.cfg.effort},
        }
        # Haiku 5.5 runs adaptive thinking by default. For a JSON task we
        # want the token budget on the answer. "disabled" is accepted at
        # effort high or below; above that, leave thinking on.
        if self.cfg.disable_thinking and self.cfg.effort in (
            "low",
            "medium",
            "high",
        ):
            request["thinking"] = {"type": "disabled"}

        msg = self._client.messages.create(**request)  # type: ignore[arg-type]
        # Concatenate text blocks defensively (response may be multi-block).
        text = "".join(
            block.text
            for block in msg.content
            if getattr(block, "type", None) == "text"
        )
        usage = getattr(msg, "usage", None)
        return Completion(
            text=text,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            stop_reason=str(getattr(msg, "stop_reason", "") or ""),
            model=str(getattr(msg, "model", self.model) or self.model),
        )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIClient(LLMClient):
    def __init__(
        self,
        cfg: ExperimentConfig,
        guard: BudgetGuard,
        ledger: UsageLedger | None = None,
    ) -> None:
        super().__init__(cfg, guard, ledger)
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
                "openai package not installed. Run: "
                'pip install -e ".[openai]"'
            ) from e
        self._client = OpenAI(api_key=key)

    def _complete(self, prompt: str) -> Completion:
        # NOTE: this path is retained for cross-provider replication and is
        # not exercised by the stub tests. Newer OpenAI models may require
        # max_completion_tokens instead of max_tokens; set RESONATOR_MODEL
        # and verify one response before trusting a full run.
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.cfg.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        finish_reason = str(getattr(choice, "finish_reason", "") or "")
        # Normalize OpenAI's truncation reason to the Anthropic spelling so
        # the verdict's max_tokens gate works across providers.
        if finish_reason == "length":
            finish_reason = "max_tokens"
        return Completion(
            text=choice.message.content or "",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            stop_reason=finish_reason,
            model=str(getattr(resp, "model", self.model) or self.model),
        )


def make_client(
    cfg: ExperimentConfig,
    guard: BudgetGuard,
    ledger: UsageLedger | None = None,
) -> LLMClient:
    client_cls = {
        "anthropic": AnthropicClient,
        "openai": OpenAIClient,
        "stub": StubClient,
    }.get(cfg.provider, StubClient)
    return client_cls(cfg, guard, ledger)
