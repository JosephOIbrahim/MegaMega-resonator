"""
The proxy cavity — a strict-additive, lossless-by-construction merge.

This is the Python stand-in for the USD/LSA cavity that Gemini conceded
(Round 5, Challenge 1) is information-theoretically sufficient to test the
gain mechanism G_t > 0, PROVIDED tasks are strict-additive (all-local).

Cavity contract
----------------
compose(S, r):
  - For each (node, attr, value) in r:
      - if (node, attr) is NOT already present in S: ADD it
      - if (node, attr) IS already present in S:     SILENTLY DROP it
  - S is never mutated in place; a new dict is returned.

This guarantees L_t = 0: no information already in S is ever destroyed or
overwritten. A wrong value placed in an early round is therefore PERMANENT
— which is exactly the substrate on which parasitic lasing becomes
observable (a locked-in falsehood the cavity faithfully preserves).

Sparsity guard (parse-first)
----------------------------
For rounds t > 1, the guard parses the response FIRST, then compares the
size of the parsed delta to ratio * size(S_{t-1}). Measuring the raw string
would reject a sparse delta wrapped in prose or code fences — a false
positive; parsing first measures the information that would actually enter
the cavity.

A response that does not parse is rejected and counted as a parse failure,
not a sparsity trigger. Parse failures and max_tokens stop reasons are
separate invalidation signals (Mile 5). The max_tokens count lives in the
UsageLedger because it applies to every call, including the baseline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from resonator.config import approx_tokens

# A cavity state is node -> {attr: value}. Same shape as megamega StateDict.
CavityState = dict[str, dict[str, object]]

VALID_ATTRS = ("region", "role", "capacity")


def parse_response_checked(raw: str) -> tuple[CavityState, bool]:
    """
    Parse a model response into a CavityState and report whether parsing
    failed.

    Returns (state, failed). `failed` is True only when no JSON object can
    be recovered. A valid JSON object with no usable attributes returns
    ({}, False): that is an empty opinion set, not a malformed response.
    """
    text = raw.strip()
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    obj: object | None = None
    parsed = False
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        parsed = True
        break

    if not parsed or not isinstance(obj, dict):
        return {}, True

    cleaned: CavityState = {}
    for node, body in obj.items():
        if not isinstance(node, str) or not isinstance(body, dict):
            continue
        attrs: dict[str, object] = {}
        for attr, val in body.items():
            if attr not in VALID_ATTRS:
                continue
            if val is None:
                continue  # null = no opinion, forces fall-through
            attrs[attr] = val
        if attrs:
            cleaned[node] = attrs
    return cleaned, False


def parse_response(raw: str) -> CavityState:
    """Backward-compatible wrapper: return the state, drop the status."""
    state, _ = parse_response_checked(raw)
    return state


def compose(state: CavityState, delta: CavityState) -> tuple[CavityState, int]:
    """
    Strict-additive merge. Returns (new_state, n_dropped).

    n_dropped counts (node, attr) opinions in `delta` that were silently
    discarded because the slot was already occupied in `state`. This is
    the empirical proxy for cavity loss: with a strict-additive cavity it
    represents redundant/conflicting opinions correctly refused, NOT
    information destroyed (L_t stays 0 — the prior opinion is preserved).
    """
    new_state: CavityState = {n: dict(a) for n, a in state.items()}
    dropped = 0
    for node, attrs in delta.items():
        slot = new_state.setdefault(node, {})
        for attr, val in attrs.items():
            if attr in slot:
                dropped += 1  # slot occupied -> silently drop, keep prior
                continue
            slot[attr] = val
    return new_state, dropped


@dataclass(frozen=True)
class Admission:
    """The guard's decision for one response."""

    admitted: bool
    delta: CavityState
    parse_failed: bool
    raw_tokens: int
    delta_tokens: int


@dataclass
class SparsityGuard:
    """
    Tracks the instrumentation signals that can void a run:
    - sparsity rejections (full-file rewrites)
    - parse failures (no recoverable JSON object)

    max_tokens stops are counted in the UsageLedger, not here.
    """

    ratio: float
    checks: int = 0  # sparsity checks (t > 1 with a non-empty prior state)
    triggers: int = 0
    parse_checks: int = 0
    parse_failures: int = 0
    per_task_triggered: list[bool] = field(default_factory=list)

    def admit(
        self,
        raw_response: str,
        prior_state: CavityState,
        round_index: int,
    ) -> Admission:
        """
        Parse r_t, then decide whether it may enter the cavity.

        Parse failures are rejected and counted separately. The sparsity
        check runs only for round_index > 1 with a non-empty prior state.
        The caller must freeze state when `admitted` is False.
        """
        delta, parse_failed = parse_response_checked(raw_response)
        self.parse_checks += 1
        raw_tokens = approx_tokens(raw_response)

        if parse_failed:
            self.parse_failures += 1
            return Admission(
                admitted=False,
                delta={},
                parse_failed=True,
                raw_tokens=raw_tokens,
                delta_tokens=0,
            )

        delta_tokens = approx_tokens(json.dumps(delta, sort_keys=True))

        if round_index <= 1 or not prior_state:
            return Admission(
                admitted=True,
                delta=delta,
                parse_failed=False,
                raw_tokens=raw_tokens,
                delta_tokens=delta_tokens,
            )

        self.checks += 1
        s_tok = approx_tokens(json.dumps(prior_state))
        admitted = delta_tokens < self.ratio * s_tok
        if not admitted:
            self.triggers += 1
        return Admission(
            admitted=admitted,
            delta=delta,
            parse_failed=False,
            raw_tokens=raw_tokens,
            delta_tokens=delta_tokens,
        )

    def record_parse(self, failed: bool) -> None:
        """Record a parse attempt made outside admit() (the baseline arm)."""
        self.parse_checks += 1
        if failed:
            self.parse_failures += 1

    @property
    def trigger_rate(self) -> float:
        return self.triggers / self.checks if self.checks else 0.0

    @property
    def parse_failure_rate(self) -> float:
        return (
            self.parse_failures / self.parse_checks
            if self.parse_checks
            else 0.0
        )
