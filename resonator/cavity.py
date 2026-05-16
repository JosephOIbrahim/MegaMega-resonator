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

Sparsity guard
--------------
For rounds t > 1, a response r_t whose size is not strictly smaller than
ratio * size(S_{t-1}) is REJECTED and the state is frozen for that round.
This defeats the "Full-File Cheat" false positive: a model that ignores
the sparse-delta instruction and regenerates the whole file every round
would bypass the cavity entirely and we'd be measuring prompt-chaining,
not cavity gain. Trigger rate is tracked; >30% across B/C invalidates the
whole run (Round 5, invalidation observable).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from resonator.config import approx_tokens

# A cavity state is node -> {attr: value}. Same shape as megamega StateDict.
CavityState = dict[str, dict[str, object]]

VALID_ATTRS = ("region", "role", "capacity")


def parse_response(raw: str) -> CavityState:
    """
    Parse a model response into a CavityState, defensively.

    Accepts a JSON object of {node: {attr: value}}. Anything malformed,
    any unknown attr, any non-dict node body is dropped silently. A
    response that doesn't parse at all yields an empty delta — the round
    contributes nothing rather than crashing the run.
    """
    raw = raw.strip()
    # Tolerate code fences and prose around the JSON.
    if "```" in raw:
        # take the largest brace-delimited span
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start : end + 1]
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # last-ditch: largest {...} span
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            obj = json.loads(raw[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return {}

    if not isinstance(obj, dict):
        return {}

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
    return cleaned


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


@dataclass
class SparsityGuard:
    """Tracks rejections of over-large (non-sparse) responses."""

    ratio: float
    checks: int = 0
    triggers: int = 0
    per_task_triggered: list[bool] = field(default_factory=list)

    def admit(self, raw_response: str, prior_state: CavityState) -> bool:
        """
        Decide whether r_t may enter the cavity. Only enforced for t > 1
        (caller passes prior_state empty for round 1 to skip the check).

        Returns True if the response is admitted, False if rejected for
        being non-sparse (caller must freeze state for that round).
        """
        if not prior_state:
            return True  # round 1 — nothing to be sparse relative to
        self.checks += 1
        r_tok = approx_tokens(raw_response)
        s_tok = approx_tokens(json.dumps(prior_state))
        admitted = r_tok < self.ratio * s_tok
        if not admitted:
            self.triggers += 1
        return admitted

    @property
    def trigger_rate(self) -> float:
        return self.triggers / self.checks if self.checks else 0.0
