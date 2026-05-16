"""
Constraint primitives for the procedural logic-grid benchmark.

The discriminated union of constraint types is the schema-level
"polarizer" that the USD substrate will eventually enforce. Keeping
constraint *shape* and constraint *evaluation* in one module ensures
the generator's self-validation and the downstream scoring layer
operate on identical semantics.

Constraint kinds
----------------
- LocalConstraint        Unary:  Node.attr == value
- CoupledEqualityConstraint  Binary: Node1.attr OP Node2.attr  (OP in {==, !=})
- CoupledSumConstraint   Binary: Node1.capacity + Node2.capacity == value

Missing-or-None semantics
-------------------------
A constraint evaluates to False when any required value is missing
or None. This is by design: sparse, partial responses must score
lower than complete-and-correct responses. Crucially, this matches
USD's "opinion absence" semantics — a node with no authored attribute
behaves identically to a Python dict with no key.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

REGIONS: tuple[str, ...] = ("us", "eu", "ap")
ROLES: tuple[str, ...] = ("web", "db", "cache")
CAPACITIES: tuple[int, ...] = (2, 4, 8, 16)

Region = Literal["us", "eu", "ap"]
Role = Literal["web", "db", "cache"]
Capacity = Literal[2, 4, 8, 16]
Attr = Literal["region", "role", "capacity"]

# A "state" is a (possibly partial) mapping from node name to attribute dict.
# Attributes that are missing or None are treated identically.
StateDict = dict[str, dict[str, Union[str, int, None]]]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class NodeState(BaseModel):
    """A fully specified node state. Used only for ground-truth instances."""

    model_config = ConfigDict(frozen=True)

    region: Region
    role: Role
    capacity: Capacity


class LocalConstraint(BaseModel):
    """Unary: Node.attr == value."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["local"] = "local"
    node: str
    attr: Attr
    op: Literal["=="] = "=="
    value: Union[str, int]


class CoupledEqualityConstraint(BaseModel):
    """Binary: Node1.attr OP Node2.attr where OP in {==, !=}."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["coupled_eq"] = "coupled_eq"
    node1: str
    node2: str
    attr: Attr
    op: Literal["==", "!="]


class CoupledSumConstraint(BaseModel):
    """Binary: Node1.capacity + Node2.capacity == value."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["coupled_sum"] = "coupled_sum"
    node1: str
    node2: str
    attr: Literal["capacity"] = "capacity"
    op: Literal["=="] = "=="
    value: int


Constraint = Annotated[
    Union[LocalConstraint, CoupledEqualityConstraint, CoupledSumConstraint],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_constraint(c: Constraint, state: StateDict) -> bool:
    """
    Evaluate a constraint against a (possibly partial) state.

    Returns False if any required value is missing or None. This is the
    explicit failure-on-absence semantics that mirrors USD's behavior
    when a sublayer fails to author an opinion for a given attribute.
    """
    if c.kind == "local":
        node_state = state.get(c.node)
        if not node_state:
            return False
        actual = node_state.get(c.attr)
        if actual is None:
            return False
        return actual == c.value

    if c.kind == "coupled_eq":
        s1 = state.get(c.node1)
        s2 = state.get(c.node2)
        if not s1 or not s2:
            return False
        v1 = s1.get(c.attr)
        v2 = s2.get(c.attr)
        if v1 is None or v2 is None:
            return False
        if c.op == "==":
            return v1 == v2
        return v1 != v2

    if c.kind == "coupled_sum":
        s1 = state.get(c.node1)
        s2 = state.get(c.node2)
        if not s1 or not s2:
            return False
        v1 = s1.get("capacity")
        v2 = s2.get("capacity")
        if not isinstance(v1, int) or not isinstance(v2, int):
            return False
        return (v1 + v2) == c.value

    # Exhaustive over discriminated union; pyright/mypy will catch new kinds.
    return False


def score_state(
    state: StateDict,
    constraints: tuple[Constraint, ...] | list[Constraint],
) -> float:
    """
    Proportional constraint satisfaction in [0.0, 1.0].

    score = (# satisfied constraints) / (# total constraints)

    By convention, an empty constraint set scores 1.0 (vacuously satisfied).
    Binary all-or-nothing scoring is rejected per the experimental design:
    statistical power requires the continuous gradient.
    """
    if not constraints:
        return 1.0
    satisfied = sum(1 for c in constraints if evaluate_constraint(c, state))
    return satisfied / len(constraints)
