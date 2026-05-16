"""Procedural benchmark generation for Track A."""

from megamega.bench.constraints import (
    CAPACITIES,
    REGIONS,
    ROLES,
    Constraint,
    CoupledEqualityConstraint,
    CoupledSumConstraint,
    LocalConstraint,
    NodeState,
    StateDict,
    evaluate_constraint,
    score_state,
)
from megamega.bench.generator import LogicTask, generate_batch, generate_task

__all__ = [
    "CAPACITIES",
    "REGIONS",
    "ROLES",
    "Constraint",
    "CoupledEqualityConstraint",
    "CoupledSumConstraint",
    "LocalConstraint",
    "LogicTask",
    "NodeState",
    "StateDict",
    "evaluate_constraint",
    "generate_batch",
    "generate_task",
    "score_state",
]
