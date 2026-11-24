"""Procedural benchmark generation for Track A."""

from megamega.bench.constraints import (
    CAPACITIES,
    REGIONS,
    ROLES,
    SINGLE_SLOT_KINDS,
    Constraint,
    CoupledEqualityConstraint,
    CoupledSumConstraint,
    LocalConstraint,
    LocalInequalityConstraint,
    LocalRangeConstraint,
    LocalSetConstraint,
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
    "LocalInequalityConstraint",
    "LocalRangeConstraint",
    "LocalSetConstraint",
    "LogicTask",
    "NodeState",
    "SINGLE_SLOT_KINDS",
    "StateDict",
    "evaluate_constraint",
    "generate_batch",
    "generate_task",
    "score_state",
]
