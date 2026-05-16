"""
Procedural generator for the Microservice Resource Allocation logic grid.

Generation strategy
-------------------
1. Sample a guaranteed-valid ground truth (GT) over N entities.
2. Derive constraints from the GT. The GT is therefore a witness that
   the task is satisfiable, removing a class of pathological tasks from
   the experimental distribution.
3. Validate at construction time that the GT satisfies every emitted
   constraint. This catches generator bugs before any cloud call.

The modularity coefficient mu controls the local/coupled split:
    mu = 1.0  ->  fully decomposable (all unary constraints)
    mu = 0.0  ->  fully entangled (all binary constraints)

Determinism
-----------
A given (seed, mu, N, M) tuple produces the same task instance bit-for-bit.
This is required for replay across experimental conditions: condition E
(USD LIVRPS) must run on the same task instances as conditions A-D.
"""

from __future__ import annotations

import random

from pydantic import BaseModel, ConfigDict, model_validator

from megamega.bench.constraints import (
    CAPACITIES,
    REGIONS,
    ROLES,
    Constraint,
    CoupledEqualityConstraint,
    CoupledSumConstraint,
    LocalConstraint,
    NodeState,
    evaluate_constraint,
)

# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------


class LogicTask(BaseModel):
    """A single benchmark task instance."""

    model_config = ConfigDict(frozen=True)

    seed: int
    n_entities: int
    m_constraints: int
    requested_mu: float
    realized_mu: float
    strict_additive: bool = False
    entities: tuple[str, ...]
    constraints: tuple[Constraint, ...]
    ground_truth: dict[str, NodeState]

    @model_validator(mode="after")
    def _strict_additive_is_all_local(self) -> "LogicTask":
        """
        Invariant for the resonator proxy-cavity test (Gemini Round 5,
        Challenge 1 concession): a strict-additive task must be solvable
        purely by ADDING correct (node, attr, value) triples. Any coupled
        constraint can create revision pressure that an append-only cavity
        cannot satisfy, which would falsely spike cavity loss L_t and kill
        the thesis for the wrong reason. So strict-additive tasks are
        all-local by construction, and the validator enforces it.
        """
        if self.strict_additive:
            for c in self.constraints:
                if c.kind != "local":
                    raise ValueError(
                        "strict_additive task contains a non-local constraint "
                        f"({c.kind}); proxy-cavity validity requires all-local."
                    )
        return self

    @model_validator(mode="after")
    def _gt_satisfies_constraints(self) -> "LogicTask":
        """
        Invariant: the ground truth must satisfy every generated constraint.
        If this validator fires, the generator has a bug. We want to know
        before the cloud bill arrives.
        """
        gt_state = {k: dict(v.model_dump()) for k, v in self.ground_truth.items()}
        for c in self.constraints:
            if not evaluate_constraint(c, gt_state):
                raise ValueError(
                    f"Generator produced unsatisfiable task at seed={self.seed}: "
                    f"ground truth violates {c!r}"
                )
        return self


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_task(
    *,
    seed: int,
    mu: float,
    n_entities: int = 8,
    m_constraints: int = 20,
    strict_additive: bool = False,
) -> LogicTask:
    """
    Generate a single logic-grid task.

    Args:
        seed: RNG seed. Same seed -> same task.
        mu: Modularity coefficient in [0.0, 1.0]. Fraction of constraints
            that are local (unary). Remainder are coupled (binary).
            Ignored when strict_additive=True (forced all-local).
        n_entities: Number of nodes. Default 8 per the experimental spec.
        m_constraints: Total constraints. Default 20 per the experimental
            spec.
        strict_additive: When True, every constraint is local (unary) and
            pins one (node, attr) slot to its ground-truth value. The task
            is then solvable purely by ADDING correct triples — never by
            retracting one. Required for the resonator proxy-cavity test:
            an append-only cavity cannot satisfy revision pressure, so
            coupled constraints would falsely inflate cavity loss. This is
            the boundary condition Gemini conceded in Round 5, Challenge 1.

    Returns:
        A frozen, validated LogicTask. The constraints are shuffled so
        their order carries no structural signal.

    Raises:
        ValueError: if arguments are out of range, or if the (N, M, mu)
            combination requires more unique constraints than the domain
            admits.
    """
    if not (0.0 <= mu <= 1.0):
        raise ValueError(f"mu must be in [0.0, 1.0], got {mu}")
    if n_entities < 1:
        raise ValueError(f"n_entities must be >= 1, got {n_entities}")
    if m_constraints < 0:
        raise ValueError(f"m_constraints must be >= 0, got {m_constraints}")

    # Strict-additive forces all-local: no coupled constraints, mu := 1.0.
    effective_mu = 1.0 if strict_additive else mu
    num_local = round(m_constraints * effective_mu)
    num_coupled = m_constraints - num_local

    if num_coupled > 0 and n_entities < 2:
        raise ValueError(
            f"Cannot generate coupled constraints with n_entities={n_entities}"
        )

    rng = random.Random(seed)
    entities = tuple(f"Node_{i}" for i in range(n_entities))

    # --- 1. Ground truth (valid by construction) -----------------------------
    ground_truth: dict[str, NodeState] = {
        e: NodeState(
            region=rng.choice(REGIONS),
            role=rng.choice(ROLES),
            capacity=rng.choice(CAPACITIES),
        )
        for e in entities
    }

    # --- 2. Sample local constraints from GT --------------------------------
    seen: set[tuple] = set()
    constraints: list[Constraint] = []

    local_attempts = 0
    max_local_attempts = max(num_local * 20, 100)
    while sum(1 for c in constraints if c.kind == "local") < num_local:
        local_attempts += 1
        if local_attempts > max_local_attempts:
            raise ValueError(
                f"Constraint domain exhausted: could not generate {num_local} "
                f"unique local constraints from N={n_entities} entities x "
                f"3 attrs = {n_entities * 3} candidates."
            )
        node = rng.choice(entities)
        attr = rng.choice(("region", "role", "capacity"))
        sig = ("local", node, attr)
        if sig in seen:
            continue
        seen.add(sig)
        gt_value = getattr(ground_truth[node], attr)
        constraints.append(LocalConstraint(node=node, attr=attr, value=gt_value))

    # --- 3. Sample coupled constraints from GT ------------------------------
    coupled_attempts = 0
    max_coupled_attempts = max(num_coupled * 20, 100)
    while (
        sum(1 for c in constraints if c.kind in ("coupled_eq", "coupled_sum"))
        < num_coupled
    ):
        coupled_attempts += 1
        if coupled_attempts > max_coupled_attempts:
            raise ValueError(
                f"Constraint domain exhausted: could not generate {num_coupled} "
                f"unique coupled constraints."
            )
        a, b = rng.sample(entities, 2)
        # Canonicalize pair ordering so (Node_3, Node_1) and (Node_1, Node_3)
        # collide on dedup.
        n1, n2 = (a, b) if a < b else (b, a)

        # 50/50 split between coupled-equality and coupled-sum.
        if rng.random() < 0.5:
            # Coupled sum (capacity-only)
            sig = ("coupled_sum", n1, n2)
            if sig in seen:
                continue
            seen.add(sig)
            total = ground_truth[n1].capacity + ground_truth[n2].capacity
            constraints.append(
                CoupledSumConstraint(node1=n1, node2=n2, value=total)
            )
        else:
            # Coupled equality / inequality
            attr = rng.choice(("region", "role", "capacity"))
            sig = ("coupled_eq", n1, n2, attr)
            if sig in seen:
                continue
            seen.add(sig)
            v1 = getattr(ground_truth[n1], attr)
            v2 = getattr(ground_truth[n2], attr)
            op = "==" if v1 == v2 else "!="
            constraints.append(
                CoupledEqualityConstraint(node1=n1, node2=n2, attr=attr, op=op)
            )

    # --- 4. Shuffle so position carries no signal ---------------------------
    rng.shuffle(constraints)

    realized_mu = (num_local / m_constraints) if m_constraints > 0 else 0.0

    return LogicTask(
        seed=seed,
        n_entities=n_entities,
        m_constraints=m_constraints,
        requested_mu=mu,
        realized_mu=realized_mu,
        strict_additive=strict_additive,
        entities=entities,
        constraints=tuple(constraints),
        ground_truth=ground_truth,
    )


def generate_batch(
    *,
    seeds: list[int] | range,
    mu: float,
    n_entities: int = 8,
    m_constraints: int = 20,
    strict_additive: bool = False,
) -> list[LogicTask]:
    """Convenience: generate N independent tasks at a fixed mu."""
    return [
        generate_task(
            seed=s,
            mu=mu,
            n_entities=n_entities,
            m_constraints=m_constraints,
            strict_additive=strict_additive,
        )
        for s in seeds
    ]
