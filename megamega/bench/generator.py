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

Strict-additive / non-pinning tasks
-----------------------------------
With strict_additive=True and constraints_per_slot > 1 (or an explicit
constraint_mix), every constraint is a single-slot local constraint:
equality, inequality, set membership, or capacity range. The prompt no
longer states the answer for every slot; the model must infer a value that
satisfies the constraints on that slot. No constraint couples two slots,
so the proxy cavity can still satisfy the task by adding correct triples
without revision.

Determinism
-----------
A given (seed, mu, N, M, constraints_per_slot, constraint_mix) tuple
produces the same task instance bit-for-bit. This is required for replay
across experimental conditions.
"""

from __future__ import annotations

import random

from pydantic import BaseModel, ConfigDict, model_validator

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
    evaluate_constraint,
)

ATTRS: tuple[str, ...] = ("region", "role", "capacity")
DOMAINS: dict[str, tuple] = {
    "region": REGIONS,
    "role": ROLES,
    "capacity": CAPACITIES,
}

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
    def _strict_additive_is_single_slot(self) -> "LogicTask":
        """
        Invariant for the resonator proxy-cavity test (Gemini Round 5,
        Challenge 1 concession): a strict-additive task must be solvable
        purely by ADDING correct (node, attr, value) triples. Any coupled
        constraint can create revision pressure that an append-only cavity
        cannot satisfy, which would falsely spike cavity loss L_t and kill
        the thesis for the wrong reason. So strict-additive tasks use only
        single-slot constraints, and the validator enforces it. Equality,
        inequality, set membership and capacity range all qualify.
        """
        if self.strict_additive:
            for c in self.constraints:
                if c.kind not in SINGLE_SLOT_KINDS:
                    raise ValueError(
                        "strict_additive task contains a non-single-slot "
                        f"constraint ({c.kind}); proxy-cavity validity "
                        "requires single-slot constraints."
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
    constraints_per_slot: int | None = None,
    constraint_mix: dict[str, float] | None = None,
) -> LogicTask:
    """
    Generate a single logic-grid task.

    Args:
        seed: RNG seed. Same seed -> same task.
        mu: Modularity coefficient in [0.0, 1.0]. Fraction of constraints
            that are local (unary). Remainder are coupled (binary).
            Ignored when strict_additive=True (forced single-slot local).
        n_entities: Number of nodes. Default 8 per the experimental spec.
        m_constraints: Total constraints for the legacy path. Default 20
            per the experimental spec. Ignored when constraints_per_slot
            is set and the per-slot path is active.
        strict_additive: When True, every constraint is a single-slot
            local constraint. The task is solvable purely by ADDING
            correct triples, never by retracting one. Required for the
            resonator proxy-cavity test. This is the boundary condition
            Gemini conceded in Round 5, Challenge 1.
        constraints_per_slot: When set (requires strict_additive=True),
            generate this many single-slot constraints per (node, attr)
            slot. Values > 1 (or an explicit constraint_mix) use the
            non-pinning generator. When None, or 1 with no mix, the
            original equality-only path is used.
        constraint_mix: Optional weights over the single-slot kinds
            ("local", "local_ineq", "local_in", "local_range"). Requires
            constraints_per_slot. The default non-pinning mix is 60%
            inequality, 25% set membership, 15% capacity range. Range is
            skipped for non-capacity attributes.

    Returns:
        A frozen, validated LogicTask. The constraints are shuffled so
        their order carries no structural signal.

    Raises:
        ValueError: if arguments are out of range, if the (N, M, mu)
            combination requires more unique constraints than the domain
            admits, or if the per-slot generator exhausts a slot's
            possible constraints.
    """
    if not (0.0 <= mu <= 1.0):
        raise ValueError(f"mu must be in [0.0, 1.0], got {mu}")
    if n_entities < 1:
        raise ValueError(f"n_entities must be >= 1, got {n_entities}")
    if m_constraints < 0:
        raise ValueError(f"m_constraints must be >= 0, got {m_constraints}")
    if constraints_per_slot is not None:
        if constraints_per_slot < 1:
            raise ValueError(
                "constraints_per_slot must be >= 1, "
                f"got {constraints_per_slot}"
            )
        if not strict_additive:
            raise ValueError(
                "constraints_per_slot requires strict_additive=True: the "
                "per-slot generator emits single-slot constraints only."
            )
    if constraint_mix is not None and constraints_per_slot is None:
        raise ValueError(
            "constraint_mix requires constraints_per_slot (the mix only "
            "applies to the per-slot generator)."
        )
    if constraint_mix is not None:
        unknown = set(constraint_mix) - set(SINGLE_SLOT_KINDS)
        if unknown:
            raise ValueError(
                f"Unknown constraint kinds in mix: {sorted(unknown)}"
            )
        if any(weight < 0 for weight in constraint_mix.values()):
            raise ValueError("constraint_mix weights must be non-negative")
        if sum(constraint_mix.values()) <= 0:
            raise ValueError("constraint_mix weights must sum to > 0")

    use_non_pinning = (
        constraints_per_slot is not None
        and (constraints_per_slot > 1 or constraint_mix is not None)
    )
    if use_non_pinning:
        per_slot = constraints_per_slot or 1
        effective_mix = constraint_mix or {
            "local_ineq": 0.60,
            "local_in": 0.25,
            "local_range": 0.15,
        }
    else:
        per_slot = 0
        effective_mix = {}

    # Strict-additive forces single-slot local constraints; mu := 1.0.
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

    seen: set[tuple] = set()
    constraints: list[Constraint] = []

    if use_non_pinning:
        # --- 2a. Single-slot constraints, N per slot ------------------------
        for node in entities:
            for attr in ATTRS:
                gt_value = getattr(ground_truth[node], attr)
                for _ in range(per_slot):
                    constraints.append(
                        _make_non_pinning_constraint(
                            rng,
                            node,
                            attr,
                            gt_value,
                            effective_mix,
                            seen,
                        )
                    )
        m_constraints = len(constraints)
        realized_mu = 1.0
    else:
        # --- 2b. Legacy path: equality locals + optional coupled -----------
        local_attempts = 0
        max_local_attempts = max(num_local * 20, 100)
        while sum(1 for c in constraints if c.kind == "local") < num_local:
            local_attempts += 1
            if local_attempts > max_local_attempts:
                raise ValueError(
                    f"Constraint domain exhausted: could not generate "
                    f"{num_local} unique local constraints from "
                    f"N={n_entities} entities x 3 attrs = "
                    f"{n_entities * 3} candidates."
                )
            node = rng.choice(entities)
            attr = rng.choice(ATTRS)
            sig = ("local", node, attr)
            if sig in seen:
                continue
            seen.add(sig)
            gt_value = getattr(ground_truth[node], attr)
            constraints.append(
                LocalConstraint(node=node, attr=attr, value=gt_value)
            )

        coupled_attempts = 0
        max_coupled_attempts = max(num_coupled * 20, 100)
        while (
            sum(
                1
                for c in constraints
                if c.kind in ("coupled_eq", "coupled_sum")
            )
            < num_coupled
        ):
            coupled_attempts += 1
            if coupled_attempts > max_coupled_attempts:
                raise ValueError(
                    f"Constraint domain exhausted: could not generate "
                    f"{num_coupled} unique coupled constraints."
                )
            a, b = rng.sample(entities, 2)
            # Canonicalize pair ordering so (Node_3, Node_1) and
            # (Node_1, Node_3) collide on dedup.
            n1, n2 = (a, b) if a < b else (b, a)

            if rng.random() < 0.5:
                # Coupled sum (capacity-only)
                sig = ("coupled_sum", n1, n2)
                if sig in seen:
                    continue
                seen.add(sig)
                total = (
                    ground_truth[n1].capacity + ground_truth[n2].capacity
                )
                constraints.append(
                    CoupledSumConstraint(node1=n1, node2=n2, value=total)
                )
            else:
                # Coupled equality / inequality
                attr = rng.choice(ATTRS)
                sig = ("coupled_eq", n1, n2, attr)
                if sig in seen:
                    continue
                seen.add(sig)
                v1 = getattr(ground_truth[n1], attr)
                v2 = getattr(ground_truth[n2], attr)
                op = "==" if v1 == v2 else "!="
                constraints.append(
                    CoupledEqualityConstraint(
                        node1=n1, node2=n2, attr=attr, op=op
                    )
                )

        realized_mu = (
            (num_local / m_constraints) if m_constraints > 0 else 0.0
        )

    # --- 3. Shuffle so position carries no signal ---------------------------
    rng.shuffle(constraints)

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


# ---------------------------------------------------------------------------
# Non-pinning constraint helpers
# ---------------------------------------------------------------------------


def _constraint_signature(c: Constraint) -> tuple:
    if c.kind == "local":
        return ("local", c.node, c.attr)
    if c.kind == "local_ineq":
        return ("local_ineq", c.node, c.attr, c.value)
    if c.kind == "local_in":
        return ("local_in", c.node, c.attr, frozenset(c.values))
    if c.kind == "local_range":
        return ("local_range", c.node, c.attr, c.op, c.value)
    if c.kind == "coupled_eq":
        return ("coupled_eq", c.node1, c.node2, c.attr)
    return ("coupled_sum", c.node1, c.node2)


def _pick_kind(
    rng: random.Random,
    mix: dict[str, float],
    attr: str,
) -> str:
    available = [
        kind
        for kind, weight in mix.items()
        if weight > 0 and (kind != "local_range" or attr == "capacity")
    ]
    weights = [mix[kind] for kind in available]
    total = sum(weights)
    if not available or total <= 0:
        raise ValueError(
            f"No usable constraint kinds for attr={attr!r} with mix={mix}"
        )
    pick = rng.random() * total
    running = 0.0
    for kind, weight in zip(available, weights):
        running += weight
        if pick <= running:
            return kind
    return available[-1]


def _try_constraint(
    rng: random.Random,
    kind: str,
    node: str,
    attr: str,
    gt_value: object,
    seen: set[tuple],
) -> Constraint | None:
    domain = DOMAINS[attr]

    if kind == "local":
        if ("local", node, attr) in seen:
            return None
        c: Constraint = LocalConstraint(node=node, attr=attr, value=gt_value)
    elif kind == "local_ineq":
        candidates = [v for v in domain if v != gt_value]
        rng.shuffle(candidates)
        for value in candidates:
            if ("local_ineq", node, attr, value) not in seen:
                c = LocalInequalityConstraint(
                    node=node, attr=attr, value=value
                )
                break
        else:
            return None
    elif kind == "local_in":
        others = [v for v in domain if v != gt_value]
        rng.shuffle(others)
        for other in others:
            values = tuple(sorted((gt_value, other)))
            if (
                "local_in",
                node,
                attr,
                frozenset(values),
            ) not in seen:
                c = LocalSetConstraint(
                    node=node, attr=attr, values=values
                )
                break
        else:
            return None
    elif kind == "local_range":
        if attr != "capacity":
            return None
        pairs: list[tuple[str, int]] = []
        # >= thresholds must include GT and exclude at least the minimum.
        pairs.extend(
            (">=", threshold)
            for threshold in domain
            if threshold > min(domain) and threshold <= gt_value
        )
        # <= thresholds must include GT and exclude at least the maximum.
        pairs.extend(
            ("<=", threshold)
            for threshold in domain
            if threshold < max(domain) and threshold >= gt_value
        )
        rng.shuffle(pairs)
        for op, threshold in pairs:
            if ("local_range", node, attr, op, threshold) not in seen:
                c = LocalRangeConstraint(
                    node=node, attr=attr, op=op, value=threshold
                )
                break
        else:
            return None
    else:
        raise ValueError(f"Unknown single-slot constraint kind: {kind}")

    seen.add(_constraint_signature(c))
    return c


def _make_non_pinning_constraint(
    rng: random.Random,
    node: str,
    attr: str,
    gt_value: object,
    mix: dict[str, float],
    seen: set[tuple],
    max_attempts: int = 100,
) -> Constraint:
    for _ in range(max_attempts):
        kind = _pick_kind(rng, mix, attr)
        c = _try_constraint(rng, kind, node, attr, gt_value, seen)
        if c is not None:
            return c
    raise ValueError(
        f"Constraint domain exhausted for {node}.{attr} after "
        f"{max_attempts} attempts; reduce constraints_per_slot or widen "
        f"the domain."
    )


def generate_batch(
    *,
    seeds: list[int] | range,
    mu: float,
    n_entities: int = 8,
    m_constraints: int = 20,
    strict_additive: bool = False,
    constraints_per_slot: int | None = None,
    constraint_mix: dict[str, float] | None = None,
) -> list[LogicTask]:
    """Convenience: generate N independent tasks at a fixed mu."""
    return [
        generate_task(
            seed=s,
            mu=mu,
            n_entities=n_entities,
            m_constraints=m_constraints,
            strict_additive=strict_additive,
            constraints_per_slot=constraints_per_slot,
            constraint_mix=constraint_mix,
        )
        for s in seeds
    ]
