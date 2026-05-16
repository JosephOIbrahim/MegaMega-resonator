"""
Tests for the benchmark generator and constraint evaluation.

Coverage targets:
- Determinism (replay safety)
- Ground-truth validity invariant (the most important check)
- Modularity coefficient accuracy
- Edge cases and input validation
- Constraint evaluation semantics, especially missing-or-None
- Scoring continuity in [0, 1]
- Serialization roundtrip via JSON
- Structural properties (no duplicates, no self-coupling)
"""

from __future__ import annotations

import pytest

from megamega.bench import (
    CAPACITIES,
    REGIONS,
    ROLES,
    CoupledEqualityConstraint,
    CoupledSumConstraint,
    LocalConstraint,
    LogicTask,
    NodeState,
    evaluate_constraint,
    generate_batch,
    generate_task,
    score_state,
)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_task(self) -> None:
        t1 = generate_task(seed=42, mu=0.5)
        t2 = generate_task(seed=42, mu=0.5)
        assert t1.ground_truth == t2.ground_truth
        assert t1.constraints == t2.constraints

    def test_same_seed_different_mu_different_task(self) -> None:
        t1 = generate_task(seed=42, mu=0.25)
        t2 = generate_task(seed=42, mu=0.75)
        # GT is sampled before constraint split, so it can match,
        # but constraints must differ in local/coupled mix.
        kinds_1 = [c.kind for c in t1.constraints]
        kinds_2 = [c.kind for c in t2.constraints]
        assert kinds_1.count("local") != kinds_2.count("local")

    def test_different_seed_differs(self) -> None:
        # Probabilistically: 100 task pairs should virtually never all collide.
        same = 0
        for s in range(100):
            t1 = generate_task(seed=s, mu=0.5)
            t2 = generate_task(seed=s + 10_000, mu=0.5)
            if t1.ground_truth == t2.ground_truth and t1.constraints == t2.constraints:
                same += 1
        assert same < 5, "Generator appears to ignore seed"


# ---------------------------------------------------------------------------
# Ground-truth validity (the critical invariant)
# ---------------------------------------------------------------------------


class TestGroundTruthValidity:
    @pytest.mark.parametrize("seed", range(50))
    @pytest.mark.parametrize("mu", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_gt_satisfies_all_constraints(self, seed: int, mu: float) -> None:
        task = generate_task(seed=seed, mu=mu)
        gt_state = {k: dict(v.model_dump()) for k, v in task.ground_truth.items()}
        for c in task.constraints:
            assert evaluate_constraint(c, gt_state), (
                f"GT violates {c!r} at seed={seed}, mu={mu}"
            )

    @pytest.mark.parametrize("seed", range(20))
    @pytest.mark.parametrize("mu", [0.0, 0.5, 1.0])
    def test_gt_scores_one(self, seed: int, mu: float) -> None:
        task = generate_task(seed=seed, mu=mu)
        gt_state = {k: dict(v.model_dump()) for k, v in task.ground_truth.items()}
        assert score_state(gt_state, task.constraints) == 1.0


# ---------------------------------------------------------------------------
# Modularity coefficient accuracy
# ---------------------------------------------------------------------------


class TestModularity:
    @pytest.mark.parametrize(
        "mu,expected_local",
        [(1.0, 20), (0.75, 15), (0.5, 10), (0.25, 5), (0.0, 0)],
    )
    def test_realized_local_count_matches_mu(
        self, mu: float, expected_local: int
    ) -> None:
        task = generate_task(seed=0, mu=mu, m_constraints=20)
        local_count = sum(1 for c in task.constraints if c.kind == "local")
        assert local_count == expected_local

    def test_realized_mu_stored(self) -> None:
        task = generate_task(seed=0, mu=0.5, m_constraints=20)
        assert task.realized_mu == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------


class TestStructure:
    @pytest.mark.parametrize("seed", range(20))
    def test_no_duplicate_constraints(self, seed: int) -> None:
        task = generate_task(seed=seed, mu=0.5)
        # Hash each constraint by its kind + relevant fields.
        sigs = set()
        for c in task.constraints:
            if c.kind == "local":
                sig = ("local", c.node, c.attr)
            elif c.kind == "coupled_eq":
                sig = ("coupled_eq", c.node1, c.node2, c.attr)
            else:  # coupled_sum
                sig = ("coupled_sum", c.node1, c.node2)
            assert sig not in sigs, f"Duplicate constraint signature {sig}"
            sigs.add(sig)

    @pytest.mark.parametrize("seed", range(20))
    def test_no_self_coupling(self, seed: int) -> None:
        task = generate_task(seed=seed, mu=0.0)
        for c in task.constraints:
            if c.kind in ("coupled_eq", "coupled_sum"):
                assert c.node1 != c.node2

    @pytest.mark.parametrize("seed", range(20))
    def test_canonical_pair_ordering(self, seed: int) -> None:
        # Coupled pairs are always (smaller_name, larger_name) for dedup.
        task = generate_task(seed=seed, mu=0.0)
        for c in task.constraints:
            if c.kind in ("coupled_eq", "coupled_sum"):
                assert c.node1 < c.node2

    def test_entity_count(self) -> None:
        task = generate_task(seed=0, mu=0.5, n_entities=8)
        assert len(task.entities) == 8
        assert len(task.ground_truth) == 8

    def test_constraint_count(self) -> None:
        task = generate_task(seed=0, mu=0.5, m_constraints=20)
        assert len(task.constraints) == 20


# ---------------------------------------------------------------------------
# Edge cases & input validation
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_mu_below_zero(self) -> None:
        with pytest.raises(ValueError, match="mu"):
            generate_task(seed=0, mu=-0.1)

    def test_mu_above_one(self) -> None:
        with pytest.raises(ValueError, match="mu"):
            generate_task(seed=0, mu=1.1)

    def test_negative_m(self) -> None:
        with pytest.raises(ValueError, match="m_constraints"):
            generate_task(seed=0, mu=0.5, m_constraints=-1)

    def test_zero_entities(self) -> None:
        with pytest.raises(ValueError, match="n_entities"):
            generate_task(seed=0, mu=1.0, n_entities=0)

    def test_zero_constraints(self) -> None:
        task = generate_task(seed=0, mu=0.5, m_constraints=0)
        assert len(task.constraints) == 0
        # Convention: vacuously satisfied
        assert task.realized_mu == 0.0

    def test_single_entity_with_coupled_fails(self) -> None:
        with pytest.raises(ValueError, match="coupled"):
            generate_task(seed=0, mu=0.0, n_entities=1, m_constraints=5)

    def test_single_entity_all_local_ok(self) -> None:
        # 1 entity * 3 attrs = 3 unique local constraints available.
        task = generate_task(seed=0, mu=1.0, n_entities=1, m_constraints=3)
        assert len(task.constraints) == 3

    def test_constraint_domain_exhaustion(self) -> None:
        # 1 entity * 3 attrs = 3 unique local constraints available;
        # asking for 10 must fail with a clear error.
        with pytest.raises(ValueError, match="domain exhausted"):
            generate_task(seed=0, mu=1.0, n_entities=1, m_constraints=10)


# ---------------------------------------------------------------------------
# Constraint evaluation semantics
# ---------------------------------------------------------------------------


class TestConstraintEvaluation:
    def test_local_correct_passes(self) -> None:
        c = LocalConstraint(node="Node_0", attr="region", value="us")
        assert evaluate_constraint(c, {"Node_0": {"region": "us"}}) is True

    def test_local_wrong_fails(self) -> None:
        c = LocalConstraint(node="Node_0", attr="region", value="us")
        assert evaluate_constraint(c, {"Node_0": {"region": "eu"}}) is False

    def test_local_missing_node_fails(self) -> None:
        c = LocalConstraint(node="Node_0", attr="region", value="us")
        assert evaluate_constraint(c, {}) is False

    def test_local_missing_attr_fails(self) -> None:
        c = LocalConstraint(node="Node_0", attr="region", value="us")
        assert evaluate_constraint(c, {"Node_0": {}}) is False

    def test_local_none_attr_fails(self) -> None:
        c = LocalConstraint(node="Node_0", attr="region", value="us")
        assert evaluate_constraint(c, {"Node_0": {"region": None}}) is False

    def test_coupled_eq_match_passes(self) -> None:
        c = CoupledEqualityConstraint(node1="A", node2="B", attr="region", op="==")
        state = {"A": {"region": "us"}, "B": {"region": "us"}}
        assert evaluate_constraint(c, state) is True

    def test_coupled_eq_mismatch_fails(self) -> None:
        c = CoupledEqualityConstraint(node1="A", node2="B", attr="region", op="==")
        state = {"A": {"region": "us"}, "B": {"region": "eu"}}
        assert evaluate_constraint(c, state) is False

    def test_coupled_neq_mismatch_passes(self) -> None:
        c = CoupledEqualityConstraint(node1="A", node2="B", attr="region", op="!=")
        state = {"A": {"region": "us"}, "B": {"region": "eu"}}
        assert evaluate_constraint(c, state) is True

    def test_coupled_eq_missing_side_fails(self) -> None:
        c = CoupledEqualityConstraint(node1="A", node2="B", attr="region", op="==")
        assert evaluate_constraint(c, {"A": {"region": "us"}}) is False

    def test_coupled_sum_correct_passes(self) -> None:
        c = CoupledSumConstraint(node1="A", node2="B", value=16)
        state = {"A": {"capacity": 8}, "B": {"capacity": 8}}
        assert evaluate_constraint(c, state) is True

    def test_coupled_sum_wrong_fails(self) -> None:
        c = CoupledSumConstraint(node1="A", node2="B", value=16)
        state = {"A": {"capacity": 8}, "B": {"capacity": 4}}
        assert evaluate_constraint(c, state) is False

    def test_coupled_sum_non_int_fails(self) -> None:
        c = CoupledSumConstraint(node1="A", node2="B", value=16)
        # If translator failed to coerce, we must fail-safe, not crash.
        state = {"A": {"capacity": "8"}, "B": {"capacity": 8}}
        assert evaluate_constraint(c, state) is False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def test_empty_constraints_score_one(self) -> None:
        assert score_state({}, []) == 1.0

    def test_all_satisfied(self) -> None:
        c = LocalConstraint(node="N", attr="region", value="us")
        assert score_state({"N": {"region": "us"}}, [c]) == 1.0

    def test_all_failed(self) -> None:
        c = LocalConstraint(node="N", attr="region", value="us")
        assert score_state({"N": {"region": "eu"}}, [c]) == 0.0

    def test_proportional_satisfaction(self) -> None:
        c1 = LocalConstraint(node="N", attr="region", value="us")
        c2 = LocalConstraint(node="N", attr="role", value="db")
        c3 = LocalConstraint(node="N", attr="capacity", value=8)
        # 2 of 3 satisfied
        state = {"N": {"region": "us", "role": "db", "capacity": 4}}
        assert score_state(state, [c1, c2, c3]) == pytest.approx(2 / 3)

    def test_empty_state_against_constraints_scores_zero(self) -> None:
        c = LocalConstraint(node="N", attr="region", value="us")
        assert score_state({}, [c]) == 0.0

    def test_score_bounded(self) -> None:
        # No matter what, score is in [0, 1].
        task = generate_task(seed=7, mu=0.5)
        gt_state = {k: dict(v.model_dump()) for k, v in task.ground_truth.items()}
        s = score_state(gt_state, task.constraints)
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_task_roundtrip_json(self) -> None:
        t1 = generate_task(seed=99, mu=0.5)
        json_str = t1.model_dump_json()
        t2 = LogicTask.model_validate_json(json_str)
        assert t1.ground_truth == t2.ground_truth
        assert t1.constraints == t2.constraints
        assert t1.seed == t2.seed
        assert t1.realized_mu == t2.realized_mu

    def test_constraints_preserve_kind_through_roundtrip(self) -> None:
        # The discriminated-union 'kind' tag must survive serialization.
        t1 = generate_task(seed=99, mu=0.5)
        t2 = LogicTask.model_validate_json(t1.model_dump_json())
        for c1, c2 in zip(t1.constraints, t2.constraints):
            assert c1.kind == c2.kind
            assert type(c1) is type(c2)


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


class TestBatch:
    def test_batch_length(self) -> None:
        tasks = generate_batch(seeds=range(10), mu=0.5)
        assert len(tasks) == 10

    def test_batch_seeds_match(self) -> None:
        tasks = generate_batch(seeds=[1, 2, 3], mu=0.5)
        assert [t.seed for t in tasks] == [1, 2, 3]

    def test_batch_deterministic(self) -> None:
        b1 = generate_batch(seeds=range(5), mu=0.5)
        b2 = generate_batch(seeds=range(5), mu=0.5)
        for t1, t2 in zip(b1, b2):
            assert t1.ground_truth == t2.ground_truth
            assert t1.constraints == t2.constraints


# ---------------------------------------------------------------------------
# Frozen / immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_task_is_frozen(self) -> None:
        task = generate_task(seed=0, mu=0.5)
        with pytest.raises(Exception):
            task.seed = 99  # type: ignore[misc]

    def test_node_state_is_frozen(self) -> None:
        ns = NodeState(region="us", role="web", capacity=8)
        with pytest.raises(Exception):
            ns.region = "eu"  # type: ignore[misc]
