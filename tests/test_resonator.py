"""
Resonator test suite.

Covers the parts that must be correct BEFORE spending money:
- Cavity is strictly additive and lossless (never overwrites)
- Response parsing is crash-proof on malformed input
- Sparsity guard math and trigger-rate accounting
- Four-way verdict precedence (the decision tree)
- Full pipeline runs end-to-end on the stub with zero network
"""

from __future__ import annotations

import pytest

from megamega.bench import generate_task
from resonator.arms import RoundRecord, TaskResult
from resonator.cavity import (
    SparsityGuard,
    compose,
    parse_response,
)
from resonator.config import ExperimentConfig, approx_tokens
from resonator.verdict import Verdict, decide


# ---------------------------------------------------------------------------
# Cavity: strict-additive, lossless
# ---------------------------------------------------------------------------


class TestCavity:
    def test_add_to_empty(self) -> None:
        s, dropped = compose({}, {"Node_0": {"region": "us"}})
        assert s == {"Node_0": {"region": "us"}}
        assert dropped == 0

    def test_never_overwrites_existing_slot(self) -> None:
        s0 = {"Node_0": {"region": "us"}}
        s1, dropped = compose(s0, {"Node_0": {"region": "eu"}})
        # The prior opinion 'us' MUST survive. 'eu' is dropped.
        assert s1 == {"Node_0": {"region": "us"}}
        assert dropped == 1

    def test_adds_new_attr_to_existing_node(self) -> None:
        s0 = {"Node_0": {"region": "us"}}
        s1, dropped = compose(s0, {"Node_0": {"role": "db"}})
        assert s1 == {"Node_0": {"region": "us", "role": "db"}}
        assert dropped == 0

    def test_input_state_not_mutated(self) -> None:
        s0 = {"Node_0": {"region": "us"}}
        compose(s0, {"Node_0": {"role": "db"}, "Node_1": {"region": "eu"}})
        assert s0 == {"Node_0": {"region": "us"}}  # unchanged

    def test_losslessness_under_many_rounds(self) -> None:
        # Once a correct value is placed, no later round can destroy it.
        s: dict = {}
        s, _ = compose(s, {"Node_0": {"region": "us"}})
        for bad in ("eu", "ap", "eu"):
            s, _ = compose(s, {"Node_0": {"region": bad}})
        assert s["Node_0"]["region"] == "us"  # first opinion preserved


# ---------------------------------------------------------------------------
# Response parsing — must never crash
# ---------------------------------------------------------------------------


class TestParse:
    def test_clean_json(self) -> None:
        assert parse_response('{"Node_0": {"region": "us"}}') == {
            "Node_0": {"region": "us"}
        }

    def test_code_fenced(self) -> None:
        raw = '```json\n{"Node_0": {"role": "db"}}\n```'
        assert parse_response(raw) == {"Node_0": {"role": "db"}}

    def test_prose_around_json(self) -> None:
        raw = 'Sure! Here:\n{"Node_1": {"capacity": 8}}\nHope that helps.'
        assert parse_response(raw) == {"Node_1": {"capacity": 8}}

    def test_garbage_returns_empty(self) -> None:
        assert parse_response("not json at all") == {}

    def test_unknown_attrs_dropped(self) -> None:
        raw = '{"Node_0": {"region": "us", "color": "blue"}}'
        assert parse_response(raw) == {"Node_0": {"region": "us"}}

    def test_null_dropped(self) -> None:
        raw = '{"Node_0": {"region": null, "role": "db"}}'
        assert parse_response(raw) == {"Node_0": {"role": "db"}}

    def test_non_dict_body_skipped(self) -> None:
        raw = '{"Node_0": "broken", "Node_1": {"region": "ap"}}'
        assert parse_response(raw) == {"Node_1": {"region": "ap"}}

    def test_top_level_list_returns_empty(self) -> None:
        assert parse_response('[1, 2, 3]') == {}


# ---------------------------------------------------------------------------
# Sparsity guard
# ---------------------------------------------------------------------------


class TestSparsityGuard:
    def test_round_one_always_admitted(self) -> None:
        g = SparsityGuard(ratio=0.5)
        assert g.admit("anything at all", {}) is True
        assert g.checks == 0  # round 1 not counted

    def test_small_delta_admitted(self) -> None:
        g = SparsityGuard(ratio=0.5)
        big_prior = {f"Node_{i}": {"region": "us"} for i in range(8)}
        assert g.admit('{"Node_0":{"role":"db"}}', big_prior) is True
        assert g.triggers == 0

    def test_full_rewrite_rejected(self) -> None:
        g = SparsityGuard(ratio=0.5)
        prior = {"Node_0": {"region": "us"}}
        huge = '{"Node_0":{"region":"us","role":"db","capacity":8}}' * 10
        assert g.admit(huge, prior) is False
        assert g.triggers == 1

    def test_trigger_rate_accounting(self) -> None:
        g = SparsityGuard(ratio=0.5)
        prior = {"Node_0": {"region": "us"}}
        g.admit("x" * 5, prior)         # tiny -> admit
        g.admit("y" * 9999, prior)      # huge -> trigger
        assert g.checks == 2
        assert g.triggers == 1
        assert g.trigger_rate == 0.5

    def test_approx_tokens_monotonic(self) -> None:
        assert approx_tokens("a" * 4) <= approx_tokens("a" * 400)


# ---------------------------------------------------------------------------
# Verdict precedence — the decision tree
# ---------------------------------------------------------------------------


def _traj(scores: list[float], vols: list[int] | None = None) -> list[RoundRecord]:
    vols = vols or [10] * len(scores)
    return [
        RoundRecord(
            round_index=i + 1,
            score=s,
            state_volume_tokens=v,
            dropped_opinions=0,
            sparsity_rejected=False,
        )
        for i, (s, v) in enumerate(zip(scores, vols))
    ]


def _result(seed: int, b: float, oracle: list[float], blind: list[float],
            blind_vols: list[int] | None = None) -> TaskResult:
    return TaskResult(
        seed=seed,
        baseline_score=b,
        oracle_trajectory=_traj(oracle),
        blind_trajectory=_traj(blind, blind_vols),
    )


class TestVerdict:
    def setup_method(self) -> None:
        self.cfg = ExperimentConfig()

    def test_invalid_overrides_everything(self) -> None:
        # Even a perfect-looking blind result is void if guard tripped >30%.
        results = [_result(i, 0.1, [0.9] * 3, [0.1, 0.5, 0.9]) for i in range(10)]
        rep = decide(results, sparsity_trigger_rate=0.5, cfg=self.cfg)
        assert rep.verdict == Verdict.INVALID

    def test_dead_when_oracle_cannot_beat_baseline(self) -> None:
        results = [_result(i, 0.6, [0.5, 0.5, 0.5], [0.4] * 3) for i in range(10)]
        rep = decide(results, 0.0, self.cfg)
        assert rep.verdict == Verdict.DEAD

    def test_true_lasing(self) -> None:
        # Oracle beats baseline AND blind climbs monotonically past it.
        results = [
            _result(i, 0.2, [0.3, 0.6, 0.9], [0.2, 0.5, 0.8]) for i in range(10)
        ]
        rep = decide(results, 0.0, self.cfg)
        assert rep.verdict == Verdict.TRUE_LASING

    def test_parasitic_lasing(self) -> None:
        # Oracle beats baseline; blind volume climbs but score stalls flat.
        results = [
            _result(i, 0.2, [0.3, 0.6, 0.9], [0.2, 0.2, 0.2],
                    blind_vols=[5, 20, 60])
            for i in range(10)
        ]
        rep = decide(results, 0.0, self.cfg)
        assert rep.verdict == Verdict.PARASITIC_LASING

    def test_oracle_dependent(self) -> None:
        # Oracle beats baseline; blind neither climbs nor shows the clean
        # parasitic signature (volume flat, score flat).
        results = [
            _result(i, 0.4, [0.5, 0.7, 0.9], [0.45, 0.45, 0.45],
                    blind_vols=[10, 10, 10])
            for i in range(10)
        ]
        rep = decide(results, 0.0, self.cfg)
        assert rep.verdict == Verdict.ORACLE_DEPENDENT

    def test_means_computed_correctly(self) -> None:
        results = [
            _result(0, 0.2, [0.0, 0.0, 0.4], [0.0, 0.0, 0.6]),
            _result(1, 0.4, [0.0, 0.0, 0.6], [0.0, 0.0, 0.8]),
        ]
        rep = decide(results, 0.0, self.cfg)
        assert rep.baseline_mean == pytest.approx(0.3)
        assert rep.oracle_mean == pytest.approx(0.5)
        assert rep.blind_mean == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Full pipeline on the stub — no network, no cost
# ---------------------------------------------------------------------------


class TestPipelineStub:
    def test_end_to_end_runs_clean(self, tmp_path) -> None:
        from resonator.run import run

        cfg = ExperimentConfig(
            n_tasks=4,
            n_rounds=3,
            baseline_samples=2,
            provider="stub",
            artifacts_dir=str(tmp_path),
        )
        artifact = run(cfg)
        assert artifact["report"]["n_tasks"] == 4
        assert artifact["report"]["verdict"] in {v.value for v in Verdict}
        # Stub is deliberately weak; it should not falsely declare lasing.
        assert artifact["report"]["verdict"] != Verdict.TRUE_LASING.value
        assert artifact["total_calls"] > 0

    def test_budget_guard_halts(self) -> None:
        from resonator.config import ExperimentConfig
        from resonator.llm import BudgetExceeded, BudgetGuard

        g = BudgetGuard(max_calls=3)
        with pytest.raises(BudgetExceeded):
            for _ in range(5):
                g.charge()

    def test_strict_additive_task_is_all_local(self) -> None:
        task = generate_task(
            seed=1, mu=1.0, strict_additive=True,
            n_entities=8, m_constraints=24,
        )
        assert all(c.kind == "local" for c in task.constraints)
        assert task.strict_additive is True
