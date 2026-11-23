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

import json

import pytest

from megamega.bench import generate_task
from resonator.arms import RoundRecord, TaskResult
from resonator.cavity import (
    SparsityGuard,
    compose,
    parse_response,
    parse_response_checked,
)
from resonator.config import ExperimentConfig, approx_tokens
from resonator.llm import BudgetGuard, UsageLedger, make_client
from resonator.verdict import Verdict, decide, paired_bootstrap_ci


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

    def test_checked_distinguishes_empty_from_malformed(self) -> None:
        assert parse_response_checked("{}") == ({}, False)
        assert parse_response_checked("not json at all") == ({}, True)
        assert parse_response_checked('{"Node_0": {"region": "us"}}') == (
            {"Node_0": {"region": "us"}},
            False,
        )


# ---------------------------------------------------------------------------
# Sparsity guard
# ---------------------------------------------------------------------------


class TestSparsityGuard:
    def test_round_one_always_admitted(self) -> None:
        g = SparsityGuard(ratio=0.5)
        a = g.admit('{"Node_0":{"region":"us"}}', {}, round_index=1)
        assert a.admitted is True
        assert a.parse_failed is False
        assert g.checks == 0  # round 1 is not a sparsity check
        assert g.parse_checks == 1
        assert g.parse_failures == 0

    def test_small_delta_admitted(self) -> None:
        g = SparsityGuard(ratio=0.5)
        big_prior = {f"Node_{i}": {"region": "us"} for i in range(8)}
        a = g.admit('{"Node_0":{"role":"db"}}', big_prior, round_index=2)
        assert a.admitted is True
        assert a.delta == {"Node_0": {"role": "db"}}
        assert a.parse_failed is False
        assert g.triggers == 0

    def test_full_rewrite_rejected(self) -> None:
        g = SparsityGuard(ratio=0.5)
        prior = {f"Node_{i}": {"region": "us"} for i in range(8)}
        huge = json.dumps(
            {
                f"Node_{i}": {"region": "us", "role": "db", "capacity": 8}
                for i in range(40)
            }
        )
        a = g.admit(huge, prior, round_index=2)
        assert a.admitted is False
        assert a.parse_failed is False
        assert g.triggers == 1

    def test_verbose_but_sparse_is_admitted(self) -> None:
        # Mile 5: the guard measures the parsed delta, not the raw string.
        g = SparsityGuard(ratio=0.5)
        prior = {f"Node_{i}": {"region": "us"} for i in range(8)}
        raw = (
            "Sure. Here is the delta:\n"
            "```json\n"
            '{"Node_0": {"role": "db"}}\n'
            "```\n"
            "This adds the missing role."
        )
        a = g.admit(raw, prior, round_index=2)
        assert a.admitted is True
        assert a.parse_failed is False
        assert a.delta == {"Node_0": {"role": "db"}}
        assert a.raw_tokens > a.delta_tokens
        assert g.triggers == 0

    def test_malformed_response_is_parse_failure_not_sparsity(self) -> None:
        g = SparsityGuard(ratio=0.5)
        prior = {f"Node_{i}": {"region": "us"} for i in range(8)}
        a = g.admit("not json at all", prior, round_index=2)
        assert a.admitted is False
        assert a.parse_failed is True
        assert g.triggers == 0
        assert g.parse_failures == 1
        assert g.parse_checks == 1
        assert g.checks == 0

    def test_empty_json_object_is_not_a_parse_failure(self) -> None:
        g = SparsityGuard(ratio=0.5)
        prior = {f"Node_{i}": {"region": "us"} for i in range(8)}
        a = g.admit("{}", prior, round_index=2)
        assert a.admitted is True
        assert a.parse_failed is False
        assert a.delta == {}

    def test_trigger_rate_accounting(self) -> None:
        g = SparsityGuard(ratio=0.5)
        prior = {f"Node_{i}": {"region": "us"} for i in range(8)}
        g.admit('{"Node_0":{"role":"db"}}', prior, round_index=2)
        huge = json.dumps(
            {
                f"Node_{i}": {"region": "us", "role": "db", "capacity": 8}
                for i in range(40)
            }
        )
        g.admit(huge, prior, round_index=2)
        assert g.checks == 2
        assert g.triggers == 1
        assert g.trigger_rate == 0.5

    def test_parse_failure_rate(self) -> None:
        g = SparsityGuard(ratio=0.5)
        g.record_parse(True)
        g.record_parse(False)
        g.record_parse(True)
        assert g.parse_failure_rate == pytest.approx(2 / 3)

    def test_approx_tokens_monotonic(self) -> None:
        assert approx_tokens("a" * 4) <= approx_tokens("a" * 400)


# ---------------------------------------------------------------------------
# Usage accounting and model resolution (Mile 4 compatibility)
# ---------------------------------------------------------------------------


class TestUsageAccounting:
    def test_stub_records_usage_and_stop_reason(self) -> None:
        cfg = ExperimentConfig(provider="stub")
        guard = BudgetGuard(max_calls=10)
        ledger = UsageLedger()
        client = make_client(cfg, guard, ledger)
        completion = client.complete("hello")
        assert completion.text
        assert completion.input_tokens > 0
        assert completion.output_tokens > 0
        assert completion.stop_reason == "end_turn"
        assert completion.model == "stub-deterministic"
        assert ledger.calls == 1
        assert ledger.input_tokens == completion.input_tokens
        summary = ledger.summary()
        assert summary["calls"] == 1
        assert summary["stop_reasons"] == {"end_turn": 1}

    def test_config_has_no_sampling_parameters(self) -> None:
        # API 2026-10-21 rejects temperature / top_p / top_k on every model.
        cfg = ExperimentConfig(provider="stub")
        assert not hasattr(cfg, "temperature_baseline")
        assert not hasattr(cfg, "temperature_iterative")
        assert cfg.max_tokens == 4096
        assert cfg.effort == "low"
        assert cfg.disable_thinking is True

    def test_cost_estimate_for_known_model(self) -> None:
        ledger = UsageLedger(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            by_model={"claude-haiku-5-5": 1},
        )
        assert ledger.estimated_cost_usd() == pytest.approx(0.60)

    def test_ledger_max_tokens_rate(self) -> None:
        ledger = UsageLedger(
            calls=4,
            stop_reasons={"max_tokens": 1, "end_turn": 3},
        )
        assert ledger.max_tokens_rate == pytest.approx(0.25)

    def test_anthropic_default_is_current_haiku(self, monkeypatch) -> None:
        monkeypatch.delenv("RESONATOR_MODEL", raising=False)
        cfg = ExperimentConfig(provider="anthropic")
        assert cfg.default_model() == "claude-haiku-5-5"

    def test_openai_requires_an_explicit_model(self, monkeypatch) -> None:
        monkeypatch.delenv("RESONATOR_MODEL", raising=False)
        cfg = ExperimentConfig(provider="openai")
        assert cfg.default_model() == ""
        with pytest.raises(RuntimeError, match="RESONATOR_MODEL"):
            cfg.require_model()


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
        # Oracle beats baseline; blind gain clears the effect floor and CI.
        results = [
            _result(i, 0.2, [0.3, 0.6, 0.9], [0.2, 0.5, 0.8]) for i in range(10)
        ]
        rep = decide(results, 0.0, self.cfg)
        assert rep.verdict == Verdict.TRUE_LASING
        assert rep.blind_gain == pytest.approx(0.6)
        assert rep.blind_gain_ci_low > 0.0

    def test_true_lasing_with_perfect_first_round(self) -> None:
        # No monotonic growth to show, but a perfect blind round is the
        # strongest verifier-free result. It must not be mislabelled.
        results = [
            _result(i, 0.2, [0.5, 0.7, 0.9], [1.0, 1.0, 1.0])
            for i in range(10)
        ]
        rep = decide(results, 0.0, self.cfg)
        assert rep.monotonic_blind_frac == 0.0
        assert rep.verdict == Verdict.TRUE_LASING

    def test_small_gain_is_not_true_lasing(self) -> None:
        # A 2-point gain is below the 5-point floor, even if consistent.
        results = [
            _result(i, 0.40, [0.5, 0.7, 0.9], [0.42, 0.42, 0.42])
            for i in range(10)
        ]
        rep = decide(results, 0.0, self.cfg)
        assert rep.verdict == Verdict.ORACLE_DEPENDENT

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
        # Oracle beats baseline; the blind arm gains only 2 points, below
        # the TRUE_LASING floor, and shows no parasitic signature.
        results = [
            _result(i, 0.40, [0.5, 0.7, 0.9], [0.42, 0.42, 0.42],
                    blind_vols=[10, 10, 10])
            for i in range(10)
        ]
        rep = decide(results, 0.0, self.cfg)
        assert rep.verdict == Verdict.ORACLE_DEPENDENT

    def test_invalid_on_parse_failure_rate(self) -> None:
        results = [_result(i, 0.2, [0.5, 0.7, 0.9], [0.9, 0.9, 0.9])
                   for i in range(10)]
        rep = decide(results, 0.0, self.cfg, parse_failure_rate=0.20)
        assert rep.verdict == Verdict.INVALID
        assert any("parse failure" in r for r in rep.invalidation_reasons)

    def test_invalid_on_max_tokens_rate(self) -> None:
        results = [_result(i, 0.2, [0.5, 0.7, 0.9], [0.9, 0.9, 0.9])
                   for i in range(10)]
        rep = decide(results, 0.0, self.cfg, max_tokens_rate=0.20)
        assert rep.verdict == Verdict.INVALID
        assert any("max_tokens" in r for r in rep.invalidation_reasons)

    def test_invalid_reports_all_reasons(self) -> None:
        results = [_result(i, 0.2, [0.5, 0.7, 0.9], [0.9, 0.9, 0.9])
                   for i in range(10)]
        rep = decide(
            results,
            0.50,
            self.cfg,
            parse_failure_rate=0.20,
            max_tokens_rate=0.20,
        )
        assert rep.verdict == Verdict.INVALID
        assert len(rep.invalidation_reasons) == 3

    def test_bootstrap_ci_constant_gain(self) -> None:
        low, high = paired_bootstrap_ci([0.2] * 20)
        assert low == pytest.approx(0.2)
        assert high == pytest.approx(0.2)

    def test_bootstrap_ci_zero_gain(self) -> None:
        low, high = paired_bootstrap_ci([0.0] * 20)
        assert low == pytest.approx(0.0)
        assert high == pytest.approx(0.0)

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
        # Mile 4: usage and stop reasons are persisted with the run.
        assert artifact["model"] == "stub-deterministic"
        assert artifact["usage"]["calls"] == artifact["total_calls"]
        assert artifact["usage"]["input_tokens"] > 0
        assert artifact["usage"]["output_tokens"] > 0
        assert artifact["usage"]["stop_reasons"] == {
            "end_turn": artifact["total_calls"]
        }
        assert artifact["per_task"][0]["baseline_usage"]["stop_reasons"]
        assert artifact["report"]["parse_failure_rate"] == 0.0
        assert artifact["report"]["max_tokens_rate"] == 0.0
        assert artifact["report"]["invalidation_reasons"]
        assert artifact["per_task"][0]["baseline_usage"]["parse_failures"] == 0

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
