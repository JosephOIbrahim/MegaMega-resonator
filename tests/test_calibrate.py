"""Tests for the calibration pre-check (Mile 6 prep)."""

from __future__ import annotations

import json

import pytest

from megamega.bench import generate_task
from resonator.calibrate import run_calibration
from resonator.config import ExperimentConfig
from resonator.llm import BudgetGuard, Completion, LLMClient, UsageLedger


class ScriptedClient(LLMClient):
    """Deterministic client that returns scripted responses in call order."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        guard: BudgetGuard,
        ledger: UsageLedger,
        script: list[str],
    ) -> None:
        super().__init__(cfg, guard, ledger)
        self.script = list(script)
        self.call_count = 0

    def _complete(self, prompt: str) -> Completion:
        text = self.script[self.call_count]
        self.call_count += 1
        return Completion(
            text=text,
            input_tokens=10,
            output_tokens=5,
            stop_reason="end_turn",
            model=self.model,
        )


def _cfg() -> ExperimentConfig:
    return ExperimentConfig(provider="stub")


def _gt_json(seed: int) -> str:
    task = generate_task(
        seed=seed,
        mu=1.0,
        n_entities=8,
        m_constraints=24,
        strict_additive=True,
    )
    return json.dumps(
        {
            node: dict(state.model_dump())
            for node, state in task.ground_truth.items()
        }
    )


def _client(script: list[str]) -> tuple[ScriptedClient, UsageLedger]:
    cfg = _cfg()
    guard = BudgetGuard(max_calls=len(script))
    ledger = UsageLedger()
    return ScriptedClient(cfg, guard, ledger, script), ledger


def test_first_shot_and_best_of_n_math() -> None:
    # Task 1: miss then hit. Task 2: miss then miss.
    script = [
        "{}",
        _gt_json(1000),
        "{}",
        "{}",
    ]
    client, ledger = _client(script)
    result = run_calibration(
        _cfg(),
        n_tasks=2,
        samples_per_task=2,
        target_b=0.75,
        target_first_shot_success=0.50,
        client=client,
        meaningful=True,
        write_artifact=False,
    )
    assert result.first_shot_mean == 0.0
    assert result.best_of_n_mean == pytest.approx(0.5)
    assert result.first_shot_success_rate == 0.0
    assert result.best_of_n_success_rate == pytest.approx(0.5)
    assert result.total_calls == 4
    assert ledger.calls == 4
    assert result.passed is True


def test_aborts_when_b_is_above_target() -> None:
    script = [_gt_json(1000), _gt_json(1000)]
    client, _ = _client(script)
    result = run_calibration(
        _cfg(),
        n_tasks=1,
        samples_per_task=2,
        target_b=0.50,
        client=client,
        meaningful=True,
        write_artifact=False,
    )
    assert result.best_of_n_mean == pytest.approx(1.0)
    assert result.passed_b is False
    assert result.passed is False
    assert "Too easy" in result.recommendation


def test_parse_failures_are_counted_and_invalidate() -> None:
    script = ["not json", _gt_json(1000)]
    client, _ = _client(script)
    result = run_calibration(
        _cfg(),
        n_tasks=1,
        samples_per_task=2,
        target_b=0.50,
        client=client,
        meaningful=True,
        write_artifact=False,
    )
    assert result.parse_failures == 1
    assert result.parse_checks == 2
    assert result.parse_failure_rate == pytest.approx(0.5)
    assert result.passed_parse is False
    assert result.passed is False
    assert "parse failure" in result.recommendation


def test_stub_mode_is_a_wiring_check_not_a_difficulty_estimate() -> None:
    result = run_calibration(
        ExperimentConfig(provider="stub"),
        n_tasks=2,
        samples_per_task=2,
        write_artifact=False,
    )
    assert result.meaningful is False
    assert result.total_calls == 4
    assert "Wiring check" in result.recommendation


def test_writes_calibration_artifact(tmp_path) -> None:
    script = ["{}", "{}", "{}", "{}"]
    client, _ = _client(script)
    result = run_calibration(
        _cfg(),
        n_tasks=2,
        samples_per_task=2,
        target_b=0.50,
        client=client,
        meaningful=True,
        write_artifact=True,
        artifacts_dir=str(tmp_path),
    )
    assert result.artifact_path is not None
    with open(result.artifact_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["result"]["n_tasks"] == 2
    assert data["result"]["best_of_n_mean"] == 0.0


def test_rejects_invalid_counts_and_targets() -> None:
    with pytest.raises(ValueError, match="n_tasks"):
        run_calibration(_cfg(), n_tasks=0, write_artifact=False)
    with pytest.raises(ValueError, match="samples_per_task"):
        run_calibration(
            _cfg(), n_tasks=1, samples_per_task=0, write_artifact=False
        )
    with pytest.raises(ValueError, match="target_b"):
        run_calibration(
            _cfg(), n_tasks=1, samples_per_task=1,
            target_b=1.5, write_artifact=False,
        )
