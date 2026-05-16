"""
Task adapter — bridges a megamega LogicTask to the resonator loop.

Responsibilities:
- Render the task as a natural-language prompt the model can answer.
- Compute the DSL diff (which constraints currently fail) for the ORACLE
  arm's feedback. This is the ground-truth-derived signal that Challenge 2
  isolates: the oracle arm gets it, the blind arm does not.
- Score a cavity state with the existing, deterministic megamega scorer.

The strict-additive task is pure completion: every constraint is
"Node_X.attr == <value>". The model's only path to a higher score is
ADDING correct (node, attr, value) triples. No constraint ever requires
changing a value already placed.
"""

from __future__ import annotations

from megamega.bench import LogicTask, evaluate_constraint, score_state
from megamega.bench.constraints import LocalConstraint
from resonator.cavity import CavityState


def render_task_prompt(task: LogicTask) -> str:
    """The base task statement, identical across all three arms."""
    lines = [
        "You are configuring a grid of microservice nodes.",
        f"There are {task.n_entities} nodes: "
        + ", ".join(task.entities)
        + ".",
        "Each node has three attributes:",
        "  - region: one of us, eu, ap",
        "  - role:   one of web, db, cache",
        "  - capacity: one of 2, 4, 8, 16",
        "",
        "The correct configuration must satisfy ALL of these constraints:",
    ]
    for i, c in enumerate(task.constraints, 1):
        assert isinstance(c, LocalConstraint)  # strict-additive => all local
        lines.append(f"  {i}. {c.node}.{c.attr} == {c.value!r}")
    lines += [
        "",
        "Respond with ONLY a JSON object mapping node name to its "
        "attributes, e.g.:",
        '{"Node_0": {"region": "us", "role": "db", "capacity": 8}}',
        "No prose, no code fences. JSON only.",
    ]
    return "\n".join(lines)


def failing_constraints_report(task: LogicTask, state: CavityState) -> str:
    """
    The ORACLE feedback signal: a list of constraints the current state
    fails. Ground-truth-derived (the DSL knows the answer). The blind arm
    never sees this.
    """
    failures: list[str] = []
    # evaluate_constraint expects {node: {attr: value}}; CavityState matches.
    typed_state = {n: dict(a) for n, a in state.items()}
    for c in task.constraints:
        if not evaluate_constraint(c, typed_state):  # type: ignore[arg-type]
            assert isinstance(c, LocalConstraint)
            failures.append(f"{c.node}.{c.attr} must be {c.value!r}")
    if not failures:
        return "(no failing constraints)"
    return "\n".join(f"  - {f}" for f in failures)


def oracle_feedback_prompt(task: LogicTask, state: CavityState) -> str:
    base = render_task_prompt(task)
    state_json = _state_json(state)
    report = failing_constraints_report(task, state)
    return (
        f"{base}\n\n"
        f"CURRENT STATE:\n{state_json}\n\n"
        f"ORACLE REPORT — the current state FAILS these constraints:\n"
        f"{report}\n\n"
        f"Emit ONLY a strictly additive JSON delta that resolves the "
        f"failing constraints. Do NOT restate nodes that are already "
        f"correct. JSON only."
    )


def blind_feedback_prompt(task: LogicTask, state: CavityState) -> str:
    base = render_task_prompt(task)
    state_json = _state_json(state)
    return (
        f"{base}\n\n"
        f"CURRENT STATE:\n{state_json}\n\n"
        f"Emit ONLY a strictly additive JSON delta that completes any "
        f"missing or incorrect constraints. Do NOT restate nodes that are "
        f"already correct. JSON only."
    )


def score(task: LogicTask, state: CavityState) -> float:
    """Proportional constraint satisfaction in [0, 1] via megamega scorer."""
    typed_state = {n: dict(a) for n, a in state.items()}
    return score_state(typed_state, task.constraints)  # type: ignore[arg-type]


def _state_json(state: CavityState) -> str:
    import json

    return json.dumps(state, indent=2, sort_keys=True)
