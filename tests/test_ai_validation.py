from __future__ import annotations

from typing import Any, Callable

import pytest

from aifilemonitoring.ai_validation import AgenticAIValidator
from aifilemonitoring.models import RuleSet


class FakeSend:
    def __init__(self, node: str, state: dict[str, Any]):
        self.node = node
        self.state = state


class FakeStateGraph:
    def __init__(self, _state_type: Any):
        self.nodes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}
        self.router: Callable[[dict[str, Any]], list[FakeSend]] | None = None

    def add_node(self, name: str, func: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.nodes[name] = func

    def add_edge(self, *_args: Any) -> None:
        return None

    def add_conditional_edges(self, _name: str, router: Callable[[dict[str, Any]], list[FakeSend]]) -> None:
        self.router = router

    def compile(self) -> FakeStateGraph:
        return self

    def invoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        state.update(self.nodes["rule_interpreter_agent"](state))
        state.update(self.nodes["chunk_planner_agent"](state))
        assert self.router is not None
        for send in self.router(state):
            result = self.nodes[send.node](send.state)
            state["chunk_results"] = state.get("chunk_results", []) + result.get("chunk_results", [])
            state["errors"] = state.get("errors", []) + result.get("errors", [])
        state.update(self.nodes["supervisor_agent"](state))
        return state


class FakeOpenAICompatibleClient:
    def complete_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        if "Normalize these JSON rules" in system_prompt:
            return {"rule_summary": "same rules", "column_rules": user_payload["rules"]["columns"]}
        return {
            "rows": [
                {"row_number": 2, "is_valid": True, "errors": []},
                {
                    "row_number": 3,
                    "is_valid": False,
                    "errors": [
                        {
                            "column": "counterparty",
                            "rule": "ai_pattern_check",
                            "value": "NEWCO",
                            "reason": "Counterparty does not match the approved onboarding pattern",
                        }
                    ],
                },
            ]
        }


class GraphBackedAgenticAIValidator(AgenticAIValidator):
    def _load_langgraph(self) -> tuple[Any, str, str, Callable[..., Any]]:
        return FakeStateGraph, "__start__", "__end__", FakeSend


def test_agentic_ai_validator_uses_native_graph_map_reduce_and_only_ai_decisions() -> None:
    rules = RuleSet(
        version="test",
        columns={
            "trade_id": {"type": "string", "required": True},
            "counterparty": {"type": "string", "required": True},
            "quantity": {"type": "integer", "required": True, "min": 1},
        },
    )
    validator = GraphBackedAgenticAIValidator(
        rules,
        FakeOpenAICompatibleClient(),  # type: ignore[arg-type]
        max_workers=1,
        chunk_size=10,
        fail_closed=True,
    )

    results = validator.validate(
        [
            (2, {"trade_id": "T1", "counterparty": "ABC", "quantity": "0"}),
            (3, {"trade_id": "T2", "counterparty": "NEWCO", "quantity": "5"}),
        ]
    )

    assert results[0].is_valid
    assert not results[1].is_valid
    assert results[1].errors[0].rule == "ai_pattern_check"


def test_agentic_ai_validator_requires_openai_compatible_client() -> None:
    rules = RuleSet(version="test", columns={"trade_id": {"type": "string", "required": True}})

    with pytest.raises(ValueError, match="AI validation requires"):
        GraphBackedAgenticAIValidator(rules, None, max_workers=1, chunk_size=10, fail_closed=True)
