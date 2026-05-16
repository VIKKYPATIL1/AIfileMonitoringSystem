from __future__ import annotations

from typing import Any

from aifilemonitoring.ai_validation import AgenticAIValidator
from aifilemonitoring.models import RuleSet
from aifilemonitoring.rules import RuleEngine


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


class LocalOnlyAgenticAIValidator(AgenticAIValidator):
    def _langgraph_available(self) -> bool:
        return False


def test_agentic_ai_validator_reconciles_ai_and_guardrail_results() -> None:
    rules = RuleSet(
        version="test",
        columns={
            "trade_id": {"type": "string", "required": True},
            "counterparty": {"type": "string", "required": True},
            "quantity": {"type": "integer", "required": True, "min": 1},
        },
    )
    engine = RuleEngine(rules)
    validator = LocalOnlyAgenticAIValidator(
        rules,
        engine,
        FakeOpenAICompatibleClient(),  # type: ignore[arg-type]
        max_workers=1,
        chunk_size=10,
        mode="assistive",
        fail_closed=True,
    )

    results = validator.validate(
        [
            (2, {"trade_id": "T1", "counterparty": "ABC", "quantity": "1"}),
            (3, {"trade_id": "T2", "counterparty": "NEWCO", "quantity": "5"}),
        ]
    )

    assert results[0].is_valid
    assert not results[1].is_valid
    assert results[1].errors[0].rule == "ai_pattern_check"


def test_agentic_ai_validator_fails_closed_without_client() -> None:
    rules = RuleSet(version="test", columns={"trade_id": {"type": "string", "required": True}})
    validator = LocalOnlyAgenticAIValidator(
        rules,
        RuleEngine(rules),
        None,
        max_workers=1,
        chunk_size=10,
        mode="assistive",
        fail_closed=True,
    )

    results = validator.validate([(2, {"trade_id": "T1"})])

    assert not results[0].is_valid
    assert results[0].errors[0].rule == "ai_client_unavailable"
