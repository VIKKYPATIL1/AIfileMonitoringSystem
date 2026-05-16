from __future__ import annotations

from typing import Any

import pytest

from aifilemonitoring.ai_validation import AgenticAIValidator
from aifilemonitoring.models import RuleSet


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


def test_agentic_ai_validator_uses_only_ai_decisions() -> None:
    rules = RuleSet(
        version="test",
        columns={
            "trade_id": {"type": "string", "required": True},
            "counterparty": {"type": "string", "required": True},
            "quantity": {"type": "integer", "required": True, "min": 1},
        },
    )
    validator = LocalOnlyAgenticAIValidator(
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
        LocalOnlyAgenticAIValidator(rules, None, max_workers=1, chunk_size=10, fail_closed=True)
