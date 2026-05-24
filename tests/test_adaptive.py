from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aifilemonitoring.adaptive import AdaptiveRuleAdvisor
from aifilemonitoring.models import RuleSet, ValidationError


class FakeAdaptiveClient:
    def complete_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "executive_summary": "Pattern repeated across files.",
            "pattern_decisions": [{"decision": "changed_business_rule"}],
            "proposed_rule_changes": [],
            "proposed_schema_changes": [],
            "approval_questions": ["Should this rule be changed?"],
        }


def _error(row_number: int) -> ValidationError:
    return ValidationError(
        row_number=row_number,
        column="symbol",
        rule="allowed",
        value="ABC",
        reason="Value is not in allowed list",
    )


def test_adaptive_threshold_counts_files_not_rows(tmp_path: Path) -> None:
    rules = RuleSet(
        version="test",
        columns={"symbol": {"type": "string", "required": True}},
        adaptive={"suggestion_threshold": 2},
    )
    advisor = AdaptiveRuleAdvisor(rules, tmp_path / "adaptive_history.json", FakeAdaptiveClient())  # type: ignore[arg-type]

    first_file = advisor.suggest("day1.csv", [_error(2), _error(3), _error(4)])
    second_file = advisor.suggest("day2.csv", [_error(2)])

    assert first_file["suggestions"] == []
    assert second_file["suggestions"][0]["file_count"] == 2
    assert second_file["suggestions"][0]["row_failure_count"] == 4
    assert second_file["llm_status"] == "used"
    history = json.loads((tmp_path / "adaptive_history.json").read_text(encoding="utf-8"))
    assert history["patterns"]["symbol|allowed|Value is not in allowed list"]["file_count"] == 2
