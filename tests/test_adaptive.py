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
            "pattern_decisions": [
                {
                    "decision": "changed_business_rule",
                    "reason": "ABC appears repeatedly and may now be approved.",
                    "recommended_change": "Add ABC to the approved symbol list after owner approval.",
                }
            ],
            "proposed_rule_changes": [{"suggested_change": "Add ABC to symbol.allowed"}],
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


def test_adaptive_suggestions_write_reviewer_table(tmp_path: Path) -> None:
    rules = RuleSet(
        version="test",
        columns={"symbol": {"type": "string", "required": True, "allowed": ["ES", "CL"]}},
        adaptive={"suggestion_threshold": 1},
    )
    advisor = AdaptiveRuleAdvisor(rules, tmp_path / "adaptive_history.json", FakeAdaptiveClient())  # type: ignore[arg-type]
    suggestions_path = tmp_path / "day1.adaptive_suggestions.json"

    advisor.write_suggestions(suggestions_path, "day1.csv", [_error(2)])

    table_path = tmp_path / "day1.adaptive_suggestions_table.csv"
    suggestions = json.loads(suggestions_path.read_text(encoding="utf-8"))
    assert suggestions_path.exists()
    assert table_path.exists()
    assert suggestions["review_table"]["csv_status"] == "created"
    assert suggestions["review_table"]["csv_path"].endswith("day1.adaptive_suggestions_table.csv")
    assert suggestions["review_table"]["png_path"].endswith("day1.adaptive_suggestions_table.png")
    assert suggestions["review_table"]["png_status"]
    table = table_path.read_text(encoding="utf-8")
    assert "Column name,Accepted format,Received value from file,Description,New change needed if accepted" in table
    assert "symbol" in table
    assert "allowed=ES, CL" in table
    assert "ABC" in table
    assert "Add ABC to symbol.allowed" in table
