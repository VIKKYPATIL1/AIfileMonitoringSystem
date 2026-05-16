from __future__ import annotations

import json
from pathlib import Path

import pytest

from aifilemonitoring.rules import RuleConfigurationError, SafeExpressionEvaluator, load_rules


def test_load_rules_validates_llm_rule_file_shape(tmp_path: Path) -> None:
    rule_file = tmp_path / "rules.json"
    rule_file.write_text(
        json.dumps(
            {
                "version": "test",
                "columns": {"symbol": {"type": "string", "required": True}},
                "combinations": [{"name": "known_symbol", "expression": "symbol in ['CL', 'ES']"}],
            }
        ),
        encoding="utf-8",
    )

    rules = load_rules(rule_file)

    assert rules.version == "test"
    assert rules.columns["symbol"]["required"] is True
    assert rules.combinations[0]["name"] == "known_symbol"


def test_safe_expression_rejects_function_calls() -> None:
    with pytest.raises(RuleConfigurationError, match="Unsupported expression element"):
        SafeExpressionEvaluator.validate("__import__('os').system('echo unsafe')")
