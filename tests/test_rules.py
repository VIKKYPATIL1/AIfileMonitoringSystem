from __future__ import annotations

from aifilemonitoring.models import RuleSet
from aifilemonitoring.rules import RuleConfigurationError, RuleEngine, SafeExpressionEvaluator


def test_rule_engine_validates_columns_and_combinations() -> None:
    rules = RuleSet(
        version="test",
        columns={
            "symbol": {"type": "string", "required": True, "allowed": ["CL", "ES"]},
            "quantity": {"type": "integer", "required": True, "min": 1},
            "price": {"type": "decimal", "required": True, "max": "500"},
        },
        combinations=[{"name": "cl_limit", "expression": "symbol != 'CL' or price <= 100", "reason": "CL too high"}],
    )
    engine = RuleEngine(rules)

    valid = engine.validate_row(2, {"symbol": "ES", "quantity": "10", "price": "200"})
    invalid = engine.validate_row(3, {"symbol": "CL", "quantity": "0", "price": "200"})

    assert valid.is_valid
    assert not invalid.is_valid
    assert {error.rule for error in invalid.errors} == {"min", "combination"}


def test_safe_expression_rejects_function_calls() -> None:
    try:
        SafeExpressionEvaluator.validate("__import__('os').system('echo unsafe')")
    except RuleConfigurationError as exc:
        assert "Unsupported expression element" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Unsafe expression was accepted")
