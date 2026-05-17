from __future__ import annotations

import ast
import json
from pathlib import Path

from .models import RuleSet


class RuleConfigurationError(ValueError):
    """Raised when a rule file cannot be loaded safely."""


class SafeExpressionEvaluator:
    """Validates that configured cross-column expressions use a safe, simple syntax."""

    ALLOWED_NODES = (
        ast.Expression,
        ast.BoolOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.List,
        ast.Tuple,
    )

    @classmethod
    def validate(cls, expression: str) -> ast.Expression:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, cls.ALLOWED_NODES):
                raise RuleConfigurationError(
                    f"Unsupported expression element: {node.__class__.__name__}"
                )
        return tree


def load_rules(path: Path) -> RuleSet:
    """Load JSON rules in an LLM-friendly structure and validate rule-file syntax."""

    with path.open("r", encoding="utf-8") as rule_stream:
        raw = json.load(rule_stream)
    columns = raw.get("columns")
    if not isinstance(columns, dict) or not columns:
        raise RuleConfigurationError("Rule file must contain a non-empty 'columns' object")
    combinations = raw.get("combinations", [])
    if not isinstance(combinations, list):
        raise RuleConfigurationError("Rule file 'combinations' must be a list")
    for combination in combinations:
        expression = combination.get("expression")
        if not expression:
            raise RuleConfigurationError("Every combination rule requires an expression")
        SafeExpressionEvaluator.validate(expression)
    return RuleSet(
        version=str(raw.get("version", "1")),
        columns=columns,
        combinations=combinations,
        adaptive=raw.get("adaptive", {}),
    )
