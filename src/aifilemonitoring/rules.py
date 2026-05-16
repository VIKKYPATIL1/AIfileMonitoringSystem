from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .models import RuleSet, ValidatedRow, ValidationError


class RuleConfigurationError(ValueError):
    """Raised when a rule file cannot be loaded safely."""


class SafeExpressionEvaluator(ast.NodeVisitor):
    """Evaluate simple boolean expressions against row values without using eval."""

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

    def __init__(self, values: dict[str, Any]):
        self.values = values

    @classmethod
    def validate(cls, expression: str) -> ast.Expression:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, cls.ALLOWED_NODES):
                raise RuleConfigurationError(
                    f"Unsupported expression element: {node.__class__.__name__}"
                )
        return tree

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_BoolOp(self, node: ast.BoolOp) -> bool:
        values = [bool(self.visit(value)) for value in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
        raise RuleConfigurationError("Unsupported boolean operator")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> bool:
        if isinstance(node.op, ast.Not):
            return not bool(self.visit(node.operand))
        raise RuleConfigurationError("Unsupported unary operator")

    def visit_Compare(self, node: ast.Compare) -> bool:
        left = self.visit(node.left)
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            right = self.visit(comparator)
            if isinstance(operator, ast.Eq):
                ok = left == right
            elif isinstance(operator, ast.NotEq):
                ok = left != right
            elif isinstance(operator, ast.Lt):
                ok = left < right
            elif isinstance(operator, ast.LtE):
                ok = left <= right
            elif isinstance(operator, ast.Gt):
                ok = left > right
            elif isinstance(operator, ast.GtE):
                ok = left >= right
            elif isinstance(operator, ast.In):
                ok = left in right
            elif isinstance(operator, ast.NotIn):
                ok = left not in right
            else:
                raise RuleConfigurationError("Unsupported comparison operator")
            if not ok:
                return False
            left = right
        return True

    def visit_Name(self, node: ast.Name) -> Any:
        return self.values.get(node.id)

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_List(self, node: ast.List) -> list[Any]:
        return [self.visit(element) for element in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> tuple[Any, ...]:
        return tuple(self.visit(element) for element in node.elts)


def load_rules(path: Path) -> RuleSet:
    """Load JSON rules in an LLM-friendly structure."""

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


class RuleEngine:
    """Applies configured column and cross-column rules to CSV rows."""

    def __init__(self, rules: RuleSet):
        self.rules = rules

    def validate_row(self, row_number: int, row: dict[str, str]) -> ValidatedRow:
        errors: list[ValidationError] = []
        typed_values: dict[str, Any] = {}
        for column, rule in self.rules.columns.items():
            raw_value = (row.get(column) or "").strip()
            required = bool(rule.get("required", False))
            nullable = bool(rule.get("nullable", not required))
            if required and raw_value == "":
                errors.append(self._error(row_number, column, "required", raw_value, "Value is required"))
                typed_values[column] = None
                continue
            if raw_value == "" and nullable:
                typed_values[column] = None
                continue
            typed_value, type_error = self._coerce(raw_value, rule)
            if type_error:
                errors.append(self._error(row_number, column, "type", raw_value, type_error))
                typed_values[column] = raw_value
                continue
            typed_values[column] = typed_value
            errors.extend(self._validate_scalar_rules(row_number, column, raw_value, typed_value, rule))

        for combination in self.rules.combinations:
            expression = combination["expression"]
            try:
                ok = bool(SafeExpressionEvaluator(typed_values).visit(SafeExpressionEvaluator.validate(expression)))
            except Exception as exc:  # malformed data can make otherwise safe comparisons fail
                ok = False
                reason = f"Combination rule could not be evaluated: {exc}"
            else:
                reason = combination.get("reason", f"Combination rule failed: {expression}")
            if not ok:
                errors.append(
                    self._error(
                        row_number,
                        combination.get("name", "combination"),
                        "combination",
                        expression,
                        reason,
                    )
                )
        return ValidatedRow(row_number=row_number, data=row, errors=errors)

    def _validate_scalar_rules(
        self, row_number: int, column: str, raw_value: str, typed_value: Any, rule: dict[str, Any]
    ) -> list[ValidationError]:
        errors: list[ValidationError] = []
        if "allowed" in rule and typed_value not in rule["allowed"]:
            errors.append(self._error(row_number, column, "allowed", raw_value, "Value is not in allowed list"))
        if "min" in rule and typed_value < self._same_type(rule["min"], typed_value):
            errors.append(self._error(row_number, column, "min", raw_value, f"Value is less than {rule['min']}"))
        if "max" in rule and typed_value > self._same_type(rule["max"], typed_value):
            errors.append(self._error(row_number, column, "max", raw_value, f"Value is greater than {rule['max']}"))
        if "regex" in rule and not re.fullmatch(str(rule["regex"]), raw_value):
            errors.append(self._error(row_number, column, "regex", raw_value, "Value does not match pattern"))
        if "min_length" in rule and len(raw_value) < int(rule["min_length"]):
            errors.append(self._error(row_number, column, "min_length", raw_value, "Value is too short"))
        if "max_length" in rule and len(raw_value) > int(rule["max_length"]):
            errors.append(self._error(row_number, column, "max_length", raw_value, "Value is too long"))
        return errors

    def _coerce(self, raw_value: str, rule: dict[str, Any]) -> tuple[Any, str | None]:
        data_type = str(rule.get("type", "string")).lower()
        try:
            if data_type == "string":
                return raw_value, None
            if data_type == "integer":
                return int(raw_value), None
            if data_type == "decimal":
                return Decimal(raw_value), None
            if data_type == "date":
                date_format = str(rule.get("format", "%Y-%m-%d"))
                return datetime.strptime(raw_value, date_format).date(), None
            if data_type == "datetime":
                date_format = str(rule.get("format", "%Y-%m-%d %H:%M:%S"))
                return datetime.strptime(raw_value, date_format), None
        except (ValueError, InvalidOperation) as exc:
            return raw_value, f"Cannot parse as {data_type}: {exc}"
        return raw_value, f"Unsupported type '{data_type}'"

    def _same_type(self, expected: Any, typed_value: Any) -> Any:
        if isinstance(typed_value, Decimal):
            return Decimal(str(expected))
        if hasattr(typed_value, "year"):
            return datetime.strptime(str(expected), "%Y-%m-%d").date()
        return type(typed_value)(expected)

    def _error(self, row_number: int, column: str, rule: str, value: str, reason: str) -> ValidationError:
        return ValidationError(row_number=row_number, column=column, rule=rule, value=value, reason=reason)
