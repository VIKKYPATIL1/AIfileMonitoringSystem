from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import RuleSet, ValidationError


class AdaptiveRuleAdvisor:
    """Creates human-reviewable rule suggestions from recurring validation failures."""

    def __init__(self, rules: RuleSet):
        self.rules = rules

    def suggest(self, errors: list[ValidationError]) -> dict[str, Any]:
        by_column: dict[str, Counter[str]] = defaultdict(Counter)
        samples: dict[str, list[str]] = defaultdict(list)
        for error in errors:
            by_column[error.column][error.rule] += 1
            if len(samples[error.column]) < 5:
                samples[error.column].append(error.value)

        suggestions = []
        threshold = int(self.rules.adaptive.get("suggestion_threshold", 10))
        for column, counter in by_column.items():
            total = sum(counter.values())
            if total < threshold:
                continue
            top_rule, count = counter.most_common(1)[0]
            suggestions.append(
                {
                    "column": column,
                    "observed_failures": total,
                    "dominant_rule": top_rule,
                    "dominant_rule_failures": count,
                    "sample_values": samples[column],
                    "recommendation": self._recommendation(column, top_rule),
                    "approval_required": True,
                }
            )
        return {"suggestions": suggestions}

    def write_suggestions(self, path: Path, errors: list[ValidationError]) -> None:
        suggestions = self.suggest(errors)
        if not suggestions["suggestions"]:
            return
        path.write_text(json.dumps(suggestions, indent=2, default=str), encoding="utf-8")

    def _recommendation(self, column: str, rule: str) -> str:
        if rule == "allowed":
            return f"Review whether the allowed values for '{column}' need a new business-approved value."
        if rule in {"min", "max"}:
            return f"Review the accepted numeric/date range for '{column}' before changing limits."
        if rule == "regex":
            return f"Review the pattern for '{column}' or ask an LLM to propose a stricter regex."
        if rule == "combination":
            return "Review cross-column business logic; do not auto-change without data-owner approval."
        return f"Review recurring '{rule}' failures for '{column}' and decide whether rules or source data changed."


def errors_to_dicts(errors: list[ValidationError]) -> list[dict[str, Any]]:
    return [asdict(error) for error in errors]
