from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm import OpenAICompatibleClient
from .models import RuleSet, ValidationError


class AdaptiveRuleAdvisor:
    """Maintains cumulative failure history and asks AI for rule-change proposals."""

    def __init__(
        self,
        rules: RuleSet,
        history_path: Path,
        client: OpenAICompatibleClient | None = None,
    ):
        self.rules = rules
        self.history_path = history_path
        self.client = client

    def observe_file(self, file_name: str, errors: list[ValidationError]) -> dict[str, Any]:
        history = self._load_history()
        now = datetime.now(timezone.utc).isoformat()
        file_patterns = self._summarize_file_patterns(errors)

        history.setdefault("version", 1)
        history.setdefault("files", [])
        history.setdefault("patterns", {})
        history["files"].append(
            {
                "file_name": file_name,
                "observed_at": now,
                "pattern_count": len(file_patterns),
                "error_count": len(errors),
            }
        )

        for pattern_key, summary in file_patterns.items():
            pattern = history["patterns"].setdefault(
                pattern_key,
                {
                    "column": summary["column"],
                    "rule": summary["rule"],
                    "reason": summary["reason"],
                    "files": [],
                    "file_count": 0,
                    "row_failure_count": 0,
                    "sample_values": [],
                    "first_seen_at": now,
                    "last_seen_at": now,
                },
            )
            if file_name not in pattern["files"]:
                pattern["files"].append(file_name)
                pattern["file_count"] = len(pattern["files"])
            pattern["row_failure_count"] += summary["row_failure_count"]
            pattern["last_seen_at"] = now
            for value in summary["sample_values"]:
                if value not in pattern["sample_values"] and len(pattern["sample_values"]) < 10:
                    pattern["sample_values"].append(value)

        self._write_history(history)
        return history

    def suggest(self, file_name: str, errors: list[ValidationError]) -> dict[str, Any]:
        history = self.observe_file(file_name, errors)
        threshold = int(self.rules.adaptive.get("suggestion_threshold", 10))
        current_keys = set(self._summarize_file_patterns(errors))
        candidates = [
            pattern
            for key, pattern in sorted(history["patterns"].items())
            if key in current_keys and int(pattern.get("file_count", 0)) >= threshold
        ]
        if not candidates:
            return {
                "file_name": file_name,
                "threshold_type": "files_with_same_failure_pattern",
                "suggestion_threshold": threshold,
                "suggestions": [],
            }

        payload = {
            "file_name": file_name,
            "threshold_type": "files_with_same_failure_pattern",
            "suggestion_threshold": threshold,
            "current_rules": self._rules_payload(),
            "candidate_patterns": candidates,
        }
        ai_response = self._ask_ai_for_suggestions(payload)
        return {
            **payload,
            "llm_status": "used" if self.client else "not_configured",
            "ai_rule_change_review": ai_response,
            "suggestions": candidates,
        }

    def write_suggestions(self, path: Path, file_name: str, errors: list[ValidationError]) -> None:
        suggestions = self.suggest(file_name, errors)
        if not suggestions["suggestions"]:
            return
        path.write_text(json.dumps(suggestions, indent=2, default=str), encoding="utf-8")

    def _summarize_file_patterns(self, errors: list[ValidationError]) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[ValidationError]] = defaultdict(list)
        for error in errors:
            grouped[self._pattern_key(error)].append(error)

        summaries = {}
        for pattern_key, grouped_errors in grouped.items():
            first = grouped_errors[0]
            value_counts = Counter(error.value for error in grouped_errors)
            summaries[pattern_key] = {
                "column": first.column,
                "rule": first.rule,
                "reason": first.reason,
                "row_failure_count": len(grouped_errors),
                "sample_values": [value for value, _count in value_counts.most_common(5)],
            }
        return summaries

    def _ask_ai_for_suggestions(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.client:
            return {
                "decision": "manual_review_required",
                "reason": "No OpenAI-compatible API client is configured for adaptive rule review.",
            }
        system_prompt = (
            "You are an adaptive data-quality governance agent. Review recurring validation failures across files, "
            "not raw row counts. Decide whether each pattern looks like bad source data, a changed business rule, "
            "or a possible database schema change. Do not weaken controls silently. Return JSON with keys: "
            "executive_summary, pattern_decisions, proposed_rule_changes, proposed_schema_changes, approval_questions."
        )
        return self.client.complete_json(system_prompt, payload)

    def _rules_payload(self) -> dict[str, Any]:
        return {
            "version": self.rules.version,
            "columns": self.rules.columns,
            "combinations": self.rules.combinations,
            "adaptive": self.rules.adaptive,
        }

    def _load_history(self) -> dict[str, Any]:
        if not self.history_path.exists():
            return {"version": 1, "files": [], "patterns": {}}
        return json.loads(self.history_path.read_text(encoding="utf-8"))

    def _write_history(self, history: dict[str, Any]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")

    def _pattern_key(self, error: ValidationError) -> str:
        return "|".join((error.column, error.rule, error.reason))


def errors_to_dicts(errors: list[ValidationError]) -> list[dict[str, Any]]:
    return [asdict(error) for error in errors]
