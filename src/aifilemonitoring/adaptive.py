from __future__ import annotations

import json
import textwrap
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
        table_rows = self._build_review_table_rows(suggestions)
        table_base_path = self._table_base_path(path)
        csv_path = table_base_path.parent / f"{table_base_path.name}.csv"
        png_path = table_base_path.parent / f"{table_base_path.name}.png"
        csv_status = self._write_review_table_csv(csv_path, table_rows)
        png_status = self._write_review_table_png(png_path, table_rows)
        suggestions["review_table"] = {
            "csv_path": str(csv_path),
            "csv_status": csv_status,
            "png_path": str(png_path),
            "png_status": png_status,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(suggestions, indent=2, default=str), encoding="utf-8")

    def _build_review_table_rows(self, suggestions: dict[str, Any]) -> list[dict[str, str]]:
        ai_review = suggestions.get("ai_rule_change_review", {})
        decisions = ai_review.get("pattern_decisions", []) if isinstance(ai_review, dict) else []
        proposed_rule_changes = ai_review.get("proposed_rule_changes", []) if isinstance(ai_review, dict) else []
        rows = []
        for index, pattern in enumerate(suggestions["suggestions"]):
            decision = decisions[index] if index < len(decisions) and isinstance(decisions[index], dict) else {}
            proposed_change = (
                proposed_rule_changes[index]
                if index < len(proposed_rule_changes) and isinstance(proposed_rule_changes[index], dict)
                else {}
            )
            column = str(pattern["column"])
            rows.append(
                {
                    "Column name": column,
                    "Accepted format": self._accepted_format(column),
                    "Received value from file": ", ".join(str(value) for value in pattern.get("sample_values", [])),
                    "Description": self._table_description(pattern, decision),
                    "New change needed if accepted": self._table_change_needed(pattern, decision, proposed_change),
                }
            )
        return rows

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

    def _accepted_format(self, column: str) -> str:
        rule = self.rules.columns.get(column)
        if not rule:
            return "Cross-column rule or AI-only validation"
        parts = [f"type={rule.get('type', 'string')}"]
        if rule.get("required"):
            parts.append("required")
        if "format" in rule:
            parts.append(f"format={rule['format']}")
        if "allowed" in rule:
            parts.append("allowed=" + ", ".join(str(value) for value in rule["allowed"]))
        if "min" in rule:
            parts.append(f"min={rule['min']}")
        if "max" in rule:
            parts.append(f"max={rule['max']}")
        if "regex" in rule:
            parts.append(f"regex={rule['regex']}")
        if "min_length" in rule:
            parts.append(f"min_length={rule['min_length']}")
        if "max_length" in rule:
            parts.append(f"max_length={rule['max_length']}")
        return "; ".join(parts)

    def _table_description(self, pattern: dict[str, Any], decision: dict[str, Any]) -> str:
        ai_reason = decision.get("reason") or decision.get("explanation")
        if ai_reason:
            return str(ai_reason)
        return (
            f"{pattern['row_failure_count']} rows across {pattern['file_count']} files failed "
            f"rule '{pattern['rule']}': {pattern['reason']}"
        )

    def _table_change_needed(
        self,
        pattern: dict[str, Any],
        decision: dict[str, Any],
        proposed_change: dict[str, Any],
    ) -> str:
        for key in ("new_change_needed_if_accepted", "suggested_change", "change", "description"):
            if proposed_change.get(key):
                return str(proposed_change[key])
        for key in ("recommended_change", "suggested_change", "action"):
            if decision.get(key):
                return str(decision[key])
        return f"Human approval required before changing rule '{pattern['rule']}' for column '{pattern['column']}'."

    def _write_review_table_csv(self, path: Path, rows: list[dict[str, str]]) -> str:
        import csv

        if not rows:
            return "skipped_no_rows"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return "created"

    def _write_review_table_png(self, path: Path, rows: list[dict[str, str]]) -> str:
        if not rows:
            return "skipped_no_rows"
        try:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
        except Exception as exc:
            return f"skipped_matplotlib_unavailable: {exc}"

        path.parent.mkdir(parents=True, exist_ok=True)
        headers = list(rows[0])
        wrapped_rows = [
            [self._wrap_cell(row[header], width=28 if header != "Received value from file" else 18) for header in headers]
            for row in rows
        ]
        fig_height = max(2.5, 1.2 + len(rows) * 0.9)
        fig, ax = plt.subplots(figsize=(16, fig_height))
        ax.axis("off")
        table = ax.table(
            cellText=wrapped_rows,
            colLabels=headers,
            loc="center",
            cellLoc="left",
            colWidths=[0.13, 0.23, 0.16, 0.24, 0.24],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2.2)
        for (row_index, _col_index), cell in table.get_celld().items():
            cell.set_edgecolor("#D1D5DB")
            if row_index == 0:
                cell.set_facecolor("#E5E7EB")
                cell.set_text_props(weight="bold", color="#111827")
            else:
                cell.set_facecolor("#FFFFFF" if row_index % 2 else "#F9FAFB")
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return "created"

    def _wrap_cell(self, value: str, width: int) -> str:
        return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False)) or ""

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

    def _table_base_path(self, suggestion_path: Path) -> Path:
        suffix = ".adaptive_suggestions"
        if suggestion_path.stem.endswith(suffix):
            return suggestion_path.with_name(f"{suggestion_path.stem.removesuffix(suffix)}.adaptive_suggestions_table")
        return suggestion_path.with_name(f"{suggestion_path.stem}_table")


def errors_to_dicts(errors: list[ValidationError]) -> list[dict[str, Any]]:
    return [asdict(error) for error in errors]
