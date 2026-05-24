from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .llm import OpenAICompatibleClient
from .models import RuleSet, ValidationError


class RejectionAnalyticsReporter:
    """Builds AI-planned visual analytics for accepted and rejected records."""

    ALLOWED_CHARTS = {
        "status_pie",
        "failed_rules_bar",
        "failed_columns_bar",
        "rule_by_column_heatmap",
    }

    def __init__(self, rules: RuleSet, client: OpenAICompatibleClient | None = None):
        self.rules = rules
        self.client = client

    def write_report(
        self,
        output_dir: Path,
        file_stem: str,
        accepted_count: int,
        rejected_count: int,
        errors: list[ValidationError],
    ) -> None:
        if accepted_count == 0 and rejected_count == 0:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = self._build_summary(accepted_count, rejected_count, errors)
        chart_plan = self._ask_ai_for_chart_plan(summary)
        rendered_charts = self._render_charts(output_dir, file_stem, summary, chart_plan)
        report = {
            "file_stem": file_stem,
            "llm_status": "used" if self.client else "not_configured",
            "summary": summary,
            "ai_chart_plan": chart_plan,
            "rendered_charts": [str(path) for path in rendered_charts],
        }
        (output_dir / f"{file_stem}.analytics.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )

    def _build_summary(
        self,
        accepted_count: int,
        rejected_count: int,
        errors: list[ValidationError],
    ) -> dict[str, Any]:
        failed_rules = Counter(error.rule for error in errors)
        failed_columns = Counter(error.column for error in errors)
        rule_by_column = Counter((error.column, error.rule) for error in errors)
        return {
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "total_count": accepted_count + rejected_count,
            "failed_rules": dict(failed_rules.most_common()),
            "failed_columns": dict(failed_columns.most_common()),
            "rule_by_column": [
                {"column": column, "rule": rule, "count": count}
                for (column, rule), count in rule_by_column.most_common()
            ],
            "sample_errors": [asdict(error) for error in errors[:20]],
            "rules": {
                "version": self.rules.version,
                "columns": list(self.rules.columns),
                "combination_rules": [rule.get("name", "combination") for rule in self.rules.combinations],
            },
        }

    def _ask_ai_for_chart_plan(self, summary: dict[str, Any]) -> dict[str, Any]:
        if not self.client:
            return {
                "decision": "fallback_default_charts",
                "reason": "No OpenAI-compatible API client is configured for analytics planning.",
                "charts": [
                    {"type": "status_pie", "title": "Accepted vs Rejected Records"},
                    {"type": "failed_rules_bar", "title": "Failed Rules"},
                    {"type": "failed_columns_bar", "title": "Failed Columns"},
                ],
            }
        system_prompt = (
            "You are a data-quality analytics agent. Choose the most useful matplotlib charts for this validation "
            "run. Pick only chart types from: status_pie, failed_rules_bar, failed_columns_bar, "
            "rule_by_column_heatmap. Return JSON with keys: decision, reason, charts. Each chart must have type "
            "and title. Prefer charts that reveal patterns in accepted records, rejected records, and failed rules."
        )
        response = self.client.complete_json(system_prompt, summary)
        charts = response.get("charts", [])
        response["charts"] = [
            chart for chart in charts if isinstance(chart, dict) and chart.get("type") in self.ALLOWED_CHARTS
        ]
        if not response["charts"]:
            response["charts"] = [{"type": "status_pie", "title": "Accepted vs Rejected Records"}]
        return response

    def _render_charts(
        self,
        output_dir: Path,
        file_stem: str,
        summary: dict[str, Any],
        chart_plan: dict[str, Any],
    ) -> list[Path]:
        try:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
        except Exception:
            return []

        rendered = []
        for index, chart in enumerate(chart_plan.get("charts", []), start=1):
            chart_type = chart.get("type")
            title = str(chart.get("title", chart_type))
            output_path = output_dir / f"{file_stem}.{index}.{chart_type}.png"
            fig, ax = plt.subplots(figsize=(8, 5))
            if chart_type == "status_pie":
                self._draw_status_pie(ax, summary, title)
            elif chart_type == "failed_rules_bar":
                self._draw_bar(ax, summary["failed_rules"], title, "Rule", "Failures")
            elif chart_type == "failed_columns_bar":
                self._draw_bar(ax, summary["failed_columns"], title, "Column", "Failures")
            elif chart_type == "rule_by_column_heatmap":
                self._draw_rule_by_column_heatmap(ax, summary, title)
            else:
                plt.close(fig)
                continue
            fig.tight_layout()
            fig.savefig(output_path, dpi=140)
            plt.close(fig)
            rendered.append(output_path)
        return rendered

    def _draw_status_pie(self, ax: Any, summary: dict[str, Any], title: str) -> None:
        values = [summary["accepted_count"], summary["rejected_count"]]
        labels = ["Accepted", "Rejected"]
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title(title)

    def _draw_bar(self, ax: Any, values: dict[str, int], title: str, xlabel: str, ylabel: str) -> None:
        if not values:
            ax.text(0.5, 0.5, "No failures", ha="center", va="center")
            ax.set_axis_off()
            return
        labels = list(values)
        counts = list(values.values())
        ax.bar(labels, counts)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)

    def _draw_rule_by_column_heatmap(self, ax: Any, summary: dict[str, Any], title: str) -> None:
        entries = summary["rule_by_column"]
        if not entries:
            ax.text(0.5, 0.5, "No failures", ha="center", va="center")
            ax.set_axis_off()
            return
        columns = sorted({entry["column"] for entry in entries})
        rules = sorted({entry["rule"] for entry in entries})
        counts = {(entry["column"], entry["rule"]): entry["count"] for entry in entries}
        matrix = [[counts.get((column, rule), 0) for rule in rules] for column in columns]
        image = ax.imshow(matrix)
        ax.set_xticks(range(len(rules)), labels=rules, rotation=30, ha="right")
        ax.set_yticks(range(len(columns)), labels=columns)
        ax.set_title(title)
        for row_index, column in enumerate(columns):
            for col_index, rule in enumerate(rules):
                ax.text(col_index, row_index, counts.get((column, rule), 0), ha="center", va="center")
        ax.figure.colorbar(image, ax=ax)
