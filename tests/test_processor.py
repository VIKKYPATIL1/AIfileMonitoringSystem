from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from aifilemonitoring.models import PipelineConfig, ValidatedRow, ValidationError
from aifilemonitoring.processor import FileProcessor


class FakeAIValidator:
    def validate(self, rows: list[tuple[int, dict[str, str]]]) -> list[ValidatedRow]:
        results: list[ValidatedRow] = []
        for row_number, row in rows:
            if row["trade_id"] == "T1":
                results.append(ValidatedRow(row_number=row_number, data=row))
                continue
            results.append(
                ValidatedRow(
                    row_number=row_number,
                    data=row,
                    errors=[
                        ValidationError(
                            row_number=row_number,
                            column="ai_validation",
                            rule="ai_rejected_row",
                            value=row["trade_id"],
                            reason="AI rejected this row",
                        )
                    ],
                )
            )
        return results


def _write_rules(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": "test",
                "columns": {
                    "trade_id": {"type": "string", "required": True},
                    "symbol": {"type": "string", "required": True, "allowed": ["ES", "CL"]},
                    "quantity": {"type": "integer", "required": True, "min": 1},
                    "price": {"type": "decimal", "required": True, "max": "500"},
                },
                "combinations": [
                    {"name": "cl_limit", "expression": "symbol != 'CL' or price <= 100", "reason": "CL too high"}
                ],
                "adaptive": {"suggestion_threshold": 1},
            }
        ),
        encoding="utf-8",
    )


def _write_input(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trade_id", "symbol", "quantity", "price"])
        writer.writerow(["T1", "ES", "2", "200"])
        writer.writerow(["T2", "CL", "1", "120"])
        writer.writerow(["T3", "BAD", "0", "5"])


def _config(tmp_path: Path, rules: Path, database_load_enabled: bool) -> PipelineConfig:
    return PipelineConfig(
        watch_dir=tmp_path / "watch",
        processing_dir=tmp_path / "processing",
        accepted_dir=tmp_path / "accepted",
        rejected_dir=tmp_path / "rejected",
        archive_dir=tmp_path / "archive",
        reason_dir=tmp_path / "reasons",
        rule_file=rules,
        database_load_enabled=database_load_enabled,
        dry_run_load_path=tmp_path / "loaded" / "rows.csv",
        max_workers=2,
        chunk_size=1,
    )


def test_processor_loads_ai_valid_rows_and_writes_ai_rejections(tmp_path: Path) -> None:
    rules = tmp_path / "rules.json"
    input_file = tmp_path / "watch" / "trades.csv"
    input_file.parent.mkdir()
    _write_rules(rules)
    _write_input(input_file)

    summary = FileProcessor(_config(tmp_path, rules, True), validator=FakeAIValidator()).process(input_file)  # type: ignore[arg-type]

    assert summary["total_rows"] == 3
    assert summary["accepted_rows"] == 1
    assert summary["rejected_rows"] == 2
    assert summary["loaded_rows"] == 1
    assert (tmp_path / "archive" / "trades.csv").exists()
    assert (tmp_path / "accepted" / "trades.csv").exists()
    assert (tmp_path / "rejected" / "trades.csv").exists()
    reasons = json.loads((tmp_path / "reasons" / "trades.reasons.json").read_text(encoding="utf-8"))
    assert {reason["rule"] for reason in reasons} == {"ai_rejected_row"}
    assert (tmp_path / "reasons" / "trades.adaptive_suggestions.json").exists()


def test_processor_can_skip_database_loading_after_ai_validation(tmp_path: Path) -> None:
    rules = tmp_path / "rules.json"
    input_file = tmp_path / "watch" / "trades.csv"
    input_file.parent.mkdir()
    _write_rules(rules)
    _write_input(input_file)

    summary = FileProcessor(_config(tmp_path, rules, False), validator=FakeAIValidator()).process(input_file)  # type: ignore[arg-type]

    assert summary["accepted_rows"] == 1
    assert summary["loaded_rows"] == 0
    assert not (tmp_path / "loaded" / "rows.csv").exists()
    assert (tmp_path / "accepted" / "trades.csv").exists()


def test_processor_requires_ai_api_configuration_without_injected_validator(tmp_path: Path) -> None:
    rules = tmp_path / "rules.json"
    _write_rules(rules)

    with pytest.raises(ValueError, match="AI validation requires"):
        FileProcessor(_config(tmp_path, rules, False))
