from __future__ import annotations

import csv
import json
from pathlib import Path

from aifilemonitoring.models import PipelineConfig
from aifilemonitoring.processor import FileProcessor


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


def test_processor_loads_valid_rows_and_writes_rejections(tmp_path: Path) -> None:
    rules = tmp_path / "rules.json"
    input_file = tmp_path / "watch" / "trades.csv"
    input_file.parent.mkdir()
    _write_rules(rules)
    _write_input(input_file)
    config = PipelineConfig(
        watch_dir=tmp_path / "watch",
        processing_dir=tmp_path / "processing",
        accepted_dir=tmp_path / "accepted",
        rejected_dir=tmp_path / "rejected",
        archive_dir=tmp_path / "archive",
        reason_dir=tmp_path / "reasons",
        rule_file=rules,
        database_load_enabled=True,
        dry_run_load_path=tmp_path / "loaded" / "rows.csv",
        max_workers=2,
        chunk_size=1,
    )

    summary = FileProcessor(config).process(input_file)

    assert summary["total_rows"] == 3
    assert summary["accepted_rows"] == 1
    assert summary["rejected_rows"] == 2
    assert summary["loaded_rows"] == 1
    assert (tmp_path / "archive" / "trades.csv").exists()
    assert (tmp_path / "accepted" / "trades.csv").exists()
    assert (tmp_path / "rejected" / "trades.csv").exists()
    reasons = json.loads((tmp_path / "reasons" / "trades.reasons.json").read_text(encoding="utf-8"))
    assert {reason["rule"] for reason in reasons} == {"combination", "allowed", "min"}
    assert (tmp_path / "reasons" / "trades.adaptive_suggestions.json").exists()


def test_processor_can_skip_database_loading(tmp_path: Path) -> None:
    rules = tmp_path / "rules.json"
    input_file = tmp_path / "watch" / "trades.csv"
    input_file.parent.mkdir()
    _write_rules(rules)
    _write_input(input_file)
    config = PipelineConfig(
        watch_dir=tmp_path / "watch",
        processing_dir=tmp_path / "processing",
        accepted_dir=tmp_path / "accepted",
        rejected_dir=tmp_path / "rejected",
        archive_dir=tmp_path / "archive",
        reason_dir=tmp_path / "reasons",
        rule_file=rules,
        database_load_enabled=False,
        dry_run_load_path=tmp_path / "loaded" / "rows.csv",
    )

    summary = FileProcessor(config).process(input_file)

    assert summary["accepted_rows"] == 1
    assert summary["loaded_rows"] == 0
    assert not (tmp_path / "loaded" / "rows.csv").exists()
    assert (tmp_path / "accepted" / "trades.csv").exists()
