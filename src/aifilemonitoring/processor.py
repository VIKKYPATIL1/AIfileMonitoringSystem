from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .adaptive_agent import AdaptiveRuleAgent
from .ai_validation import AgenticAIValidator
from .llm import OpenAICompatibleClient
from .loaders import CsvDryRunLoader, Loader, OracleLoader
from .models import PipelineConfig, ValidatedRow
from .rules import load_rules


class FileProcessor:
    """Coordinates validation, rejection output, archive, and database loading."""

    def __init__(self, config: PipelineConfig, loader: Loader | None = None, validator: AgenticAIValidator | None = None):
        self.config = config
        self.config.ensure_directories()
        self.rules = load_rules(config.rule_file)
        self.ai_client = self._build_ai_client(config)
        self.agent_pool = validator or self._build_validation_pool(config)
        self.loader = (loader or self._build_loader(config)) if config.database_load_enabled else None
        self.adaptive_agent = (
            AdaptiveRuleAgent(config, self.rules, self.ai_client) if config.adaptive_rule_change_enabled else None
        )

    def process(self, input_path: Path) -> dict[str, int | str]:
        processing_path = self._move_to_processing(input_path)
        rows, fieldnames = self._read_rows(processing_path)
        validated = self.agent_pool.validate(rows)
        accepted_rows = [item.data for item in validated if item.is_valid]
        rejected = [item for item in validated if not item.is_valid]
        loaded_count = self.loader.load(accepted_rows) if self.loader else 0
        if accepted_rows:
            self._write_csv(self.config.accepted_dir / processing_path.name, accepted_rows, fieldnames)
        if rejected:
            self._write_rejected(processing_path, rejected, fieldnames)
            if self.adaptive_agent:
                self.adaptive_agent.observe_failures([error for item in rejected for error in item.errors])
        archive_path = self.config.archive_dir / processing_path.name
        shutil.move(str(processing_path), archive_path)
        return {
            "file": str(archive_path),
            "total_rows": len(rows),
            "accepted_rows": len(accepted_rows),
            "rejected_rows": len(rejected),
            "loaded_rows": loaded_count,
        }

    def _build_ai_client(self, config: PipelineConfig) -> OpenAICompatibleClient | None:
        return OpenAICompatibleClient.from_env(
            config.openai_compatible_base_url,
            config.openai_compatible_api_key,
            config.openai_compatible_model,
            config.llm_timeout_seconds,
            config.llm_temperature,
        )

    def _build_validation_pool(self, config: PipelineConfig) -> AgenticAIValidator:
        return AgenticAIValidator(
            self.rules,
            self.ai_client,
            config.max_workers,
            config.chunk_size,
            config.ai_fail_closed,
        )

    def _build_loader(self, config: PipelineConfig) -> Loader:
        if config.dry_run_load_path:
            return CsvDryRunLoader(config.dry_run_load_path)
        if config.oracle_user and config.oracle_password and config.oracle_dsn and config.oracle_table:
            return OracleLoader(config.oracle_user, config.oracle_password, config.oracle_dsn, config.oracle_table)
        return CsvDryRunLoader(config.accepted_dir / "loaded_rows.csv")

    def _move_to_processing(self, input_path: Path) -> Path:
        destination = self.config.processing_dir / input_path.name
        if destination.exists():
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            destination = destination.with_name(f"{destination.stem}_{timestamp}{destination.suffix}")
        shutil.move(str(input_path), destination)
        return destination

    def _read_rows(self, input_path: Path) -> tuple[list[tuple[int, dict[str, str]]], list[str]]:
        with input_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream, delimiter=self.config.delimiter)
            fieldnames = reader.fieldnames or []
            return [(index, dict(row)) for index, row in enumerate(reader, start=2)], fieldnames

    def _write_csv(self, path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_rejected(self, processing_path: Path, rejected: list[ValidatedRow], fieldnames: list[str]) -> None:
        rejected_rows = [item.data for item in rejected]
        self._write_csv(self.config.rejected_dir / processing_path.name, rejected_rows, fieldnames)
        errors = [error for item in rejected for error in item.errors]
        reason_path = self.config.reason_dir / f"{processing_path.stem}.reasons.json"
        reason_path.write_text(
            json.dumps([asdict(error) for error in errors], indent=2, default=str), encoding="utf-8"
        )
        legacy_suggestions_path = self.config.reason_dir / f"{processing_path.stem}.adaptive_suggestions.json"
        legacy_suggestions_path.write_text(
            json.dumps({"message": "Adaptive rule-change proposals are tracked in rule_change_dir when enabled."}, indent=2),
            encoding="utf-8",
        )
