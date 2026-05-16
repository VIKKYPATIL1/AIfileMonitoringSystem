from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PipelineConfig:
    """Runtime configuration for the file-monitoring pipeline."""

    watch_dir: Path = Path("/data/futures")
    processing_dir: Path = Path("/data/futures/processing")
    accepted_dir: Path = Path("/data/futures/accepted")
    rejected_dir: Path = Path("/data/futures/rejected")
    archive_dir: Path = Path("/data/futures/archive")
    reason_dir: Path = Path("/data/futures/reasons")
    rule_file: Path = Path("rules.json")
    poll_seconds: float = 5.0
    stable_seconds: float = 2.0
    chunk_size: int = 5_000
    max_workers: int = 4
    input_glob: str = "*.csv"
    delimiter: str = ","
    oracle_dsn: str | None = None
    oracle_user: str | None = None
    oracle_password: str | None = None
    oracle_table: str | None = None
    database_load_enabled: bool = False
    dry_run_load_path: Path | None = None
    ai_validation_enabled: bool = False
    ai_validation_mode: str = "assistive"
    ai_fail_closed: bool = True
    openai_compatible_base_url: str | None = None
    openai_compatible_api_key: str | None = None
    openai_compatible_model: str | None = None
    llm_timeout_seconds: int = 120
    llm_temperature: float = 0.0

    def ensure_directories(self) -> None:
        for directory in (
            self.watch_dir,
            self.processing_dir,
            self.accepted_dir,
            self.rejected_dir,
            self.archive_dir,
            self.reason_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        if self.dry_run_load_path:
            self.dry_run_load_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class ValidationError:
    row_number: int
    column: str
    rule: str
    value: str
    reason: str


@dataclass(slots=True)
class ValidatedRow:
    row_number: int
    data: dict[str, str]
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass(slots=True)
class RuleSet:
    version: str
    columns: dict[str, dict[str, Any]]
    combinations: list[dict[str, Any]] = field(default_factory=list)
    adaptive: dict[str, Any] = field(default_factory=dict)
