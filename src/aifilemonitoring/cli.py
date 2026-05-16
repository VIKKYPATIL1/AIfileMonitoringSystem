from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from .models import PipelineConfig
from .processor import FileProcessor
from .watcher import PollingFileWatcher


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive AI file monitoring system")
    parser.add_argument("--config", type=Path, help="Optional JSON pipeline configuration file")
    parser.add_argument("--watch-dir", type=Path, help="Directory to watch, defaults to /data/futures")
    parser.add_argument("--rules", type=Path, help="Rules JSON file")
    parser.add_argument("--once", type=Path, help="Process one file and exit")
    parser.add_argument(
        "--load-to-database",
        "--load-to-db",
        dest="database_load_enabled",
        type=parse_bool,
        choices=[True, False],
        metavar="true|false",
        help="Enable or disable loading accepted rows to Oracle/dry-run loader",
    )
    parser.add_argument("--dry-run-load-path", type=Path, help="Append accepted rows to this CSV instead of Oracle")
    parser.add_argument("--openai-compatible-base-url", help="Base URL ending in /v1 for qwen/gpt-oss/gemma compatible APIs")
    parser.add_argument("--openai-compatible-api-key", help="API key for the OpenAI-compatible model gateway")
    parser.add_argument("--openai-compatible-model", help="Model name for AI validation")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser


def load_config(config_path: Path | None, args: argparse.Namespace) -> PipelineConfig:
    data: dict[str, Any] = {}
    if config_path:
        data.update(json.loads(config_path.read_text(encoding="utf-8")))
    env_map = {
        "oracle_dsn": "ORACLE_DSN",
        "oracle_user": "ORACLE_USER",
        "oracle_password": "ORACLE_PASSWORD",
        "oracle_table": "ORACLE_TABLE",
        "database_load_enabled": "DATABASE_LOAD_ENABLED",
        "openai_compatible_base_url": "OPENAI_COMPATIBLE_BASE_URL",
        "openai_compatible_api_key": "OPENAI_COMPATIBLE_API_KEY",
        "openai_compatible_model": "OPENAI_COMPATIBLE_MODEL",
    }
    for key, env_name in env_map.items():
        if os.getenv(env_name):
            data[key] = parse_bool(os.environ[env_name]) if key == "database_load_enabled" else os.environ[env_name]
    if args.watch_dir:
        data["watch_dir"] = args.watch_dir
    if args.rules:
        data["rule_file"] = args.rules
    if args.database_load_enabled is not None:
        data["database_load_enabled"] = args.database_load_enabled
    if args.dry_run_load_path:
        data["dry_run_load_path"] = args.dry_run_load_path
    if args.openai_compatible_base_url:
        data["openai_compatible_base_url"] = args.openai_compatible_base_url
    if args.openai_compatible_api_key:
        data["openai_compatible_api_key"] = args.openai_compatible_api_key
    if args.openai_compatible_model:
        data["openai_compatible_model"] = args.openai_compatible_model
    path_keys = [
        "watch_dir",
        "processing_dir",
        "accepted_dir",
        "rejected_dir",
        "archive_dir",
        "reason_dir",
        "rule_file",
        "dry_run_load_path",
    ]
    for key in path_keys:
        if key in data and data[key] is not None:
            data[key] = Path(data[key])
    return PipelineConfig(**data)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config, args)
    if args.once:
        summary = FileProcessor(config).process(args.once)
        print(json.dumps(summary, indent=2))
        return
    PollingFileWatcher(config).run_forever()


if __name__ == "__main__":
    main()
