from __future__ import annotations

import hashlib
import json
import smtplib
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from .llm import OpenAICompatibleClient
from .models import PipelineConfig, RuleSet, ValidationError


class NotificationSender:
    """Sends approval requests by SMTP when configured, otherwise writes mail files to an outbox."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.outbox_dir = config.notification_outbox_dir
        self.outbox_dir.mkdir(parents=True, exist_ok=True)

    def send(self, subject: str, body: str) -> Path | None:
        if not self.config.notification_email_to:
            return self._write_outbox(subject, body)
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.config.notification_email_from
        message["To"] = self.config.notification_email_to
        message.set_content(body)
        if self.config.smtp_host:
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port, timeout=30) as smtp:
                smtp.send_message(message)
            return None
        return self._write_outbox(subject, body, message)

    def _write_outbox(self, subject: str, body: str, message: EmailMessage | None = None) -> Path:
        safe_subject = "".join(character if character.isalnum() else "_" for character in subject)[:80]
        path = self.outbox_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{safe_subject}.eml"
        if message is None:
            message = EmailMessage()
            message["Subject"] = subject
            message["From"] = self.config.notification_email_from
            message["To"] = self.config.notification_email_to or "unconfigured@example.com"
            message.set_content(body)
        path.write_text(message.as_string(), encoding="utf-8")
        return path


class AdaptiveRuleAgent:
    """Tracks repeated AI validation failures and asks AI to propose rule/schema changes."""

    def __init__(self, config: PipelineConfig, rules: RuleSet, client: OpenAICompatibleClient | None):
        if client is None:
            raise ValueError("Adaptive rule changes require the same OpenAI-compatible API client used for validation")
        self.config = config
        self.rules = rules
        self.client = client
        self.notifier = NotificationSender(config)
        self.config.adaptive_history_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.rule_change_dir.mkdir(parents=True, exist_ok=True)

    def observe_failures(self, errors: list[ValidationError]) -> list[Path]:
        if not self.config.adaptive_rule_change_enabled or not errors:
            return []
        history = self._load_history()
        today = datetime.now(timezone.utc).date().isoformat()
        for error in errors:
            key = self._failure_key(error)
            entry = history.setdefault(
                key,
                {
                    "column": error.column,
                    "rule": error.rule,
                    "reason": error.reason,
                    "sample_values": [],
                    "dates": [],
                },
            )
            if error.value and len(entry["sample_values"]) < 20:
                entry["sample_values"].append(error.value)
            if today not in entry["dates"]:
                entry["dates"].append(today)
            entry["dates"] = sorted(set(entry["dates"]))
        self._write_history(history)

        proposals = []
        for key, entry in history.items():
            if self._consecutive_days(entry["dates"]) >= self.config.adaptive_day_threshold:
                proposals.append(self._create_proposal(key, entry))
        return proposals

    def _create_proposal(self, key: str, entry: dict[str, Any]) -> Path:
        proposal_id = hashlib.sha256(
            f"{key}:{entry['dates'][-1]}:{self.rules.version}".encode("utf-8")
        ).hexdigest()[:12]
        proposal_path = self.config.rule_change_dir / f"{proposal_id}.json"
        if proposal_path.exists():
            return proposal_path
        prompt = (
            "You are an adaptive data-quality rule-change agent. A validation failure has repeated for the configured "
            "number of consecutive days. Decide whether rules should change. Return only JSON with keys: "
            "summary, updated_rules, db_schema_changes, reviewer_notes. updated_rules must be the full replacement "
            "rules JSON object if a rule change is recommended, otherwise null. db_schema_changes must be a list of "
            "objects with sql and reason when database schema changes are needed."
        )
        ai_proposal = self.client.complete_json(
            prompt,
            {
                "failure_pattern": entry,
                "threshold_days": self.config.adaptive_day_threshold,
                "current_rules": self._rules_payload(),
            },
        )
        proposal = {
            "proposal_id": proposal_id,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "failure_key": key,
            "failure_pattern": entry,
            "ai_proposal": ai_proposal,
        }
        proposal_path.write_text(json.dumps(proposal, indent=2, default=str), encoding="utf-8")
        self.notifier.send(
            f"Rule change approval required: {proposal_id}",
            "An adaptive AI agent detected repeated validation failures and proposed a rule change.\n\n"
            f"Proposal file: {proposal_path}\n\n"
            f"Failure pattern:\n{json.dumps(entry, indent=2)}\n\n"
            f"AI proposal:\n{json.dumps(ai_proposal, indent=2, default=str)}\n\n"
            f"Approve with: ai-file-monitor --config <config> --approve-rule-change {proposal_id}",
        )
        return proposal_path

    def _failure_key(self, error: ValidationError) -> str:
        return hashlib.sha256(f"{error.column}|{error.rule}|{error.reason}".encode("utf-8")).hexdigest()

    def _consecutive_days(self, dates: list[str]) -> int:
        date_set = {date.fromisoformat(item) for item in dates}
        if not date_set:
            return 0
        current = max(date_set)
        count = 0
        while current in date_set:
            count += 1
            current -= timedelta(days=1)
        return count

    def _load_history(self) -> dict[str, Any]:
        if not self.config.adaptive_history_file.exists():
            return {}
        return json.loads(self.config.adaptive_history_file.read_text(encoding="utf-8"))

    def _write_history(self, history: dict[str, Any]) -> None:
        self.config.adaptive_history_file.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")

    def _rules_payload(self) -> dict[str, Any]:
        return {
            "version": self.rules.version,
            "columns": self.rules.columns,
            "combinations": self.rules.combinations,
            "adaptive": self.rules.adaptive,
        }


class RuleChangeApprovalManager:
    """Applies human-approved adaptive rule and optional database schema changes."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.notifier = NotificationSender(config)
        self.config.rule_change_dir.mkdir(parents=True, exist_ok=True)

    def approve(self, proposal_id: str, apply_db_schema_change: bool = False) -> dict[str, Any]:
        proposal_path = self.config.rule_change_dir / f"{proposal_id}.json"
        if not proposal_path.exists():
            raise FileNotFoundError(f"Rule change proposal not found: {proposal_path}")
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        ai_proposal = proposal.get("ai_proposal", {})
        updated_rules = ai_proposal.get("updated_rules")
        if updated_rules:
            self.config.rule_file.write_text(json.dumps(updated_rules, indent=2, default=str), encoding="utf-8")
        db_schema_changes = ai_proposal.get("db_schema_changes") or []
        sql_path = None
        if db_schema_changes:
            sql_path = self.config.rule_change_dir / f"{proposal_id}.schema.sql"
            sql_path.write_text(
                "\n".join(change.get("sql", "") + ";" for change in db_schema_changes if change.get("sql")),
                encoding="utf-8",
            )
            self.notifier.send(
                f"Database schema change review: {proposal_id}",
                "The approved rule change includes database schema changes.\n\n"
                f"SQL file: {sql_path}\n\n"
                f"Changes:\n{json.dumps(db_schema_changes, indent=2, default=str)}",
            )
            if apply_db_schema_change:
                self._apply_schema_changes(db_schema_changes)
        proposal["status"] = "approved"
        proposal["approved_at"] = datetime.now(timezone.utc).isoformat()
        proposal["db_schema_applied"] = bool(db_schema_changes and apply_db_schema_change)
        proposal_path.write_text(json.dumps(proposal, indent=2, default=str), encoding="utf-8")
        return {
            "proposal_id": proposal_id,
            "rules_updated": bool(updated_rules),
            "schema_sql_file": str(sql_path) if sql_path else None,
            "db_schema_applied": proposal["db_schema_applied"],
        }

    def _apply_schema_changes(self, db_schema_changes: list[dict[str, Any]]) -> None:
        if not (self.config.oracle_user and self.config.oracle_password and self.config.oracle_dsn):
            raise ValueError("Oracle connection settings are required to apply database schema changes")
        import oracledb

        with oracledb.connect(
            user=self.config.oracle_user,
            password=self.config.oracle_password,
            dsn=self.config.oracle_dsn,
        ) as connection:
            with connection.cursor() as cursor:
                for change in db_schema_changes:
                    sql = str(change.get("sql", "")).strip().rstrip(";")
                    if sql:
                        cursor.execute(sql)
            connection.commit()
