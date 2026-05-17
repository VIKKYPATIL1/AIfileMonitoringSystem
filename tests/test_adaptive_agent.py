from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from aifilemonitoring.adaptive_agent import AdaptiveRuleAgent, RuleChangeApprovalManager
from aifilemonitoring.models import PipelineConfig, RuleSet, ValidationError


class FakeProposalClient:
    def complete_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        updated_rules = dict(user_payload["current_rules"])
        updated_rules["columns"] = dict(updated_rules["columns"])
        updated_rules["columns"]["symbol"] = {"type": "string", "required": True, "allowed": ["ES", "CL", "NG"]}
        return {
            "summary": "Add NG as an allowed symbol after repeated failures.",
            "updated_rules": updated_rules,
            "db_schema_changes": [{"sql": "ALTER TABLE FUTURES_TRADES ADD SYMBOL_REASON VARCHAR2(100)", "reason": "Track new rule reason"}],
            "reviewer_notes": "Needs data-owner approval.",
        }


def _config(tmp_path: Path, rule_file: Path) -> PipelineConfig:
    return PipelineConfig(
        rule_file=rule_file,
        adaptive_rule_change_enabled=True,
        adaptive_day_threshold=2,
        adaptive_history_file=tmp_path / "adaptive" / "history.json",
        rule_change_dir=tmp_path / "rule_changes",
        notification_outbox_dir=tmp_path / "outbox",
        notification_email_to="owner@example.com",
    )


def test_adaptive_agent_creates_proposal_after_consecutive_failure_threshold(tmp_path: Path) -> None:
    rule_file = tmp_path / "rules.json"
    rule_file.write_text(json.dumps({"version": "test", "columns": {"symbol": {"type": "string"}}}), encoding="utf-8")
    rules = RuleSet(version="test", columns={"symbol": {"type": "string"}})
    config = _config(tmp_path, rule_file)
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    error = ValidationError(2, "symbol", "allowed", "NG", "Symbol is not allowed")
    failure_key = AdaptiveRuleAgent(config, rules, FakeProposalClient())._failure_key(error)  # type: ignore[arg-type]
    config.adaptive_history_file.parent.mkdir(parents=True, exist_ok=True)
    config.adaptive_history_file.write_text(
        json.dumps({failure_key: {"column": "symbol", "rule": "allowed", "reason": "Symbol is not allowed", "sample_values": ["NG"], "dates": [yesterday]}}),
        encoding="utf-8",
    )

    proposals = AdaptiveRuleAgent(config, rules, FakeProposalClient()).observe_failures([error])  # type: ignore[arg-type]

    assert len(proposals) == 1
    proposal = json.loads(proposals[0].read_text(encoding="utf-8"))
    assert proposal["status"] == "pending"
    assert proposal["ai_proposal"]["updated_rules"]["columns"]["symbol"]["allowed"] == ["ES", "CL", "NG"]
    assert list(config.notification_outbox_dir.glob("*.eml"))


def test_approval_manager_applies_rule_file_and_writes_schema_sql(tmp_path: Path) -> None:
    rule_file = tmp_path / "rules.json"
    rule_file.write_text(json.dumps({"version": "old", "columns": {"symbol": {"type": "string"}}}), encoding="utf-8")
    config = _config(tmp_path, rule_file)
    config.rule_change_dir.mkdir(parents=True)
    proposal_id = "abc123"
    (config.rule_change_dir / f"{proposal_id}.json").write_text(
        json.dumps(
            {
                "proposal_id": proposal_id,
                "status": "pending",
                "ai_proposal": {
                    "updated_rules": {"version": "new", "columns": {"symbol": {"type": "string", "allowed": ["NG"]}}},
                    "db_schema_changes": [{"sql": "ALTER TABLE FUTURES_TRADES ADD SYMBOL_REASON VARCHAR2(100)", "reason": "Need reason"}],
                },
            }
        ),
        encoding="utf-8",
    )

    summary = RuleChangeApprovalManager(config).approve(proposal_id, apply_db_schema_change=False)

    assert summary["rules_updated"] is True
    assert json.loads(rule_file.read_text(encoding="utf-8"))["version"] == "new"
    assert Path(summary["schema_sql_file"]).exists()
    assert summary["db_schema_applied"] is False
