from __future__ import annotations

import importlib
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, TypedDict

from .agents import chunked
from .llm import OpenAICompatibleClient
from .models import RuleSet, ValidatedRow, ValidationError
from .rules import RuleEngine


class ValidationGraphState(TypedDict, total=False):
    rows: list[tuple[int, dict[str, str]]]
    rules: dict[str, Any]
    interpreted_rules: dict[str, Any]
    deterministic_results: list[ValidatedRow]
    ai_results: list[ValidatedRow]
    reconciled_results: list[ValidatedRow]
    errors: list[str]


class AgenticAIValidator:
    """LangGraph-style multi-agent validator for AI-assisted rule checks."""

    def __init__(
        self,
        rules: RuleSet,
        engine: RuleEngine,
        client: OpenAICompatibleClient | None,
        max_workers: int = 4,
        chunk_size: int = 5_000,
        mode: str = "assistive",
        fail_closed: bool = True,
    ):
        self.rules = rules
        self.engine = engine
        self.client = client
        self.max_workers = max(1, max_workers)
        self.chunk_size = max(1, chunk_size)
        self.mode = mode
        self.fail_closed = fail_closed
        self._compiled_graph = self._build_langgraph() if self._langgraph_available() else None

    def validate(self, rows: list[tuple[int, dict[str, str]]]) -> list[ValidatedRow]:
        if not rows:
            return []
        state: ValidationGraphState = {"rows": rows, "rules": self._rules_payload(), "errors": []}
        if self._compiled_graph:
            result = self._compiled_graph.invoke(state)
        else:
            result = self._run_local_graph(state)
        return sorted(result["reconciled_results"], key=lambda item: item.row_number)

    def _langgraph_available(self) -> bool:
        return importlib.util.find_spec("langgraph") is not None and importlib.util.find_spec("langgraph.graph") is not None

    def _build_langgraph(self) -> Any:
        graph_module = importlib.import_module("langgraph.graph")
        state_graph = graph_module.StateGraph(ValidationGraphState)
        state_graph.add_node("rule_interpreter_agent", self._rule_interpreter_agent)
        state_graph.add_node("deterministic_guardrail_agent", self._deterministic_guardrail_agent)
        state_graph.add_node("ai_chunk_validator_agents", self._ai_chunk_validator_agents)
        state_graph.add_node("supervisor_reconciliation_agent", self._supervisor_reconciliation_agent)
        state_graph.add_edge(graph_module.START, "rule_interpreter_agent")
        state_graph.add_edge("rule_interpreter_agent", "deterministic_guardrail_agent")
        state_graph.add_edge("deterministic_guardrail_agent", "ai_chunk_validator_agents")
        state_graph.add_edge("ai_chunk_validator_agents", "supervisor_reconciliation_agent")
        state_graph.add_edge("supervisor_reconciliation_agent", graph_module.END)
        return state_graph.compile()

    def _run_local_graph(self, state: ValidationGraphState) -> ValidationGraphState:
        for node in (
            self._rule_interpreter_agent,
            self._deterministic_guardrail_agent,
            self._ai_chunk_validator_agents,
            self._supervisor_reconciliation_agent,
        ):
            state.update(node(state))
        return state

    def _rule_interpreter_agent(self, state: ValidationGraphState) -> ValidationGraphState:
        if not self.client:
            return {"interpreted_rules": state["rules"]}
        system_prompt = (
            "You are a rule interpretation agent for a financial data-quality workflow. Normalize these JSON rules "
            "into clear validation instructions. Do not invent columns or relax any rule. Return JSON with keys: "
            "rule_summary, column_rules, combination_rules, risk_notes."
        )
        interpreted = self.client.complete_json(system_prompt, {"rules": state["rules"]})
        return {"interpreted_rules": interpreted}

    def _deterministic_guardrail_agent(self, state: ValidationGraphState) -> ValidationGraphState:
        return {
            "deterministic_results": [
                self.engine.validate_row(row_number, row) for row_number, row in state["rows"]
            ]
        }

    def _ai_chunk_validator_agents(self, state: ValidationGraphState) -> ValidationGraphState:
        if not self.client:
            if self.fail_closed:
                return {
                    "ai_results": [
                        self._ai_unavailable_result(row_number, row) for row_number, row in state["rows"]
                    ],
                    "errors": ["AI validation is enabled, but no OpenAI-compatible API client is configured."],
                }
            return {"ai_results": state["deterministic_results"]}

        results: list[ValidatedRow] = []
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="ai-rule-agent") as executor:
            futures = [
                executor.submit(self._validate_chunk_with_ai, chunk, state["interpreted_rules"])
                for chunk in chunked(state["rows"], self.chunk_size)
            ]
            for future in as_completed(futures):
                chunk_results, chunk_errors = future.result()
                results.extend(chunk_results)
                errors.extend(chunk_errors)
        return {"ai_results": results, "errors": errors}

    def _supervisor_reconciliation_agent(self, state: ValidationGraphState) -> ValidationGraphState:
        deterministic = {item.row_number: item for item in state["deterministic_results"]}
        ai = {item.row_number: item for item in state["ai_results"]}
        reconciled: list[ValidatedRow] = []
        for row_number, row in state["rows"]:
            deterministic_row = deterministic[row_number]
            ai_row = ai.get(row_number)
            if not ai_row:
                reconciled.append(self._missing_ai_result(row_number, row))
                continue
            if self.mode == "authoritative":
                reconciled.append(ai_row)
                continue
            errors = list(deterministic_row.errors)
            errors.extend(self._new_ai_errors(ai_row, errors))
            reconciled.append(ValidatedRow(row_number=row_number, data=row, errors=errors))
        return {"reconciled_results": reconciled}

    def _validate_chunk_with_ai(
        self, chunk: list[tuple[int, dict[str, str]]], interpreted_rules: dict[str, Any]
    ) -> tuple[list[ValidatedRow], list[str]]:
        system_prompt = (
            "You are one validation agent in a parallel financial file-monitoring graph. Validate every provided row "
            "against the interpreted business rules. Return only JSON with this shape: "
            "{\"rows\":[{\"row_number\":2,\"is_valid\":true,\"errors\":[{\"column\":\"name\","
            "\"rule\":\"rule_id\",\"value\":\"bad\",\"reason\":\"short reason\"}]}]}. "
            "Include every input row exactly once. Do not change row values."
        )
        payload = {
            "interpreted_rules": interpreted_rules,
            "raw_rules": self._rules_payload(),
            "rows": [{"row_number": row_number, "data": row} for row_number, row in chunk],
        }
        try:
            response = self.client.complete_json(system_prompt, payload) if self.client else {"rows": []}
            return self._parse_ai_rows(response, chunk), []
        except Exception as exc:
            if self.fail_closed:
                return [self._ai_exception_result(row_number, row, exc) for row_number, row in chunk], [str(exc)]
            return [self.engine.validate_row(row_number, row) for row_number, row in chunk], [str(exc)]

    def _parse_ai_rows(self, response: dict[str, Any], chunk: list[tuple[int, dict[str, str]]]) -> list[ValidatedRow]:
        rows_by_number = {row_number: row for row_number, row in chunk}
        parsed: list[ValidatedRow] = []
        for item in response.get("rows", []):
            row_number = int(item["row_number"])
            if row_number not in rows_by_number:
                continue
            errors = []
            for error in item.get("errors", []):
                errors.append(
                    ValidationError(
                        row_number=row_number,
                        column=str(error.get("column", "ai_validation")),
                        rule=str(error.get("rule", "ai_rule")),
                        value=str(error.get("value", "")),
                        reason=str(error.get("reason", "AI validation rejected the row")),
                    )
                )
            parsed.append(ValidatedRow(row_number=row_number, data=rows_by_number[row_number], errors=errors))
        parsed_numbers = {item.row_number for item in parsed}
        for row_number, row in chunk:
            if row_number not in parsed_numbers:
                parsed.append(self._missing_ai_result(row_number, row))
        return parsed

    def _new_ai_errors(self, ai_row: ValidatedRow, existing: list[ValidationError]) -> list[ValidationError]:
        existing_keys = {(error.column, error.rule, error.value, error.reason) for error in existing}
        return [
            error
            for error in ai_row.errors
            if (error.column, error.rule, error.value, error.reason) not in existing_keys
        ]

    def _rules_payload(self) -> dict[str, Any]:
        return {
            "version": self.rules.version,
            "columns": self.rules.columns,
            "combinations": self.rules.combinations,
            "adaptive": self.rules.adaptive,
        }

    def _missing_ai_result(self, row_number: int, row: dict[str, str]) -> ValidatedRow:
        return ValidatedRow(
            row_number=row_number,
            data=row,
            errors=[
                ValidationError(
                    row_number=row_number,
                    column="ai_validation",
                    rule="missing_ai_result",
                    value="",
                    reason="AI validation did not return a decision for this row",
                )
            ],
        )

    def _ai_unavailable_result(self, row_number: int, row: dict[str, str]) -> ValidatedRow:
        return ValidatedRow(
            row_number=row_number,
            data=row,
            errors=[
                ValidationError(
                    row_number=row_number,
                    column="ai_validation",
                    rule="ai_client_unavailable",
                    value="",
                    reason="AI validation is enabled but no OpenAI-compatible API client is configured",
                )
            ],
        )

    def _ai_exception_result(self, row_number: int, row: dict[str, str], exc: Exception) -> ValidatedRow:
        return ValidatedRow(
            row_number=row_number,
            data=row,
            errors=[
                ValidationError(
                    row_number=row_number,
                    column="ai_validation",
                    rule="ai_validation_exception",
                    value="",
                    reason=f"AI validation failed closed: {exc}",
                )
            ],
        )


def validation_rows_to_dicts(rows: list[ValidatedRow]) -> list[dict[str, Any]]:
    return [
        {
            "row_number": row.row_number,
            "data": row.data,
            "errors": [asdict(error) for error in row.errors],
            "is_valid": row.is_valid,
        }
        for row in rows
    ]
