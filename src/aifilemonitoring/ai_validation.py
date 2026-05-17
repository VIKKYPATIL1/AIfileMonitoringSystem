from __future__ import annotations

import operator
from dataclasses import asdict
from typing import Annotated, Any, Callable, TypedDict

from .agents import chunked
from .llm import OpenAICompatibleClient
from .models import RuleSet, ValidatedRow, ValidationError


class ValidationChunk(TypedDict):
    chunk_id: int
    rows: list[tuple[int, dict[str, str]]]


class ChunkValidationResult(TypedDict):
    chunk_id: int
    rows: list[dict[str, Any]]


class ValidationGraphState(TypedDict, total=False):
    rows: list[tuple[int, dict[str, str]]]
    rules: dict[str, Any]
    interpreted_rules: dict[str, Any]
    chunks: list[ValidationChunk]
    chunk: ValidationChunk
    chunk_results: Annotated[list[ChunkValidationResult], operator.add]
    reconciled_results: list[ValidatedRow]
    errors: Annotated[list[str], operator.add]


class AgenticAIValidator:
    """Native LangGraph multi-agent validator that validates rows only through AI."""

    def __init__(
        self,
        rules: RuleSet,
        client: OpenAICompatibleClient | None,
        max_workers: int = 4,
        chunk_size: int = 5_000,
        fail_closed: bool = True,
    ):
        if client is None:
            raise ValueError(
                "AI validation requires an OpenAI-compatible API client. Configure "
                "OPENAI_COMPATIBLE_BASE_URL, OPENAI_COMPATIBLE_API_KEY, and OPENAI_COMPATIBLE_MODEL."
            )
        self.rules = rules
        self.client = client
        self.max_workers = max(1, max_workers)
        self.chunk_size = max(1, chunk_size)
        self.fail_closed = fail_closed
        self._state_graph, self._start, self._end, self._send = self._load_langgraph()
        self._compiled_graph = self._build_langgraph()

    def validate(self, rows: list[tuple[int, dict[str, str]]]) -> list[ValidatedRow]:
        if not rows:
            return []
        state: ValidationGraphState = {
            "rows": rows,
            "rules": self._rules_payload(),
            "chunk_results": [],
            "errors": [],
        }
        result = self._compiled_graph.invoke(state, config={"max_concurrency": self.max_workers})
        return sorted(result["reconciled_results"], key=lambda item: item.row_number)

    def _load_langgraph(self) -> tuple[Any, str, str, Callable[..., Any]]:
        try:
            from langgraph.graph import END, START, StateGraph
            from langgraph.types import Send
        except ImportError as exc:
            raise RuntimeError(
                "Native LangGraph is required for AI validation. Install project dependencies with "
                "`python -m pip install -e .` or install `langgraph>=1.0`."
            ) from exc
        return StateGraph, START, END, Send

    def _build_langgraph(self) -> Any:
        state_graph = self._state_graph(ValidationGraphState)
        state_graph.add_node("rule_interpreter_agent", self._rule_interpreter_agent)
        state_graph.add_node("chunk_planner_agent", self._chunk_planner_agent)
        state_graph.add_node("validator_agent", self._validator_agent)
        state_graph.add_node("supervisor_agent", self._supervisor_agent)
        state_graph.add_edge(self._start, "rule_interpreter_agent")
        state_graph.add_edge("rule_interpreter_agent", "chunk_planner_agent")
        state_graph.add_conditional_edges("chunk_planner_agent", self._route_chunks)
        state_graph.add_edge("validator_agent", "supervisor_agent")
        state_graph.add_edge("supervisor_agent", self._end)
        return state_graph.compile()

    def _rule_interpreter_agent(self, state: ValidationGraphState) -> ValidationGraphState:
        system_prompt = (
            "You are a rule interpretation agent for a financial data-quality workflow. Normalize these JSON rules "
            "into clear validation instructions for downstream AI validators. Do not invent columns or relax any rule. "
            "Return JSON with keys: rule_summary, column_rules, combination_rules, risk_notes."
        )
        interpreted = self.client.complete_json(system_prompt, {"rules": state["rules"]})
        return {"interpreted_rules": interpreted}

    def _chunk_planner_agent(self, state: ValidationGraphState) -> ValidationGraphState:
        chunks = [
            {"chunk_id": index, "rows": chunk}
            for index, chunk in enumerate(chunked(state["rows"], self.chunk_size), start=1)
        ]
        return {"chunks": chunks}

    def _route_chunks(self, state: ValidationGraphState) -> list[Any]:
        return [
            self._send(
                "validator_agent",
                {
                    "chunk": chunk,
                    "rules": state["rules"],
                    "interpreted_rules": state["interpreted_rules"],
                    "chunk_results": [],
                    "errors": [],
                },
            )
            for chunk in state["chunks"]
        ]

    def _validator_agent(self, state: ValidationGraphState) -> ValidationGraphState:
        chunk = state["chunk"]
        chunk_id = int(chunk["chunk_id"])
        rows = chunk["rows"]
        system_prompt = (
            "You are one validation agent in a native LangGraph map-reduce file-monitoring graph. "
            "Validate every provided row against the interpreted business rules using AI reasoning only. "
            "Return only JSON with this shape: "
            "{\"rows\":[{\"row_number\":2,\"is_valid\":true,\"errors\":[{\"column\":\"name\","
            "\"rule\":\"rule_id\",\"value\":\"bad\",\"reason\":\"short reason\"}]}]}. "
            "Include every input row exactly once. Do not change row values."
        )
        payload = {
            "chunk_id": chunk_id,
            "interpreted_rules": state["interpreted_rules"],
            "raw_rules": state["rules"],
            "rows": [{"row_number": row_number, "data": row} for row_number, row in rows],
        }
        try:
            response = self.client.complete_json(system_prompt, payload)
            parsed_rows = validation_rows_to_dicts(self._parse_ai_rows(response, rows))
            return {"chunk_results": [{"chunk_id": chunk_id, "rows": parsed_rows}], "errors": []}
        except Exception as exc:
            if not self.fail_closed:
                raise
            failed_rows = [self._ai_exception_result(row_number, row, exc) for row_number, row in rows]
            return {
                "chunk_results": [{"chunk_id": chunk_id, "rows": validation_rows_to_dicts(failed_rows)}],
                "errors": [str(exc)],
            }

    def _supervisor_agent(self, state: ValidationGraphState) -> ValidationGraphState:
        results_by_row: dict[int, ValidatedRow] = {}
        duplicate_rows: set[int] = set()
        for chunk_result in state.get("chunk_results", []):
            for item in chunk_result.get("rows", []):
                row_number = int(item["row_number"])
                row = dict(item.get("data", {}))
                errors = [
                    ValidationError(
                        row_number=int(error.get("row_number", row_number)),
                        column=str(error.get("column", "ai_validation")),
                        rule=str(error.get("rule", "ai_rule")),
                        value=str(error.get("value", "")),
                        reason=str(error.get("reason", "AI validation rejected the row")),
                    )
                    for error in item.get("errors", [])
                ]
                if row_number in results_by_row:
                    duplicate_rows.add(row_number)
                results_by_row[row_number] = ValidatedRow(row_number=row_number, data=row, errors=errors)

        reconciled: list[ValidatedRow] = []
        for row_number, row in state["rows"]:
            result = results_by_row.get(row_number) or self._missing_ai_result(row_number, row)
            if row_number in duplicate_rows:
                result.errors.append(
                    ValidationError(
                        row_number=row_number,
                        column="ai_validation",
                        rule="duplicate_ai_result",
                        value="",
                        reason="Multiple AI validator agents returned a decision for this row",
                    )
                )
            reconciled.append(result)
        return {"reconciled_results": reconciled}

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
