from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from .models import ValidatedRow
from .rules import RuleEngine


def chunked(items: list[tuple[int, dict[str, str]]], size: int) -> Iterable[list[tuple[int, dict[str, str]]]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


class ValidationAgentPool:
    """Divides rows across worker agents so rule checks can run in parallel."""

    def __init__(self, engine: RuleEngine, max_workers: int = 4, chunk_size: int = 5_000):
        self.engine = engine
        self.max_workers = max(1, max_workers)
        self.chunk_size = max(1, chunk_size)

    def validate(self, rows: list[tuple[int, dict[str, str]]]) -> list[ValidatedRow]:
        if not rows:
            return []
        results: list[ValidatedRow] = []
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="rule-agent") as executor:
            futures = [executor.submit(self._validate_chunk, chunk) for chunk in chunked(rows, self.chunk_size)]
            for future in as_completed(futures):
                results.extend(future.result())
        return sorted(results, key=lambda item: item.row_number)

    def _validate_chunk(self, rows: list[tuple[int, dict[str, str]]]) -> list[ValidatedRow]:
        return [self.engine.validate_row(row_number, row) for row_number, row in rows]
