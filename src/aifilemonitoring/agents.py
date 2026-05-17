from __future__ import annotations

from typing import Iterable


def chunked(items: list[tuple[int, dict[str, str]]], size: int) -> Iterable[list[tuple[int, dict[str, str]]]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]
