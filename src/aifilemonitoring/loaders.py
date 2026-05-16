from __future__ import annotations

import csv
from pathlib import Path
from typing import Protocol


class Loader(Protocol):
    def load(self, rows: list[dict[str, str]]) -> int:
        """Load accepted rows and return the inserted row count."""


class CsvDryRunLoader:
    """Platform-independent loader for development and tests."""

    def __init__(self, path: Path):
        self.path = path

    def load(self, rows: list[dict[str, str]]) -> int:
        if not rows:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists()
        fieldnames = list(rows[0].keys())
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
        return len(rows)


class OracleLoader:
    """Oracle SQL loader using python-oracledb when installed."""

    def __init__(self, user: str, password: str, dsn: str, table: str):
        self.user = user
        self.password = password
        self.dsn = dsn
        self.table = table

    def load(self, rows: list[dict[str, str]]) -> int:
        if not rows:
            return 0
        import oracledb

        columns = list(rows[0].keys())
        placeholders = ", ".join(f":{index + 1}" for index in range(len(columns)))
        column_sql = ", ".join(columns)
        sql = f"INSERT INTO {self.table} ({column_sql}) VALUES ({placeholders})"
        values = [tuple(row.get(column) for column in columns) for row in rows]
        with oracledb.connect(user=self.user, password=self.password, dsn=self.dsn) as connection:
            with connection.cursor() as cursor:
                cursor.executemany(sql, values)
            connection.commit()
        return len(rows)
