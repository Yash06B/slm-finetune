"""Safe, read-only SQL execution and result comparison (execution accuracy)."""

from __future__ import annotations

import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .schema import DB_PATH, build_db

MAX_ROWS = 5000
TIMEOUT_S = 5.0

_WRITE_RE = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum)\b", re.I)


@dataclass
class ExecResult:
    ok: bool
    rows: list[tuple] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    error: str | None = None
    truncated: bool = False


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    build_db(path)
    # mode=ro: even a malicious completion cannot modify the database.
    return sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True, check_same_thread=False)


def execute(sql: str, con: sqlite3.Connection | None = None, timeout_s: float = TIMEOUT_S) -> ExecResult:
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        return ExecResult(False, error="empty query")
    if not re.match(r"^(select|with)\b", sql, re.I):
        return ExecResult(False, error="only SELECT/WITH queries are allowed")
    if _WRITE_RE.search(re.sub(r"'[^']*'", "''", sql)):
        return ExecResult(False, error="write/DDL keyword in query")
    own = con is None
    con = con or connect()
    deadline = time.monotonic() + timeout_s
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(MAX_ROWS + 1)
        cols = [d[0] for d in cur.description or []]
        truncated = len(rows) > MAX_ROWS
        return ExecResult(True, rows[:MAX_ROWS], cols, truncated=truncated)
    except sqlite3.Warning as e:  # e.g. "You can only execute one statement at a time."
        return ExecResult(False, error=f"sqlite warning: {e}")
    except sqlite3.Error as e:
        msg = "timeout" if "interrupted" in str(e) else str(e)
        return ExecResult(False, error=msg)
    finally:
        con.set_progress_handler(None, 0)
        if own:
            con.close()


def _norm_value(v):
    if isinstance(v, float):
        v = round(v, 2)
        return int(v) if v == int(v) else v
    if isinstance(v, str):
        return v.strip()
    return v


def _norm_row(row: tuple) -> tuple:
    # Sort values within a row so that a different column order still matches
    # (SELECT name, total vs SELECT total, name). Values are keyed by (type, repr).
    return tuple(sorted((_norm_value(v) for v in row), key=lambda x: (type(x).__name__, repr(x))))


def has_order_by(sql: str) -> bool:
    # Only the outermost ORDER BY matters; a heuristic is fine: last ORDER BY not inside parentheses.
    depth, s = 0, sql.lower()
    for i, ch in enumerate(s):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and s.startswith("order by", i):
            return True
    return False


def results_match(gold: ExecResult, pred: ExecResult, gold_sql: str) -> bool:
    """Execution accuracy: same multiset of rows (ordered iff the gold query has an outer ORDER BY)."""
    if not (gold.ok and pred.ok):
        return False
    g = [_norm_row(r) for r in gold.rows]
    p = [_norm_row(r) for r in pred.rows]
    if has_order_by(gold_sql):
        return g == p
    return Counter(g) == Counter(p)
