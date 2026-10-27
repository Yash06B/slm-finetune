"""The single prompt format used for training, evaluation and serving.

Keeping this in one place guarantees the fine-tuned model is evaluated and served
with exactly the prompt it was trained on, and that API models see the same schema.
"""

from __future__ import annotations

import re

from .schema import schema_prompt

SYSTEM_PROMPT = (
    "You translate questions into SQL for a SQLite e-commerce database.\n"
    "Reply with exactly one SQLite SELECT query and nothing else: no explanation, no markdown.\n\n"
    "Schema:\n" + schema_prompt()
)


def build_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def training_example(question: str, sql: str) -> dict:
    """A chat-format record for SFT: system + user + the gold SQL as the assistant turn."""
    return {"messages": build_messages(question) + [{"role": "assistant", "content": normalize_sql(sql)}]}


def normalize_sql(sql: str) -> str:
    sql = re.sub(r"\s+", " ", sql.strip()).rstrip(";").strip()
    return sql + ";"


_FENCE_RE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.S | re.I)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_START_RE = re.compile(r"\b(with|select)\b", re.I)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply (handles fences, <think> blocks and chatty preambles)."""
    text = _THINK_RE.sub("", text or "").strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    m = _START_RE.search(text)
    if not m:
        return text.strip()
    text = text[m.start():]
    # Stop at the first semicolon that is not inside a string literal.
    in_str = False
    for i, ch in enumerate(text):
        if ch == "'":
            in_str = not in_str
        elif ch == ";" and not in_str:
            return text[: i + 1].strip()
    return text.strip()
