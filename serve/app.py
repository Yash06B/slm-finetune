"""Ask questions about the shop database in plain English, answered by the local model.

CLI:
  python serve/app.py ask "Which 5 customers spent the most in 2025?"
  python serve/app.py ask --model llama3.2:3b "How many orders were cancelled?"

HTTP (FastAPI):
  python serve/app.py serve --port 8008
  curl -X POST localhost:8008/query -H "content-type: application/json" -d '{"question": "How many customers are in Japan?"}'

The model only writes the SQL. The app runs it against a read-only connection and
returns the rows, so a bad generation can never modify data.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from sqlbench.execution import connect, execute  # noqa: E402
from sqlbench.llm import OllamaBackend  # noqa: E402
from sqlbench.prompts import build_messages, extract_sql  # noqa: E402

DEFAULT_MODEL = os.environ.get("SLM_MODEL", "sql-qwen")
_DB_LOCK = threading.Lock()  # FastAPI runs sync handlers in a thread pool; one sqlite connection


def answer(question: str, backend: OllamaBackend, con) -> dict:
    comp = backend.chat(build_messages(question))
    if comp.error:
        return {"question": question, "error": f"model call failed: {comp.error}"}
    sql = extract_sql(comp.text)
    with _DB_LOCK:
        res = execute(sql, con)
    return {
        "question": question, "sql": sql, "columns": res.columns, "rows": res.rows[:200],
        "row_count": len(res.rows), "error": res.error, "model_latency_ms": round(comp.latency_s * 1000),
        "model": backend.model,
    }


def print_table(columns: list[str], rows: list[tuple], limit: int = 20) -> None:
    cells = [[str(c) for c in columns]] + [["" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
                                            for v in r] for r in rows[:limit]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(columns))]
    for j, row in enumerate(cells):
        print("  " + " | ".join(v.ljust(w) for v, w in zip(row, widths)))
        if j == 0:
            print("  " + "-+-".join("-" * w for w in widths))
    if len(rows) > limit:
        print(f"  ... {len(rows) - limit} more rows")


class Query(BaseModel):
    question: str


def create_app(model: str) -> FastAPI:
    backend, con = OllamaBackend(model), connect()
    app = FastAPI(title="slm-finetune text-to-SQL", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model}

    @app.post("/query")
    def query(q: Query):
        if not q.question.strip():
            raise HTTPException(400, "question is empty")
        return answer(q.question, backend, con)

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="answer one question in the terminal")
    a.add_argument("question")
    a.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name (default: $SLM_MODEL or sql-qwen)")
    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8008)
    args = ap.parse_args()

    if args.cmd == "ask":
        out = answer(args.question, OllamaBackend(args.model), connect())
        if "sql" not in out:
            raise SystemExit(out["error"])
        print(f"\nSQL ({out['model']}, {out['model_latency_ms']} ms):\n  {out['sql']}\n")
        if out["error"]:
            print(f"Query failed: {out['error']}")
        else:
            print_table(out["columns"], out["rows"])
    else:
        import uvicorn

        uvicorn.run(create_app(args.model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
