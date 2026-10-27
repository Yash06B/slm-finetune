"""Score a model on a test set: execution accuracy, latency and cost.

Every model gets the same zero-shot prompt (system prompt with the schema + the
question; see sqlbench/prompts.py). A prediction counts as correct when its SQL
executes and returns the same rows as the gold SQL.

  # local GGUF through Ollama (free)
  python eval/run_eval.py --backend ollama:sql-qwen --label "Qwen2.5-1.5B QLoRA (Q4_K_M)"

  # base model in Colab with transformers, and the fine-tuned adapter
  python eval/run_eval.py --backend hf:unsloth/Qwen2.5-1.5B-Instruct --load-in-4bit --gpu-hourly-usd 0.35
  python eval/run_eval.py --backend hf:unsloth/Qwen2.5-1.5B-Instruct --adapter outputs/qwen2.5-1.5b-sql/lora_adapter --load-in-4bit

  # API models (paid)
  python eval/run_eval.py --backend anthropic:claude-haiku-5-5 --workers 4

Outputs results/<run-name>/predictions.jsonl and summary.json. Re-running with the
same run name resumes and skips examples already scored.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlbench.execution import connect, execute, results_match  # noqa: E402
from sqlbench.llm import make_backend  # noqa: E402
from sqlbench.pricing import request_cost  # noqa: E402
from sqlbench.prompts import build_messages, extract_sql  # noqa: E402

TEST_SETS = {"gold": ROOT / "data/gold_test.jsonl", "synth": ROOT / "data/generated/test_synth.jsonl"}


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(preds: list[dict], meta: dict) -> dict:
    n = len(preds)
    lat = [p["latency_s"] for p in preds if not p.get("api_error")]
    costs = [p["cost_usd"] for p in preds if p.get("cost_usd") is not None]
    by_diff: dict[str, list[bool]] = {}
    for p in preds:
        by_diff.setdefault(p.get("difficulty", "?"), []).append(p["correct"])
    return {
        **meta,
        "n": n,
        "execution_accuracy": round(sum(p["correct"] for p in preds) / max(1, n), 4),
        "valid_sql_rate": round(sum(p["executed"] for p in preds) / max(1, n), 4),
        "accuracy_by_difficulty": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_diff.items())},
        "api_errors": sum(1 for p in preds if p.get("api_error")),
        "latency_p50_s": round(percentile(lat, 50), 3),
        "latency_p95_s": round(percentile(lat, 95), 3),
        "latency_mean_s": round(statistics.fmean(lat), 3) if lat else None,
        "avg_input_tokens": round(statistics.fmean(p["input_tokens"] for p in preds), 1) if n else None,
        "avg_output_tokens": round(statistics.fmean(p["output_tokens"] for p in preds), 1) if n else None,
        "cost_per_1k_usd": round(1000 * statistics.fmean(costs), 4) if costs and len(costs) == n else None,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, help="kind:model, e.g. ollama:sql-qwen, anthropic:claude-haiku-5-5")
    ap.add_argument("--test-set", default="gold", help="'gold', 'synth' or a path to a jsonl file")
    ap.add_argument("--label", default=None, help="display name in the results table")
    ap.add_argument("--run-name", default=None, help="results/<run-name>/ (default derived from backend + test set)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N examples (smoke tests)")
    ap.add_argument("--workers", type=int, default=1, help="parallel requests (API backends only)")
    ap.add_argument("--max-tokens", type=int, default=0, help="default 512 local, 4096 for Claude (thinking)")
    ap.add_argument("--effort", default="low", help="Claude effort level")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir for hf backends")
    ap.add_argument("--load-in-4bit", action="store_true", help="hf backends: load base in 4-bit like QLoRA")
    ap.add_argument("--gpu-hourly-usd", type=float, default=None,
                    help="self-hosted cost model: hourly price of the GPU this runs on")
    ap.add_argument("--hardware", default=None, help="free-text hardware note for the table, e.g. 'Colab T4'")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    test_path = TEST_SETS.get(args.test_set, Path(args.test_set))
    examples = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        examples = examples[: args.limit]

    kind = args.backend.split(":", 1)[0]
    kwargs = {}
    if kind == "anthropic":
        kwargs["effort"] = args.effort
    if kind == "hf":
        kwargs.update(adapter=args.adapter, load_in_4bit=args.load_in_4bit)
    backend = make_backend(args.backend, **kwargs)
    max_tokens = args.max_tokens or (4096 if kind == "anthropic" else 512)
    if kind != "anthropic" and args.workers > 1:
        print("note: forcing --workers 1 for local backends so latency is per-request, not contended")
        args.workers = 1

    test_name = test_path.stem
    run_name = args.run_name or re.sub(r"[^A-Za-z0-9._-]+", "_", f"{backend.name}__{test_name}").strip("_")
    out_dir = ROOT / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    done = {}
    if pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                p = json.loads(line)
                if not p.get("api_error"):
                    done[p["id"]] = p
    todo = [ex for ex in examples if ex["id"] not in done]
    print(f"{backend.name}: {len(examples)} examples ({len(done)} cached) from {test_path.name}")

    if todo and not args.no_warmup and hasattr(backend, "warmup"):
        backend.warmup()

    con = connect()
    gold_cache = {ex["id"]: execute(ex["sql"], con) for ex in examples}
    lock = threading.Lock()
    results = dict(done)

    def score(ex: dict) -> dict:
        comp = backend.chat(build_messages(ex["question"]), max_tokens=max_tokens)
        sql = extract_sql(comp.text)
        with lock:  # sqlite connection shared across threads
            pred = execute(sql, con)
        return {
            "id": ex["id"], "difficulty": ex.get("difficulty"), "question": ex["question"],
            "gold_sql": ex["sql"], "pred_sql": sql, "raw_output": comp.text,
            "executed": pred.ok, "exec_error": pred.error,
            "correct": results_match(gold_cache[ex["id"]], pred, ex["sql"]),
            "latency_s": round(comp.latency_s, 4), "input_tokens": comp.input_tokens,
            "output_tokens": comp.output_tokens,
            "cost_usd": request_cost(backend.name, comp.input_tokens, comp.output_tokens, comp.latency_s,
                                     args.gpu_hourly_usd),
            "api_error": comp.error if comp.error and not comp.text else None,
        }

    with pred_path.open("a", encoding="utf-8", newline="\n") as f, ThreadPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(score, todo), 1):
            results[r["id"]] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            mark = "ok " if r["correct"] else ("ERR" if r["api_error"] else "x  ")
            print(f"  [{i:3d}/{len(todo)}] {mark} {r['latency_s']:6.2f}s  {r['id']}")

    preds = [results[ex["id"]] for ex in examples if ex["id"] in results]
    meta = {
        "label": args.label or backend.name, "backend": backend.name, "test_set": test_name,
        "adapter": args.adapter, "effort": args.effort if kind == "anthropic" else None,
        "gpu_hourly_usd": args.gpu_hourly_usd,
        "hardware": args.hardware or ("API" if kind == "anthropic" else "local"),
    }
    summary = summarize(preds, meta)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
