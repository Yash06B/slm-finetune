"""Generate synthetic (question, SQL) training pairs with a large LLM, then validate,
deduplicate and split them.

Pipeline
  1. Sample a "focus" (business topic x SQL feature x difficulty) for diversity.
  2. Ask the teacher model for a batch of pairs as JSON.
  3. Keep a pair only if the SQL is a read-only SELECT that executes on db/shop.db
     and returns at least one row, and the question uses explicit dates.
  4. Deduplicate: exact SQL, exact question, and near-duplicate questions
     (token Jaccard), and drop anything that overlaps the hand-written gold test set.
  5. Hold out a synthetic test split, then a validation split; the rest is train.

Examples
  # real run with Claude (paid; ~3k pairs)
  python data/generate.py --backend anthropic:claude-opus-5-5 --target 3000 --workers 8

  # free: a local Ollama teacher (runs for hours; --resume continues after a stop)
  python data/generate.py --backend ollama:qwen2.5-coder:7b --target 3000 --workers 1
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlbench.env import load_dotenv  # noqa: E402
from sqlbench.execution import connect, execute  # noqa: E402
from sqlbench.llm import make_backend  # noqa: E402
from sqlbench.pricing import request_cost  # noqa: E402
from sqlbench.prompts import normalize_sql, training_example  # noqa: E402
from sqlbench.schema import schema_prompt  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent
GOLD_PATH = DATA_DIR / "gold_test.jsonl"

TOPICS = [
    "revenue and sales totals", "customer geography (countries, cities)", "loyalty tiers",
    "product catalog and pricing", "inventory and stock levels", "the category hierarchy (parent/child categories)",
    "product reviews and ratings", "payments, payment methods and refunds", "order status and fulfilment",
    "trends over time (by year or month)", "shipping costs", "discounts", "brand performance",
    "customer purchase behaviour (repeat buyers, first orders, inactive customers)", "profit and margins",
]
FEATURES = [
    "a simple WHERE filter", "aggregation (COUNT/SUM/AVG/MIN/MAX)", "GROUP BY with HAVING",
    "a join across three or more tables", "a correlated or uncorrelated subquery",
    "an anti-join (NOT EXISTS / NOT IN / LEFT JOIN ... IS NULL)", "a CTE (WITH clause)",
    "a window function (RANK, ROW_NUMBER, running SUM)", "CASE WHEN expressions",
    "date functions (strftime) for filtering or grouping", "a self-join on categories",
    "top-N with ORDER BY and LIMIT", "percentages or ratios", "COUNT(DISTINCT ...)",
]
DIFFICULTY_FOR_FEATURE = {
    "a simple WHERE filter": "easy", "aggregation (COUNT/SUM/AVG/MIN/MAX)": "easy",
    "top-N with ORDER BY and LIMIT": "easy", "COUNT(DISTINCT ...)": "medium",
    "GROUP BY with HAVING": "medium", "a join across three or more tables": "medium",
    "date functions (strftime) for filtering or grouping": "medium", "CASE WHEN expressions": "medium",
    "percentages or ratios": "medium", "a correlated or uncorrelated subquery": "hard",
    "an anti-join (NOT EXISTS / NOT IN / LEFT JOIN ... IS NULL)": "hard", "a CTE (WITH clause)": "hard",
    "a window function (RANK, ROW_NUMBER, running SUM)": "hard", "a self-join on categories": "hard",
}
STYLES = [
    "plain business English", "terse, like a dashboard search box", "casual, like a Slack message to an analyst",
    "formal, like a request from a finance manager", "a mix of the above",
]

GEN_SYSTEM = """You write training data for a text-to-SQL model that works on ONE fixed SQLite database.

Schema:
{schema}

Rules for every pair:
- The SQL must be valid SQLite, a single read-only SELECT (CTEs allowed), and must answer the question exactly.
- Select only the columns the question asks for, in a natural order. No SELECT *.
- Use explicit dates or years ("in 2024", "since 2025-03-01"). Never say "last month", "this year", "recently" or "today".
- If the question asks for a top-N or "the most", it must not depend on how ties are broken; add a tie-breaker to the question and SQL (e.g. "then by name") if needed.
- Only use values that exist in the schema notes (statuses, tiers, methods) and the category, brand, country and city names listed below.
- Vary the wording; do not start every question the same way.

Category names: {categories}
Brand names: {brands}
Countries and cities: {places}

Reply with a JSON array only, no prose: [{{"question": "...", "sql": "...", "difficulty": "easy|medium|hard"}}, ...]"""

GEN_USER = """Write {n} new question/SQL pairs.
Focus topic: {topic}.
Each SQL should use {feature} (difficulty: {difficulty}).
Question style: {style}.
Random seed for variety: {seed}."""

RELATIVE_DATE_RE = re.compile(r"\b(last|this|past|previous|next)\s+(week|month|quarter|year|\d+\s+days?)\b|\b(today|yesterday|recent(ly)?|currently this)\b", re.I)
WORD_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def sql_key(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.lower()).strip().rstrip(";").strip()


def parse_pairs(text: str) -> list[dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict) and isinstance(d.get("question"), str) and isinstance(d.get("sql"), str)]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class Pool:
    """Thread-safe accepted set with dedup against itself and the gold test set."""

    def __init__(self, gold: list[dict], near_dup: float, test_overlap: float):
        self.lock = threading.Lock()
        self.items: list[dict] = []
        self.sql_keys: set[str] = set()
        self.q_tokens: list[set[str]] = []
        self.gold_tokens = [tokens(g["question"]) for g in gold]
        self.gold_sql = {sql_key(g["sql"]) for g in gold}
        self.near_dup, self.test_overlap = near_dup, test_overlap
        self.rejects: Counter = Counter()

    def try_add(self, item: dict, con) -> str | None:
        q, sql = item["question"].strip(), item["sql"].strip()
        if not (8 <= len(q) <= 400):
            return "question_length"
        if RELATIVE_DATE_RE.search(q):
            return "relative_date"
        res = execute(sql, con)
        if not res.ok:
            return "sql_error"
        if not res.rows:
            return "empty_result"
        if all(v is None for row in res.rows for v in row):
            return "null_result"  # e.g. SUM over no rows: a filter value that does not exist
        if res.truncated:
            return "too_many_rows"
        k, toks = sql_key(sql), tokens(q)
        with self.lock:
            if k in self.gold_sql or any(jaccard(toks, g) >= self.test_overlap for g in self.gold_tokens):
                return "overlaps_gold_test"
            if k in self.sql_keys:
                return "duplicate_sql"
            if any(jaccard(toks, t) >= self.near_dup for t in self.q_tokens):
                return "near_duplicate_question"
            self.sql_keys.add(k)
            self.q_tokens.append(toks)
            # Use the difficulty we asked for (tied to the SQL feature); teachers' self-labels drift to "medium".
            diff = DIFFICULTY_FOR_FEATURE.get(item.get("_feature"), item.get("difficulty"))
            diff = diff if diff in ("easy", "medium", "hard") else "medium"
            self.items.append({"question": q, "sql": normalize_sql(sql), "difficulty": diff,
                               "topic": item.get("_topic"), "feature": item.get("_feature")})
        return None


def split(items: list[dict], test_frac: float, val_frac: float, seed: int, near_dup: float):
    """Stratified (by difficulty) test/val/train split. Train/val drop near-dups of test questions."""
    rng = random.Random(seed)
    by_diff: dict[str, list[dict]] = {}
    for it in items:
        by_diff.setdefault(it["difficulty"], []).append(it)
    test, val, train = [], [], []
    for group in by_diff.values():
        rng.shuffle(group)
        n_test, n_val = round(len(group) * test_frac), round(len(group) * val_frac)
        test += group[:n_test]
        val += group[n_test:n_test + n_val]
        train += group[n_test + n_val:]
    test_toks = [tokens(t["question"]) for t in test]
    keep = lambda it: not any(jaccard(tokens(it["question"]), t) >= near_dup for t in test_toks)  # noqa: E731
    dropped = len(train) + len(val)
    train, val = [x for x in train if keep(x)], [x for x in val if keep(x)]
    dropped -= len(train) + len(val)
    for i, t in enumerate(test):
        t["id"] = f"s{i + 1:04d}"
    return train, val, test, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="anthropic:claude-opus-5-5", help="teacher model, kind:model")
    ap.add_argument("--target", type=int, default=3000, help="accepted pairs to collect before splitting")
    ap.add_argument("--batch", type=int, default=10, help="pairs requested per LLM call")
    ap.add_argument("--workers", type=int, default=4, help="parallel LLM calls (use 1 for Ollama)")
    ap.add_argument("--max-calls", type=int, default=0, help="stop after this many calls (default: 3x needed)")
    ap.add_argument("--effort", default="medium", help="Claude effort level for the teacher")
    ap.add_argument("--temperature", type=float, default=0.8, help="sampling temperature for Ollama teachers")
    ap.add_argument("--near-dup", type=float, default=0.85, help="question Jaccard threshold for duplicates")
    ap.add_argument("--test-overlap", type=float, default=0.7, help="Jaccard threshold vs test questions")
    ap.add_argument("--test-frac", type=float, default=0.08)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "generated")
    ap.add_argument("--resume", action="store_true", help="reuse accepted pairs from <out>/accepted.jsonl")
    args = ap.parse_args()
    load_dotenv()

    kwargs = {}
    if args.backend.startswith("anthropic:"):
        kwargs["effort"] = args.effort
    elif args.backend.startswith("ollama:"):
        kwargs["temperature"] = args.temperature  # greedy decoding would repeat itself across calls
    backend = make_backend(args.backend, **kwargs)
    con = connect()
    gold = load_jsonl(GOLD_PATH)
    pool = Pool(gold, args.near_dup, args.test_overlap)
    if args.resume:
        for it in load_jsonl(args.out / "accepted.jsonl"):
            pool.try_add({**it, "_topic": it.get("topic"), "_feature": it.get("feature")}, con)
        print(f"resumed {len(pool.items)} accepted pairs")

    categories = ", ".join(r[0] for r in con.execute("SELECT name FROM categories ORDER BY category_id"))
    brands = ", ".join(r[0] for r in con.execute("SELECT DISTINCT brand FROM products ORDER BY brand"))
    places = "; ".join(f"{c}: {cities}" for c, cities in con.execute(
        "SELECT country, GROUP_CONCAT(DISTINCT city) FROM customers GROUP BY country ORDER BY country"))
    system = GEN_SYSTEM.format(schema=schema_prompt(), categories=categories, brands=brands, places=places)
    rng = random.Random(args.seed)
    max_calls = args.max_calls or max(3, 3 * (args.target - len(pool.items)) // args.batch + 1)

    raw_path = args.out / "raw.jsonl"
    args.out.mkdir(parents=True, exist_ok=True)
    raw_f = raw_path.open("a", encoding="utf-8", newline="\n")
    stats = {"calls": 0, "failed_calls": 0, "proposed": 0, "cost_usd": 0.0, "in_tok": 0, "out_tok": 0}

    def one_call(i: int):
        topic, feature, style = rng.choice(TOPICS), rng.choice(FEATURES), rng.choice(STYLES)
        user = GEN_USER.format(n=args.batch, topic=topic, feature=feature,
                               difficulty=DIFFICULTY_FOR_FEATURE[feature], style=style, seed=rng.randint(0, 10**6))
        comp = backend.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                            max_tokens=8000)
        return topic, feature, comp

    calls_made = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        while len(pool.items) < args.target and calls_made < max_calls:
            n = min(args.workers * 2, max_calls - calls_made)
            futures = [ex.submit(one_call, calls_made + j) for j in range(n)]
            calls_made += n
            for fut in as_completed(futures):
                topic, feature, comp = fut.result()
                stats["calls"] += 1
                stats["in_tok"] += comp.input_tokens
                stats["out_tok"] += comp.output_tokens
                stats["cost_usd"] += request_cost(backend.name, comp.input_tokens, comp.output_tokens, comp.latency_s) or 0.0
                if comp.error:
                    stats["failed_calls"] += 1
                    print(f"  call failed: {comp.error}", file=sys.stderr)
                    continue
                for pair in parse_pairs(comp.text):
                    stats["proposed"] += 1
                    pair.update(_topic=topic, _feature=feature)
                    reason = pool.try_add(pair, con)
                    pool.rejects[reason or "accepted"] += 1
                    raw_f.write(json.dumps({"question": pair["question"], "sql": pair["sql"], "topic": topic,
                                            "feature": feature, "status": reason or "accepted"}) + "\n")
            raw_f.flush()
            print(f"calls={stats['calls']:4d} accepted={len(pool.items):5d}/{args.target} "
                  f"proposed={stats['proposed']} cost=${stats['cost_usd']:.2f}")
    raw_f.close()

    write_jsonl(args.out / "accepted.jsonl", pool.items)
    train, val, test, dropped = split(pool.items, args.test_frac, args.val_frac, args.seed, args.test_overlap)
    write_jsonl(args.out / "train.jsonl", [{**training_example(x["question"], x["sql"]), **x} for x in train])
    write_jsonl(args.out / "val.jsonl", [{**training_example(x["question"], x["sql"]), **x} for x in val])
    write_jsonl(args.out / "test_synth.jsonl", test)

    summary = {
        "teacher": backend.name, **stats, "accepted": len(pool.items),
        "acceptance_rate": round(len(pool.items) / max(1, stats["proposed"]), 3),
        "reject_reasons": dict(pool.rejects), "train": len(train), "val": len(val), "test_synth": len(test),
        "dropped_for_test_overlap": dropped,
        "difficulty": dict(Counter(x["difficulty"] for x in pool.items)),
    }
    (args.out / "stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
