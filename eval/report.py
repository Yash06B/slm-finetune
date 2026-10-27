"""Collect results/*/summary.json into a markdown table (with bootstrap 95% CIs).

  python eval/report.py                 # print the table
  python eval/report.py --update-readme # replace the block between RESULTS markers in README.md

Runs whose directory name starts with "smoke" are skipped unless --include-smoke.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def bootstrap_ci(correct: list[bool], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not correct:
        return float("nan"), float("nan")
    rng, n = random.Random(seed), len(correct)
    means = sorted(sum(rng.choices(correct, k=n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def fmt_cost(c):
    if c is None:
        return "n/a"
    return f"${c:.3f}" if c < 1 else f"${c:.2f}"


def build_table(include_smoke: bool) -> str:
    runs = []
    for s in sorted((ROOT / "results").glob("*/summary.json")):
        if s.parent.name.startswith("smoke") and not include_smoke:
            continue
        summary = json.loads(s.read_text(encoding="utf-8"))
        preds = [json.loads(line) for line in (s.parent / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        summary["ci"] = bootstrap_ci([p["correct"] for p in preds])
        runs.append(summary)
    if not runs:
        return "_No results yet. Run `eval/run_eval.py` (see Reproduce)._"

    out = []
    for test_set in sorted({r["test_set"] for r in runs}):
        rows = sorted((r for r in runs if r["test_set"] == test_set), key=lambda r: -r["execution_accuracy"])
        out.append(f"**Test set: `{test_set}`** (n={rows[0]['n']})\n")
        out.append("| Model | Hardware | Exec. accuracy (95% CI) | Easy / Med / Hard | Valid SQL | p50 latency | p95 latency | Cost / 1k requests |")
        out.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            lo, hi = r["ci"]
            d = r["accuracy_by_difficulty"]
            diff = " / ".join(f"{d[k] * 100:.0f}" if k in d else "-" for k in ("easy", "medium", "hard"))
            out.append(
                f"| {r['label']} | {r['hardware']} | **{r['execution_accuracy'] * 100:.1f}%** "
                f"({lo * 100:.0f}-{hi * 100:.0f}) | {diff} | {r['valid_sql_rate'] * 100:.0f}% | "
                f"{r['latency_p50_s']:.2f}s | {r['latency_p95_s']:.2f}s | {fmt_cost(r['cost_per_1k_usd'])} |")
        out.append("")
    return "\n".join(out).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update-readme", action="store_true")
    ap.add_argument("--include-smoke", action="store_true")
    args = ap.parse_args()
    table = build_table(args.include_smoke)
    print(table)
    if args.update_readme:
        readme = ROOT / "README.md"
        text = readme.read_text(encoding="utf-8")
        if START not in text or END not in text:
            raise SystemExit(f"README.md has no {START} ... {END} block")
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        readme.write_text(f"{head}{START}\n{table}\n{END}{tail}", encoding="utf-8")
        print("README.md updated")


if __name__ == "__main__":
    main()
