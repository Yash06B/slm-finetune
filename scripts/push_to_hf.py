"""Publish the adapter, the GGUF and the dataset to the Hugging Face Hub.

Dry run by default: it prints what would be uploaded. Add --yes to actually push.
Needs `huggingface-cli login` (or HF_TOKEN) with a write token.

  python scripts/push_to_hf.py --user <hf-username>            # dry run
  python scripts/push_to_hf.py --user <hf-username> --yes      # push

Creates (names configurable):
  <user>/sql-qwen2.5-1.5b-lora      LoRA adapter + model card with the results table
  <user>/sql-qwen2.5-1.5b-GGUF      quantized GGUF + Modelfile, runnable with Ollama
  <user>/shop-text-to-sql           train/val/test_synth + gold test (dataset repo)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.report import build_table  # noqa: E402
from serve.make_modelfile import render as render_modelfile  # noqa: E402

CARD = """---
base_model: {base_model}
library_name: peft
license: apache-2.0
language: [en]
tags: [text-to-sql, sqlite, qlora, unsloth, lora]
datasets: [{dataset_repo}]
---

# {repo_name}

QLoRA fine-tune of `{base_model}` that turns English questions into SQLite queries for one fixed
e-commerce schema (7 tables: customers, orders, order_items, products, categories, payments, reviews).
Built to show how far a small, local model trained for one narrow task gets on 340 synthetic examples
and a free GPU. Quantized GGUF for Ollama: [{gguf_repo}](https://huggingface.co/{gguf_repo}).
Dataset: [{dataset_repo}](https://huggingface.co/datasets/{dataset_repo}).
{github_line}
## Results

Execution accuracy = the generated SQL returns the same rows as the gold SQL on the database.

{results}

## Training

{hyperparams}

## Use with Ollama

{ollama_usage}

## Limitations

Trained for this one schema only. It will not generalize to arbitrary databases without retraining.
The training data is synthetic (written by `qwen2.5-coder:7b` and filtered by execution), so it inherits
that model's blind spots. Large API models have not been evaluated on this test set yet.
"""

OLLAMA_USAGE = """The model expects the system prompt it was trained with (instructions + the full schema). The
Modelfile in the GGUF repo bakes it in, so create the model from it rather than with `ollama run hf.co/...`:

```bash
hf download {gguf_repo} --local-dir sql-qwen
cd sql-qwen
ollama create sql-qwen -f Modelfile
ollama run sql-qwen "Which 3 countries have the most customers?"
```"""

GGUF_CARD = """---
base_model: {lora_repo}
license: apache-2.0
language: [en]
tags: [text-to-sql, sqlite, gguf, ollama]
---

# {repo_name}

Q4_K_M GGUF of [{lora_repo}](https://huggingface.co/{lora_repo}), a QLoRA fine-tune of Qwen2.5-1.5B-Instruct
that writes SQLite for one fixed e-commerce schema. Runs on a laptop GPU through Ollama.

{ollama_usage}

Results, training details and limitations: see the [adapter's model card](https://huggingface.co/{lora_repo}).
"""

DATASET_CARD = """---
license: apache-2.0
language: [en]
task_categories: [text-generation]
tags: [text-to-sql, sqlite, synthetic]
---

# {repo_name}

Question → SQLite pairs for one fixed 7-table e-commerce schema, used to fine-tune
[{lora_repo}](https://huggingface.co/{lora_repo}).

| File | Rows | What it is |
|---|---|---|
| `train.jsonl` / `val.jsonl` | {n_train} / {n_val} | Synthetic pairs in chat format (`messages`) plus `question`, `sql`, `difficulty` |
| `test_synth.jsonl` | {n_test} | Held-out synthetic pairs, near-duplicates removed from train |
| `gold_test.jsonl` | {n_gold} | Hand-written test questions, never used for training |
| `stats.json` | | Generation stats: proposed, accepted, reject reasons |

Synthetic pairs were written by `qwen2.5-coder:7b` (Ollama) and kept only if the SQL is read-only, executes on the
database and returns non-empty, non-NULL rows. The database itself is generated deterministically by
`sqlbench/schema.py` in the code repo.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True, help="Hugging Face username or org")
    ap.add_argument("--run-dir", type=Path, default=ROOT / "outputs/qwen2.5-1.5b-sql")
    ap.add_argument("--gguf-dir", type=Path, default=ROOT / "serve/gguf")
    ap.add_argument("--model-name", default="sql-qwen2.5-1.5b")
    ap.add_argument("--dataset-name", default="shop-text-to-sql")
    ap.add_argument("--github-url", default="https://github.com/Yash06B/slm-finetune")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--yes", action="store_true", help="actually upload (default is a dry run)")
    args = ap.parse_args()

    adapter = args.run_dir / "lora_adapter"
    hp_path = args.run_dir / "hyperparams.json"
    hp = json.loads(hp_path.read_text()) if hp_path.exists() else {}
    base_model = hp.get("model", "unsloth/Qwen2.5-1.5B-Instruct")
    lora_repo = f"{args.user}/{args.model_name}-lora"
    gguf_repo = f"{args.user}/{args.model_name}-GGUF"
    data_repo = f"{args.user}/{args.dataset_name}"

    hp_md = "\n".join(f"- `{k}`: {v}" for k, v in hp.items() if k not in ("train_file", "val_file", "output_dir")) \
        or "_see train/train.py defaults_"
    ollama_usage = OLLAMA_USAGE.format(gguf_repo=gguf_repo)
    github_line = "" if "<you>" in args.github_url else f"Code, data pipeline and eval harness: {args.github_url}\n"
    card = CARD.format(base_model=base_model, dataset_repo=data_repo, repo_name=f"{args.model_name}-lora",
                       github_line=github_line, results=build_table(include_smoke=False),
                       hyperparams=hp_md, gguf_repo=gguf_repo, ollama_usage=ollama_usage)

    ggufs = sorted(args.gguf_dir.glob("*.gguf"))
    data_paths = [ROOT / "data/generated/train.jsonl", ROOT / "data/generated/val.jsonl",
                  ROOT / "data/generated/test_synth.jsonl", ROOT / "data/gold_test.jsonl",
                  ROOT / "data/generated/stats.json"]
    data_files = [p for p in data_paths if p.exists()]
    n = {p.stem: sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
         for p in data_files if p.suffix == ".jsonl"}

    # Each upload item: (local path or generated text, path in repo, description). On the Hub the files sit flat,
    # so the GGUF repo gets its own Modelfile pointing at ./<file>.gguf.
    lora_items = [(adapter, ".", "LoRA adapter folder")] if adapter.exists() else []
    if lora_items:
        lora_items.append((card, "README.md", "model card with results table"))
    gguf_items = []
    if ggufs:
        family = "llama" if "llama" in args.model_name.lower() else "qwen"
        gguf_items = [(g, g.name, f"{g.stat().st_size / 1e6:.0f} MB") for g in ggufs] + [
            (render_modelfile(f"./{ggufs[0].name}", family), "Modelfile", "Modelfile with the training system prompt"),
            (GGUF_CARD.format(lora_repo=lora_repo, repo_name=f"{args.model_name}-GGUF", ollama_usage=ollama_usage),
             "README.md", "model card"),
        ]
    data_items = [(p, p.name, f"{n.get(p.stem, '-')} rows" if p.suffix == ".jsonl" else "") for p in data_files]
    if data_items:
        data_items.append((DATASET_CARD.format(
            repo_name=args.dataset_name, lora_repo=lora_repo, n_train=n.get("train", 0), n_val=n.get("val", 0),
            n_test=n.get("test_synth", 0), n_gold=n.get("gold_test", 0)), "README.md", "dataset card"))

    plan = [(lora_repo, "model", lora_items), (gguf_repo, "model", gguf_items), (data_repo, "dataset", data_items)]
    for repo, kind, items in plan:
        url = f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}"
        print(f"{kind:7s} {url}  {'(private)' if args.private else '(public)'}")
        for src, dest, desc in items:
            where = src.relative_to(ROOT) if isinstance(src, Path) else "generated"
            print(f"          + {dest:38s} {desc:42s} <- {where}")
        if not items:
            print("          ! nothing to upload yet")
    if not args.yes:
        print("\nDry run. Re-run with --yes to upload.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    for repo, kind, items in plan:
        if not items:
            continue
        api.create_repo(repo, repo_type=kind, private=args.private, exist_ok=True)
        for src, dest, _ in items:
            if isinstance(src, Path) and src.is_dir():
                api.upload_folder(repo_id=repo, repo_type=kind, folder_path=str(src), ignore_patterns=["README.md"])
            else:
                data = src.encode() if isinstance(src, str) else str(src)
                api.upload_file(repo_id=repo, repo_type=kind, path_or_fileobj=data, path_in_repo=dest)
        print(f"uploaded {repo}")
    print(f"\nDone: https://huggingface.co/{lora_repo}")


if __name__ == "__main__":
    main()
