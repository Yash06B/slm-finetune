# slm-finetune: a 1.5B model that writes SQL for one database

Fine-tune a small open model (Qwen2.5-1.5B-Instruct) with QLoRA for one narrow job, turning English questions into SQLite
queries for a fixed e-commerce schema, and measure it against large API models zero-shot on **accuracy, latency and cost**.
The fine-tuned model runs locally through Ollama, so a question never leaves your machine.

Part of a series: [claude-overlay](https://github.com/<you>/claude-overlay) (local-first desktop copilot, Claude + Ollama) ·
RAG with an eval harness · MCP agent for issue triage · **this repo** (model-level work: data, training, eval, serving).

## Why this task

Text-to-SQL on a known schema is a narrow, high-volume task where a small specialist should be able to beat a generalist,
and it has an objective metric: **execution accuracy**. Run the generated SQL against the database and compare the rows
with the gold query's rows. No LLM-as-judge, no string matching.

## Results

> **No results yet.** The table below is filled in by `python eval/report.py --update-readme` from real runs in
> `results/`. Nothing in it is estimated or hand-entered.

<!-- RESULTS:START -->
_No results yet. Run `eval/run_eval.py` (see Reproduce)._
<!-- RESULTS:END -->

How to read it:

- **Exec. accuracy**: share of questions whose SQL returns exactly the gold rows (order-sensitive only when the gold query
  has an outer `ORDER BY`; column order and float rounding to 2 decimals are ignored). The 95% CI is a bootstrap over questions;
  with 50 gold questions it is wide, so the synthetic test split is reported as well.
- **Test sets**: `gold_test` is 50 hand-written questions (15 easy, 20 medium, 15 hard) checked against the database and
  never shown to the teacher model or used in training. `test_synth` is a held-out slice of the generated data
  (in-distribution); near-duplicates of any test question are removed from train.
- **Latency**: wall clock per request, one request at a time, after a warm-up call. p50 / p95 over the test set.
- **Cost per 1k requests**: API models use list prices × measured tokens (thinking tokens included). Self-hosted models
  use seconds per request × the GPU's hourly price (`--gpu-hourly-usd`), which assumes no batching and so overstates
  the cost of a real server. A laptop running Ollama is shown as `n/a` unless you set an hourly price.

## Architecture

```mermaid
flowchart LR
    subgraph data["data/"]
        S[(shop.db<br/>7 tables, seeded)] --> G
        T[Teacher LLM<br/>Claude] -->|question + SQL pairs| G[generate.py<br/>execute · filter · dedup]
        G --> TR[train.jsonl / val.jsonl]
        G --> TS[test_synth.jsonl]
        GOLD[gold_test.jsonl<br/>50 hand-written] -.->|overlap removed| G
    end
    subgraph train["train/ (Colab T4)"]
        TR --> Q[Unsloth QLoRA<br/>Qwen2.5-1.5B, 4-bit base]
        Q --> A[LoRA adapter]
    end
    subgraph serve["serve/"]
        A --> M[merge + GGUF Q4_K_M]
        M --> O[Ollama<br/>Modelfile]
        O --> API[FastAPI / CLI<br/>read-only SQL execution]
    end
    subgraph eval["eval/"]
        TS & GOLD --> H[run_eval.py]
        A & O & C[Claude API models] --> H
        H --> R[results/*.json → README table]
    end
```

One prompt format (`sqlbench/prompts.py`: system prompt with the full schema, then the question) is shared by data
generation, training, every evaluated model and the server, so the comparison is apples to apples and the served model
sees exactly what it was trained on.

## Repo layout

| Path | What it does |
|---|---|
| `sqlbench/schema.py` | Schema DDL + deterministic database builder (same bytes on every machine) |
| `sqlbench/execution.py` | Read-only SQL execution with timeout, and the execution-accuracy comparison |
| `sqlbench/prompts.py` | The single prompt format and SQL extraction from model output |
| `sqlbench/llm.py` | Backends with one interface: `ollama:`, `anthropic:`, `openai:` (any compatible server), `hf:` (transformers + LoRA) |
| `data/generate.py` | Synthetic pair generation with a teacher LLM, validation, dedup, train/val/test split |
| `data/gold_test.jsonl` | 50 hand-written test questions |
| `train/train.py` | QLoRA fine-tuning (Unsloth + TRL), CSV logging (+ optional W&B) |
| `train/colab_train.ipynb` | End-to-end notebook: train → eval → GGUF → download / push |
| `eval/run_eval.py` | Scores any backend on a test set: accuracy, p50/p95 latency, tokens, cost |
| `eval/report.py` | Builds the results table (bootstrap CIs) and writes it into this README |
| `serve/export_gguf.py` | Merge adapter, quantize to GGUF, write the Modelfile |
| `serve/app.py` | CLI and FastAPI demo backed by the local Ollama model |
| `scripts/push_to_hf.py` | Publish adapter, GGUF and dataset to Hugging Face (dry run unless `--yes`) |

## Training setup

| Hyperparameter | Value | Why |
|---|---|---|
| Base model | `unsloth/Qwen2.5-1.5B-Instruct` | Apache-2.0, strong at code for its size, fits a free T4 easily |
| Method | QLoRA: 4-bit NF4 base, LoRA on all attention + MLP projections | Full fine-tune quality at a fraction of the memory |
| LoRA rank / alpha / dropout | 16 / 32 / 0 | Standard starting point; the task is narrow |
| Learning rate, schedule | 2e-4, cosine, 3% warmup | Usual LoRA range |
| Effective batch | 16 (4 × 4 grad. accumulation) | |
| Epochs | 2 | Watch `eval_loss` in `train_log.csv`; synthetic data overfits fast |
| Max sequence length | 2048 | Schema prompt is ~650 tokens, answers are short |
| Loss | Assistant tokens only (`train_on_responses_only`) | Don't spend capacity re-learning the fixed schema prompt |
| Optimizer | AdamW 8-bit, weight decay 0.01 | |

Every run writes `hyperparams.json`, `train_log.csv` and `train_metrics.json` (wall clock, GPU, peak VRAM) next to the adapter.

## Reproduce

### 0. Setup (Windows, macOS or Linux)

```bash
python -m venv .venv
```
```bash
.venv/Scripts/python -m pip install -r requirements.txt
```
(On macOS/Linux use `.venv/bin/python`.) Install [Ollama](https://ollama.com) for local serving and evaluation.

```bash
.venv/Scripts/python -m sqlbench.schema
```

### 1. Generate training data

With Claude as the teacher (paid; prints the running cost). Set `ANTHROPIC_API_KEY` first.

```bash
.venv/Scripts/python data/generate.py --backend anthropic:claude-opus-5-5 --target 3000 --workers 8
```

Free smoke test with a local model (lower quality, for checking the pipeline only):

```bash
.venv/Scripts/python data/generate.py --backend ollama:qwen3:4b-instruct --target 30 --batch 5 --workers 1 --out data/generated/sample
```

Output: `data/generated/{train,val,test_synth}.jsonl` and `stats.json` (acceptance rate, reject reasons, cost).
Commit `data/generated/` so the Colab notebook can clone it.

### 2. Train on a free GPU (Colab)

1. Push this repo to GitHub (including `data/generated/`).
2. Open `train/colab_train.ipynb` in Colab (File → Upload notebook, or open it from GitHub).
3. Runtime → Change runtime type → **T4 GPU**.
4. In the first code cell set `REPO_URL` to your repo, then Runtime → **Run all**.
5. The notebook trains, evaluates base vs fine-tuned on both test sets, exports `serve/gguf/*.gguf` and downloads
   `slm-artifacts.zip`. Unzip it into this folder on your PC.

Kaggle works the same way (Settings → Accelerator → GPU, Internet on). A local NVIDIA GPU with 6 GB+ also fits the 1.5B
QLoRA run: use WSL2 or Linux, `pip install unsloth`, then `python train/train.py`.

### 3. Serve locally with Ollama

```bash
ollama create sql-qwen -f serve/Modelfile
```
```bash
.venv/Scripts/python serve/app.py ask "Which 5 customers spent the most on delivered orders in 2025?"
```
```bash
.venv/Scripts/python serve/app.py serve --port 8008
```

### 4. Evaluate

```bash
.venv/Scripts/python eval/run_eval.py --backend ollama:sql-qwen --label "Qwen2.5-1.5B QLoRA, Q4_K_M (ours)" --hardware "RTX 4050 laptop, Ollama"
```
```bash
.venv/Scripts/python eval/run_eval.py --backend anthropic:claude-haiku-5-5 --workers 4
```
```bash
.venv/Scripts/python eval/run_eval.py --backend anthropic:claude-sonnet-5-5 --workers 4
```
```bash
.venv/Scripts/python eval/report.py --update-readme
```

Add `--test-set synth` to any eval command for the in-distribution split, and `--limit 5` for a quick check.
The Claude runs cost real money: 50 gold questions × ~700 input tokens is cents on Haiku 5.5, a bit more on Sonnet 5.5.

### 5. Publish to Hugging Face

```bash
.venv/Scripts/python scripts/push_to_hf.py --user <hf-username>
```

That is a dry run listing every file and repo. Add `--yes` to upload. The model card embeds the results table.

## Design decisions

- **Execution accuracy over exact match.** Many SQL strings are correct; the rows are what matter.
- **Hand-written gold test set.** A test set produced by the same teacher that wrote the training data would reward
  copying the teacher's style. The gold set is written separately and overlap with it is filtered out of training.
- **Read-only by construction.** Queries run on a `mode=ro` SQLite connection with a statement allow-list and a timeout,
  in eval and in the server.
- **Deterministic database.** Seeded generator, so results reproduce on any machine without shipping the `.db`.

## Next steps

- LLM-judge pass on generated pairs (does the SQL answer the question?) to cut label noise, and measure its effect.
- Ablations: rank 8 vs 16 vs 64, 1 vs 3 epochs, 500 vs 3k vs 10k pairs (accuracy vs data size curve).
- Qwen2.5-3B and Llama-3.2-3B with the same pipeline; Q4_K_M vs Q8_0 accuracy/latency trade-off.
- Throughput numbers with batching (vLLM via the `openai:` backend) for a fairer self-hosted cost.
- Plug the model into claude-overlay as a local "ask your database" tool.
