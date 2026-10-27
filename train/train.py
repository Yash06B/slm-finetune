"""QLoRA fine-tuning with Unsloth + TRL. Needs a CUDA GPU (free Colab T4 is enough for 1.5B).

  python train/train.py                                   # defaults below
  python train/train.py --model unsloth/Qwen2.5-3B-Instruct --epochs 2 --report-to wandb

Writes to --output-dir:
  lora_adapter/        the LoRA weights + tokenizer (small, what you push to HF)
  train_log.csv        every logged step: loss, eval_loss, learning rate, grad norm
  hyperparams.json     the exact config of this run
  merged_16bit/        (with --save-merged) full model, input for GGUF export
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Response-only loss: mask everything before the assistant turn so the model is
# trained on the SQL, not on reproducing the (long, fixed) schema prompt.
RESPONSE_MARKERS = {
    "qwen": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    "llama": ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train-file", type=Path, default=ROOT / "data/generated/train.jsonl")
    ap.add_argument("--val-file", type=Path, default=ROOT / "data/generated/val.jsonl")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs/qwen2.5-1.5b-sql")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4, help="per-device micro batch")
    ap.add_argument("--grad-accum", type=int, default=4, help="effective batch = batch-size * grad-accum")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--report-to", default="none", choices=["none", "wandb"], help="CSV is always written")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--save-merged", action="store_true", help="also save a merged 16-bit model")
    ap.add_argument("--max-steps", type=int, default=-1, help="override epochs (for quick tests)")
    return ap.parse_args()


def main():
    args = parse_args()
    # Unsloth must be imported before transformers/trl to apply its patches.
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    from datasets import load_dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    family = "llama" if "llama" in args.model.lower() else "qwen"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_length, load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    data = load_dataset("json", data_files={"train": str(args.train_file), "validation": str(args.val_file)})

    def to_text(batch):
        return {"text": [tokenizer.apply_chat_template(m, tokenize=False) for m in batch["messages"]]}

    data = data.map(to_text, batched=True, remove_columns=data["train"].column_names)

    # TRL renamed a few arguments across versions; pass whichever this install accepts.
    cfg_params = inspect.signature(SFTConfig.__init__).parameters
    cfg = dict(
        output_dir=str(args.output_dir),
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.scheduler,
        optim="adamw_8bit",
        logging_steps=5,
        eval_steps=args.eval_steps,
        save_strategy="no",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        seed=args.seed,
        report_to=args.report_to,
        run_name=args.run_name or args.output_dir.name,
        packing=False,
    )
    cfg["max_length" if "max_length" in cfg_params else "max_seq_length"] = args.max_seq_length
    cfg["eval_strategy" if "eval_strategy" in cfg_params else "evaluation_strategy"] = "steps"

    trainer_kwargs = dict(model=model, train_dataset=data["train"], eval_dataset=data["validation"],
                          args=SFTConfig(**cfg))
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    instruction_part, response_part = RESPONSE_MARKERS[family]
    trainer = train_on_responses_only(trainer, instruction_part=instruction_part, response_part=response_part)

    log_path = args.output_dir / "train_log.csv"

    class CSVLogger(TrainerCallback):
        fields = ["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm", "time_s"]

        def __init__(self):
            self.t0 = time.time()
            with log_path.open("w", newline="") as f:
                csv.DictWriter(f, self.fields).writeheader()

        def on_log(self, _args, state, _control, logs=None, **_):
            row = {k: (logs or {}).get(k) for k in self.fields}
            row.update(step=state.global_step, epoch=round(state.epoch or 0, 3), time_s=round(time.time() - self.t0, 1))
            with log_path.open("a", newline="") as f:
                csv.DictWriter(f, self.fields).writerow(row)

    trainer.add_callback(CSVLogger())

    hp = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    hp.update(effective_batch=args.batch_size * args.grad_accum, n_train=len(data["train"]),
              n_val=len(data["validation"]), family=family)
    (args.output_dir / "hyperparams.json").write_text(json.dumps(hp, indent=2))

    t0 = time.time()
    result = trainer.train()
    metrics = {**result.metrics, **trainer.evaluate(), "wall_clock_min": round((time.time() - t0) / 60, 1)}
    import torch

    if torch.cuda.is_available():
        metrics["gpu"] = torch.cuda.get_device_name(0)
        metrics["peak_vram_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    (args.output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    model.save_pretrained(str(args.output_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "lora_adapter"))
    if args.save_merged:
        model.save_pretrained_merged(str(args.output_dir / "merged_16bit"), tokenizer, save_method="merged_16bit")
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
