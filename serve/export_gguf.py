"""Merge the LoRA adapter into the base model and export a quantized GGUF for Ollama.

Run where you trained (Colab/Kaggle GPU); Unsloth builds llama.cpp on first use.

  python serve/export_gguf.py --adapter outputs/qwen2.5-1.5b-sql/lora_adapter --quant q4_k_m

Produces serve/gguf/sql-qwen2.5-1.5b.Q4_K_M.gguf and serve/Modelfile.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, default=ROOT / "outputs/qwen2.5-1.5b-sql/lora_adapter")
    ap.add_argument("--quant", default="q4_k_m", help="q4_k_m (default), q5_k_m, q8_0, f16 ...")
    ap.add_argument("--name", default="sql-qwen2.5-1.5b")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "serve/gguf")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    args = ap.parse_args()

    from unsloth import FastLanguageModel

    # Loading the adapter dir pulls the base model named in adapter_config.json.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter), max_seq_length=args.max_seq_length, load_in_4bit=False, dtype=None)
    tmp = args.out_dir / "_unsloth_export"
    model.save_pretrained_gguf(str(tmp), tokenizer, quantization_method=args.quant)

    # Unsloth's output filename differs between versions; find the quantized file.
    candidates = sorted(tmp.parent.glob("**/*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = [p for p in candidates if args.quant.lower() in p.name.lower()] or candidates
    if not candidates:
        raise SystemExit(f"no .gguf produced under {args.out_dir}")
    final = args.out_dir / f"{args.name}.{args.quant.upper()}.gguf"
    shutil.move(str(candidates[0]), final)
    print(f"GGUF: {final} ({final.stat().st_size / 1e9:.2f} GB)")

    family = "llama" if "llama" in str(args.name).lower() else "qwen"
    subprocess.run([sys.executable, str(ROOT / "serve/make_modelfile.py"), "--gguf", str(final),
                    "--family", family], check=True)


if __name__ == "__main__":
    main()
