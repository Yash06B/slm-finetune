"""Write an Ollama Modelfile for the fine-tuned GGUF.

The system prompt is taken from sqlbench/prompts.py, so the served model sees the
exact prompt it was trained on. Clients that send their own system message (like
eval/run_eval.py) override it; clients that send only the question get it for free.

  python serve/make_modelfile.py --gguf serve/sql-qwen2.5-1.5b.Q4_K_M.gguf
  ollama create sql-qwen -f serve/Modelfile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlbench.prompts import SYSTEM_PROMPT  # noqa: E402

TEMPLATES = {
    # ChatML (Qwen2.5). Ollama puts the SYSTEM below into .Messages when the
    # request has no system message, so ranging over .Messages is enough.
    "qwen": (
        '{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}<|im_start|>assistant\n',
        ["<|im_end|>", "<|im_start|>"],
    ),
    "llama": (
        '{{- range .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>\n\n{{ .Content }}<|eot_id|>'
        '{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\n',
        ["<|eot_id|>", "<|start_header_id|>"],
    ),
}


def render(gguf: str, family: str) -> str:
    template, stops = TEMPLATES[family]
    system = SYSTEM_PROMPT.replace('"""', "'''")
    lines = [
        f"FROM {gguf}",
        f'TEMPLATE """{template}"""',
        f'SYSTEM """{system}"""',
        "PARAMETER temperature 0",
        "PARAMETER num_ctx 4096",
        "PARAMETER num_predict 512",
        *[f'PARAMETER stop "{s}"' for s in stops],
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="path to the .gguf, relative to the Modelfile or absolute")
    ap.add_argument("--family", choices=list(TEMPLATES), default="qwen")
    ap.add_argument("--out", type=Path, default=ROOT / "serve" / "Modelfile")
    args = ap.parse_args()
    gguf = Path(args.gguf)
    if not args.gguf.endswith(".gguf"):
        ref = args.gguf  # a registry model name such as llama3.2:3b (handy for testing the template)
    elif gguf.is_absolute():
        ref = gguf.as_posix()
    else:
        # Ollama resolves FROM relative to the Modelfile's directory; "./" marks it as a file path.
        try:
            ref = "./" + gguf.resolve().relative_to(args.out.resolve().parent).as_posix()
        except ValueError:
            ref = gguf.resolve().as_posix()
    args.out.write_text(render(ref, args.family), encoding="utf-8", newline="\n")
    print(f"wrote {args.out}\nnext: ollama create sql-qwen -f {args.out}")


if __name__ == "__main__":
    main()
