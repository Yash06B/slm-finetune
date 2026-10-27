"""Minimal model backends with one interface: chat(messages) -> Completion.

Every backend reports wall-clock latency and token counts so the eval harness can
compute p50/p95 latency and cost the same way for local and API models.

  ollama:<model>          local Ollama server (GGUF), e.g. ollama:sql-qwen
  anthropic:<model>       Claude API, e.g. anthropic:claude-haiku-5-5
  openai:<model>          any OpenAI-compatible /v1/chat/completions server
                          (vLLM, llama.cpp server, LM Studio...); set OPENAI_BASE_URL
  hf:<model-or-path>      transformers on a local GPU, optionally with a LoRA
                          adapter (pass adapter=...), used in Colab
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    error: str | None = None


class Backend:
    name: str

    def chat(self, messages: list[dict], max_tokens: int = 512) -> Completion:  # pragma: no cover
        raise NotImplementedError


def _post_json(url: str, payload: dict, headers: dict | None = None, timeout: float = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class OllamaBackend(Backend):
    def __init__(self, model: str, host: str | None = None, num_ctx: int = 4096):
        self.model, self.name = model, f"ollama:{model}"
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.num_ctx = num_ctx

    def chat(self, messages, max_tokens=512):
        t0 = time.perf_counter()
        try:
            r = _post_json(f"{self.host}/api/chat", {
                "model": self.model, "messages": messages, "stream": False, "think": False,
                "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": self.num_ctx},
            })
        except Exception as e:  # noqa: BLE001 - report and keep evaluating
            return Completion("", 0, 0, time.perf_counter() - t0, error=str(e))
        return Completion(r["message"]["content"], r.get("prompt_eval_count", 0), r.get("eval_count", 0),
                          time.perf_counter() - t0)

    def warmup(self):
        """Load the model into memory so the first timed request isn't a cold start."""
        self.chat([{"role": "user", "content": "SELECT 1"}], max_tokens=1)


class AnthropicBackend(Backend):
    def __init__(self, model: str, effort: str = "low", max_retries: int = 4):
        import anthropic

        self.model, self.name, self.effort = model, f"anthropic:{model}", effort
        self.client = anthropic.Anthropic(max_retries=max_retries)

    def chat(self, messages, max_tokens=2048):
        import anthropic

        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        t0 = time.perf_counter()
        try:
            # No temperature: current Claude models reject sampling parameters.
            # effort=low keeps thinking short; text-to-SQL does not need deep reasoning.
            resp = self.client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system, messages=convo,
                output_config={"effort": self.effort},
            )
        except anthropic.APIStatusError as e:
            return Completion("", 0, 0, time.perf_counter() - t0, error=f"{e.status_code}: {e.message}")
        except anthropic.APIConnectionError as e:
            return Completion("", 0, 0, time.perf_counter() - t0, error=f"connection: {e}")
        latency = time.perf_counter() - t0
        text = "".join(b.text for b in resp.content if b.type == "text")
        err = "refusal" if resp.stop_reason == "refusal" else None
        u = resp.usage
        in_tok = u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
        return Completion(text, in_tok, u.output_tokens, latency, error=err)


class OpenAICompatBackend(Backend):
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        self.model, self.name = model, f"openai:{model}"
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "none")

    def chat(self, messages, max_tokens=512):
        t0 = time.perf_counter()
        try:
            r = _post_json(f"{self.base_url}/chat/completions",
                           {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0},
                           headers={"Authorization": f"Bearer {self.api_key}"})
        except Exception as e:  # noqa: BLE001
            return Completion("", 0, 0, time.perf_counter() - t0, error=str(e))
        u = r.get("usage") or {}
        return Completion(r["choices"][0]["message"]["content"] or "", u.get("prompt_tokens", 0),
                          u.get("completion_tokens", 0), time.perf_counter() - t0)


class HFBackend(Backend):
    """Greedy decoding with transformers. Use load_in_4bit=True to match the QLoRA base."""

    def __init__(self, model: str, adapter: str | None = None, load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf:{model}" + (f"+{adapter}" if adapter else "")
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(adapter or model)
        kwargs = {"torch_dtype": "auto", "device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def chat(self, messages, max_tokens=512):
        torch = self.torch
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        new = out[0, ids.shape[1]:]
        return Completion(self.tok.decode(new, skip_special_tokens=True), ids.shape[1], int(new.shape[0]), latency)


def make_backend(spec: str, **kwargs) -> Backend:
    """Build a backend from 'kind:model', e.g. 'ollama:llama3.2:3b' or 'anthropic:claude-haiku-5-5'."""
    kind, _, model = spec.partition(":")
    if not model:
        raise ValueError(f"backend spec must look like kind:model, got {spec!r}")
    if kind == "ollama":
        return OllamaBackend(model, **kwargs)
    if kind == "anthropic":
        return AnthropicBackend(model, **kwargs)
    if kind == "openai":
        return OpenAICompatBackend(model, **kwargs)
    if kind == "hf":
        return HFBackend(model, **kwargs)
    raise ValueError(f"unknown backend kind {kind!r} (expected ollama, anthropic, openai or hf)")
