"""Per-token API prices and the cost model used in the results table.

API models: cost = input_tokens * in_price + output_tokens * out_price.
Self-hosted models: there is no per-token bill, so we report an *amortized GPU
cost*: wall-clock seconds per request * (hourly GPU price / 3600). That assumes
one request at a time (batch size 1), which overstates the cost of a server that
batches requests, so it is a conservative upper bound.

Prices in USD per 1M tokens. Claude prices as of 2026-10 (platform.claude.com
pricing page); re-check before publishing numbers.
"""

from __future__ import annotations

API_PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),  # prompts up to 100K tokens
}


def model_id(backend_name: str) -> str:
    return backend_name.split(":", 1)[1] if ":" in backend_name else backend_name


def request_cost(backend_name: str, input_tokens: int, output_tokens: int, latency_s: float,
                 gpu_hourly_usd: float | None = None) -> float | None:
    """USD for one request, or None when no price is known."""
    if backend_name.startswith(("ollama:", "hf:")) or (backend_name.startswith("openai:") and gpu_hourly_usd):
        return latency_s * gpu_hourly_usd / 3600 if gpu_hourly_usd is not None else None
    prices = API_PRICES.get(model_id(backend_name))
    if prices is None:
        return None
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000
