"""
Accumulates OpenAI token usage across pipeline stages and reports cost.
Usage:
    from src.indexing.token_counter import record, report, save_csv
    record("extraction", response.usage)
"""

import csv
import os

_usage: dict[str, dict] = {}

# gpt-4o-mini pricing (USD per 1M tokens, May 2025)
_PRICE_IN  = 0.15
_PRICE_OUT = 0.60


def record(stage: str, usage) -> None:
    """Pass the response.usage object from any OpenAI completion call."""
    if stage not in _usage:
        _usage[stage] = {"prompt": 0, "completion": 0, "calls": 0}
    _usage[stage]["prompt"]     += usage.prompt_tokens
    _usage[stage]["completion"] += usage.completion_tokens
    _usage[stage]["calls"]      += 1


def _cost(prompt: int, completion: int) -> float:
    return (prompt * _PRICE_IN + completion * _PRICE_OUT) / 1_000_000


def report() -> str:
    if not any(d["calls"] for d in _usage.values()):
        return "(no LLM calls recorded)"
    col = "{:<20} {:>6} {:>10} {:>12} {:>12}"
    lines = [col.format("Stage", "Calls", "Prompt", "Completion", "Cost (USD)"),
             "-" * 64]
    total_cost = 0.0
    for stage, d in _usage.items():
        if not d["calls"]:
            continue
        c = _cost(d["prompt"], d["completion"])
        total_cost += c
        lines.append(col.format(stage, d["calls"],
                                f"{d['prompt']:,}", f"{d['completion']:,}",
                                f"${c:.4f}"))
    lines.append("-" * 64)
    tp = sum(d["prompt"]     for d in _usage.values())
    tc = sum(d["completion"] for d in _usage.values())
    lines.append(col.format("TOTAL", "", f"{tp:,}", f"{tc:,}", f"${total_cost:.4f}"))
    return "\n".join(lines)


def save_csv(path: str = "outputs/results/token_usage.csv") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "stage", "calls", "prompt_tokens", "completion_tokens", "cost_usd"])
        w.writeheader()
        for stage, d in _usage.items():
            if not d["calls"]:
                continue
            w.writerow({"stage": stage, "calls": d["calls"],
                        "prompt_tokens": d["prompt"],
                        "completion_tokens": d["completion"],
                        "cost_usd": round(_cost(d["prompt"], d["completion"]), 6)})
    print(f"Token usage saved: {path}")
