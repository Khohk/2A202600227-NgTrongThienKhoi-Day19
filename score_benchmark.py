"""
Auto-score flat_rag_correct / hybrid_rag_correct in benchmark.csv.

Scoring logic:
  - "Not found" / empty answer      → N
  - Key phrase from expected present → Y
  - Otherwise                        → N

Run: python score_benchmark.py
"""

import csv
import re

CSV_PATH = "outputs/results/benchmark.csv"

# Key phrases that MUST appear in the answer for it to count as correct.
# One match from the list is enough.
KEY_PHRASES: list[list[str]] = [
    # Q1
    ["vice president of research", "vp of research"],
    # Q2
    ["amazon", "aws"],
    # Q3
    ["30 billion", "$30"],
    # Q4
    ["ceo of microsoft ai", "microsoft ai"],
    # Q5
    ["copilot"],
    # Q6
    ["tpu"],
    # Q7
    ["openai", "dod", "200m", "department of defense"],
    # Q8
    ["mustafa suleyman", "ceo of microsoft ai"],
    # Q9
    ["openai"],
    # Q10
    ["anthropic"],
    # Q11
    ["openai", "anthropic"],
    # Q12
    ["microsoft", "antitrust", "doj", "eu"],
    # Q13
    ["azure", "aws", "amazon web services"],
    # Q14
    ["openai", "sam altman"],
    # Q15
    ["openai", "jony ive", "apple designer"],
    # Q16
    ["anthropic", "openai"],
    # Q17
    ["claude", "anthropic", "gemini"],
    # Q18
    ["852", "openai"],
    # Q19
    ["anthropic", "nvidia"],
    # Q20
    ["alphago", "alphafold"],
]

_NOT_FOUND_RE = re.compile(r"not found|no relevant|no information", re.IGNORECASE)


def score(answer: str, phrases: list[str]) -> str:
    if not answer or _NOT_FOUND_RE.search(answer):
        return "N"
    a = answer.lower()
    return "Y" if any(p in a for p in phrases) else "N"


def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows):
        kp = KEY_PHRASES[i] if i < len(KEY_PHRASES) else []
        row["flat_rag_correct"]   = score(row.get("flat_rag_answer",   ""), kp)
        row["hybrid_rag_correct"] = score(row.get("hybrid_rag_answer", ""), kp)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    flat_y   = sum(1 for r in rows if r["flat_rag_correct"]   == "Y")
    hybrid_y = sum(1 for r in rows if r["hybrid_rag_correct"] == "Y")
    n = len(rows)
    print(f"Scored {n} questions:")
    print(f"  Flat RAG   : {flat_y}/{n} correct ({flat_y/n*100:.0f}%)")
    print(f"  Hybrid RAG : {hybrid_y}/{n} correct ({hybrid_y/n*100:.0f}%)")
    print(f"Saved -> {CSV_PATH}")


if __name__ == "__main__":
    main()
