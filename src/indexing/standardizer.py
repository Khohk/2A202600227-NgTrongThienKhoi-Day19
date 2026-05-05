"""
Second Pass — Entity Standardization.
- Basic: text normalization + rapidfuzz dedup
- LLM: review all unique entities, group same-concept ones
"""

import json
import os
import re
from rapidfuzz import fuzz
from openai import OpenAI
from dotenv import load_dotenv
from src.indexing.token_counter import record as _record_tokens

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

STANDARDIZE_PROMPT = """You are a Knowledge Graph expert.
Below is a list of entity names extracted from multiple text chunks.
Many of them refer to the SAME real-world entity but are written differently.

Task: Return a JSON mapping from each variant → canonical (official) name.
- Use the most common / official form as the canonical name.
- Only merge entities you are CONFIDENT refer to the same thing.
- Do not merge entities that are genuinely different.

Entities:
{entities}

Output ONLY valid JSON:
{{"mapping": {{"variant_name": "canonical_name", ...}}}}
"""


# ── Basic: fuzzy dedup ───────────────────────────────────────────────────────

def _fuzzy_mapping(entities: list[str], threshold: int = 88) -> dict[str, str]:
    canonical: dict[str, str] = {}
    for name in sorted(entities, key=len):          # shorter = more canonical
        matched = False
        for seen in canonical:
            if fuzz.ratio(name.lower(), seen.lower()) >= threshold:
                canonical[name] = canonical[seen]   # map to existing canonical
                matched = True
                break
        if not matched:
            canonical[name] = name
    return canonical


# ── LLM: semantic grouping ───────────────────────────────────────────────────

def _llm_mapping(entities: list[str]) -> dict[str, str]:
    entity_list = "\n".join(f"- {e}" for e in entities)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a Knowledge Graph expert. Always respond with valid JSON."},
            {"role": "user",   "content": STANDARDIZE_PROMPT.format(entities=entity_list)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    _record_tokens("standardization", resp.usage)
    return json.loads(resp.choices[0].message.content).get("mapping", {})


# ── Apply mapping to triples ─────────────────────────────────────────────────

SKIP_PREDICATES = {"IS", "BE", "HAS", "HAVE", "DO", "GET", "WAS", "ARE", "YEAR", "EVENT", "SAID", "NOTED"}
MAX_OBJECT_LEN  = 40   # object node quá dài là sentence → bỏ

# Phrase-like objects that are NOT valid entity names
_NOISE_OBJ_RE = re.compile(
    r"^\$"                                             # monetary: "$500 million"
    r"|^millions?\s+of\b"                              # quantity: "millions of books"
    r"|\s+a\s+\w"                                     # article mid-phrase: "Anthropic a supply chain"
    r"|\b(developed|selling|pirated|processing|delegating|strike)\b"  # verb phrases
    r"|\b\d{4}\s+\w+\s+(war|attack|crisis|conflict)\b",  # news events: "2026 Iran war"
    re.IGNORECASE,
)

def _apply(triples: list[dict], mapping: dict[str, str]) -> list[dict]:
    result = []
    seen = set()
    for t in triples:
        s = mapping.get(t["subject"], t["subject"]).strip()
        p = t["predicate"].strip().upper()
        o = mapping.get(t["object"],  t["object"]).strip()

        # Lọc noise
        if not s or not o:
            continue
        if p in SKIP_PREDICATES:
            continue
        if len(o) > MAX_OBJECT_LEN:      # object là cả câu → bỏ
            continue
        if _NOISE_OBJ_RE.search(o):   # phrase/monetary/event → bỏ
            continue

        key = (s, p, o)
        if key in seen:
            continue
        seen.add(key)

        # Giữ lại toàn bộ field phụ (source_file, subject_type, year...)
        new_t = {**t, "subject": s, "predicate": p, "object": o}
        result.append(new_t)
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def standardize(triples: list[dict], use_llm: bool = True) -> list[dict]:
    entities = list({t["subject"] for t in triples} | {t["object"] for t in triples})

    # Step 1: fuzzy baseline
    mapping = _fuzzy_mapping(entities)

    # Step 2: LLM semantic grouping (optional)
    if use_llm:
        print(f"  [Standardize] Sending {len(entities)} entities to LLM...")
        llm_map = _llm_mapping(entities)
        # LLM mapping overrides fuzzy where confident
        mapping.update(llm_map)

    before = len(triples)
    triples = _apply(triples, mapping)
    print(f"  [Standardize] {before} → {len(triples)} triples after dedup")
    return triples
