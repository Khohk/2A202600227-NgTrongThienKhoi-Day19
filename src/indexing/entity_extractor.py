"""
First Pass — SPO Extraction with overlapping chunks.
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from src.indexing.token_counter import record as _record_tokens

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SPO_PROMPT = """You are a Knowledge Graph extraction expert.
Extract Subject-Predicate-Object triples AND classify each entity type.

=== RULES ===

[1] COREFERENCE — resolve every pronoun to the actual entity name.
    "He founded OpenAI" → subject = "Sam Altman", NOT "He" or "The CEO".

[2] DISAMBIGUATION — append tag only when truly ambiguous.
    "Apple" (tech company) → "Apple_Inc"
    "Apple" (fruit)        → "Apple_Fruit"

[3] NORMALIZE entity names — one canonical form across all triples.
    "Open AI" / "OAI" / "open-ai" → "OpenAI"

[4] PREDICATE must be SHORT UPPER_SNAKE_CASE verb phrases (max 3 words).
    Good : FOUNDED_BY  CEO_OF  INVESTED_IN  DEVELOPED  PARTNERED_WITH  ACQUIRED
    Bad  : MAKES_AVAILABLE  IS  FED_LLMS_WITH  HAS_BEEN_DESCRIBED_AS
    Map  : "IS CEO OF" → CEO_OF | "MAKES AVAILABLE" → PROVIDES | "IS" → skip

[5] NODE TYPE — classify every entity:
    "Person"       : individual humans (Sam Altman, Dario Amodei)
    "Organization" : companies, labs, agencies (OpenAI, Google, DoD)
    "Product"      : software, models, hardware (GPT-4, Claude, H100)
    "Location"     : cities, countries (San Francisco, France)
    "Concept"      : abstract ideas (AI safety, LLM)
    "Date"         : years or dates — use as edge property NOT as object node

[6] DATE RULE — never create a node for a year/date.
    Wrong: {{"subject":"OpenAI","predicate":"FOUNDED_IN","object":"2015"}}
    Right: {{"subject":"OpenAI","predicate":"FOUNDED_IN","object":"2015","year":2015}}
    → Put the date in the "year" field; keep "object" as the date string.

[7] AMOUNT RULE — for investment/funding/revenue amounts, store in "amount" field, NOT as object.
    Wrong: {{"subject":"Amazon","predicate":"INVESTED","object":"$4 billion"}}
    Right: {{"subject":"Amazon","predicate":"INVESTED_IN","object":"Anthropic","amount":"$4 billion","year":2023}}
    → object must be the ENTITY receiving the investment; dollar value goes in "amount".
    Wrong: {{"subject":"Nvidia","predicate":"INVESTED","object":"$30 billion in OpenAI"}}
    Right: {{"subject":"Nvidia","predicate":"INVESTED_IN","object":"OpenAI","amount":"$30 billion"}}

[8] OBJECT must be a SHORT ENTITY NAME (max 5 words), NOT a full sentence.
    Wrong: (Anthropic) --[ANNOUNCED]--> (stop selling products to Chinese entities)
    Right: skip this triple — the object is a policy statement, not an entity.

[8] Skip generic/vague triples where subject or object is:
    pronouns, "the company", "the firm", "it", "both", numbers alone.

=== OUTPUT FORMAT ===
{{"triples": [
  {{"subject": "Sam Altman",  "subject_type": "Person",       "predicate": "CEO_OF",      "object": "OpenAI",    "object_type": "Organization"}},
  {{"subject": "OpenAI",     "subject_type": "Organization",  "predicate": "DEVELOPED",   "object": "GPT-4",     "object_type": "Product"}},
  {{"subject": "OpenAI",     "subject_type": "Organization",  "predicate": "FOUNDED_IN",  "object": "2015",        "object_type": "Date",         "year": 2015}},
  {{"subject": "Amazon",     "subject_type": "Organization",  "predicate": "INVESTED_IN", "object": "Anthropic",   "object_type": "Organization", "amount": "$4 billion", "year": 2023}}
]}}

=== PASSAGE ===
\"\"\"{text}\"\"\"
"""


def _chunk_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += chunk_size - overlap
    return chunks


def extract_triples(text: str) -> list[dict]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a Knowledge Graph extraction expert. Always respond with valid JSON."},
            {"role": "user",   "content": SPO_PROMPT.format(text=text)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    _record_tokens("extraction", resp.usage)
    return json.loads(resp.choices[0].message.content).get("triples", [])


def extract_from_file(filepath: str, chunk_size: int = 200, overlap: int = 20) -> list[dict]:
    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    source_file = os.path.basename(filepath)
    chunks = _chunk_words(text, chunk_size, overlap)
    all_triples = []

    for i, chunk in enumerate(chunks):
        print(f"  chunk {i+1}/{len(chunks)}...", end=" ", flush=True)
        triples = extract_triples(chunk)

        # Best Practice: pointer ngược về Document Chunk gốc (slide Best Practices)
        for t in triples:
            t["source_file"]  = source_file
            t["source_chunk"] = i

        all_triples.extend(triples)
        print(f"{len(triples)} triples")

    return all_triples
