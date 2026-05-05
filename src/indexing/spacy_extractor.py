"""
No-LLM extractor using spaCy NER + dependency parsing.
Replaces entity_extractor.py for testing without API key.

Install: pip install spacy && python -m spacy download en_core_web_sm
"""

import re
import spacy

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _chunk_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += chunk_size - overlap
    return chunks


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_triples_spacy(text: str) -> list[dict]:
    nlp  = _get_nlp()
    doc  = nlp(text)
    triples = []

    # ── Strategy 1: NER pairs linked by root verb ────────────────────────────
    ents = [e for e in doc.ents if e.label_ in {
        "ORG", "PERSON", "GPE", "PRODUCT", "WORK_OF_ART", "EVENT", "DATE", "MONEY", "CARDINAL"
    }]

    for sent in doc.sents:
        sent_ents = [e for e in ents if e.start >= sent.start and e.end <= sent.end]
        if len(sent_ents) < 2:
            continue

        root = [t for t in sent if t.dep_ == "ROOT"]
        predicate = root[0].lemma_.upper() if root else "RELATED_TO"

        # Link first ORG/PERSON to every other entity in same sentence
        subjects   = [e for e in sent_ents if e.label_ in {"ORG", "PERSON"}]
        objects    = [e for e in sent_ents if e not in subjects]

        if not subjects:
            subjects = sent_ents[:1]
            objects  = sent_ents[1:]

        for subj in subjects[:1]:
            for obj in objects:
                s = _clean(subj.text)
                o = _clean(obj.text)
                if s != o:
                    triples.append({"subject": s, "predicate": predicate, "object": o})

    # ── Strategy 2: nsubj → root → dobj/attr dependency arcs ────────────────
    for token in doc:
        if token.dep_ == "ROOT":
            subj_toks = [c for c in token.children if c.dep_ in {"nsubj", "nsubjpass"}]
            obj_toks  = [c for c in token.children if c.dep_ in {"dobj", "attr", "pobj", "appos"}]

            for s_tok in subj_toks:
                for o_tok in obj_toks:
                    s = _clean(s_tok.text)
                    o = _clean(o_tok.text)
                    p = token.lemma_.upper()
                    if len(s) > 1 and len(o) > 1 and s != o:
                        triples.append({"subject": s, "predicate": p, "object": o})

    return triples


def extract_from_file_spacy(filepath: str, chunk_size: int = 200, overlap: int = 20) -> list[dict]:
    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    # Strip markdown syntax
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*\*|\*|__|_|`", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\n{2,}", " ", text)

    chunks = _chunk_words(text, chunk_size, overlap)
    all_triples = []
    for i, chunk in enumerate(chunks):
        print(f"  chunk {i+1}/{len(chunks)}...", end=" ", flush=True)
        triples = extract_triples_spacy(chunk)
        all_triples.extend(triples)
        print(f"{len(triples)} triples")

    return all_triples
