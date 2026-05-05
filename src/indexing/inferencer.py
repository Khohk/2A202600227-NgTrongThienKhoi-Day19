"""
Third Pass — Relationship Inference.
- Rule-based: transitive closure (A→B, B→C ⟹ A→C) for selected predicates
- LLM-based: infer links between disconnected graph communities
"""

import json
import os
import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv
from src.indexing.token_counter import record as _record_tokens

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Chỉ áp transitive cho các relation có nghĩa khi chuỗi
TRANSITIVE_PREDICATES = {"PART_OF", "SUBSIDIARY_OF", "OWNED_BY", "BELONGS_TO"}

INFERENCE_PROMPT = """You are a Knowledge Graph expert.
The graph below has {n} disconnected components.
Here are representative entities from each component:

{components}

Task: Infer plausible relationships BETWEEN components that are not already in the graph.
Only infer relationships you are confident about based on general knowledge.

Output ONLY valid JSON:
{{"inferred": [
  {{"subject": "EntityA", "predicate": "RELATION", "object": "EntityB"}},
  ...
]}}
"""


# ── Rule-based: transitive closure ───────────────────────────────────────────

def apply_transitive(triples: list[dict]) -> list[dict]:
    G = nx.DiGraph()
    for t in triples:
        G.add_edge(t["subject"], t["object"], relation=t["predicate"])

    new_triples = []
    for pred in TRANSITIVE_PREDICATES:
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == pred]
        for u, v in edges:
            for _, w, d in G.out_edges(v, data=True):
                if d.get("relation") == pred and not G.has_edge(u, w):
                    new_triples.append({"subject": u, "predicate": pred, "object": w})

    if new_triples:
        print(f"  [Inference] Transitive: +{len(new_triples)} triples")
    return triples + new_triples


# ── LLM-based: bridge disconnected communities ───────────────────────────────

def _get_communities(G: nx.DiGraph, max_per_community: int = 5) -> list[list[str]]:
    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))
    # Only look at components that are actually disconnected (>1 component)
    if len(components) <= 1:
        return []
    # Take top-degree nodes from each component as representatives
    reps = []
    for comp in sorted(components, key=len, reverse=True)[:8]:  # max 8 components
        sub = G.subgraph(comp)
        top = sorted(comp, key=lambda n: sub.degree(n), reverse=True)[:max_per_community]
        reps.append(top)
    return reps


def apply_llm_inference(triples: list[dict]) -> list[dict]:
    G = nx.DiGraph()
    for t in triples:
        G.add_edge(t["subject"], t["object"], relation=t["predicate"])

    communities = _get_communities(G)
    if not communities:
        print("  [Inference] Graph is fully connected — skipping LLM inference")
        return triples

    formatted = "\n".join(
        f"Component {i+1}: {', '.join(c)}"
        for i, c in enumerate(communities)
    )
    print(f"  [Inference] Sending {len(communities)} disconnected components to LLM...")

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": INFERENCE_PROMPT.format(
            n=len(communities), components=formatted
        )}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    _record_tokens("inference", resp.usage)
    inferred = json.loads(resp.choices[0].message.content).get("inferred", [])
    print(f"  [Inference] LLM inferred: +{len(inferred)} triples")
    return triples + inferred


# ── Main ─────────────────────────────────────────────────────────────────────

def infer(triples: list[dict], apply_transitive_flag: bool = True, use_llm: bool = True) -> list[dict]:
    if apply_transitive_flag:
        triples = apply_transitive(triples)
    if use_llm:
        triples = apply_llm_inference(triples)
    return triples
