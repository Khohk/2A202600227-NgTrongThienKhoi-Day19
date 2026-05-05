"""
Test toàn bộ pipeline KHÔNG dùng LLM.
Chạy: python test_no_llm.py
"""

import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import networkx as nx

DATA_RAW     = "data/raw"
TRIPLES_PATH = "data/processed/triples_test.json"
GRAPH_PATH   = "outputs/graphs/kg_test.graphml"

with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)

# ── Pass 1: spaCy extraction ─────────────────────────────────────────────────
from src.indexing.spacy_extractor import extract_from_file_spacy

print("=" * 55)
print("Pass 1 — SPO Extraction (spaCy, no LLM)")
print("=" * 55)
all_triples = []

files = sorted(f for f in os.listdir(DATA_RAW) if f.endswith(".md"))
for fname in files:
    print(f"\n[ {fname} ]")
    triples = extract_from_file_spacy(
        os.path.join(DATA_RAW, fname),
        chunk_size=cfg["chunking"]["chunk_size"],
        overlap=cfg["chunking"]["overlap"],
    )
    all_triples.extend(triples)
    print(f"  → {len(triples)} triples from this file")

print(f"\nPass 1 total: {len(all_triples)} raw triples")

# ── Pass 2: Standardization (fuzzy only, no LLM) ────────────────────────────
from src.indexing.standardizer import standardize

print("\n" + "=" * 55)
print("Pass 2 — Standardization (rapidfuzz only)")
print("=" * 55)
all_triples = standardize(all_triples, use_llm=False)

# ── Pass 3: Inference (transitive rules only, no LLM) ───────────────────────
from src.indexing.inferencer import infer

print("\n" + "=" * 55)
print("Pass 3 — Inference (transitive rules only)")
print("=" * 55)
all_triples = infer(all_triples, apply_transitive_flag=True, use_llm=False)

# ── Save triples ─────────────────────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
with open(TRIPLES_PATH, "w", encoding="utf-8") as f:
    json.dump(all_triples, f, ensure_ascii=False, indent=2)
print(f"\nSaved {len(all_triples)} triples → {TRIPLES_PATH}")

# ── Build graph ──────────────────────────────────────────────────────────────
from src.graph.builder import build_graph, save_graph, visualize_html

print("\n" + "=" * 55)
print("Building NetworkX graph...")
print("=" * 55)
G = build_graph(all_triples)
save_graph(G, GRAPH_PATH)
visualize_html(G, "outputs/graphs/kg_test.html")

# ── Visualize (matplotlib) ───────────────────────────────────────────────────
from src.graph.visualizer import draw_graph

draw_graph(G, title="Tech Company KG — spaCy (no LLM)", output_path="outputs/graphs/kg_test.png")

# ── Quick stats ──────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Graph Stats")
print("=" * 55)
print(f"  Nodes : {G.number_of_nodes()}")
print(f"  Edges : {G.number_of_edges()}")
print(f"  Components (weakly connected): {nx.number_weakly_connected_components(G)}")

top5 = sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)[:5]
print(f"  Top-5 nodes by degree: {top5}")

print("\nDone. Check outputs/graphs/kg_test.png")
