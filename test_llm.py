"""
Test pipeline dùng LLM (GPT).
Chạy thử với 1 file để kiểm tra chất lượng trước khi chạy toàn bộ.
Chạy: python test_llm.py [--file OpenAI] [--all]
"""

import json, os, sys, argparse
sys.stdout.reconfigure(encoding="utf-8")

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from dotenv import load_dotenv
load_dotenv()

if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "sk-...":
    print("[ERROR] Chưa điền OPENAI_API_KEY vào file .env")
    sys.exit(1)

DATA_RAW     = "data/raw"
OUT_TRIPLES  = "data/processed/triples_llm.json"
OUT_GRAPH    = "outputs/graphs/kg_llm.graphml"
OUT_PNG      = "outputs/graphs/kg_llm.png"
OUT_HTML     = "outputs/graphs/kg_llm.html"

with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--file", default="Anthropic",
                    help="Tên file không có .md (default: Anthropic)")
parser.add_argument("--all", action="store_true",
                    help="Chạy toàn bộ 9 file (tốn nhiều token hơn)")
args = parser.parse_args()

# ── Chọn file cần chạy ───────────────────────────────────────────────────────
all_files = sorted(f for f in os.listdir(DATA_RAW) if f.endswith(".md"))
if args.all:
    target_files = all_files
else:
    fname = args.file if args.file.endswith(".md") else args.file + ".md"
    if fname not in all_files:
        print(f"[ERROR] Không tìm thấy {fname} trong {DATA_RAW}/")
        print(f"Các file có sẵn: {all_files}")
        sys.exit(1)
    target_files = [fname]

# ── Pass 1: LLM Extraction ───────────────────────────────────────────────────
from src.indexing.entity_extractor import extract_from_file

print("=" * 55)
print(f"Pass 1 — LLM Extraction  (model: {os.getenv('OPENAI_MODEL')})")
print("=" * 55)

all_triples = []
for fname in target_files:
    print(f"\n[ {fname} ]")
    triples = extract_from_file(
        os.path.join(DATA_RAW, fname),
        chunk_size=cfg["chunking"]["chunk_size"],
        overlap=cfg["chunking"]["overlap"],
    )
    all_triples.extend(triples)
    print(f"  → {len(triples)} triples")

print(f"\nPass 1 total: {len(all_triples)} raw triples")

# ── Pass 2: Standardization ──────────────────────────────────────────────────
from src.indexing.standardizer import standardize

print("\n" + "=" * 55)
print(f"Pass 2 — Standardization (LLM={cfg['standardization']['use_llm_for_entities']})")
print("=" * 55)
all_triples = standardize(
    all_triples,
    use_llm=cfg["standardization"]["use_llm_for_entities"],
)

# ── Pass 3: Inference ────────────────────────────────────────────────────────
from src.indexing.inferencer import infer

print("\n" + "=" * 55)
print(f"Pass 3 — Inference (LLM={cfg['inference']['use_llm_for_inference']})")
print("=" * 55)
all_triples = infer(
    all_triples,
    apply_transitive_flag=cfg["inference"]["apply_transitive"],
    use_llm=cfg["inference"]["use_llm_for_inference"],
)

# ── Save triples ─────────────────────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
with open(OUT_TRIPLES, "w", encoding="utf-8") as f:
    json.dump(all_triples, f, ensure_ascii=False, indent=2)

# ── Build & visualize graph ──────────────────────────────────────────────────
from src.graph.builder import build_graph, save_graph, visualize_html
from src.graph.visualizer import draw_graph
import networkx as nx

print("\n" + "=" * 55)
print("Building graph...")
print("=" * 55)
G = build_graph(all_triples)
os.makedirs("outputs/graphs", exist_ok=True)
save_graph(G, OUT_GRAPH)
visualize_html(G, OUT_HTML)
draw_graph(G, title="Tech Company KG — LLM", output_path=OUT_PNG)

# ── Quality report ───────────────────────────────────────────────────────────
from collections import Counter

preds = Counter(t["predicate"] for t in all_triples)
noise_nodes = {"it","he","she","they","this","that","one","two","three",
               "company","firm","organization","group","team","both"}
clean_nodes = [n for n in G.nodes if n.lower() not in noise_nodes and len(n) > 2]

print("\n" + "=" * 55)
print("Quality Report")
print("=" * 55)
print(f"  Triples        : {len(all_triples)}")
print(f"  Nodes          : {G.number_of_nodes()}  (clean: {len(clean_nodes)})")
print(f"  Edges          : {G.number_of_edges()}")
print(f"  Components     : {nx.number_weakly_connected_components(G)}")

print(f"\n  Top 10 predicates:")
for p, c in preds.most_common(10):
    print(f"    {p}: {c}")

top5 = sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)[:5]
print(f"\n  Top 5 nodes by degree: {top5}")

print(f"\n  Sample triples:")
for t in all_triples[:15]:
    print(f"    ({t['subject']}) --[{t['predicate']}]--> ({t['object']})")

print(f"\nDone.")
print(f"  Triples : {OUT_TRIPLES}")
print(f"  Graph   : {OUT_GRAPH}")
print(f"  PNG     : {OUT_PNG}")
print(f"  HTML    : {OUT_HTML}")

# ── Token usage & cost report ────────────────────────────────────────────────
from src.indexing.token_counter import report as token_report, save_csv as token_save_csv

print(f"\n{'='*55}")
print("Token Usage & Cost Report")
print(f"{'='*55}")
print(token_report())
token_save_csv("outputs/results/token_usage.csv")
