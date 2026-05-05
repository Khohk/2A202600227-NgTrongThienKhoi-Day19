"""
Entry point — chạy toàn bộ pipeline theo thứ tự.
Usage: python main.py [--step 1|2|3|4|5|all]
"""

import argparse
import json
import os
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

DATA_RAW     = "data/raw"
TRIPLES_PATH = "data/processed/triples.json"
GRAPH_PATH   = "outputs/graphs/kg.graphml"


def _load_config() -> dict:
    with open("config.toml", "rb") as f:
        return tomllib.load(f)


def step1_fetch():
    from fetch_corpus import main as fetch
    fetch()


def step2_index():
    from src.indexing.entity_extractor import extract_from_file
    from src.indexing.standardizer   import standardize
    from src.indexing.inferencer     import infer

    cfg = _load_config()
    os.makedirs("data/processed", exist_ok=True)

    # ── Pass 1: SPO Extraction ───────────────────────────────────────────────
    print("\n=== Pass 1: SPO Extraction ===")
    all_triples = []
    chunk_size = cfg["chunking"]["chunk_size"]
    overlap    = cfg["chunking"]["overlap"]

    files = [f for f in os.listdir(DATA_RAW) if f.endswith(".md")]
    for fname in files:
        print(f"\nExtracting: {fname}")
        triples = extract_from_file(
            os.path.join(DATA_RAW, fname),
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_triples.extend(triples)
    print(f"\nPass 1 done: {len(all_triples)} raw triples")

    # ── Pass 2: Entity Standardization ──────────────────────────────────────
    if cfg["standardization"]["enabled"]:
        print("\n=== Pass 2: Entity Standardization ===")
        all_triples = standardize(
            all_triples,
            use_llm=cfg["standardization"]["use_llm_for_entities"],
        )

    # ── Pass 3: Relationship Inference ──────────────────────────────────────
    if cfg["inference"]["enabled"]:
        print("\n=== Pass 3: Relationship Inference ===")
        all_triples = infer(
            all_triples,
            apply_transitive_flag=cfg["inference"]["apply_transitive"],
            use_llm=cfg["inference"]["use_llm_for_inference"],
        )

    with open(TRIPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(all_triples, f, ensure_ascii=False, indent=2)
    print(f"\nFinal triples: {len(all_triples)} → saved to {TRIPLES_PATH}")


def step3_build():
    from src.graph.builder import build_graph, save_graph, visualize_html

    with open(TRIPLES_PATH, encoding="utf-8") as f:
        triples = json.load(f)

    G = build_graph(triples)
    save_graph(G, GRAPH_PATH)
    visualize_html(G, "outputs/graphs/kg.html")


def step4_query_demo():
    from src.graph.builder import load_graph
    from src.rag.graph_rag import query_graph_rag

    G = load_graph(GRAPH_PATH)
    questions = [
        "Who founded OpenAI?",
        "What is the relationship between Microsoft and OpenAI?",
        "Which companies are related to Google DeepMind?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {query_graph_rag(q, G)}")


def step5_evaluate():
    from src.graph.builder import load_graph
    from src.evaluation.benchmark import run_benchmark
    from src.rag.flat_rag import build_flat_index

    print("Building Flat RAG index...")
    build_flat_index()

    # Prefer LLM-extracted graph if available
    llm_graph = "outputs/graphs/kg_llm.graphml"
    graph_path = llm_graph if os.path.exists(llm_graph) else GRAPH_PATH
    print(f"Loading graph from: {graph_path}")
    G = load_graph(graph_path)
    run_benchmark(G)


STEPS = {
    "1": ("Fetch corpus", step1_fetch),
    "2": ("Index → Extract triples", step2_index),
    "3": ("Build graph", step3_build),
    "4": ("Query demo", step4_query_demo),
    "5": ("Evaluate (Flat RAG vs GraphRAG)", step5_evaluate),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="all", help="1|2|3|4|5|all")
    args = parser.parse_args()

    if args.step == "all":
        for key, (name, fn) in STEPS.items():
            print(f"\n{'='*50}\nStep {key}: {name}\n{'='*50}")
            fn()
    elif args.step in STEPS:
        name, fn = STEPS[args.step]
        print(f"\nRunning Step {args.step}: {name}")
        fn()
    else:
        print(f"Unknown step '{args.step}'. Choose from: {list(STEPS.keys())} or 'all'")
