"""
20 benchmark questions designed from 5 data files:
  OpenAI, Anthropic, Microsoft, Nvidia, Google_DeepMind

Failure modes covered:
  [MH] Multi-hop  — answer requires chaining 2+ entities across hops
  [CD] Cross-doc  — answer spans multiple .md files
  [GT] Global thematic — requires surveying all documents
"""

import csv
import os
import sys
import time
import networkx as nx
from src.rag.flat_rag import load_flat_index, query_flat_rag
from src.rag.graph_rag import query_hybrid_rag

# Each entry: (question, expected_answer_hint, failure_mode, why_flat_rag_fails)

QUESTIONS = [
    # ── Multi-hop (MH) ────────────────────────────────────────────────────────
    (
        "What role did Dario Amodei hold at OpenAI before he co-founded Anthropic?",
        "Vice President of Research",
        "MH",
        "Dario appears in Anthropic.md as founder; his OpenAI role is only in OpenAI.md. "
        "Flat RAG retrieves one chunk and misses the cross-entity link.",
    ),
    (
        "Anthropic uses AWS as its primary cloud provider. "
        "Which company owns AWS and who are AWS's main cloud competitors in this corpus?",
        "Amazon owns AWS; competitors include Microsoft Azure and Google Cloud",
        "MH",
        "Requires 2-hop: Anthropic -> uses AWS -> owned by Amazon -> competes with Azure/GCP. "
        "Flat RAG retrieves the Anthropic chunk but cannot follow the ownership chain.",
    ),
    (
        "Nvidia gifted its first DGX-1 supercomputer to OpenAI in 2016. "
        "How much did Nvidia later invest in OpenAI's 2026 funding round?",
        "$30 billion",
        "MH",
        "Two facts about Nvidia<->OpenAI are in distant chunks. "
        "Flat RAG retrieves one but rarely both; GraphRAG links the Nvidia node across hops.",
    ),
    (
        "Mustafa Suleyman co-founded DeepMind in 2010. "
        "What role did he later take at Microsoft in 2024?",
        "CEO of Microsoft AI",
        "MH + CD",
        "DeepMind co-founders are in Google_DeepMind.md; Suleyman's Microsoft role is in Microsoft.md. "
        "Flat RAG cannot connect the same person across two documents.",
    ),
    (
        "OpenAI's services are hosted on Microsoft Azure. "
        "What product did Microsoft build using OpenAI's GPT-4 model?",
        "Copilot, integrated into Windows 11 and Microsoft 365",
        "MH",
        "Requires chaining: OpenAI -> hosted on Azure -> Microsoft used GPT-4 -> Copilot. "
        "Flat RAG finds the hosting fact but misses the downstream product link.",
    ),
    (
        "Google invested in Anthropic and also provides compute hardware. "
        "What type of hardware does Google give Anthropic access to for model training?",
        "Custom TPU chips",
        "MH",
        "Investment fact and TPU hardware fact are in separate chunks of Anthropic.md. "
        "GraphRAG connects Google -> INVESTED_IN -> Anthropic and Google -> PROVIDED_TPU -> Anthropic.",
    ),
    (
        "Anthropic partnered with Palantir and AWS to serve US intelligence agencies. "
        "Which other company in this corpus also received a DoD AI contract in July 2025?",
        "OpenAI ($200M DoD contract)",
        "MH + CD",
        "DoD contracts are in both Anthropic.md and OpenAI.md. "
        "Flat RAG retrieves one document's chunk; GraphRAG traverses the shared DoD node.",
    ),
    (
        "A co-founder of DeepMind later left to co-found Inflection AI, then joined Microsoft. "
        "Who is this person and what role did they take at Microsoft?",
        "Mustafa Suleyman became CEO of Microsoft AI in 2024",
        "MH + CD",
        "Requires linking DeepMind founders (Google_DeepMind.md) to Microsoft role (Microsoft.md). "
        "Flat RAG cannot bridge two separate documents through a shared person node.",
    ),

    # ── Cross-document (CD) ───────────────────────────────────────────────────
    (
        "Which companies in this corpus has Microsoft directly invested in?",
        "OpenAI ($1B in 2019, $10B in 2023)",
        "CD",
        "Investment facts are in OpenAI.md and Microsoft.md. "
        "Flat RAG retrieves whichever chunk is most similar and may miss the other file.",
    ),
    (
        "Which single company in this corpus has received large investments from "
        "both Amazon and Google?",
        "Anthropic (Amazon $4B+$3.5B, Google $500M+$300M)",
        "CD",
        "Both investment facts live in Anthropic.md in separate chunks. "
        "GraphRAG connects Anthropic -> INVESTED_BY -> Amazon and Anthropic -> INVESTED_BY -> Google.",
    ),
    (
        "Compare how OpenAI and Anthropic differ in their relationship with "
        "the US Department of Defense.",
        "OpenAI accepted a $200M DoD contract; Anthropic rejected Pentagon demands "
        "and was designated a supply-chain risk.",
        "CD",
        "Requires reading both OpenAI.md and Anthropic.md and comparing two stances. "
        "Flat RAG retrieves one company's chunk and ignores the other.",
    ),
    (
        "Which companies in this corpus are facing or have faced antitrust or "
        "regulatory investigations?",
        "Microsoft (EU antitrust on Teams bundling, US DOJ investigation)",
        "GT",
        "Regulatory facts are in different sections of Microsoft.md. "
        "GraphRAG aggregates via INVESTIGATED_BY edges across company nodes.",
    ),
    (
        "What primary cloud service does each AI company in this corpus rely on "
        "for its infrastructure?",
        "OpenAI->Microsoft Azure, Anthropic->Amazon Web Services",
        "GT + CD",
        "Cloud provider facts are in OpenAI.md and Anthropic.md separately. "
        "GraphRAG can traverse USES_CLOUD edges from each company node.",
    ),
    (
        "Jan Leike and John Schulman both left the same company in 2024 to join Anthropic. "
        "What company did they leave, and who is its CEO?",
        "They left OpenAI; Sam Altman is CEO of OpenAI",
        "CD",
        "Both facts (researchers leaving OpenAI) appear across OpenAI.md and Anthropic.md. "
        "Flat RAG retrieves whichever chunk is more similar and may miss the connection.",
    ),
    (
        "Which company in this corpus acquired a hardware startup co-founded by "
        "a former Apple designer?",
        "OpenAI acquired 'io', co-founded by Jony Ive (former Apple designer)",
        "MH",
        "Requires chaining: acquisition -> founder background -> Apple. "
        "Flat RAG retrieves the acquisition fact but drops the Apple designer detail.",
    ),

    # ── Global thematic (GT) ─────────────────────────────────────────────────
    (
        "Which companies in this corpus were founded by people who previously worked "
        "at other companies also in this corpus?",
        "Anthropic (founded by ex-OpenAI employees including Dario and Daniela Amodei)",
        "GT + CD",
        "Requires cross-referencing founder backgrounds across documents. "
        "Flat RAG cannot aggregate this relationship across the entire corpus.",
    ),
    (
        "Which AI model product lines exist in this corpus, and which company produced each?",
        "GPT/DALL-E/Sora->OpenAI, Claude->Anthropic, Gemini/AlphaGo/AlphaFold->Google DeepMind",
        "GT + CD",
        "Product lines are in separate documents. "
        "Flat RAG retrieves top-k similar chunks; GraphRAG traverses DEVELOPED_BY edges.",
    ),
    (
        "Which company in this corpus has the highest post-money valuation as of 2026, "
        "and who led its most recent funding round?",
        "OpenAI ($852B valuation); Amazon invested $50B in February 2026",
        "MH",
        "Valuation and lead-investor facts are in separate paragraphs of OpenAI.md. "
        "Flat RAG may retrieve the valuation chunk but miss the investor chain.",
    ),
    (
        "Nvidia invested in OpenAI's 2026 round. "
        "Which other company in this corpus is Nvidia also expected to invest in?",
        "Anthropic (Nvidia and Microsoft expected to invest up to $15B)",
        "MH + CD",
        "Nvidia->OpenAI is in OpenAI.md; Nvidia->Anthropic is in Anthropic.md. "
        "Flat RAG retrieves one; GraphRAG connects Nvidia as a shared investor node.",
    ),
    (
        "Google DeepMind developed Gemini to compete with OpenAI. "
        "Name two other AI products from Google DeepMind that are NOT language models.",
        "AlphaGo (Go-playing AI) and AlphaFold (protein structure prediction)",
        "GT",
        "Requires reading all of Google_DeepMind.md and filtering non-LLM products. "
        "Flat RAG returns whichever chunk scores highest on 'AI products' similarity.",
    ),
]

OUTPUT_PATH = "outputs/results/benchmark.csv"


# ── Runner ────────────────────────────────────────────────────────────────────

def run_benchmark(G: nx.DiGraph):
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    db = load_flat_index()

    rows = []
    for i, (question, expected, mode, why_flat_fails) in enumerate(QUESTIONS, 1):
        print(f"\n[{i:02d}/{len(QUESTIONS)}] [{mode}] {question[:80]}...")

        t0 = time.time()
        flat_ans = query_flat_rag(question, db)
        flat_time = round(time.time() - t0, 2)

        t0 = time.time()
        graph_ans = query_hybrid_rag(question, G, db)
        graph_time = round(time.time() - t0, 2)

        rows.append({
            "id": i,
            "failure_mode": mode,
            "question": question,
            "expected_answer": expected,
            "flat_rag_answer": flat_ans,
            "flat_rag_time_s": flat_time,
            "flat_rag_correct": "",
            "hybrid_rag_answer": graph_ans,
            "hybrid_rag_time_s": graph_time,
            "hybrid_rag_correct": "",
            "why_flat_rag_fails": why_flat_fails,
            "notes": "",
        })

        print(f"  Expected : {expected[:80]}")
        print(f"  Flat RAG ({flat_time}s): {flat_ans[:80]}...")
        print(f"  GraphRAG ({graph_time}s): {graph_ans[:80]}...")

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    mh = sum(1 for _, _, m, _ in QUESTIONS if "MH" in m)
    cd = sum(1 for _, _, m, _ in QUESTIONS if "CD" in m)
    gt = sum(1 for _, _, m, _ in QUESTIONS if "GT" in m)
    print(f"\nBenchmark saved -> {OUTPUT_PATH}")
    print(f"Question breakdown: {mh} Multi-hop | {cd} Cross-doc | {gt} Global thematic")
    print("Fill 'flat_rag_correct' and 'graph_rag_correct' columns manually (Y/N).")
