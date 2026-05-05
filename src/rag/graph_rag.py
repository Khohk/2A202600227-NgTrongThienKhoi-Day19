"""
GraphRAG: query via 2-hop BFS on NetworkX graph, then answer with LLM.
Best Practices applied:
  - Source pointer: trả về file gốc cùng answer
  - Max 50 edges per traversal
  - Multi-entity start: BFS từ nhiều entity để cover cross-doc connections
  - Fallback for global questions: dùng top-degree nodes khi không match entity
"""

import os
import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv
from src.graph.querier import two_hop_context

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ANSWER_PROMPT = """You are a helpful assistant. Use ONLY the graph context below to answer.
If the context does not contain enough information, say "Not found in graph."

Graph context (Knowledge Graph triples):
{context}

Question: {question}
Answer:"""

HYBRID_PROMPT = """You are a helpful assistant with access to two complementary information sources.

## GRAPH CONTEXT (structured entity relationships):
{graph_context}

## DOCUMENT CONTEXT (detailed text passages):
{doc_context}

Instructions:
- Use Graph Context for relationship structure and entity connections.
- Use Document Context for specific details: dollar amounts, dates, names, quantities.
- Combine both to give a complete, accurate answer.
- If neither source contains the answer, say "Not found."

Question: {question}
Answer:"""


def extract_entities_from_question(question: str, G: nx.DiGraph,
                                   max_entities: int = 3) -> list[str]:
    """
    Trả về tối đa max_entities nodes khớp với câu hỏi (longest-match first,
    không overlap). Fallback: top-5 degree nodes cho global/thematic questions.
    """
    q_lower = question.lower()
    found   = []
    covered = set()   # character positions đã dùng

    for node in sorted(G.nodes, key=len, reverse=True):
        node_lower = node.lower()
        if len(node_lower) < 3:
            continue
        idx = q_lower.find(node_lower)
        if idx == -1:
            continue
        span = set(range(idx, idx + len(node_lower)))
        if span & covered:          # overlap với match trước → bỏ qua
            continue
        found.append(node)
        covered |= span
        if len(found) >= max_entities:
            break

    # Fallback cho global questions ("each company", "product lines", ...)
    if not found:
        found = sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)[:5]

    return found


def _collect_graph_context(question: str, G: nx.DiGraph) -> tuple[str, list[str]]:
    """Shared helper: BFS từ matched entities, trả về (context_str, sources)."""
    entities   = extract_entities_from_question(question, G)
    seen_lines : set[str]  = set()
    all_lines  : list[str] = []
    all_sources: list[str] = []

    for entity in entities:
        ctx, srcs = two_hop_context(G, entity, hops=2)
        for line in ctx.splitlines():
            if line and line not in seen_lines:
                seen_lines.add(line)
                all_lines.append(line)
        for s in srcs:
            if s not in all_sources:
                all_sources.append(s)

    return "\n".join(all_lines[:80]), all_sources


def query_graph_rag(question: str, G: nx.DiGraph) -> str:
    context, all_sources = _collect_graph_context(question, G)

    if not context:
        return "No relevant information found in the knowledge graph."

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": ANSWER_PROMPT.format(
            context=context, question=question
        )}],
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()

    if all_sources:
        answer += f"\n\n[Sources: {', '.join(all_sources)}]"

    return answer


def query_hybrid_rag(question: str, G: nx.DiGraph, db) -> str:
    """Hybrid: graph structure + vector detail → LLM tổng hợp cả hai."""
    # Graph context (relationships + amounts from edge attrs)
    graph_context, all_sources = _collect_graph_context(question, G)
    if not graph_context:
        graph_context = "No graph context found."

    # Vector context (chi tiết từ raw documents)
    docs = db.similarity_search(question, k=5)
    doc_context = "\n\n---\n".join(d.page_content for d in docs)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": HYBRID_PROMPT.format(
            graph_context=graph_context,
            doc_context=doc_context,
            question=question,
        )}],
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()

    if all_sources:
        answer += f"\n\n[Sources: {', '.join(all_sources)}]"

    return answer
