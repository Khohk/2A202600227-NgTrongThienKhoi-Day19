"""
Query the Knowledge Graph via BFS traversal (NetworkX).
Best Practices (slide):
  - Giới hạn duyệt: 2 hop, tối đa MAX_EDGES cạnh
  - Trả về source pointer (file + chunk) cùng context
"""

import networkx as nx

MAX_EDGES = 50   # Best Practice: "duyệt 2 hop, nhưng tối đa 50 cạnh"


def find_entity(G: nx.DiGraph, name: str) -> str | None:
    name_lower = name.lower()
    # Exact match trước
    for node in G.nodes:
        if node.lower() == name_lower:
            return node
    # Partial match fallback
    for node in G.nodes:
        if name_lower in node.lower() or node.lower() in name_lower:
            return node
    return None


def two_hop_context(
    G: nx.DiGraph,
    entity: str,
    hops: int = 2,
    max_edges: int = MAX_EDGES,
) -> tuple[str, list[str]]:
    """
    BFS tối đa `hops` bước, không vượt quá `max_edges` cạnh.

    Returns:
        context  : chuỗi triple đã textualize để gửi LLM
        sources  : danh sách source file để truy xuất chunk gốc
    """
    start = find_entity(G, entity)
    if not start:
        return "", []

    visited  = set()
    frontier = [start]
    lines    = []
    sources  = []
    edge_count = 0

    for _ in range(hops):
        if edge_count >= max_edges:
            break
        next_frontier = []
        for node in frontier:
            if node in visited:
                continue
            visited.add(node)

            for nb in G.successors(node):
                if edge_count >= max_edges:
                    break
                d      = G[node][nb]
                rel    = d.get("relation", "RELATED_TO")
                src    = d.get("source_file", "")
                amount = d.get("amount", "")
                suffix = f" [{amount}]" if amount else ""
                lines.append(f"({node}) --[{rel}{suffix}]--> ({nb})")
                if src:
                    sources.append(src)
                next_frontier.append(nb)
                edge_count += 1

            for nb in G.predecessors(node):
                if edge_count >= max_edges:
                    break
                d      = G[nb][node]
                rel    = d.get("relation", "RELATED_TO")
                src    = d.get("source_file", "")
                amount = d.get("amount", "")
                suffix = f" [{amount}]" if amount else ""
                lines.append(f"({nb}) --[{rel}{suffix}]--> ({node})")
                if src:
                    sources.append(src)
                next_frontier.append(nb)
                edge_count += 1

        frontier = next_frontier

    context = "\n".join(lines)
    sources  = list(dict.fromkeys(sources))   # dedup, preserve order
    return context, sources
