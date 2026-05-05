"""
Build a NetworkX DiGraph from extracted triples.
Nodes carry 'node_type' (Person/Organization/Product/Location/Concept/Date).
Edges carry 'relation' and optional 'year' / 'source_file' / 'source_chunk'.
"""

import re
import os
import networkx as nx

NODE_TYPE_COLORS = {
    "Person":       "#4A90D9",
    "Organization": "#E85D4A",
    "Product":      "#27AE60",
    "Location":     "#F39C12",
    "Concept":      "#9B59B6",
    "Date":         "#95A5A6",
    "Unknown":      "#BDC3C7",
}

SKIP_PREDICATES = {
    "IS","BE","HAS","HAVE","DO","GET","WAS","ARE",
    "YEAR","EVENT","SAID","NOTED","STATED",
}

GENERIC_NODES = {
    "it","he","she","they","we","this","that","both",
    "the company","the firm","the organization","the startup",
    "developers","users","non-technical users","general availability",
}

# Phrase/monetary/event nodes that should not be graph nodes
_NOISE_NODE_RE = re.compile(
    r"^\$"                                             # monetary: "$4 billion"
    r"|^millions?\s+of\b"                              # quantity phrase
    r"|\s+a\s+\w"                                     # article mid-phrase
    r"|\b(developed|selling|pirated|processing|delegating|strike)\b"  # verb phrases
    r"|\b\d{4}\s+\w+\s+(war|attack|crisis|conflict)\b",  # news events
    re.IGNORECASE,
)

# Regex để nhận diện node là date/year thuần
_DATE_RE = re.compile(
    r"^(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|\d{4}|\d{1,2}/\d{4}|q[1-4]\s*\d{4}|fy\d{4})",
    re.IGNORECASE,
)

# Known orgs / persons để hỗ trợ heuristic type inference
_ORG_KEYWORDS  = {"Inc","Corp","LLC","Ltd","AI","Labs","Technologies","Institute",
                  "University","Foundation","Group","Agency","Department","DoD","DoJ"}
_PERSON_NAMES  = {"Sam Altman","Dario Amodei","Daniela Amodei","Elon Musk","Bill Gates",
                  "Satya Nadella","Jensen Huang","Mark Zuckerberg","Demis Hassabis",
                  "Jan Leike","John Schulman","Andrej Karpathy","Jack Clark"}
_LOCATION_KEYS = {"San Francisco","New York","London","Paris","Washington","Seattle",
                  "California","France","UK","US","EU"}
_PRODUCT_KEYS  = {"GPT","Claude","Gemini","LLaMA","Mistral","Copilot","Azure","AWS",
                  "Windows","ChatGPT","Bard","Grok","H100","DGX","TPU","API"}


def _infer_type(name: str, llm_type: str) -> str:
    """Dùng LLM type nếu hợp lệ, fallback heuristic nếu Unknown/None."""
    if llm_type and llm_type not in {"Unknown", "?", ""}:
        return llm_type

    if name in _PERSON_NAMES or re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+$", name):
        return "Person"
    if any(k in name for k in _LOCATION_KEYS):
        return "Location"
    if any(k in name for k in _PRODUCT_KEYS):
        return "Product"
    if any(name.endswith(k) or k in name for k in _ORG_KEYWORDS):
        return "Organization"
    if _DATE_RE.match(name):
        return "Date"
    return "Organization"   # safe default cho entity viết hoa


def build_graph(triples: list[dict]) -> nx.DiGraph:
    G = nx.DiGraph()

    for t in triples:
        s = t.get("subject", "").strip()
        p = t.get("predicate", "").strip().upper()
        o = t.get("object",  "").strip()

        if not s or not o:
            continue
        if p in SKIP_PREDICATES:
            continue
        if s.lower() in GENERIC_NODES or o.lower() in GENERIC_NODES:
            continue
        if _NOISE_NODE_RE.search(s) or _NOISE_NODE_RE.search(o):
            continue
        # Bỏ object là date node — đã lưu vào edge property rồi
        if _DATE_RE.match(o) and t.get("year"):
            o = None   # sẽ skip add_edge bên dưới

        if not o:
            continue

        s_type = _infer_type(s, t.get("subject_type", ""))
        o_type = _infer_type(o, t.get("object_type", ""))

        if not G.has_node(s):
            G.add_node(s, node_type=s_type, color=NODE_TYPE_COLORS.get(s_type, "#BDC3C7"))
        if not G.has_node(o):
            G.add_node(o, node_type=o_type, color=NODE_TYPE_COLORS.get(o_type, "#BDC3C7"))

        edge_attrs = {"relation": p}
        if t.get("year"):
            edge_attrs["year"] = t["year"]
        if t.get("amount"):
            edge_attrs["amount"] = t["amount"]
        if t.get("source_file"):
            edge_attrs["source_file"]  = t["source_file"]
            edge_attrs["source_chunk"] = t.get("source_chunk", -1)

        G.add_edge(s, o, **edge_attrs)

    return G


def save_graph(G: nx.DiGraph, path: str = "outputs/graphs/kg.graphml"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nx.write_graphml(G, path)
    print(f"Graph saved: {path}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


def load_graph(path: str = "outputs/graphs/kg.graphml") -> nx.DiGraph:
    return nx.read_graphml(path)


def visualize_html(G: nx.DiGraph, path: str = "outputs/graphs/kg.html"):
    try:
        from pyvis.network import Network
    except ImportError:
        print("pyvis not installed — skipping HTML visualization")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    net = Network(height="750px", width="100%", directed=True, notebook=False)
    net.repulsion(node_distance=250, central_gravity=0.1,
                  spring_length=200, spring_strength=0.04, damping=0.09)
    for node, data in G.nodes(data=True):
        color = data.get("color", "#BDC3C7")
        net.add_node(node, label=node, color=color,
                     title=data.get("node_type", "Unknown"))
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, label=data.get("relation", ""),
                     title=data.get("source_file", ""))
    net.save_graph(path)
    print(f"Visualization saved: {path}")
