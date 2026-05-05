"""
Visualize Knowledge Graph using Matplotlib + NetworkX.
Output: outputs/graphs/kg_plot.png
"""

import os
import math
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from adjustText import adjust_text as _adjust_text
    _HAS_ADJUST_TEXT = True
except ImportError:
    _HAS_ADJUST_TEXT = False


# Node color từ node_type attribute (gán bởi builder.py)
_TYPE_COLORS = {
    "Person":       "#4A90D9",  # blue
    "Organization": "#E85D4A",  # red
    "Product":      "#27AE60",  # green
    "Location":     "#F39C12",  # orange
    "Concept":      "#9B59B6",  # purple
    "Date":         "#95A5A6",  # grey
}

def _node_color(node: str, G: nx.DiGraph = None) -> str:
    if G and G.nodes[node].get("node_type"):
        return _TYPE_COLORS.get(G.nodes[node]["node_type"], "#BDC3C7")
    return "#4A90D9" if any(c.islower() for c in node) else "#E85D4A"


def _push_apart(pos: dict, min_dist: float, iterations: int = 120) -> dict:
    """Push node centres apart until no pair is closer than min_dist."""
    nodes = list(pos.keys())
    xy = [[pos[n][0], pos[n][1]] for n in nodes]
    for _ in range(iterations):
        moved = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                dx = xy[j][0] - xy[i][0]
                dy = xy[j][1] - xy[i][1]
                d = math.hypot(dx, dy)
                if d < min_dist and d > 1e-9:
                    push = (min_dist - d) / d * 0.5
                    xy[i][0] -= dx * push
                    xy[i][1] -= dy * push
                    xy[j][0] += dx * push
                    xy[j][1] += dy * push
                    moved = True
        if not moved:
            break
    return {nodes[i]: (xy[i][0], xy[i][1]) for i in range(len(nodes))}


def draw_graph(
    G: nx.DiGraph,
    title: str = "Tech Company Knowledge Graph",
    output_path: str = "outputs/graphs/kg_plot.png",
    max_nodes: int = 80,
    figsize: tuple = (46, 36),
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    # ── Layout ──────────────────────────────────────────────────────────────────
    # 1. Kamada-Kawai distributes nodes more uniformly than spring
    # 2. _push_apart then guarantees no two centres are closer than min_dist,
    #    removing the dense cluster that spring/KK leaves at the centre
    try:
        pos = nx.kamada_kawai_layout(G, scale=6.0)
    except Exception:
        n = G.number_of_nodes()
        pos = nx.spring_layout(G, seed=42, k=6.0 / math.sqrt(max(n, 1)),
                               iterations=200, scale=6.0)

    # min_dist ≈ width of a typical label in data coords
    # scale=6 → range 12 units across figsize[0] inches (≈80 % axes)
    # pts_per_unit ≈ figsize[0]*0.8*72 / 12
    pts_per_unit = figsize[0] * 0.8 * 72 / 12
    avg_label_pts = 11 * 6          # ~11 chars × 6 pt per char
    min_dist = avg_label_pts / pts_per_unit * 2.2   # 2.2× label width
    pos = _push_apart(pos, min_dist=min_dist)

    node_list   = list(G.nodes())
    node_colors = [_node_color(n, G) for n in node_list]
    node_sizes  = [900 + G.degree(n) * 220 for n in node_list]
    short_labels = {n: (n[:20] + "…" if len(n) > 22 else n) for n in node_list}

    # ── Edges ────────────────────────────────────────────────────────────────
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#666666", arrows=True, arrowsize=18,
        width=1.1, connectionstyle="arc3,rad=0.12", alpha=0.7,
    )

    # ── Nodes ────────────────────────────────────────────────────────────────
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors, node_size=node_sizes,
        alpha=0.93, linewidths=0.8, edgecolors="#FFFFFF",
    )

    # ── Node labels via ax.text so adjustText can reposition them ────────────
    data_per_inch = 12.0 / (figsize[0] * 0.8)   # data units per inch
    texts = []
    for i, node in enumerate(node_list):
        x, y = pos[node]
        radius_inch = math.sqrt(node_sizes[i] / math.pi) / 72.0
        offset = radius_inch * data_per_inch * 1.4
        t = ax.text(x, y + offset, short_labels[node],
                    fontsize=9, color="white", fontweight="bold",
                    ha="center", va="bottom", zorder=5)
        texts.append(t)

    if _HAS_ADJUST_TEXT:
        node_xs = [pos[n][0] for n in node_list]
        node_ys = [pos[n][1] for n in node_list]
        _adjust_text(
            texts, x=node_xs, y=node_ys, ax=ax,
            force_text=(0.8, 1.0),
            force_points=(0.6, 0.8),
            expand_text=(1.4, 1.6),
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5, alpha=0.55),
        )

    # ── Edge labels ──────────────────────────────────────────────────────────
    # Only show on edges where both endpoints have degree ≥ 2 (avoids clutter)
    edge_labels = {
        (u, v): d.get("relation", "")
        for u, v, d in G.edges(data=True)
        if G.degree(u) >= 2 and G.degree(v) >= 2
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_size=7, font_color="#E0E0E0", rotate=False,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#1A1D27",
                  edgecolor="none", alpha=0.78),
    )

    # ── Legend ───────────────────────────────────────────────────────────────
    legend = [
        mpatches.Patch(color="#E85D4A", label="Organization"),
        mpatches.Patch(color="#4A90D9", label="Person"),
        mpatches.Patch(color="#27AE60", label="Product"),
        mpatches.Patch(color="#F39C12", label="Location"),
        mpatches.Patch(color="#9B59B6", label="Concept"),
        mpatches.Patch(color="#95A5A6", label="Date"),
    ]
    ax.legend(handles=legend, loc="upper left", framealpha=0.45,
              labelcolor="white", facecolor="#1A1D27",
              edgecolor="#666666", fontsize=13)

    ax.set_title(title, color="white", fontsize=22, pad=18, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Graph plot saved: {output_path}")


# ── Demo: vẽ sample graph nhỏ để test không cần data thật ───────────────────

def draw_sample():
    sample_triples = [
        ("OpenAI", "FOUNDED_BY", "Sam Altman"),
        ("OpenAI", "FOUNDED_BY", "Elon Musk"),
        ("OpenAI", "FOUNDED_IN", "2015"),
        ("OpenAI", "PARTNERED_WITH", "Microsoft"),
        ("Microsoft", "INVESTED_IN", "OpenAI"),
        ("Microsoft", "OWNS", "Azure"),
        ("Anthropic", "FOUNDED_BY", "Dario Amodei"),
        ("Anthropic", "FOUNDED_BY", "Daniela Amodei"),
        ("Dario Amodei", "FORMERLY_AT", "OpenAI"),
        ("Google", "ACQUIRED", "DeepMind"),
        ("Google DeepMind", "DEVELOPED", "Gemini"),
        ("Nvidia", "PRODUCES", "H100 GPU"),
        ("OpenAI", "USES", "H100 GPU"),
        ("Hugging Face", "HOSTS", "Open-source models"),
        ("Meta Platforms", "RELEASED", "LLaMA"),
        ("Meta Platforms", "FOUNDED_BY", "Mark Zuckerberg"),
        ("Amazon Web Services", "PART_OF", "Amazon"),
        ("Amazon Web Services", "COMPETES_WITH", "Azure"),
    ]

    G = nx.DiGraph()
    for s, p, o in sample_triples:
        G.add_edge(s, o, relation=p)

    draw_graph(G, title="Tech Company Knowledge Graph (Sample)", output_path="outputs/graphs/kg_plot.png")


if __name__ == "__main__":
    import sys

    if "--sample" in sys.argv:
        draw_sample()
    else:
        # Load graph thật từ graphml
        graph_path = "outputs/graphs/kg.graphml"
        if not os.path.exists(graph_path):
            print(f"Graph not found at {graph_path}. Run: python main.py --step 3")
            print("Falling back to sample graph...")
            draw_sample()
        else:
            G = nx.read_graphml(graph_path)
            draw_graph(G)
