#!/usr/bin/env python3
"""
Generate a clean illustration of a Cell Ontology snippet showing the neural
and immune lineage of selected cell types, highlighting the dual parentage
of microglial cell.  Each node shows its Zhou (2008) IC value.

Uses graphviz for automatic DAG layout.  Color scheme matches ontology_ic_tree.png.
"""

import graphviz


def ic_color(ic_val):
    """Map IC value (0..1) to a high-contrast cream -> teal -> navy gradient.

    Uses the actual IC range in this snippet (0.004 .. 0.715) stretched to
    full visual range so the top and bottom nodes look clearly different.
    """
    # Stretch to 0-1 based on the data range in this figure
    lo, hi = 0.0, 0.72
    t = max(0.0, min(1.0, (ic_val - lo) / (hi - lo)))

    # Three-stop gradient: cream (#FFFDE7) -> teal (#26A69A) -> navy (#0D2B52)
    if t < 0.5:
        s = t / 0.5
        r = int(255 * (1 - s) + 38 * s)
        g = int(253 * (1 - s) + 166 * s)
        b = int(231 * (1 - s) + 154 * s)
    else:
        s = (t - 0.5) / 0.5
        r = int(38 * (1 - s) + 13 * s)
        g = int(166 * (1 - s) + 43 * s)
        b = int(154 * (1 - s) + 82 * s)
    return f"#{r:02X}{g:02X}{b:02X}"


def font_color(ic_val):
    """White text for dark backgrounds (high IC), black otherwise."""
    return "white" if ic_val > 0.45 else "black"


def draw_ontology_snippet(output_path="ontology_snippet"):
    dot = graphviz.Digraph(
        "CL_snippet",
        format="png",
        engine="dot",
        graph_attr={
            "rankdir": "TB",
            "bgcolor": "white",
            "fontname": "Helvetica",
            "label": "",
            "pad": "0.4",
            "nodesep": "1.2",
            "ranksep": "0.65",
            "size": "14,8",
            "ratio": "compress",
            "dpi": "180",
        },
        node_attr={
            "shape": "box",
            "style": "rounded,filled",
            "fontname": "Helvetica-Bold",
            "fontsize": "13",
            "margin": "0.22,0.12",
            "penwidth": "1.5",
            "color": "#888888",
        },
        edge_attr={
            "color": "#AAAAAA",
            "penwidth": "1.8",
            "arrowsize": "0.9",
        },
    )

    # --- IC values (precomputed with Zhou 2008, k=0.5 on cl-basic.obo) ---
    ic = {
        "cell":             0.004,
        "eukaryotic":       0.142,
        "neural":           0.290,
        "neuron_assoc":     0.453,
        "neuron":           0.364,
        "glial":            0.504,
        "macroglial":       0.573,
        "astrocyte":        0.628,
        "oligodendrocyte":  0.673,
        "microglial":       0.715,
        "macrophage":       0.547,
        "tissue_macro":     0.600,
        "cns_macro":        0.600,
    }

    def make_node(key, name):
        v = ic[key]
        dot.node(key, f"{name}\\nIC = {v:.3f}",
                 fillcolor=ic_color(v), fontcolor=font_color(v))

    # --- Nodes ---
    make_node("cell",             "cell")
    make_node("eukaryotic",       "eukaryotic cell")
    make_node("neural",           "neural cell")
    make_node("neuron_assoc",     "neuron assoc. cell")
    make_node("neuron",           "neuron")
    make_node("glial",            "glial cell")
    make_node("macroglial",       "macroglial cell")
    make_node("astrocyte",        "astrocyte")
    make_node("oligodendrocyte",  "oligodendrocyte")
    make_node("microglial",       "microglial cell")
    make_node("macrophage",       "macrophage")
    make_node("tissue_macro",     "tissue-resident macrophage")
    make_node("cns_macro",        "CNS macrophage")

    # --- Rank constraints for clean layout ---
    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("neural")
        s.node("macrophage")

    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("neuron")
        s.node("neuron_assoc")

    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("glial")
        s.node("cns_macro")

    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("macroglial")
        s.node("microglial")

    with dot.subgraph() as s:
        s.attr(rank="same")
        s.node("astrocyte")
        s.node("oligodendrocyte")

    # --- Edges ---
    # Neural lineage (solid gray)
    dot.edge("cell", "eukaryotic")
    dot.edge("eukaryotic", "neural")
    dot.edge("neural", "neuron_assoc")
    dot.edge("neural", "neuron")
    dot.edge("neuron_assoc", "glial")
    dot.edge("glial", "macroglial")
    dot.edge("macroglial", "astrocyte")
    dot.edge("macroglial", "oligodendrocyte")

    # Glial cell -> microglial (neural parent)
    dot.edge("glial", "microglial")

    # Immune lineage (dashed to indicate elided intermediate terms)
    dot.edge("eukaryotic", "macrophage", style="dashed",
             label="  (via myeloid\n   leukocyte ...)  ",
             fontname="Helvetica", fontsize="9", fontcolor="#999999")
    dot.edge("macrophage", "tissue_macro")
    dot.edge("tissue_macro", "cns_macro")

    # CNS macrophage -> microglial (immune parent)
    dot.edge("cns_macro", "microglial")


    # Render
    dot.render(output_path, cleanup=True)
    print(f"Saved to {output_path}.png")


if __name__ == "__main__":
    draw_ontology_snippet()
