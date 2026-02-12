# Information Content (IC) Formula Analysis for Cell Ontology

## Background

The original benchmarking code used shortest-path distance on an undirected graph to measure
semantic distance between predicted and ground-truth cell types. This approach has fundamental
problems when applied to the Cell Ontology (CL), which is a **DAG (directed acyclic graph)**
with significant multiple inheritance (33.5% of terms have multiple `is_a` parents).

We evaluated replacing path-based distance with **Information Content (IC) based semantic
similarity**, specifically the **Lin similarity** measure using the **MICA (Most Informative
Common Ancestor)**.

## Why Path-Based Distance Fails on the CL DAG

1. **Shortest undirected path underestimates distance.** In a DAG with multiple inheritance,
   the undirected shortest path can shortcut through sibling/cousin nodes that bridge separate
   branches. Among 19,900 sampled cell type pairs, 717 (3.6%) had shortest path < LCA distance.

2. **LCA is ambiguous in a DAG.** Two cell types can have multiple "lowest" common ancestors
   at the same depth, with neither being an ancestor of the other. Among sampled pairs, 3,040
   (15.3%) had multiple independent LCAs.

3. **Parent-child distance inflated by LCA.** When comparing `neuron` to `GABAergic neuron`
   (a direct parent-child pair), the LCA-based distance is 3 instead of 1, because `neuron`
   has 3 parents and the LCA must go up to a grandparent.

## IC-Based Semantic Similarity

Instead of counting edges, we measure how "informative" the shared ancestor is.

### Lin Similarity Formula

    Sim_Lin(A, B) = 2 * IC(MICA) / (IC(A) + IC(B))

Where MICA = the common ancestor with the highest IC (most specific/informative).
Result is in [0, 1] where 1 = identical terms, 0 = completely unrelated.

### IC Formulas Evaluated

Four intrinsic IC formulas were compared (no external corpus needed):

**1. Seco (2004)**

    IC(t) = 1 - log(desc(t) + 1) / log(N)

Pure descendant count. N = total terms in ontology.

**2. Zhou (2008)**

    IC(t) = k * Seco_IC(t) + (1-k) * log(depth(t) + 1) / log(max_depth + 1)

Blends descendant count with depth. Parameter k controls the weight (k=0.5 = equal blend).

**3. Sanchez (2011)**

    IC(t) = -log( (leaves(t)/desc(t) + 1) / (max_leaves + 1) )

Uses ratio of leaf descendants to total descendants.

**4. Depth-only**

    IC(t) = depth(t) / max_depth

Pure structural depth baseline.

## Results on CL Ontology (3,238 non-obsolete terms)

### IC Values for Neuroscience Cell Types

    Cell Type                                Depth  Desc Leaves |   Seco   Zhou Sanchez  Depth
    ===========================================================================================
    cell (root)                                  0  3228   2039 |  0.000  0.000   7.131  0.000
    secretory cell                               1   614    388 |  0.206  0.238   7.131  0.083
    neuron                                       3   746    510 |  0.181  0.361   7.100  0.250
    GABAergic neuron                             2   159    104 |  0.372  0.400   7.117  0.167
    glutamatergic neuron                         2    98     64 |  0.431  0.430   7.118  0.167
    astrocyte                                    6    59     41 |  0.493  0.626   7.093  0.500
    oligodendrocyte                              6    28     19 |  0.583  0.671   7.103  0.500
    microglial cell                              4     4      3 |  0.801  0.714   7.061  0.333
    Purkinje cell                                3     4      2 |  0.801  0.671   7.215  0.250
    glial cell                                   4   154    101 |  0.376  0.502   7.116  0.333
    macroglial cell                              5    89     60 |  0.443  0.571   7.105  0.417
    astrocyte of the cerebral cortex             6     6      4 |  0.759  0.759   7.110  0.500

### Lin Similarity Comparison (Neuroscience Cell Type Pairs)

    Cell A                    Cell B                    Seco   Zhou  Sanchez  Depth
    ================================================================================
    GABAergic neuron          glutamatergic neuron      0.511  0.870  1.002   1.500
    neuron                    GABAergic neuron          0.656  0.948  1.003   1.200
    GABAergic neuron          Purkinje cell             0.634  0.747  0.995   1.200
    astrocyte                 oligodendrocyte           0.823  0.880  1.005   0.833
    astrocyte                 microglial cell           0.581  0.749  1.008   0.800
    neuron                    astrocyte                 0.433  0.582  1.005   0.444
    neuron                    microglial cell           0.298  0.534  1.007   0.571
    cortical astrocyte        astrocyte                 0.788  0.904  1.004   1.000
    glutamatergic neuron      Purkinje cell             0.334  0.656  0.995   1.200

### MICA Selection (Which Common Ancestor is Picked)

    Cell A                    Cell B                    Seco MICA              Zhou MICA
    =======================================================================================
    GABAergic neuron          glutamatergic neuron      secretory cell         neuron
    neuron                    GABAergic neuron          neuron                 neuron
    GABAergic neuron          Purkinje cell             GABAergic neuron       GABAergic neuron
    astrocyte                 oligodendrocyte           macroglial cell        macroglial cell
    astrocyte                 microglial cell           glial cell             glial cell
    neuron                    astrocyte                 neural cell            neural cell
    neuron                    microglial cell           neural cell            neural cell

## Formula Evaluation

### Sanchez: BROKEN for this ontology

All Lin similarities are ~1.0 (range: 0.994-1.008). The MICA is always `cell` (root).
The leaves/descendants ratio does not differentiate well in CL because most terms have
similar branching patterns. This formula provides no discriminative power.

### Depth-only: BROKEN

Produces Lin similarities > 1.0 (e.g., 1.5 for GABAergic vs glutamatergic). This occurs
because in a DAG with multiple inheritance, the MICA (e.g., `neuron` at depth 3) can be
deeper than the query terms (e.g., `GABAergic neuron` at depth 2, reached via a shorter
path through `secretory cell` at depth 1). Depth alone cannot produce valid Lin similarity.

### Seco: Works but semantically incorrect MICA selection

`neuron` (depth 3, 746 descendants) gets IC=0.181, while `secretory cell` (depth 1, 614
descendants) gets IC=0.206. This means `secretory cell` is picked as the MICA for
GABAergic vs glutamatergic neuron — semantically wrong, since both are neuron subtypes
and `neuron` is clearly the more meaningful common ancestor.

The problem: Seco IC is dominated by descendant count. `neuron` has more descendants (746)
than `secretory cell` (614), so its IC is lower despite being deeper and more specific in
context. The formula ignores structural depth entirely.

### Zhou (k=0.5): BEST — biologically intuitive results

By blending descendant count with depth:

- `neuron` gets IC=0.361 (depth component boosts it despite 746 descendants)
- `secretory cell` gets IC=0.238 (shallow depth pulls it down despite fewer descendants)
- MICA for GABAergic vs glutamatergic is correctly `neuron` (IC=0.361 > 0.238)

Biological intuition check:
- GABAergic vs glutamatergic: **0.870** (high — both neuron subtypes, MICA=neuron)
- GABAergic vs Purkinje: **0.747** (high — Purkinje is a GABAergic neuron)
- astrocyte vs oligodendrocyte: **0.880** (high — both macroglial cells)
- astrocyte vs microglial cell: **0.749** (moderate-high — both glia but different lineages)
- neuron vs astrocyte: **0.582** (moderate — different cell classes, both neural)
- neuron vs microglial cell: **0.534** (moderate — neural vs immune-derived)
- cortical astrocyte vs astrocyte: **0.904** (very high — subtype vs parent)

These rankings match biological expectations: closely related cell types score high,
distantly related ones score lower, and the ordering is sensible.

## Decision

**Use Zhou (k=0.5) IC with Lin similarity** as the ontology-based semantic similarity
metric in the benchmarking framework.

## Correlation Between IC Similarity and Shortest-Path Distance

To understand how the two ontology scoring methods relate, we computed both metrics for
2,000 random CL term pairs (1,788 valid — both methods returned a result) and measured
correlation.

### Summary

| Metric | Value |
|--------|-------|
| Pearson r | **-0.34** |
| Spearman r (rank) | **-0.47** |

The two methods are only **moderately anti-correlated**. They agree on the broad ranking
of term pairs (nearby pairs score similarly under both) but disagree on roughly half the
fine-grained ordering.

### Mean IC Similarity at Each Shortest-Path Distance

    distance  mean_sim  median_sim  std     n
    =========================================
         1     0.7372       —        —       1
         2     0.5966    0.6405    0.1922   15
         3     0.5329    0.5992    0.2204   44
         4     0.3349    0.2591    0.2252  112
         5     0.2843    0.2105    0.1912  201
         6     0.2593    0.1986    0.1627  281
         7     0.2259    0.1857    0.1284  322
         8     0.2257    0.1824    0.1120  309
         9     0.2171    0.1766    0.1116  194
        10     0.2075    0.1760    0.0806  126
        11     0.1832    0.1703    0.0656   75
        12     0.1802    0.1704    0.0645   54
        13     0.1763    0.1649    0.0588   33
        14     0.1622    0.1612    0.0038   10
        15     0.1615    0.1611    0.0041    9
        16     0.1597       —        —       1
        17     0.1554       —        —       1

### Interpretation

1. **High variance at mid-range distances.** At distance=4, the IC similarity standard
   deviation is 0.23 on a mean of 0.33. Two term pairs that are 4 hops apart can have
   vastly different semantic similarity — some 4-hop paths cross meaningful biological
   boundaries while others stay within a single branch.

2. **Diminishing sensitivity at longer distances.** Shortest-path distances 7 through 12
   all produce roughly similar IC similarity values (~0.18–0.23). The path method keeps
   counting hops, but IC correctly saturates — terms that far apart are all "distant"
   regardless of a few extra edges.

3. **IC captures edge weight implicitly.** A hop near the leaves (between specific cell
   subtypes like Purkinje cell → GABAergic neuron) carries more semantic weight than a
   hop near the root (between broad categories like cell → native cell). Shortest-path
   treats all edges as equally meaningful.

These results confirm that IC similarity and shortest-path distance capture partially
overlapping but distinct information about ontological relatedness. The IC method is
recommended as the default because it better reflects biological intuition, is
well-defined on DAGs with multiple inheritance, and avoids the ambiguities of
undirected path shortcutting.

### Reproducibility

Analysis was run on the full Cell Ontology (3,434 CL terms), sampling 2,000 random pairs
with seed=42. Zhou k=0.5 IC with Lin similarity vs. shortest undirected path distance.

## References

- Lin, D. (1998). An Information-Theoretic Definition of Similarity. In *Proceedings of
  the 15th International Conference on Machine Learning (ICML 1998)*, Vol. 98, pp. 296-304.
- Zhou, Z., Wang, Y., & Gu, J. (2008). A New Model of Information Content for Semantic
  Similarity in WordNet. In *2008 Second International Conference on Future Generation
  Communication and Networking Symposia*, Hainan, China, pp. 85-89.
  doi: 10.1109/FGCNS.2008.16.
- Seco, N., Veale, T., & Hayes, J. (2004). An Intrinsic Information Content Metric for
  Semantic Similarity in WordNet. In *Proceedings of the 16th European Conference on
  Artificial Intelligence (ECAI 2004)*, pp. 1089-1090.
- Sanchez, D., Batet, M., & Isern, D. (2011). Ontology-based information content
  computation. *Knowledge-Based Systems*, 24(2), pp. 297-303.

## Reproducibility

Run `python ic_formula_comparison.py` to regenerate all results.
Raw output saved in `ic_formula_comparison_results.txt`.
