# Bug Fix Notes — Code Review Feedback

This document records the bugs found during code review of the benchmarking framework,
along with fixes applied and lessons for the team. Work is being done on the
`review/student-code-fixes` branch.

---

## Status Summary

| Step | Bug | Status |
|------|-----|--------|
| 0 | Ontology DAG analysis (research) | Done |
| 1 | FAISS -1 neighbor indices wrap around | **Fixed** |
| 2 | Cosine metric doesn't compute cosine | **Fixed** |
| 3 | `score_batch` returns (0.0, 0.0) on failure | **Fixed** |
| 4 | Empty-string predictions corrupt F1 | **Fixed** |
| 5 | Add IC-based similarity as option alongside path distance | **Fixed** |
| 6 | Graph rebuilt on every cell (performance) | **Fixed** (folded into Step 5) |

---

## Step 1: FAISS -1 Neighbor Indices Silently Wrap Around

**Files:** `prediction_module.py` (line 83), `main_benchmark.py` (line 276)

**What was wrong:**
FAISS returns `-1` for neighbors it can't find (e.g., when k exceeds the index size or
search fails partially). NumPy interprets `-1` as "last element" when used as an array
index, so `term_ids[-1]` silently returns the last reference cell's label instead of
signaling an error. This injects a wrong label into the majority vote.

**What we fixed:**
Added a filter `valid_mask = row_indices >= 0` before indexing into the reference
annotations. Cells with zero valid neighbors get an empty prediction.

**Lesson:**
NumPy negative indexing is a feature, not a bug — but it becomes a bug when your data
uses -1 as a sentinel value. Always filter sentinel values before using arrays as indices.
This is a common FAISS + NumPy pitfall.

---

## Step 2: Cosine Metric Doesn't Actually Compute Cosine

**Files:** `prediction_module.py` (`execute_query()`), `main_benchmark.py` (line 32)

**What was wrong:**
The code normalized only the query vectors (`faiss.normalize_L2(queries)`) while the
FAISS index was built with L2 (Euclidean) metric. This computes:

    ||q_normalized - v_unnormalized||²

which is neither Euclidean distance nor cosine similarity — it's a meaningless hybrid.
For true cosine similarity, you need to build the index with `METRIC_INNER_PRODUCT` and
normalize **all** vectors (both index and query) at index-build time.

**What we fixed:**
- `execute_query()` now raises a `ValueError` if `metric='cosine'` is passed, with an
  explanation of why it can't work with an L2-built index.
- Removed `'cosine'` from the default metrics list in `main_benchmark.py`.
- Removed dead normalization-check code.

**Lesson:**
The distance metric in FAISS is baked into the index at build time. You cannot change it
at query time by transforming only one side of the comparison. Understanding the math
behind distance metrics (L2 vs inner product vs cosine) prevents this kind of silent
error. Also: if embeddings are already L2-normalized, euclidean and cosine rankings are
mathematically identical — `||q-v||² = 2(1 - q·v)`.

---

## Step 3: `score_batch` Returns (0.0, 0.0) on Failure

**File:** `ontology_utils.py` (line 162)

**What was wrong:**
When `score_batch()` can't compute any valid ontology distances (all IDs missing from
the ontology, data mismatch, etc.), it returned `(0.0, 0.0)`. A distance of 0.0 means
"perfect match" — so total failure looks like perfect performance.

**What we fixed:**
Changed `return 0.0, 0.0` to `return float('nan'), float('nan')`. NaN propagates
visibly through DataFrames and CSV output, making it obvious that something went wrong.

**Lesson:**
Never use a valid value as a sentinel for failure. Zero is a meaningful distance (exact
match). Use NaN, None, or raise an exception to signal "no data" — otherwise failures
masquerade as success and are invisible in results tables.

---

## Step 4: Empty-String Predictions Corrupt F1 Scores

**File:** `main_benchmark.py` (before `calculate_accuracy()` call)

**What was wrong:**
When all of a cell's neighbors have invalid labels, `vote_neighbors()` returns `''`
(empty string) as the prediction. This empty string becomes a phantom class in sklearn's
F1 calculation. Since no ground truth cell has label `''`, the phantom class gets
precision=0, recall=0, F1=0 — and macro F1 averages this zero across all real classes,
dragging down the score.

Example: 10 real cell types with perfect F1=1.0, plus one `''` phantom class →
macro F1 = 10/11 = 0.909 instead of 1.0.

**What we fixed:**
Filter out cells with empty-string predictions (and their corresponding ground truth
and neighbor labels) before passing to `calculate_accuracy()`. A log message reports
how many cells were filtered.

**Lesson:**
Be aware of how sklearn metrics handle unexpected classes. Macro F1 averages equally
across all unique labels it sees — including garbage labels you didn't intend. Always
validate your prediction list before computing metrics.

---

## Step 5: Add IC-Based Semantic Similarity as Alternative to Path Distance

**Files:** `ontology_utils.py`, `main_benchmark.py`, `unit-tests/test_ontology_utils.py`

**What was wrong:**
The original `calculate_graph_distance()` converts the directed ontology graph to
undirected and finds the shortest path. The docstring claims LCA-based distance, but
the code computes shortest undirected path. These differ on DAGs — the Cell Ontology
has 33.5% of terms with multiple parents.

Shortest undirected path can shortcut across separate branches of the hierarchy,
underestimating true semantic distance.

**What we fixed:**
Added IC-based Lin similarity as an alternative method, selectable via
`--ontology-method` CLI option (choices: `ic` or `shortest_path`, default `ic`).
The original path-based distance is preserved as the `shortest_path` option.

New functions in `ontology_utils.py`:
- `precompute_ic(graph, k=0.5)` — precomputes Zhou IC for all terms (done once)
- `calculate_lin_similarity(graph, pred, truth, ic_values)` — pairwise Lin similarity
- `_compute_pairwise_score(graph, pred, truth, method, ic_values)` — unified dispatcher
- All batch functions (`score_batch`, `calculate_per_cell_distances`,
  `calculate_avg_neighbor_distances`) accept `method` and `ic_values` parameters

The IC method uses:

- **Zhou (2008) weighted intrinsic IC:**
  IC(t) = k * (1 - log(desc+1)/log(N)) + (1-k) * log(depth+1)/log(max_depth+1)
  with k=0.5 (equal blend of descendant count and structural depth).
  Reference: Zhou, Z., Wang, Y., & Gu, J. (2008). A New Model of Information Content
  for Semantic Similarity in WordNet. In *2008 Second International Conference on Future
  Generation Communication and Networking Symposia*, Hainan, China, pp. 85-89.
  doi: 10.1109/FGCNS.2008.16.

- **Lin (1998) similarity:**
  Sim_Lin(A,B) = 2 * IC(MICA) / (IC(A) + IC(B))
  where MICA = Most Informative Common Ancestor (highest IC among common ancestors).
  Reference: Lin, D. (1998). An Information-Theoretic Definition of Similarity. In
  *Proceedings of the 15th International Conference on Machine Learning (ICML 1998)*,
  Vol. 98, pp. 296-304.

**Lesson:**
When working with ontologies that are DAGs (not trees), path-based distances become
ambiguous due to multiple inheritance. IC-based similarity uses the information content
of the Most Informative Common Ancestor (MICA), which is well-defined regardless of
graph structure. The choice of IC formula matters — see `IC_FORMULA_ANALYSIS.md` for
why Zhou k=0.5 was selected over Seco, Sanchez, and depth-only formulas.

---

## Step 6: Graph Rebuilt on Every Cell (Performance)

**File:** `ontology_utils.py` — `calculate_graph_distance()`

**What was wrong:**
`graph.to_undirected()` was called once per cell inside `calculate_graph_distance()`.
For 10,000 cells x 30 neighbors = 300,000 graph copies. Pure waste.

**What we fixed:**
Added `_undirected_cache` dict parameter to `calculate_graph_distance()` that caches
the undirected conversion by graph ID. The conversion now happens once per graph
object, not per call. (The IC method doesn't use undirected graphs at all, so this
only applies to `--ontology-method shortest_path`.)

**Lesson:**
When a function is called in a tight loop, check whether it does expensive work that
could be hoisted out or cached. Converting a graph with thousands of edges 300,000
times is the kind of hidden performance bug that's easy to miss in code review but
obvious in a profiler.

---

## Also Fixed: Unit Tests Used Fake Ontology IDs

**File:** `unit-tests/test_ontology_utils.py`

The original tests used sequential CL IDs (CL:0000001 = "neuron", CL:0000002 = "astrocyte")
that don't exist in the real Cell Ontology. CL:0000001 is actually "primary cultured cell"
and CL:0000002 is obsolete. These tests had never been run with pronto/networkx installed.

Rewrote all 21 tests with correct CL IDs verified against the actual ontology. All pass.

---

## Research Artifacts

- `ic_formula_comparison.py` — Script comparing 4 IC formulas on the CL ontology
- `ic_formula_comparison_results.txt` — Raw output of the comparison
- `IC_FORMULA_ANALYSIS.md` — Full analysis document with rationale for choosing Zhou k=0.5
