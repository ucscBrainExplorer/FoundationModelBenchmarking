# Changelog

---

## 2026-02-10 — Fix 6 logic bugs and add IC-based ontology similarity metric

**Commit:** `25bb1d0` on branch `review/student-code-fixes`

### Modified files (4)

#### `prediction_module.py` — `execute_query()` and `vote_neighbors()`

- **Step 1:** Filter out FAISS -1 sentinel indices before indexing into reference annotations (prevents silent wrap-around to last element)
- **Step 2:** Cosine metric now raises `ValueError` with explanation instead of silently computing a meaningless hybrid distance. Removed dead normalization-check code. Cleaned up docstring.

#### `main_benchmark.py` — `run_benchmark()`

- **Step 1:** Filter -1 indices in the top-k neighbor labels section
- **Step 2:** Removed `'cosine'` from default metrics list. Removed the normalized-embeddings warning block (no longer needed)
- **Step 4:** Filter out empty-string predictions before passing to `calculate_accuracy()`, with a log message reporting how many were filtered and the cell IDs of affected cells (logged to `benchmark.log`)
- **Step 5:** Added `precompute_ic` import. New `ontology_method` parameter (default `'ic'`). Precomputes IC values when ontology is loaded. Passes `method` and `ic_values` to all ontology scoring calls. Uses `if/elif/else` to set column names (`mean_ontology_similarity` for IC, `mean_ontology_dist` for shortest_path). New `--ontology-method` CLI argument with `choices=["ic", "shortest_path"]`.

#### `ontology_utils.py` — complete refactor

- **Step 3:** `score_batch()` returns `(NaN, NaN)` instead of `(0.0, 0.0)` when no valid distances
- **Step 5:** Added IC-based Lin similarity as alternative method:
  - `_get_all_ancestors()`, `_get_all_descendants()`, `_get_shortest_depth()` — graph traversal helpers
  - `precompute_ic(graph, k=0.5)` — Zhou (2008) weighted IC for all terms
  - `calculate_lin_similarity(graph, pred, truth, ic_values)` — Lin (1998) similarity via MICA
  - `_compute_pairwise_score()` — unified dispatcher for both methods
  - `score_batch()`, `calculate_per_cell_distances()`, `calculate_avg_neighbor_distances()` — all accept `method` and `ic_values` parameters
  - Full academic citations in docstrings and section header comments
- **Step 6:** `calculate_graph_distance()` caches the undirected graph conversion via `_undirected_cache` dict

#### `unit-tests/test_ontology_utils.py`

- Rewrote all tests with correct CL IDs (original used fake IDs that don't exist in the real ontology)
- Added `TestLinSimilarity` class (8 tests): identical terms, missing terms, neuron subtypes high similarity, sibling glia high similarity, neuron vs astrocyte moderate, biological ordering, symmetry, root has lowest IC
- Added `TestScoreBatchIC` class (3 tests): perfect predictions, all-invalid returns NaN, similar pairs high score
- Total: **32 tests**, all passing

### New files (4)

| File | Purpose |
|------|---------|
| `BUGFIX_NOTES.md` | Documents all 6 bugs with what was wrong, what was fixed, and lessons for students |
| `IC_FORMULA_ANALYSIS.md` | Analysis of 4 IC formulas with rationale for choosing Zhou k=0.5 |
| `ic_formula_comparison.py` | Script that generated the IC formula comparison |
| `ic_formula_comparison_results.txt` | Raw output of the comparison script |

### Test results

**44 tests pass** (32 ontology + 12 evaluation). The 3 faiss-dependent test files can't run locally (faiss not installed).
