# Changelog

---

## 2026-02-12 — Split benchmarking into three standalone composable programs

**Branch:** `review/student-code-fixes`

### Summary

Major refactoring that splits the monolithic `main_benchmark.py` into three standalone, composable programs: `predict.py` (prediction without ground truth), `evaluate.py` (evaluation with ontology metrics), and `annotate_cl_terms.py` (readable names → CL IDs with LLM fallback). Enables flexible workflows including evaluating external predictions and predicting on unlabeled data. `main_benchmark.py` preserved for backwards compatibility.

### New Programs

#### `predict.py` — Standalone prediction
- Predict cell types using FAISS k-NN + majority voting
- **No ground truth required** — works on unlabeled data
- Outputs TSV with: predicted CL term IDs, readable names, vote confidence, mean euclidean distance, neighbor distances, neighbor cell types
- Reuses: `data_loader`, `prediction_module`, `obo_parser`

#### `evaluate.py` — Standalone evaluation
- Evaluate predictions using ontology-based semantic metrics (IC similarity or shortest-path distance)
- **Accepts any predictions** — not just from `predict.py`
- Configurable column names via `--pred_id_col`, `--truth_id_col` for external prediction formats
- Outputs: `evaluation_summary.tsv`, `per_cell_evaluation.tsv`, `ontology_analysis_report.txt`, `ontology_distance_analysis.png`
- Reuses: `ontology_utils`, `analyze_ontology_results`, `obo_parser`
- Reports exact CL term match rate as supplementary statistic

#### `annotate_cl_terms.py` — Standalone annotation
- Map readable cell type names → CL ontology term IDs
- Multi-step matching: exact match → synonym match → fuzzy normalization → LLM-assisted fallback
- LLM consensus: queries both Claude and OpenAI APIs, auto-accepts if both agree, interactive prompt if they disagree
- Fuzzy normalization: strips hyphens/underscores/plurals (e.g., "T-cells" → "T cell", "oligodendrocytes" → "oligodendrocyte")
- Outputs original TSV + added `cell_type_ontology_term_id` column
- Requires `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` for LLM fallback

### New Module

#### `obo_parser.py` — OBO parsing
- Extracted `parse_obo_names()` from `utility/normalize_cell_types.py` to eliminate code duplication
- Single source of truth for OBO file parsing
- Now imported by: `predict.py`, `evaluate.py`, `annotate_cl_terms.py`, `utility/normalize_cell_types.py`

### Modified Files

#### `utility/normalize_cell_types.py`
- Removed local `parse_obo_names()` definition
- Now imports from `obo_parser` module

#### `README.md` — Complete rewrite
- Three standalone programs prominently featured at top
- Three workflow examples (A: full benchmark, B: external predictions, C: predict only)
- Moved `main_benchmark.py` to "Legacy" section (still documented for Kubernetes + local)
- Updated project structure showing new files
- Fixed misleading statement: `evaluate.py` uses ontology metrics only; `main_benchmark.py` computes accuracy/F1
- Links to new `docs/` folder

#### `HOW_TO_RUN.md` — Removed
- Content consolidated into README and `docs/` folder

### New Documentation (`docs/`)

#### `docs/QUICK_START.md`
- Quick start guide for all three programs
- Three workflow examples with copy-paste commands
- Ontology metrics explanation (IC similarity vs. shortest-path distance)
- Common arguments tables

#### `docs/USAGE.md`
- Detailed usage for each program with multiple examples
- File format specifications
- Troubleshooting section (common errors and solutions)
- Output file descriptions

#### `docs/API.md`
- Full programmatic API reference for all modules
- Function signatures, args, returns, raises
- Complete example workflow at end

### New Tests (`tests/`)

Modern pytest-based test suite:
- `tests/conftest.py` — Shared fixtures (temp dirs, sample OBO, embeddings, annotations, predictions, ground truth)
- `tests/test_obo_parser.py` — OBO parsing tests
- `tests/test_predict.py` — Argument parsing and output format tests
- `tests/test_evaluate.py` — Argument parsing and workflow tests
- `tests/test_annotate_cl_terms.py` — Matching logic tests (exact, synonym, fuzzy, LLM)
- `tests/README.md` — Test suite documentation

Legacy `unit-tests/` directory preserved unchanged.

### Documentation Files

- `IMPLEMENTATION_SUMMARY.md` — Complete overview of refactoring decisions and design
- `VERIFICATION_CHECKLIST.md` — Testing and validation checklist

### Files Removed

- `analyze_per_cell_results.py` — Functionality now in `analyze_ontology_results.py` (not imported anywhere)
- `ic_formula_comparison.py` — Standalone research script (not imported anywhere)
- `verify_setup.py` — Development script (not imported anywhere)
- `HOW_TO_RUN.md` — Content consolidated into README and `docs/`
- `docs/README_UPDATE.md` — Source content for README update (no longer needed)

### Key Design Decisions

1. **Composability** — Each program has single responsibility, can be used independently or chained
2. **Backwards compatibility** — `main_benchmark.py` unchanged, all existing modules unchanged
3. **Column name flexibility** — `evaluate.py` accepts custom column names for external predictions
4. **Ontology-first evaluation** — Exact match is supplementary; primary metrics are semantic similarity
5. **LLM-assisted annotation** — Dual API (Claude + OpenAI) for robustness with consensus-based auto-accept

### Workflows Enabled

**Workflow A: Full benchmark (predict + evaluate)**
```bash
python3 predict.py --index idx.faiss --ref_annot ref.tsv --obo cl.obo --embeddings test.npy --output preds.tsv
python3 evaluate.py --predictions preds.tsv --ground_truth truth.tsv --obo cl.obo --output eval_results/
```

**Workflow B: External predictions (map names → evaluate)**
```bash
python3 annotate_cl_terms.py --obo cl.obo --input external_preds.tsv --output annotated.tsv
python3 evaluate.py --predictions annotated.tsv --ground_truth truth.tsv --obo cl.obo --output eval_results/
```

**Workflow C: Predict only (no ground truth)**
```bash
python3 predict.py --index idx.faiss --ref_annot ref.tsv --obo cl.obo --embeddings unlabeled.npy --output preds.tsv
```

---

## 2026-02-10 — Simplify S3/local workflow, clean up CLI args and docs

**Commit:** `0cc8922` on branch `review/student-code-fixes`

### Summary

Simplified to two modes: S3 (cluster) or local (`--no-s3`). Removed PVC support, `--results-dest`, and `DATA_ROOT` detection. S3 download/upload is now automatic unless `--no-s3`.

### Changes

- **`main_benchmark.py`** — Removed `--results-dest` and `DATA_ROOT`. S3 enabled = download + upload; `--no-s3` = local only. New `--index` arg (single file) replaces unused `--indices_config`. S3 download places files at CLI arg paths instead of hardcoded subdirectories. Now downloads `cl.obo` and all `.faiss` files. Removed baked-in S3 defaults.
- **`data_loader.py`** — Cleaned up `load_test_batch()`: documented `{dataset_id}` naming convention, removed dead code. S3 download skips `benchmark_results/` keys.
- **`benchmarking-job.yaml`** — Renamed job to `fm-benchmark-job`, image to `jzhu647/fm_benchmark:latest`. Simplified args to match S3 filenames. Removed PVC volume mount.
- **`README.md`** — Trimmed to ~210 lines. Removed PVC, `--results-dest`, `--no-upload` references.
- **`HOW_TO_RUN.md`** — Rewritten ~155 lines. Options table shows S3 source for each arg. Removed PVC sections.
- **`requirements.txt`** / **`Dockerfile`** — Unpinned `faiss-cpu`.
- **`run_benchmark.sh`** — Deleted.
- **`pvc.yaml`** / **`pvc-access-pod.yaml`** — Deleted.

---

## 2026-02-10 — Add S3 upload, timestamped results, and documentation updates

**Commit:** `fd4fdab` on branch `review/student-code-fixes`

### Modified files (5)

#### `main_benchmark.py` — timestamped results + S3 upload + logging fix

- All results now write to `{DATA_ROOT}/benchmark_results/{timestamp}/` instead of flat in `DATA_ROOT`. Each run is isolated; previous results are never overwritten.
- Replaced `# TODO: Add file to s3 - Suhas` with actual S3 upload. Results upload automatically to `s3://{bucket}/{prefix}/benchmark_results/{timestamp}/` after each run.
- New `--no-upload` CLI flag to skip S3 upload independently of `--no-s3`.
- Moved `logging.basicConfig` after `DATA_ROOT` definition; log file now writes into the timestamped results directory.

#### `data_loader.py` — S3 upload function

- `_get_s3_client()` — reusable S3 client creation (env vars on cluster, AWS profile locally).
- `upload_results_to_s3(bucket, prefix, results_dir, timestamp)` — walks the local results directory and uploads all files to S3 under the same timestamped structure.

#### `README.md` — updated for current code

- Removed cosine from supported metrics; updated `ontology_utils.py` module description.
- Replaced Ontology-Aware Metrics section with both IC and shortest-path documentation, including score interpretation table (**higher = more similar** for IC).
- Updated output columns table with `mean/median_ontology_similarity`.
- Added `--ontology-method` and `--no-upload` CLI examples.
- Replaced Generated Results section with full directory tree (local + S3) and retrieval commands.
- Updated Quick Start with timestamped paths and S3 download instructions.
- Added Lin (1998) and Zhou (2008) to References.

#### `IC_FORMULA_ANALYSIS.md` — correlation analysis

- New section: "Correlation Between IC Similarity and Shortest-Path Distance"
  - Pearson r = -0.34, Spearman r = -0.47 (moderately anti-correlated)
  - Mean IC similarity at each path distance (1–17)
  - Interpretation: high variance at mid-range, diminishing sensitivity at long range, IC captures implicit edge weight
  - Reproducibility note (3,434 CL terms, 2,000 pairs, seed=42)

#### `HOW_TO_RUN.md` — rewritten for current code

- Removed hardcoded student path (`/Users/Suhas/...`).
- All result paths updated to `benchmark_results/{timestamp}/`.
- Step 8 reorganized into 3 retrieval options: S3 download (recommended), PVC pod, kubectl cp.
- New Step 9: results directory structure, summary and per-cell CSV column descriptions, similarity interpretation table.
- Added S3 upload to log monitoring messages and troubleshooting.
- Removed references to nonexistent `TECHNICAL_WALKTHROUGH.md` and cosine metric.

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
