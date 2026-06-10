#!/usr/bin/env python3
"""
run_mmc.py — Reusable MapMyCells (MMC) runner

Workflow:
  1. Prepare raw-count h5ads (writes temp files if X is log-normalized)
  2. Build precomputed_stats.h5 from reference (skipped if already present)
  3. Run hierarchical cell type mapping via on-the-fly marker discovery
  4. Post-process to produce mmc_results_cl.tsv with CL ontology term IDs
  5. Save run_config.json for reproducibility

Usage:
  python3.11 run_mmc.py \\
    --query  /path/to/query.h5ad \\
    --reference /path/to/reference.h5ad \\
    --ref_label_col cell_label \\
    --ref_cl_col cell_label_ontology_term_id \\
    --output_dir /path/to/output/ \\
    --n_processors 4
"""

import argparse
import json
import os
import pathlib
import time

import anndata
import numpy as np
import pandas as pd
import scipy.sparse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_raw_counts(h5ad_path):
    """
    Return True if X looks like raw integer counts.
    Samples the first 100 rows in backed mode to avoid loading the full file.
    """
    adata = anndata.read_h5ad(h5ad_path, backed="r")
    X = adata.X
    sample = X[:100]
    if scipy.sparse.issparse(sample):
        sample = sample.toarray()
    else:
        sample = np.array(sample)
    adata.file.close()
    return np.allclose(sample, np.round(sample), atol=1e-3)


def prepare_raw_h5ad(h5ad_path, tmp_dir, label="file"):
    """
    If X is not integer raw counts, write a temp h5ad using .raw.X as X.
    Returns path to use (either original or temp file).
    """
    print(f"  Checking {label} for raw counts...", flush=True)
    if check_raw_counts(h5ad_path):
        print(f"    X appears to be raw counts — using as-is.")
        return h5ad_path

    print(f"    X is not raw counts — extracting .raw.X to temp h5ad...")
    adata = anndata.read_h5ad(h5ad_path)
    if adata.raw is None:
        raise ValueError(
            f".raw is None in {h5ad_path}. "
            "Cannot prepare raw-count h5ad. "
            "Re-run with --normalization log2CPM or supply a file with raw counts in X."
        )
    raw_adata = adata.raw.to_adata()
    tmp_path = os.path.join(tmp_dir, pathlib.Path(h5ad_path).stem + "_raw.h5ad")
    raw_adata.write_h5ad(tmp_path)
    print(f"    Wrote temp raw h5ad: {tmp_path}")
    return tmp_path


# ---------------------------------------------------------------------------
# Step 2: Build precomputed stats
# ---------------------------------------------------------------------------

def build_precomputed_stats(ref_path, output_path, label_col,
                            n_processors, normalization, clobber):
    from cell_type_mapper.cli.precompute_stats_scrattch import (
        PrecomputationScrattchRunner,
    )

    config = {
        "h5ad_path": ref_path,
        "output_path": output_path,
        "hierarchy": [label_col],
        "normalization": normalization,
        "n_processors": n_processors,
        "clobber": clobber,
        "tmp_dir": None,
        "layer": "X",
        "gene_id_col": None,
    }
    runner = PrecomputationScrattchRunner(args=[], input_data=config)
    runner.run()


# ---------------------------------------------------------------------------
# Step 3: Run mapping
# ---------------------------------------------------------------------------

def run_mapping(query_path, precomputed_stats_path, output_dir,
                n_processors, normalization):
    from cell_type_mapper.cli.map_to_on_the_fly_markers import OnTheFlyMapper

    csv_path = os.path.join(output_dir, "mmc_results.csv")
    json_path = os.path.join(output_dir, "mmc_results.json")

    config = {
        "query_path": query_path,
        "precomputed_stats": {
            "path": precomputed_stats_path,
            "log_level": "ERROR",
        },
        "reference_markers": {
            "precomputed_path_list": [precomputed_stats_path],
            "log_level": "ERROR",
        },
        "query_markers": {
            "log_level": "ERROR",
        },
        "type_assignment": {
            "algorithm": "hierarchical",
            "normalization": normalization,
            "bootstrap_factor": 0.9,
            "bootstrap_iteration": 100,
            "min_markers": 10,
            "n_runners_up": 5,
            "chunk_size": 10000,
            "rng_seed": 42,
            "log_level": "ERROR",
        },
        "csv_result_path": csv_path,
        "extended_result_path": json_path,
        "n_processors": n_processors,
        "log_level": "ERROR",
    }
    runner = OnTheFlyMapper(args=[], input_data=config)
    runner.run()
    return csv_path, json_path


# ---------------------------------------------------------------------------
# Step 4: Post-process → mmc_results_cl.tsv
# ---------------------------------------------------------------------------

def postprocess(csv_path, reference_h5ad, label_col, cl_col, output_dir):
    df = pd.read_csv(csv_path, comment="#")

    label_col_name = f"{label_col}_label"
    prob_col_name = f"{label_col}_bootstrapping_probability"

    if label_col_name not in df.columns:
        raise ValueError(
            f"Expected column '{label_col_name}' in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    # Build CL ID lookup from reference obs
    ref = anndata.read_h5ad(reference_h5ad, backed="r")
    if cl_col not in ref.obs.columns:
        raise ValueError(
            f"Column '{cl_col}' not found in reference obs. "
            f"Available columns: {list(ref.obs.columns)}"
        )
    cl_lookup = (
        ref.obs[[label_col, cl_col]]
        .drop_duplicates()
        .set_index(label_col)[cl_col]
        .to_dict()
    )
    ref.file.close()

    out = pd.DataFrame({
        "cell_id": df["cell_id"],
        "mmc_cell_type": df[label_col_name],
        "mmc_cell_type_ontology_term_id": df[label_col_name].map(cl_lookup),
        "mmc_bootstrapping_probability": df[prob_col_name],
    })

    out_path = os.path.join(output_dir, "mmc_results_cl.tsv")
    out.to_csv(out_path, sep="\t", index=False)
    print(f"  Wrote {len(out):,} rows → {out_path}")

    unmapped = out["mmc_cell_type_ontology_term_id"].isna().sum()
    if unmapped:
        print(f"  WARNING: {unmapped:,} cells have no CL term ID mapping")
    else:
        print(f"  All cells mapped to CL term IDs.")

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run MapMyCells hierarchical cell type mapping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", required=True,
                        help="Path to query h5ad")
    parser.add_argument("--reference", required=True,
                        help="Path to reference h5ad")
    parser.add_argument("--ref_label_col", default="cell_label",
                        help="obs column for cell type labels in reference")
    parser.add_argument("--ref_cl_col", default="cell_label_ontology_term_id",
                        help="obs column for CL ontology term IDs in reference")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory (created if absent)")
    parser.add_argument("--n_processors", type=int, default=4)
    parser.add_argument("--normalization", default="raw",
                        choices=["raw", "log2CPM"])
    parser.add_argument("--clobber", action="store_true",
                        help="Overwrite existing precomputed_stats.h5")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    precomputed_stats_path = os.path.join(args.output_dir, "precomputed_stats.h5")

    t_total = time.time()

    # Use output_dir for temp files (avoids filling /tmp)
    tmp_dir = os.path.join(args.output_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # Step 1: Ensure raw counts in X
        print("\n[1/4] Checking for raw counts...")
        query_path = prepare_raw_h5ad(args.query, tmp_dir, label="query")
        ref_path = prepare_raw_h5ad(args.reference, tmp_dir, label="reference")

        # Step 2: Build precomputed stats
        if os.path.exists(precomputed_stats_path) and not args.clobber:
            print(f"\n[2/4] Precomputed stats already exist — skipping.")
            print(f"      {precomputed_stats_path}")
            print(f"      (use --clobber to rebuild)")
        else:
            print(f"\n[2/4] Building precomputed stats from reference...")
            t0 = time.time()
            build_precomputed_stats(
                ref_path=ref_path,
                output_path=precomputed_stats_path,
                label_col=args.ref_label_col,
                n_processors=args.n_processors,
                normalization=args.normalization,
                clobber=args.clobber,
            )
            print(f"  Done in {time.time()-t0:.0f}s → {precomputed_stats_path}")

        # Step 3: Run mapping
        print(f"\n[3/4] Running hierarchical mapping...")
        t0 = time.time()
        csv_path, json_path = run_mapping(
            query_path=query_path,
            precomputed_stats_path=precomputed_stats_path,
            output_dir=args.output_dir,
            n_processors=args.n_processors,
            normalization=args.normalization,
        )
        print(f"  Done in {time.time()-t0:.0f}s → {csv_path}")

    finally:
        # Clean up temp files
        import shutil
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Step 4: Post-process
    print(f"\n[4/4] Post-processing results...")
    postprocess(
        csv_path=csv_path,
        reference_h5ad=args.reference,
        label_col=args.ref_label_col,
        cl_col=args.ref_cl_col,
        output_dir=args.output_dir,
    )

    # Save run config
    run_config = {
        "query": args.query,
        "reference": args.reference,
        "ref_label_col": args.ref_label_col,
        "ref_cl_col": args.ref_cl_col,
        "output_dir": args.output_dir,
        "n_processors": args.n_processors,
        "normalization": args.normalization,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = os.path.join(args.output_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nRun config → {config_path}")
    print(f"Total time: {(time.time()-t_total)/60:.1f} min")
    print("Done!")


if __name__ == "__main__":
    main()
