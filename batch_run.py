#!/usr/bin/env python3
"""
Batch runner: predict (+ optionally evaluate) over a set of query/reference pairs.

Config file format (YAML):

  defaults:                           # applied to every run unless overridden
    method: enrichment_weighted_knn
    k: 30
    obo: /path/to/cl.obo
    ontology_method: ic
    output_dir: results/               # top-level base directory

  runs:
    - name: predict_only              # output goes to results/predict_only/
      index:      /path/to/index.faiss
      adata:      /path/to/query.h5ad
      ref_annot:  /path/to/ref_annot.tsv

    - name: predict_and_eval          # output goes to results/predict_and_eval/
      index:        /path/to/index.faiss
      npy:          /path/to/query.npy     # alternative to adata
      obs:          /path/to/query_obs.tsv
      ref_annot:    /path/to/ref_annot.tsv
      ground_truth: /path/to/ground_truth.tsv   # presence triggers evaluation
      pred_id_col:  weighted_cell_type_ontology_term_id   # required per run
      truth_id_col: mapped_cell_label_ontology_term_id    # required per run
      method: distance_weighted_knn             # per-run override
      subdir: custom_subdir_name        # optional: override subdir (default: name)

Usage:
  python3 batch_run.py config.yaml
  python3 batch_run.py config.yaml --dry-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


PREDICT_SCRIPT  = Path(__file__).parent / "cell_labelling" / "predict.py"
EVALUATE_SCRIPT = Path(__file__).parent / "evaluate.py"

PREDICT_DEFAULTS = {
    "method": "enrichment_weighted_knn",
    "k": 30,
}

EVALUATE_DEFAULTS = {
    "ontology_method": "ic",
}


def merge(defaults: dict, run: dict) -> dict:
    merged = {**defaults}
    merged.update(run)
    return merged


def build_predict_cmd(cfg: dict, labels_path: str) -> list:
    cmd = [sys.executable, str(PREDICT_SCRIPT)]
    cmd += ["--index",    str(cfg["index"])]
    if "npy" in cfg:
        cmd += ["--npy", str(cfg["npy"]), "--obs", str(cfg["obs"])]
    else:
        cmd += ["--adata", str(cfg["adata"])]
    cmd += ["--ref_annot", str(cfg["ref_annot"])]
    cmd += ["--method",    str(cfg.get("method", PREDICT_DEFAULTS["method"]))]
    cmd += ["--k",         str(cfg.get("k",      PREDICT_DEFAULTS["k"]))]
    cmd += ["--output",    labels_path]
    return cmd


def build_evaluate_cmd(cfg: dict, labels_path: str, eval_dir: str) -> list:
    cmd = [sys.executable, str(EVALUATE_SCRIPT)]
    cmd += ["--predictions",     labels_path]
    cmd += ["--ground_truth",    str(cfg["ground_truth"])]
    cmd += ["--obo",             str(cfg["obo"])]
    cmd += ["--ontology-method", str(cfg.get("ontology_method", EVALUATE_DEFAULTS["ontology_method"]))]
    cmd += ["--pred_id_col",     str(cfg["pred_id_col"])]
    cmd += ["--truth_id_col",    str(cfg["truth_id_col"])]
    cmd += ["--output-dir",      eval_dir]
    return cmd


def run_cmd(cmd: list, log_path: str, dry_run: bool) -> bool:
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        return True
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(proc.stdout)
        sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        print(f"  ERROR (exit {proc.returncode}) — see {log_path}")
    return proc.returncode == 0


def validate_run_cfg(cfg: dict, name: str) -> list:
    errors = []
    if "index" not in cfg:
        errors.append("missing: index")
    if "adata" not in cfg and "npy" not in cfg:
        errors.append("missing: adata or npy")
    if "npy" in cfg and "obs" not in cfg:
        errors.append("missing: obs (required with npy)")
    if "ref_annot" not in cfg:
        errors.append("missing: ref_annot")
    if "ground_truth" in cfg:
        if "obo" not in cfg:
            errors.append("missing: obo (required when ground_truth is set)")
        if "pred_id_col" not in cfg:
            errors.append("missing: pred_id_col (required when ground_truth is set)")
        if "truth_id_col" not in cfg:
            errors.append("missing: truth_id_col (required when ground_truth is set)")
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Batch predict + evaluate over query/reference pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config",     help="YAML config file")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print commands without running them")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    global_defaults = {**PREDICT_DEFAULTS, **EVALUATE_DEFAULTS}
    global_defaults.update(config.get("defaults", {}))
    runs = config.get("runs", [])

    if not runs:
        print("No runs defined in config.")
        sys.exit(1)

    if "output_dir" not in global_defaults:
        print("Config error: output_dir must be set in defaults")
        sys.exit(1)

    # validate all runs upfront before starting any work
    all_valid = True
    for i, run in enumerate(runs):
        name = run.get("name", f"run_{i+1}")
        cfg  = merge(global_defaults, run)
        errors = validate_run_cfg(cfg, name)
        if errors:
            print(f"Config error in '{name}': {'; '.join(errors)}")
            all_valid = False
    if not all_valid:
        sys.exit(1)

    results = []

    for i, run in enumerate(runs):
        name       = run.get("name", f"run_{i+1}")
        cfg        = merge(global_defaults, run)
        base_dir   = Path(cfg.get("output_dir", "results"))
        subdir     = run.get("subdir", name)
        output_dir = base_dir / subdir
        labels_path = str(output_dir / "labels.tsv")
        eval_dir    = str(output_dir / "eval")

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(runs)}] {name}")
        print(f"  output_dir: {output_dir}")
        print(f"{'='*60}")

        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

        # predict
        print("\n[predict]")
        predict_ok = run_cmd(
            build_predict_cmd(cfg, labels_path),
            str(output_dir / "predict.log"),
            args.dry_run,
        )

        # evaluate — only if ground_truth is specified
        eval_ok = None
        if "ground_truth" not in cfg:
            print("\n[evaluate] skipped (no ground_truth in config)")
        elif not predict_ok:
            print("\n[evaluate] skipped (predict failed)")
            eval_ok = False
        else:
            print("\n[evaluate]")
            eval_ok = run_cmd(
                build_evaluate_cmd(cfg, labels_path, eval_dir),
                str(output_dir / "evaluate.log"),
                args.dry_run,
            )

        results.append((name, predict_ok, eval_ok))

    # summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for name, predict_ok, eval_ok in results:
        pred_str = "ok"     if predict_ok else "FAILED"
        if eval_ok is None:
            eval_str = "skipped"
        else:
            eval_str = "ok" if eval_ok else "FAILED"
        row_ok = predict_ok and (eval_ok is not False)
        status = "OK" if row_ok else "FAILED"
        print(f"  {name:30s}  predict={pred_str:<6}  eval={eval_str:<7}  [{status}]")
        if not row_ok:
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
