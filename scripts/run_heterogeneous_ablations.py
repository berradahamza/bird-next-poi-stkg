"""Five structural ablations, identical splits/seeds/protocol, validation only."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ("poi_only", "poi_room", "poi_room_spatial", "poi_subject", "poi_room_subject")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument("--split-seed", type=int, default=2022)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--history", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--gnn-layers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--only", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/heterogeneous_ablations")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--spatial-json", type=Path, default=ROOT / "data/raw/bird_adjacency.json")
    args = parser.parse_args()
    if args.gnn_layers == 1 and any(v in args.only for v in ("poi_room_spatial", "poi_room_subject")):
        parser.error("Spatial variants require two layers for ROOM -> ROOM -> POI propagation")
    # Absolute paths survive subprocess cwd changes, including paths with spaces.
    args.output_root = args.output_root.resolve()
    args.data_dir = args.data_dir.resolve()
    args.spatial_json = args.spatial_json.resolve()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.only)) != len(args.only):
        parser.error("Seeds and variants must be unique")
    runs = [(v, seed, args.output_root / f"{v}_split{args.split_seed}_init{seed}") for v in args.only for seed in args.seeds]
    if any((run / "config.json").exists() for _, _, run in runs):
        parser.error("A planned run already exists; choose a new --output-root")
    rows, reference = [], None
    for variant, seed, run in runs:
        command = [sys.executable, "-m", "stkg_cll_poi", "train", "--variant", variant,
                   "--seed", str(seed), "--split-seed", str(args.split_seed), "--epochs", str(args.epochs),
                   "--hidden", str(args.hidden), "--history", str(args.history), "--gnn-layers", str(args.gnn_layers),
                   "--output", str(run), "--data-dir", str(args.data_dir), "--spatial-json", str(args.spatial_json), "--bootstrap", "0"]
        subprocess.run(command, cwd=ROOT, check=True)
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        split = json.loads((run / "split.json").read_text(encoding="utf-8"))
        graph = json.loads((run / "graph_train.json").read_text(encoding="utf-8"))
        # Compare actual split, transition statistics, normalizer and input hashes.
        invariants = {"split": split, "inputs": config["processed_input_sha256"],
                      "spatial": config["spatial_input_sha256"], "examples": metrics["n_examples"],
                      "transitions": graph["transition_edges"],
                      "normalizer": json.loads((run / "feature_normalizer.json").read_text(encoding="utf-8"))}
        if reference is None:
            reference = invariants
        if invariants != reference:
            raise AssertionError("Ablation protocol or train-only statistics drifted")
        row = {"variant": variant, "split_seed": args.split_seed, "init_seed": seed,
               "selected_epoch": metrics["selected_epoch"]}
        for scope, key in [("micro", "micro"), ("macro", "macro_by_visitor")]:
            for metric in ["acc1", "hit5", "hit10", "mrr"]:
                row[f"validation_{scope}_{metric}"] = metrics["validation"][key][metric]
        rows.append(row)
        with (args.output_root / "validation_matrix.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
    print("Validation matrix:", args.output_root / "validation_matrix.csv")


if __name__ == "__main__":
    main()
