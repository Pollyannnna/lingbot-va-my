#!/usr/bin/env python3
"""Estimate RoboCasa action quantiles for LingBot-VA configs.

This script scans LeRobot parquet files and computes approximate q01/q99 stats
for RoboCasa manipulator actions, then expands them to a 30-dim format
compatible with LingBot-VA action channel mapping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_int_list(csv_value: str) -> list[int]:
    values = [v.strip() for v in csv_value.split(",") if v.strip()]
    if not values:
        raise ValueError("action index list is empty")

    parsed = [int(v) for v in values]
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"action indices must be unique, got: {parsed}")
    if min(parsed) < 0:
        raise ValueError(f"action indices must be non-negative, got: {parsed}")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute RoboCasa action q01/q99 from LeRobot parquet files.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root. Supports either a single lerobot dataset root, or a directory containing many task/*/lerobot datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--action-key",
        type=str,
        default="action",
        help="Parquet column name for action vectors.",
    )
    parser.add_argument(
        "--action-indices",
        type=str,
        default="5,6,7,8,9,10,11",
        help=(
            "Comma-separated raw RoboCasa action indices to normalize. "
            "Default selects manipulator channels: ee xyz, ee rot xyz, gripper."
        ),
    )
    parser.add_argument(
        "--target-action-dim",
        type=int,
        default=30,
        help="Expanded LingBot-VA action dim for output q01_30/q99_30.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=2000,
        help="Max sampled rows per parquet file (approximate quantiles, keeps memory bounded).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for per-file sampling.",
    )
    parser.add_argument(
        "--min-span",
        type=float,
        default=1e-3,
        help="Minimum allowed q99-q01 per selected action dim before falling back to [-1, 1].",
    )
    return parser.parse_args()


def _find_parquet_files(dataset_root: Path) -> list[Path]:
    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")

    single_dataset_pattern = dataset_root.glob("data/chunk-*/episode_*.parquet")
    single_dataset_files = sorted(single_dataset_pattern)
    if single_dataset_files:
        return single_dataset_files

    multi_dataset_files = sorted(
        dataset_root.glob("**/lerobot/data/chunk-*/episode_*.parquet")
    )
    if multi_dataset_files:
        return multi_dataset_files

    fallback_files = sorted(dataset_root.glob("**/data/chunk-*/episode_*.parquet"))
    return fallback_files


def _load_actions_from_parquet(
    parquet_path: Path,
    action_key: str,
) -> np.ndarray:
    df = pd.read_parquet(parquet_path, columns=[action_key])
    if action_key not in df.columns:
        raise KeyError(f"missing action key '{action_key}' in {parquet_path}")

    actions = np.asarray(
        [np.asarray(v, dtype=np.float32) for v in df[action_key].tolist()],
        dtype=np.float32,
    )
    if actions.ndim != 2:
        raise ValueError(
            f"unexpected action shape from {parquet_path}: {actions.shape}, expected [T, A]"
        )
    return actions


def main() -> int:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    action_indices = _parse_int_list(args.action_indices)
    used_action_dim = len(action_indices)

    parquet_files = _find_parquet_files(args.dataset_root)
    if not parquet_files:
        raise FileNotFoundError(
            f"no parquet files found under {args.dataset_root} "
            "(expected **/lerobot/data/chunk-*/episode_*.parquet)"
        )

    sampled_rows: list[np.ndarray] = []
    total_rows = 0
    valid_files = 0

    for p in parquet_files:
        try:
            actions = _load_actions_from_parquet(p, args.action_key)
        except Exception as exc:
            print(f"[WARN] skip {p}: {exc}")
            continue

        if actions.shape[1] <= max(action_indices):
            print(
                f"[WARN] skip {p}: action dim {actions.shape[1]} does not cover indices {action_indices}"
            )
            continue

        valid_files += 1
        total_rows += actions.shape[0]
        actions = actions[:, action_indices]

        if actions.shape[0] > args.max_rows_per_file:
            idx = rng.choice(actions.shape[0], size=args.max_rows_per_file, replace=False)
            actions = actions[idx]
        sampled_rows.append(actions)

    if not sampled_rows:
        raise RuntimeError("no valid action rows collected")

    stacked = np.concatenate(sampled_rows, axis=0)
    q01_7 = np.quantile(stacked, 0.01, axis=0).astype(np.float64).tolist()
    q99_7 = np.quantile(stacked, 0.99, axis=0).astype(np.float64).tolist()

    collapsed_dims = []
    for i, (q01, q99) in enumerate(zip(q01_7, q99_7)):
        if (q99 - q01) < args.min_span:
            collapsed_dims.append(i)
            q01_7[i] = -1.0
            q99_7[i] = 1.0

    if collapsed_dims:
        print(
            "[WARN] collapsed quantile span for selected dims "
            f"{collapsed_dims}; fallback to [-1, 1] for those dims."
        )

    tail_dim = max(args.target_action_dim - used_action_dim, 0)
    q01_30 = q01_7 + [0.0] * tail_dim
    q99_30 = q99_7 + [1.0] * tail_dim

    out = {
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "num_parquet_files_total": len(parquet_files),
        "num_parquet_files_used": valid_files,
        "num_rows_total": int(total_rows),
        "num_rows_sampled": int(stacked.shape[0]),
        "action_indices": action_indices,
        "used_action_dim": int(used_action_dim),
        "target_action_dim": int(args.target_action_dim),
        "q01_7": q01_7,
        "q99_7": q99_7,
        "q01_30": q01_30,
        "q99_30": q99_30,
    }

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"saved: {output_path}")
    print(f"files used: {valid_files}/{len(parquet_files)}")
    print(f"rows sampled: {stacked.shape[0]} (from total rows: {total_rows})")
    print(f"action_indices: {action_indices}")
    print(f"q01_7: {q01_7}")
    print(f"q99_7: {q99_7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
