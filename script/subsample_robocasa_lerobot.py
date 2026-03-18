#!/usr/bin/env python3
"""Create a balanced RoboCasa LeRobot subset for LingBot-VA training.

This script scans a directory containing many `*/lerobot` datasets, keeps a
fixed number of episodes per dataset, and writes a clean subset with episode
indices renumbered from zero. Episode parquet files and videos are hardlinked
by default to avoid duplicating storage.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.datasets.compute_stats import aggregate_stats


STATIC_META_FILES = (
    "tasks.jsonl",
    "embodiment.json",
    "modality.json",
)
DATASET_MAPPING_NAME = "source_episode_mapping.jsonl"
ROOT_MANIFEST_NAME = "selected_episode_manifest.jsonl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subsample RoboCasa LeRobot datasets with balanced task coverage.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Root directory containing many */lerobot datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root for the subsampled datasets.",
    )
    parser.add_argument(
        "--episodes-per-dataset",
        type=int,
        default=50,
        help="Target number of episodes to keep for each lerobot dataset.",
    )
    parser.add_argument(
        "--link-mode",
        type=str,
        default="hardlink",
        choices=("hardlink", "symlink", "copy"),
        help="How to materialize selected parquet/video files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output dataset root if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned subsets without writing files.",
    )
    return parser.parse_args()


def _find_lerobot_roots(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if (root / "meta" / "info.json").exists() and (root / "videos").exists():
        return [root]

    roots: list[Path] = []
    for info_path in sorted(root.glob("**/meta/info.json")):
        dataset_root = info_path.parent.parent
        if (dataset_root / "videos").exists() and (dataset_root / "data").exists():
            roots.append(dataset_root)
    return roots


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=4), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _primary_task(record: dict[str, Any]) -> str:
    tasks = record.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    if isinstance(tasks, str):
        return tasks
    return ""


def _allocate_evenly(groups: OrderedDict[str, list[dict[str, Any]]], target: int) -> dict[str, int]:
    target = min(target, sum(len(items) for items in groups.values()))
    alloc = {key: 0 for key in groups}
    keys = list(groups.keys())

    if target <= len(keys):
        for i in range(target):
            pos = int((i + 0.5) * len(keys) / target)
            if pos >= len(keys):
                pos = len(keys) - 1
            alloc[keys[pos]] = 1
        return alloc

    remaining = target

    while remaining > 0:
        progressed = False
        for key, items in groups.items():
            if alloc[key] >= len(items):
                continue
            alloc[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break

    return alloc


def _choose_evenly(items: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)

    n = len(items)
    selected = []
    for i in range(k):
        pos = int((i + 0.5) * n / k)
        if pos >= n:
            pos = n - 1
        selected.append(items[pos])
    return selected


def _select_records(records: list[dict[str, Any]], target: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for record in records:
        key = _primary_task(record)
        grouped.setdefault(key, []).append(record)

    alloc = _allocate_evenly(grouped, target)
    selected: list[dict[str, Any]] = []
    for key, items in grouped.items():
        selected.extend(_choose_evenly(items, alloc[key]))

    selected.sort(key=lambda row: int(row["episode_index"]))
    return selected, alloc


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    shutil.copy2(src, dst)


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _stats_to_numpy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _stats_to_numpy(v) for k, v in value.items()}
    if isinstance(value, list):
        return np.asarray(value)
    return value


def _copy_static_meta(src_meta: Path, dst_meta: Path) -> None:
    for name in STATIC_META_FILES:
        src = src_meta / name
        if src.exists():
            dst = dst_meta / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _build_subset_dataset(
    src_root: Path,
    dst_root: Path,
    episodes_per_dataset: int,
    link_mode: str,
    dry_run: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    src_meta = src_root / "meta"
    dst_meta = dst_root / "meta"
    src_info = _read_json(src_meta / "info.json")
    src_episodes = _read_jsonl(src_meta / "episodes.jsonl")
    src_episode_stats = _read_jsonl(src_meta / "episodes_stats.jsonl")
    episode_stats_by_index = {
        int(row["episode_index"]): row["stats"] for row in src_episode_stats
    }

    selected_records, alloc = _select_records(src_episodes, episodes_per_dataset)
    video_keys = [
        key for key, feature in src_info["features"].items() if feature.get("dtype") == "video"
    ]
    chunks_size = int(src_info.get("chunks_size", 1000))

    summary = {
        "source_root": str(src_root),
        "output_root": str(dst_root),
        "source_episodes": len(src_episodes),
        "selected_episodes": len(selected_records),
        "selection_by_task": {k: v for k, v in alloc.items() if v > 0},
        "selected_source_episode_indices": [
            int(record["episode_index"]) for record in selected_records
        ],
        "mapping_file": str(dst_meta / DATASET_MAPPING_NAME),
    }
    if dry_run:
        manifest_rows = []
        for new_index, record in enumerate(selected_records):
            old_index = int(record["episode_index"])
            old_chunk = old_index // chunks_size
            new_chunk = new_index // chunks_size
            manifest_rows.append(
                {
                    "source_dataset_root": str(src_root),
                    "output_dataset_root": str(dst_root),
                    "source_episode_index": old_index,
                    "output_episode_index": new_index,
                    "primary_task": _primary_task(record),
                    "tasks": copy.deepcopy(record.get("tasks", [])),
                    "length": int(record["length"]),
                    "source_parquet": str(
                        src_root / "data" / f"chunk-{old_chunk:03d}" / f"episode_{old_index:06d}.parquet"
                    ),
                    "output_parquet": str(
                        dst_root / "data" / f"chunk-{new_chunk:03d}" / f"episode_{new_index:06d}.parquet"
                    ),
                }
            )
        return summary, manifest_rows

    dst_root.mkdir(parents=True, exist_ok=True)
    dst_meta.mkdir(parents=True, exist_ok=True)
    _copy_static_meta(src_meta, dst_meta)

    selected_episode_rows: list[dict[str, Any]] = []
    selected_episode_stats_rows: list[dict[str, Any]] = []
    dataset_manifest_rows: list[dict[str, Any]] = []
    total_frames = 0

    for new_index, record in enumerate(selected_records):
        old_index = int(record["episode_index"])
        old_chunk = old_index // chunks_size
        new_chunk = new_index // chunks_size

        new_record = copy.deepcopy(record)
        new_record["episode_index"] = new_index
        selected_episode_rows.append(new_record)
        total_frames += int(new_record["length"])

        episode_stats = copy.deepcopy(episode_stats_by_index[old_index])
        selected_episode_stats_rows.append(
            {
                "episode_index": new_index,
                "stats": episode_stats,
            }
        )

        mapping_row = {
            "source_dataset_root": str(src_root),
            "output_dataset_root": str(dst_root),
            "source_episode_index": old_index,
            "output_episode_index": new_index,
            "source_chunk_index": old_chunk,
            "output_chunk_index": new_chunk,
            "primary_task": _primary_task(record),
            "tasks": copy.deepcopy(record.get("tasks", [])),
            "length": int(record["length"]),
            "source_episode_record": copy.deepcopy(record),
            "output_episode_record": copy.deepcopy(new_record),
            "source_parquet": str(
                src_root / "data" / f"chunk-{old_chunk:03d}" / f"episode_{old_index:06d}.parquet"
            ),
            "output_parquet": str(
                dst_root / "data" / f"chunk-{new_chunk:03d}" / f"episode_{new_index:06d}.parquet"
            ),
            "source_videos": {},
            "output_videos": {},
        }

        src_parquet = src_root / "data" / f"chunk-{old_chunk:03d}" / f"episode_{old_index:06d}.parquet"
        dst_parquet = dst_root / "data" / f"chunk-{new_chunk:03d}" / f"episode_{new_index:06d}.parquet"
        _link_or_copy(src_parquet, dst_parquet, link_mode)

        for video_key in video_keys:
            src_video = (
                src_root
                / "videos"
                / f"chunk-{old_chunk:03d}"
                / video_key
                / f"episode_{old_index:06d}.mp4"
            )
            dst_video = (
                dst_root
                / "videos"
                / f"chunk-{new_chunk:03d}"
                / video_key
                / f"episode_{new_index:06d}.mp4"
            )
            mapping_row["source_videos"][video_key] = str(src_video)
            mapping_row["output_videos"][video_key] = str(dst_video)
            _link_or_copy(src_video, dst_video, link_mode)

        dataset_manifest_rows.append(mapping_row)

    subset_info = copy.deepcopy(src_info)
    subset_info["total_episodes"] = len(selected_episode_rows)
    subset_info["total_frames"] = total_frames
    subset_info["total_videos"] = len(selected_episode_rows) * len(video_keys)
    subset_info["total_chunks"] = math.ceil(len(selected_episode_rows) / chunks_size) if selected_episode_rows else 0
    subset_info["splits"] = {"train": f"0:{len(selected_episode_rows)}"}

    aggregated_stats = aggregate_stats(
        [_stats_to_numpy(row["stats"]) for row in selected_episode_stats_rows]
    )

    _write_json(dst_meta / "info.json", subset_info)
    _write_jsonl(dst_meta / "episodes.jsonl", selected_episode_rows)
    _write_jsonl(dst_meta / "episodes_stats.jsonl", selected_episode_stats_rows)
    _write_json(dst_meta / "stats.json", _jsonify(aggregated_stats))
    _write_jsonl(dst_meta / DATASET_MAPPING_NAME, dataset_manifest_rows)

    return summary, dataset_manifest_rows


def main() -> int:
    args = _parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"source root does not exist: {source_root}")

    lerobot_roots = _find_lerobot_roots(source_root)
    if not lerobot_roots:
        raise FileNotFoundError(f"no lerobot datasets found under {source_root}")

    if output_root.exists():
        has_existing = any(output_root.iterdir())
        if has_existing and not args.overwrite:
            raise FileExistsError(
                f"output root is not empty: {output_root}. Use --overwrite to recreate it."
            )
        if has_existing and args.overwrite:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    root_manifest_rows: list[dict[str, Any]] = []
    for src_root in lerobot_roots:
        rel = src_root.relative_to(source_root)
        dst_root = output_root / rel
        summary, dataset_manifest_rows = _build_subset_dataset(
            src_root=src_root,
            dst_root=dst_root,
            episodes_per_dataset=args.episodes_per_dataset,
            link_mode=args.link_mode,
            dry_run=args.dry_run,
        )
        summaries.append(summary)
        root_manifest_rows.extend(dataset_manifest_rows)
        print(
            "[DATASET]",
            f"src={src_root}",
            f"dst={dst_root}",
            f"selected={summary['selected_episodes']}",
            f"source={summary['source_episodes']}",
            f"alloc={summary['selection_by_task']}",
        )

    summary_path = output_root / "subset_summary.json"
    manifest_path = output_root / ROOT_MANIFEST_NAME
    summary_payload = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "episodes_per_dataset": args.episodes_per_dataset,
        "link_mode": args.link_mode,
        "dataset_count": len(summaries),
        "root_manifest_file": str(manifest_path),
        "datasets": summaries,
    }
    _write_json(summary_path, summary_payload)
    _write_jsonl(manifest_path, root_manifest_rows)

    total_selected = sum(row["selected_episodes"] for row in summaries)
    print(
        "[SUMMARY]",
        f"datasets={len(summaries)}",
        f"selected_episodes={total_selected}",
        f"summary={summary_path}",
        f"manifest={manifest_path}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
