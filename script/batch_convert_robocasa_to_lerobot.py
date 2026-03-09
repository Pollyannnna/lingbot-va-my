#!/usr/bin/env python3
"""Batch convert RoboCasa hdf5 datasets to LeRobot format with dedup + resume.

Key features:
- Recursively scan a root dir (e.g. /data/share/lijiang/data/robocasa0.2)
- Choose one preferred hdf5 per dataset folder (favor image-rich *_im*.hdf5)
- Skip already-converted datasets via marker + integrity checks
- Optional post-step: add action_config to meta/episodes.jsonl
- Write progress/status files for realtime monitoring
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py


IMAGE_KEYS = (
    "robot0_eye_in_hand_image",
    "robot0_agentview_left_image",
    "robot0_agentview_right_image",
)
MARKER_NAME = ".lerobot_convert_done.json"


@dataclass
class Task:
    source_hdf5: Path
    dataset_dir: Path
    lerobot_dir: Path
    marker_path: Path
    score: tuple[int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch convert RoboCasa hdf5 files to LeRobot format.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory to scan for *.hdf5 files.",
    )
    parser.add_argument(
        "--converter-python",
        type=Path,
        default=Path(sys.executable),
        help="Python binary used to run convert_hdf5_lerobot.py.",
    )
    parser.add_argument(
        "--converter-script",
        type=Path,
        required=True,
        help="Path to convert_hdf5_lerobot.py.",
    )
    parser.add_argument(
        "--add-action-config",
        action="store_true",
        help="Run add_action_config.py after each successful conversion.",
    )
    parser.add_argument(
        "--action-config-python",
        type=Path,
        default=Path(sys.executable),
        help="Python binary used for add_action_config.py.",
    )
    parser.add_argument(
        "--action-config-script",
        type=Path,
        default=None,
        help="Path to add_action_config.py (required with --add-action-config).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip datasets already converted and validated (default: true).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reconvert even if marker+validation pass.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned tasks, do not run conversion.",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=Path("./logs/robocasa_batch_convert.status.jsonl"),
        help="JSONL status output path.",
    )
    parser.add_argument(
        "--progress-json",
        type=Path,
        default=Path("./logs/robocasa_batch_convert.progress.json"),
        help="Progress json output path.",
    )
    parser.add_argument(
        "--task-log-dir",
        type=Path,
        default=Path("./logs/robocasa_batch_convert_tasks"),
        help="Per-task log directory.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue converting remaining tasks when one task fails (default: true).",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_status(status_file: Path, payload: dict[str, Any]) -> None:
    ensure_parent(status_file)
    with status_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_progress(progress_json: Path, payload: dict[str, Any]) -> None:
    ensure_parent(progress_json)
    progress_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_raw_stats(hdf5_path: Path) -> tuple[int, int]:
    """Return (num_episodes, total_frames)."""
    with h5py.File(hdf5_path, "r") as f:
        demos = list(f["data"].keys())
        total_frames = 0
        for d in demos:
            total_frames += len(f["data"][d]["actions"])
    return len(demos), total_frames


def has_image_obs(hdf5_path: Path) -> bool:
    try:
        with h5py.File(hdf5_path, "r") as f:
            data = f.get("data")
            if data is None or len(data.keys()) == 0:
                return False
            first_demo = data[list(data.keys())[0]]
            obs = first_demo.get("obs")
            if obs is None:
                return False
            return all(k in obs for k in IMAGE_KEYS)
    except Exception:
        return False


def select_score(hdf5_path: Path) -> tuple[int, str]:
    name = hdf5_path.name.lower()
    # Lower score = higher priority.
    if name == "demo_im128.hdf5":
        base = 0
    elif "im128" in name:
        base = 1
    elif "_im" in name:
        base = 2
    elif name == "demo.hdf5":
        base = 4
    else:
        base = 5
    if has_image_obs(hdf5_path):
        base -= 2
    return base, name


def scan_tasks(root: Path) -> list[Task]:
    files = sorted(root.rglob("*.hdf5"))
    by_dir: dict[Path, list[Path]] = {}
    for f in files:
        by_dir.setdefault(f.parent, []).append(f)

    tasks: list[Task] = []
    for dataset_dir, group in sorted(by_dir.items()):
        scored = sorted(((select_score(p), p) for p in group), key=lambda x: x[0])
        best_score, best_file = scored[0]
        tasks.append(
            Task(
                source_hdf5=best_file,
                dataset_dir=dataset_dir,
                lerobot_dir=dataset_dir / "lerobot",
                marker_path=dataset_dir / MARKER_NAME,
                score=best_score,
            )
        )
    return tasks


def validate_output(task: Task) -> tuple[bool, dict[str, Any]]:
    lerobot_dir = task.lerobot_dir
    meta_dir = lerobot_dir / "meta"
    data_dir = lerobot_dir / "data"
    videos_dir = lerobot_dir / "videos"

    required = [
        meta_dir / "info.json",
        meta_dir / "episodes.jsonl",
        meta_dir / "tasks.jsonl",
    ]
    for p in required:
        if not p.exists():
            return False, {"reason": f"missing_file:{p}"}

    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    video_files = sorted(videos_dir.glob("chunk-*/*/episode_*.mp4"))
    if not parquet_files:
        return False, {"reason": "no_parquet"}
    if not video_files:
        return False, {"reason": "no_videos"}

    raw_episodes, raw_total_frames = read_raw_stats(task.source_hdf5)
    ep_lines = [
        x
        for x in (meta_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

    if len(parquet_files) != raw_episodes:
        return (
            False,
            {
                "reason": "parquet_episode_mismatch",
                "raw_episodes": raw_episodes,
                "parquet_files": len(parquet_files),
            },
        )
    if len(ep_lines) != raw_episodes:
        return (
            False,
            {
                "reason": "episodes_jsonl_mismatch",
                "raw_episodes": raw_episodes,
                "episodes_jsonl": len(ep_lines),
            },
        )

    return (
        True,
        {
            "raw_episodes": raw_episodes,
            "raw_total_frames": raw_total_frames,
            "parquet_files": len(parquet_files),
            "video_files": len(video_files),
        },
    )


def marker_matches_source(marker_path: Path, source_hdf5: Path) -> bool:
    if not marker_path.exists():
        return False
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    stat = source_hdf5.stat()
    return (
        payload.get("source_hdf5") == str(source_hdf5)
        and payload.get("source_size") == stat.st_size
        and payload.get("source_mtime_ns") == stat.st_mtime_ns
    )


def build_converter_cmd(args: argparse.Namespace, source_hdf5: Path) -> list[str]:
    return [
        str(args.converter_python),
        str(args.converter_script),
        "--raw_dataset_path",
        str(source_hdf5),
    ]


def build_action_config_cmd(args: argparse.Namespace, lerobot_dir: Path) -> list[str]:
    assert args.action_config_script is not None
    return [
        str(args.action_config_python),
        str(args.action_config_script),
        "--dataset-root",
        str(lerobot_dir),
        "--normalize-style",
        "--write-episodes-ori",
    ]


def run_command(cmd: list[str], log_path: Path) -> tuple[int, float]:
    ensure_parent(log_path)
    start = time.time()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[CMD] {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode, time.time() - start


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    converter_script = args.converter_script.expanduser().resolve()
    converter_python = args.converter_python.expanduser().resolve()
    status_file = args.status_file.expanduser().resolve()
    progress_json = args.progress_json.expanduser().resolve()
    task_log_dir = args.task_log_dir.expanduser().resolve()

    if not root.exists():
        print(f"[ERROR] root not found: {root}")
        return 2
    if not converter_script.exists():
        print(f"[ERROR] converter script not found: {converter_script}")
        return 2
    if not converter_python.exists():
        print(f"[ERROR] converter python not found: {converter_python}")
        return 2
    if args.add_action_config:
        if args.action_config_script is None:
            print("[ERROR] --add-action-config requires --action-config-script")
            return 2
        ac_script = args.action_config_script.expanduser().resolve()
        ac_python = args.action_config_python.expanduser().resolve()
        if not ac_script.exists():
            print(f"[ERROR] action config script not found: {ac_script}")
            return 2
        if not ac_python.exists():
            print(f"[ERROR] action config python not found: {ac_python}")
            return 2
        args.action_config_script = ac_script
        args.action_config_python = ac_python

    args.root = root
    args.converter_script = converter_script
    args.converter_python = converter_python

    tasks = scan_tasks(root)
    print(f"[INFO] root={root}")
    print(f"[INFO] scanned_hdf5_task_dirs={len(tasks)}")

    progress: dict[str, Any] = {
        "root": str(root),
        "total": len(tasks),
        "started_at": int(time.time()),
        "processed": 0,
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "running_source": None,
        "last_event": None,
    }
    write_progress(progress_json, progress)

    append_status(
        status_file,
        {
            "ts": int(time.time()),
            "type": "scan_done",
            "root": str(root),
            "total_tasks": len(tasks),
            "dry_run": args.dry_run,
        },
    )

    if args.dry_run:
        for i, t in enumerate(tasks, start=1):
            print(
                f"[DRY-RUN] {i:04d}/{len(tasks):04d} "
                f"dir={t.dataset_dir} source={t.source_hdf5.name} score={t.score}"
            )
        return 0

    task_log_dir.mkdir(parents=True, exist_ok=True)
    any_fail = False

    for i, task in enumerate(tasks, start=1):
        progress["running_source"] = str(task.source_hdf5)
        progress["last_event"] = f"start:{task.source_hdf5.name}"
        write_progress(progress_json, progress)

        print(
            f"[TASK] {i:04d}/{len(tasks):04d} source={task.source_hdf5} "
            f"(score={task.score})"
        )
        append_status(
            status_file,
            {
                "ts": int(time.time()),
                "type": "task_start",
                "index": i,
                "total": len(tasks),
                "source_hdf5": str(task.source_hdf5),
                "dataset_dir": str(task.dataset_dir),
                "score": task.score,
            },
        )

        if args.skip_existing and not args.force:
            if marker_matches_source(task.marker_path, task.source_hdf5):
                ok, detail = validate_output(task)
                if ok:
                    progress["processed"] += 1
                    progress["skipped"] += 1
                    progress["last_event"] = f"skip:{task.source_hdf5.name}"
                    write_progress(progress_json, progress)
                    print(f"[SKIP] already converted and validated: {task.dataset_dir}")
                    append_status(
                        status_file,
                        {
                            "ts": int(time.time()),
                            "type": "task_skip",
                            "source_hdf5": str(task.source_hdf5),
                            "dataset_dir": str(task.dataset_dir),
                            "detail": detail,
                        },
                    )
                    continue

        log_name = (
            f"{time.strftime('%Y%m%d_%H%M%S')}"
            f"_{task.dataset_dir.name}_{task.source_hdf5.name}.log"
        )
        task_log_path = task_log_dir / log_name

        cmd = build_converter_cmd(args, task.source_hdf5)
        rc, elapsed = run_command(cmd, task_log_path)
        if rc != 0:
            any_fail = True
            progress["processed"] += 1
            progress["failed"] += 1
            progress["last_event"] = f"fail_convert:{task.source_hdf5.name}"
            write_progress(progress_json, progress)
            print(f"[FAIL] converter rc={rc} source={task.source_hdf5}")
            append_status(
                status_file,
                {
                    "ts": int(time.time()),
                    "type": "task_fail",
                    "stage": "convert",
                    "returncode": rc,
                    "elapsed_sec": round(elapsed, 3),
                    "source_hdf5": str(task.source_hdf5),
                    "dataset_dir": str(task.dataset_dir),
                    "task_log": str(task_log_path),
                },
            )
            if not args.continue_on_error:
                break
            continue

        if args.add_action_config:
            ac_cmd = build_action_config_cmd(args, task.lerobot_dir)
            rc_ac, elapsed_ac = run_command(ac_cmd, task_log_path)
            if rc_ac != 0:
                any_fail = True
                progress["processed"] += 1
                progress["failed"] += 1
                progress["last_event"] = f"fail_action_config:{task.source_hdf5.name}"
                write_progress(progress_json, progress)
                print(f"[FAIL] action_config rc={rc_ac} source={task.source_hdf5}")
                append_status(
                    status_file,
                    {
                        "ts": int(time.time()),
                        "type": "task_fail",
                        "stage": "action_config",
                        "returncode": rc_ac,
                        "elapsed_sec": round(elapsed_ac, 3),
                        "source_hdf5": str(task.source_hdf5),
                        "dataset_dir": str(task.dataset_dir),
                        "task_log": str(task_log_path),
                    },
                )
                if not args.continue_on_error:
                    break
                continue

        ok, detail = validate_output(task)
        if not ok:
            any_fail = True
            progress["processed"] += 1
            progress["failed"] += 1
            progress["last_event"] = f"fail_validate:{task.source_hdf5.name}"
            write_progress(progress_json, progress)
            print(f"[FAIL] validation failed source={task.source_hdf5} detail={detail}")
            append_status(
                status_file,
                {
                    "ts": int(time.time()),
                    "type": "task_fail",
                    "stage": "validate",
                    "source_hdf5": str(task.source_hdf5),
                    "dataset_dir": str(task.dataset_dir),
                    "detail": detail,
                    "task_log": str(task_log_path),
                },
            )
            if not args.continue_on_error:
                break
            continue

        stat = task.source_hdf5.stat()
        marker_payload = {
            "source_hdf5": str(task.source_hdf5),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "converted_at": int(time.time()),
            "detail": detail,
            "converter_script": str(args.converter_script),
        }
        task.marker_path.write_text(
            json.dumps(marker_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        progress["processed"] += 1
        progress["success"] += 1
        progress["last_event"] = f"success:{task.source_hdf5.name}"
        write_progress(progress_json, progress)
        print(f"[OK] {task.source_hdf5} -> {task.lerobot_dir}")
        append_status(
            status_file,
            {
                "ts": int(time.time()),
                "type": "task_success",
                "source_hdf5": str(task.source_hdf5),
                "dataset_dir": str(task.dataset_dir),
                "task_log": str(task_log_path),
                "detail": detail,
            },
        )

    progress["running_source"] = None
    progress["finished_at"] = int(time.time())
    progress["last_event"] = "finished"
    write_progress(progress_json, progress)

    append_status(
        status_file,
        {
            "ts": int(time.time()),
            "type": "finished",
            "summary": progress,
        },
    )

    print(
        "[SUMMARY] "
        f"total={progress['total']} processed={progress['processed']} "
        f"success={progress['success']} skipped={progress['skipped']} failed={progress['failed']}"
    )
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

