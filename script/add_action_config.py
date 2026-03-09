#!/usr/bin/env python3
"""Add action_config to LeRobot episodes.jsonl files for LingBotVA training.

This script is idempotent:
- Existing valid action_config entries are kept unchanged by default.
- Missing action_config entries are filled with one full-episode segment.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FileStats:
    path: Path
    total_lines: int = 0
    added_lines: int = 0
    style_fixed_lines: int = 0
    invalid_lines: int = 0
    changed: bool = False
    created_ori: bool = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill missing action_config in LeRobot episodes.jsonl files.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory to recursively search for meta/episodes.jsonl.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify format; do not modify files.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing action_config with one full-episode segment.",
    )
    parser.add_argument(
        "--backup-suffix",
        type=str,
        default=".bak",
        help="Backup suffix for changed files (ignored by --dry-run).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code if any invalid line is found.",
    )
    parser.add_argument(
        "--skill",
        type=str,
        default="",
        help='Default skill string to write in action_config segments (RobotWin style uses "").',
    )
    parser.add_argument(
        "--normalize-style",
        action="store_true",
        help="Normalize existing valid action_config entries by filling missing action_text/skill fields.",
    )
    parser.add_argument(
        "--no-normalize-style",
        action="store_false",
        dest="normalize_style",
        help="Disable style normalization for existing valid action_config.",
    )
    parser.add_argument(
        "--write-episodes-ori",
        action="store_true",
        help="When writing changes, create sibling episodes_ori.jsonl if absent, preserving original lines.",
    )
    parser.add_argument(
        "--no-write-episodes-ori",
        action="store_false",
        dest="write_episodes_ori",
        help="Do not create episodes_ori.jsonl.",
    )
    parser.add_argument(
        "--episodes-ori-name",
        type=str,
        default="episodes_ori.jsonl",
        help="Filename used for preserved original metadata.",
    )
    parser.add_argument(
        "--max-error-print",
        type=int,
        default=50,
        help="Max number of per-line error messages to print per file.",
    )
    parser.set_defaults(normalize_style=True, write_episodes_ori=True)
    return parser.parse_args()


def _safe_action_text(record: dict[str, Any]) -> str:
    tasks = record.get("tasks")
    if isinstance(tasks, list) and tasks:
        first = tasks[0]
        return first if isinstance(first, str) else str(first)
    if isinstance(tasks, str):
        return tasks
    return ""


def _valid_action_config(record: dict[str, Any]) -> bool:
    if "action_config" not in record:
        return False
    cfg = record["action_config"]
    if not isinstance(cfg, list) or len(cfg) == 0:
        return False
    length = record.get("length")
    if not isinstance(length, int) or length <= 0:
        return False
    for seg in cfg:
        if not isinstance(seg, dict):
            return False
        if "start_frame" not in seg or "end_frame" not in seg:
            return False
        start_frame = seg["start_frame"]
        end_frame = seg["end_frame"]
        if not isinstance(start_frame, int) or not isinstance(end_frame, int):
            return False
        if not (0 <= start_frame < end_frame <= length):
            return False
    return True


def _default_action_config(record: dict[str, Any]) -> list[dict[str, Any]]:
    length = int(record["length"])
    return [
        {
            "start_frame": 0,
            "end_frame": length,
            "action_text": _safe_action_text(record),
            "skill": "",
        }
    ]


def _normalize_style(record: dict[str, Any], default_skill: str) -> bool:
    cfg = record.get("action_config")
    if not isinstance(cfg, list):
        return False
    changed = False
    default_text = _safe_action_text(record)
    for seg in cfg:
        if not isinstance(seg, dict):
            continue
        if not isinstance(seg.get("action_text"), str):
            seg["action_text"] = default_text
            changed = True
        if "skill" not in seg:
            seg["skill"] = default_skill
            changed = True
    return changed


def _process_file(path: Path, args: argparse.Namespace) -> FileStats:
    stats = FileStats(path=path)
    printed_errors = 0
    raw_lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    stats.total_lines = len(raw_lines)
    if stats.total_lines == 0:
        return stats

    out_lines: list[str] = []
    any_change = False

    for idx, line in enumerate(raw_lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            stats.invalid_lines += 1
            out_lines.append(line)
            if printed_errors < args.max_error_print:
                print(f"[INVALID JSON] {path}:{idx}")
                printed_errors += 1
            continue

        if not isinstance(record, dict):
            stats.invalid_lines += 1
            out_lines.append(line)
            if printed_errors < args.max_error_print:
                print(f"[INVALID RECORD] {path}:{idx} is not a JSON object")
                printed_errors += 1
            continue

        length = record.get("length")
        if not isinstance(length, int) or length <= 0:
            stats.invalid_lines += 1
            out_lines.append(line)
            if printed_errors < args.max_error_print:
                print(f"[INVALID LENGTH] {path}:{idx} length={length!r}")
                printed_errors += 1
            continue

        has_valid = _valid_action_config(record)

        if args.verify_only:
            if not has_valid:
                stats.invalid_lines += 1
                if printed_errors < args.max_error_print:
                    print(f"[MISSING/INVALID action_config] {path}:{idx}")
                    printed_errors += 1
            out_lines.append(json.dumps(record, ensure_ascii=False))
            continue

        if args.overwrite_existing or not has_valid:
            record["action_config"] = _default_action_config(record)
            if args.skill != "":
                record["action_config"][0]["skill"] = args.skill
            stats.added_lines += 1
            any_change = True
        elif args.normalize_style:
            if _normalize_style(record, args.skill):
                stats.style_fixed_lines += 1
                any_change = True

        out_lines.append(json.dumps(record, ensure_ascii=False))

    stats.changed = any_change

    if args.verify_only:
        return stats

    if args.dry_run or not any_change:
        return stats

    if args.write_episodes_ori:
        ori_path = path.with_name(args.episodes_ori_name)
        if not ori_path.exists():
            ori_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
            stats.created_ori = True

    backup_path = path.with_suffix(path.suffix + args.backup_suffix)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return stats


def main() -> int:
    args = _parse_args()
    root = args.dataset_root.expanduser().resolve()
    if not root.exists():
        print(f"dataset root not found: {root}")
        return 2

    files = sorted(root.rglob("meta/episodes.jsonl"))
    if not files:
        print(f"no meta/episodes.jsonl found under: {root}")
        return 2

    total_lines = 0
    total_added = 0
    total_style_fixed = 0
    total_invalid = 0
    changed_files = 0
    created_ori_files = 0

    for path in files:
        stats = _process_file(path, args)
        total_lines += stats.total_lines
        total_added += stats.added_lines
        total_style_fixed += stats.style_fixed_lines
        total_invalid += stats.invalid_lines
        changed_files += int(stats.changed)
        created_ori_files += int(stats.created_ori)
        print(
            f"[FILE] {path} | lines={stats.total_lines} added={stats.added_lines} style_fixed={stats.style_fixed_lines} "
            f"invalid={stats.invalid_lines} changed={stats.changed} created_ori={stats.created_ori}"
        )

    mode = "verify-only" if args.verify_only else ("dry-run" if args.dry_run else "write")
    print(
        f"[SUMMARY] mode={mode} files={len(files)} changed_files={changed_files} created_ori_files={created_ori_files} "
        f"lines={total_lines} added={total_added} style_fixed={total_style_fixed} invalid={total_invalid}"
    )

    if args.strict and total_invalid > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
