#!/usr/bin/env python3
"""RoboCasa benchmark client for LingBot-VA websocket inference server.

This script follows the RoboCasa environment setup and success metric logic used in
`other/cosmos-policy/cosmos_policy/experiments/robot/robocasa/run_robocasa_eval.py`,
while reusing LingBot-VA's websocket client protocol.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import imageio
import numpy as np
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROBOCASA_REPO_ROOT = WORKSPACE_ROOT / "robocasa"
DEFAULT_ROBOSUITE_REPO_ROOT = WORKSPACE_ROOT / "robosuite"

for repo_root in (
    str(WORKSPACE_ROOT),
    str(DEFAULT_ROBOCASA_REPO_ROOT),
    str(DEFAULT_ROBOSUITE_REPO_ROOT),
):
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import robosuite
from robocasa.utils import dataset_registry as robocasa_dataset_registry

from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy

DEFAULT_CONTROLLER_CONFIG_PATH = str(
    WORKSPACE_ROOT
    / "other/cosmos-policy/cosmos_policy/experiments/robot/robocasa/robocasa_controller_configs.pkl"
)

LEGACY_TASK_MAX_STEPS = {
    "PnPCounterToCab": 500,
    "PnPCabToCounter": 500,
    "PnPCounterToSink": 700,
    "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600,
    "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500,
    "PnPStoveToCounter": 500,
    "OpenSingleDoor": 500,
    "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000,
    "CloseDoubleDoor": 700,
    "OpenDrawer": 500,
    "CloseDrawer": 500,
    "TurnOnStove": 500,
    "TurnOffStove": 500,
    "TurnOnSinkFaucet": 500,
    "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500,
    "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600,
    "CoffeePressButton": 300,
    "TurnOnMicrowave": 500,
    "TurnOffMicrowave": 500,
}


def _build_default_task_registry() -> Dict[str, Dict[str, Any]]:
    task_registry: Dict[str, Dict[str, Any]] = {}

    atomic_tasks = getattr(robocasa_dataset_registry, "ATOMIC_TASK_DATASETS", {})
    composite_tasks = getattr(robocasa_dataset_registry, "COMPOSITE_TASK_DATASETS", {})
    legacy_single_stage_tasks = getattr(
        robocasa_dataset_registry, "SINGLE_STAGE_TASK_DATASETS", {}
    )
    legacy_multi_stage_tasks = getattr(
        robocasa_dataset_registry, "MULTI_STAGE_TASK_DATASETS", {}
    )

    if atomic_tasks or composite_tasks:
        task_registry.update(atomic_tasks)
        task_registry.update(composite_tasks)
    else:
        task_registry.update(legacy_single_stage_tasks)
        task_registry.update(legacy_multi_stage_tasks)

    return task_registry


DEFAULT_TASK_REGISTRY = _build_default_task_registry()
TASK_SET_REGISTRY: Mapping[str, Sequence[str]] = getattr(
    robocasa_dataset_registry,
    "TASK_SET_REGISTRY",
    {},
)


def str2bool(value: str) -> bool:
    value_lower = value.lower()
    if value_lower in {"true", "1", "yes", "y"}:
        return True
    if value_lower in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from '{value}'")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def infer_tasks_from_dataset_root(dataset_root: str) -> List[str]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"--dataset-root does not exist: {root}")

    task_names: List[str] = []
    for task_group in ("atomic", "composite"):
        task_dir = root / task_group
        if not task_dir.is_dir():
            continue
        task_names.extend(sorted(path.name for path in task_dir.iterdir() if path.is_dir()))

    if not task_names:
        raise ValueError(
            f"--dataset-root does not contain atomic/composite task folders: {root}"
        )
    return task_names


def resolve_available_tasks(dataset_root: str, task_set: str) -> List[str]:
    dataset_root = dataset_root.strip()
    task_set = task_set.strip()

    if dataset_root:
        return infer_tasks_from_dataset_root(dataset_root)

    if task_set:
        if task_set not in TASK_SET_REGISTRY:
            raise ValueError(
                f"Unknown --task-set '{task_set}'. Available task sets: {sorted(TASK_SET_REGISTRY)}"
            )
        return list(TASK_SET_REGISTRY[task_set])

    return sorted(DEFAULT_TASK_REGISTRY.keys())


def parse_task_names(task_name_arg: str, available_tasks: Sequence[str]) -> List[str]:
    available_task_set = set(available_tasks)
    if task_name_arg.strip().lower() == "all":
        return list(available_tasks)

    task_names = [item.strip() for item in task_name_arg.split(",") if item.strip()]
    invalid = [name for name in task_names if name not in available_task_set]
    if invalid:
        raise ValueError(
            "Invalid task name(s): "
            f"{invalid}. Available tasks: {list(available_tasks)}"
        )
    return task_names


def normalize_obj_instance_split(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None

    value = raw_value.strip()
    if not value:
        return None

    normalized = value.lower()
    split_aliases: Dict[str, str | None] = {
        "a": "pretrain",
        "pretrain": "pretrain",
        "train": "pretrain",
        "b": "target",
        "target": "target",
        "test": "target",
        "none": None,
        "all": None,
    }
    if normalized in split_aliases:
        return split_aliases[normalized]

    raise ValueError(
        "--obj-instance-split must be one of "
        "{target, pretrain, none} or legacy aliases {A, B, train, test, all}."
    )


def get_task_horizon(task_name: str) -> int:
    task_cfg = DEFAULT_TASK_REGISTRY.get(task_name, {})
    horizon = task_cfg.get("horizon")
    if horizon is not None:
        return int(horizon)
    return LEGACY_TASK_MAX_STEPS.get(task_name, 500)


def load_controller_configs(path: str, controller_name: str, robots: str) -> Tuple[Any, str]:
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f), path

    try:
        from robosuite.controllers import load_composite_controller_config

        controller_configs = load_composite_controller_config(controller="BASIC", robot=robots)
        return controller_configs, "robosuite.controllers.load_composite_controller_config"
    except Exception as exc:  # pragma: no cover - env-dependent
        raise FileNotFoundError(
            "Cannot load RoboCasa controller configs. Either provide a valid "
            f"--controller-config-path (current: {path}) or ensure robosuite controller loading works."
        ) from exc


def select_layout_and_style_ids(raw_ids: str, episode_idx: int) -> Tuple[Tuple[int, int], ...] | None:
    if not raw_ids:
        return None

    all_ids = ast.literal_eval(raw_ids)
    if not all_ids:
        return None

    scene_index = (episode_idx // 10) % len(all_ids)
    return (all_ids[scene_index],)


def create_robocasa_env(
    *,
    task_name: str,
    env_img_res: int,
    robots: str,
    obj_instance_split: str | None,
    randomize_cameras: bool,
    layout_and_style_ids: str,
    seed: int | None,
    episode_idx: int,
    controller_configs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    scene_ids = select_layout_and_style_ids(layout_and_style_ids, episode_idx)
    env_kwargs = {
        "env_name": task_name,
        "robots": robots,
        "controller_configs": controller_configs,
        "camera_names": ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        "camera_widths": env_img_res,
        "camera_heights": env_img_res,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": True,
        "camera_depths": False,
        "seed": seed,
        "obj_instance_split": obj_instance_split,
        "generative_textures": None,
        "randomize_cameras": randomize_cameras,
        "layout_and_style_ids": scene_ids,
        "translucent_robot": False,
    }
    env = robosuite.make(**env_kwargs)
    return env, env_kwargs


def _safe_uint8_image(img: np.ndarray, flip_images: bool) -> np.ndarray:
    arr = np.asarray(img)
    if flip_images:
        arr = np.flipud(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def build_obs_payload(obs: Dict[str, Any], prompt: str, flip_images: bool) -> Dict[str, Any]:
    left_img = _safe_uint8_image(obs["robot0_agentview_left_image"], flip_images)
    right_img = _safe_uint8_image(obs["robot0_agentview_right_image"], flip_images)
    wrist_img = _safe_uint8_image(obs["robot0_eye_in_hand_image"], flip_images)

    proprio = np.concatenate(
        [
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1),
        ],
        axis=0,
    )

    # Include multiple aliases so the payload can work with different server obs_cam_keys configs.
    return {
        "observation.images.cam_high": left_img,
        "observation.images.cam_left_wrist": right_img,
        "observation.images.cam_right_wrist": wrist_img,
        "observation.images.primary": left_img,
        "observation.images.secondary": right_img,
        "observation.images.wrist": wrist_img,
        "observation.images.robot0_agentview_left": left_img,
        "observation.images.robot0_agentview_right": right_img,
        "observation.images.robot0_eye_in_hand": wrist_img,
        "observation.state": proprio,
        "task": prompt,
    }


class ActionMapper:
    def __init__(
        self,
        mode: str,
        env_action_dim: int,
        fixed_base_motion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        fixed_control_mode: float = -1.0,
    ):
        self.mode = mode
        self.env_action_dim = env_action_dim
        self.fixed_base_motion = np.asarray(fixed_base_motion, dtype=np.float32)
        self.fixed_control_mode = float(fixed_control_mode)
        self._warned: set[str] = set()

    def _warn_once(self, key: str, msg: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        print(f"[WARN] {msg}")

    def _first7_to_robocasa12(self, action: np.ndarray) -> np.ndarray:
        # RoboCasa / robosuite runtime env action layout for PandaOmron:
        #   0:3 ee_position, 3:6 ee_rotation, 6:7 gripper_close,
        #   7:11 base_motion, 11:12 control_mode.
        #
        # Note that this differs from the LeRobot dataset storage layout in
        # meta/modality.json, which stores base / control first. The env.step
        # call expects the runtime order here.
        out = np.zeros((12,), dtype=np.float32)
        out[0:3] = action[0:3]
        out[3:6] = action[3:6]
        out[6] = action[6]
        out[7:11] = self.fixed_base_motion
        out[11] = self.fixed_control_mode
        return out

    def map(self, raw_action: np.ndarray) -> np.ndarray:
        action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        dim = action.shape[0]

        if self.mode == "identity":
            if dim != self.env_action_dim:
                raise ValueError(
                    f"identity mode requires action dim={self.env_action_dim}, but got {dim}."
                )
            return action

        if self.mode == "first12":
            if dim < 12:
                raise ValueError(f"first12 mode requires at least 12 dims, but got {dim}.")
            out = np.zeros((self.env_action_dim,), dtype=np.float32)
            used = min(self.env_action_dim, 12)
            out[:used] = action[:used]
            return out

        if self.mode == "first7":
            if self.env_action_dim != 12:
                raise ValueError("first7 mode currently only supports env_action_dim=12")
            if dim < 7:
                raise ValueError(f"first7 mode requires at least 7 dims, but got {dim}.")
            return self._first7_to_robocasa12(action)

        if self.env_action_dim == 12:
            if dim == 12:
                return action[:12]
            if dim >= 7:
                if dim in {14, 16, 30}:
                    self._warn_once(
                        f"auto_{dim}",
                        f"Server returned {dim}-dim action. Auto mode will map first 7 dims to RoboCasa runtime action "
                        f"[ee_pos, ee_rot, gripper] and append fixed base/control {self.fixed_base_motion.tolist() + [self.fixed_control_mode]}.",
                    )
                elif dim not in {7, 8}:
                    self._warn_once(
                        f"auto_other_{dim}",
                        "Server returned an unexpected action dim. Auto mode falls back to RoboCasa runtime first7 mapping.",
                    )
                return self._first7_to_robocasa12(action)

            self._warn_once(
                "auto_short",
                "Server returned fewer than 7 dims. Auto mode will pad RoboCasa runtime action and keep fixed base/control defaults.",
            )
            out = np.zeros((12,), dtype=np.float32)
            used = min(dim, 7)
            out[:used] = action[:used]
            out[7:11] = self.fixed_base_motion
            out[11] = self.fixed_control_mode
            return out

        out = np.zeros((self.env_action_dim,), dtype=np.float32)
        used = min(dim, self.env_action_dim)
        out[:used] = action[:used]
        if dim != self.env_action_dim:
            self._warn_once(
                f"auto_pad_{dim}_{self.env_action_dim}",
                f"Server returned {dim}-dim action for env dim {self.env_action_dim}. Auto mode will truncate/pad.",
            )
        return out


def resize_image_like(img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if img.shape[0] == target_h and img.shape[1] == target_w:
        return img
    return np.asarray(Image.fromarray(img).resize((target_w, target_h), Image.BILINEAR))


def compose_rollout_frame(obs_payload: Dict[str, Any]) -> np.ndarray:
    left = obs_payload["observation.images.cam_high"]
    right = obs_payload["observation.images.cam_left_wrist"]
    wrist = obs_payload["observation.images.cam_right_wrist"]

    target_hw = (left.shape[0], left.shape[1])
    right = resize_image_like(right, target_hw)
    wrist = resize_image_like(wrist, target_hw)

    return np.concatenate([left, right, wrist], axis=1)


def save_rollout_video(frames: Sequence[np.ndarray], path: Path, fps: int = 10) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, list(frames), fps=fps)


def ensure_uint8_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    arr = arr.astype(np.float32)
    if arr.max() <= 1.0001 and arr.min() >= 0.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _split_triptych_frame(img_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = img_frame.shape[:2]
    single_w = w // 3
    left = img_frame[:, 0:single_w]
    right = img_frame[:, single_w : single_w * 2]
    wrist = img_frame[:, single_w * 2 : single_w * 3]
    return left, right, wrist


def add_title_bar(img: np.ndarray, text: str, bar_height: int = 36) -> np.ndarray:
    h, w = img.shape[:2]
    title_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    cv2 = __import__("cv2")
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    text_x = max((w - text_w) // 2, 4)
    text_y = (bar_height + text_h) // 2 - 4
    cv2.putText(
        title_bar,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([title_bar, img])


def save_imagined_stream_video(
    imagined_video_chunks: Sequence[np.ndarray],
    path: Path,
    fps: int = 10,
) -> None:
    if not imagined_video_chunks:
        return

    frames: List[np.ndarray] = []
    for chunk in imagined_video_chunks:
        chunk_arr = np.asarray(chunk)
        if chunk_arr.ndim != 4:
            continue
        for frame in chunk_arr:
            frames.append(ensure_uint8_frame(frame))
    save_rollout_video(frames, path, fps=fps)


def save_comparison_video(
    real_obs_payloads: Sequence[Dict[str, Any]],
    imagined_video_chunks: Sequence[np.ndarray],
    chunk_obs_counts: Sequence[int],
    path: Path,
    fps: int = 10,
) -> None:
    if not real_obs_payloads or not imagined_video_chunks:
        return

    frame_map: List[Tuple[int, int]] = []
    for chunk_id, real_count in enumerate(chunk_obs_counts):
        imagined_chunk = np.asarray(imagined_video_chunks[chunk_id]) if chunk_id < len(imagined_video_chunks) else None
        imagined_count = int(imagined_chunk.shape[0]) if imagined_chunk is not None and imagined_chunk.ndim == 4 else 0
        for local_idx in range(real_count):
            if imagined_count > 0:
                imagined_idx = min(int(local_idx / max(real_count, 1) * imagined_count), imagined_count - 1)
                frame_map.append((chunk_id, imagined_idx))
            else:
                frame_map.append((chunk_id, -1))

    comparison_frames: List[np.ndarray] = []
    for real_idx, obs_payload in enumerate(real_obs_payloads):
        left = obs_payload["observation.images.robot0_agentview_left"]
        right = obs_payload["observation.images.robot0_agentview_right"]
        wrist = obs_payload["observation.images.robot0_eye_in_hand"]
        real_row = np.concatenate([left, right, wrist], axis=1)
        real_row = add_title_bar(real_row, f"Real Observation [{real_idx + 1}/{len(real_obs_payloads)}]")

        imagined_row: np.ndarray
        chunk_id, frame_id = frame_map[real_idx] if real_idx < len(frame_map) else (-1, -1)
        if chunk_id >= 0 and frame_id >= 0 and chunk_id < len(imagined_video_chunks):
            frame = ensure_uint8_frame(imagined_video_chunks[chunk_id][frame_id])
            im_left, im_right, im_wrist = _split_triptych_frame(frame)
            imagined_row = np.concatenate([im_left, im_right, im_wrist], axis=1)
            imagined_row = add_title_bar(imagined_row, f"Imagined Video Stream [chunk {chunk_id}]")
        else:
            imagined_row = np.zeros_like(real_row)
            imagined_row = add_title_bar(imagined_row, "Imagined Video Stream [unavailable]")

        width = max(real_row.shape[1], imagined_row.shape[1])
        if real_row.shape[1] != width:
            cv2 = __import__("cv2")
            real_row = cv2.resize(real_row, (width, real_row.shape[0]))
        if imagined_row.shape[1] != width:
            cv2 = __import__("cv2")
            imagined_row = cv2.resize(imagined_row, (width, imagined_row.shape[0]))
        comparison_frames.append(np.vstack([real_row, imagined_row]))

    save_rollout_video(comparison_frames, path, fps=fps)


@dataclass
class EpisodeResult:
    success: bool
    length: int
    task_description: str
    video_path: str | None
    imagined_video_path: str | None = None
    comparison_video_path: str | None = None


def _round_float(value: float) -> float:
    return round(float(value), 4)


def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def _flatten_episode_rows(run_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task_result in run_state.get("results", []):
        for episode in task_result.get("episode_results", []):
            row = {
                "task_name": task_result.get("task_name"),
                "num_trials": task_result.get("num_trials"),
                "completed_trials": task_result.get("completed_trials", 0),
                "episode_idx": episode.get("episode_idx"),
                "seed": episode.get("seed"),
                "success": episode.get("success"),
                "length": episode.get("length"),
                "task_description": episode.get("task_description"),
                "video_path": episode.get("video_path"),
                "imagined_video_path": episode.get("imagined_video_path"),
                "comparison_video_path": episode.get("comparison_video_path"),
            }
            rows.append(row)
    return rows


def _write_episode_csv(rows: Sequence[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_name",
        "num_trials",
        "completed_trials",
        "episode_idx",
        "seed",
        "success",
        "length",
        "task_description",
        "video_path",
        "imagined_video_path",
        "comparison_video_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_episode_jsonl(rows: Sequence[Dict[str, Any]], jsonl_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(jsonl_path)


def _refresh_task_metrics(task_result: Dict[str, Any]) -> None:
    episodes = task_result.get("episode_results", [])
    successes = [bool(ep.get("success")) for ep in episodes]
    lengths = [float(ep.get("length", 0.0)) for ep in episodes]
    completed_trials = len(episodes)
    num_successes = int(sum(successes))

    task_result["completed_trials"] = completed_trials
    task_result["num_successes"] = num_successes
    task_result["success_rate"] = (
        _round_float(num_successes / completed_trials) if completed_trials else 0.0
    )
    task_result["avg_episode_length"] = (
        _round_float(sum(lengths) / completed_trials) if completed_trials else 0.0
    )


def _refresh_run_metrics(run_state: Dict[str, Any]) -> None:
    task_results = run_state.get("results", [])
    completed_tasks = 0
    completed_episodes = 0
    success_rates: List[float] = []
    total_successes = 0
    total_trials = 0

    for task_result in task_results:
        _refresh_task_metrics(task_result)
        completed_episodes += int(task_result.get("completed_trials", 0))
        total_successes += int(task_result.get("num_successes", 0))
        total_trials += int(task_result.get("num_trials", 0))
        if task_result.get("completed_trials", 0) >= task_result.get("num_trials", 0):
            completed_tasks += 1
        if task_result.get("completed_trials", 0) > 0:
            success_rates.append(float(task_result.get("success_rate", 0.0)))

    run_state["completed_tasks"] = completed_tasks
    run_state["completed_episodes"] = completed_episodes
    run_state["total_successes"] = total_successes
    run_state["total_trials"] = total_trials
    run_state["final_success_rate"] = (
        _round_float(sum(success_rates) / len(success_rates)) if success_rates else 0.0
    )


class IncrementalRunRecorder:
    def __init__(self, save_dir: Path, run_state: Dict[str, Any]):
        self.save_dir = save_dir
        self.summary_dir = save_dir / "summary"
        self.run_state = run_state
        timestamp = run_state["timestamp"]
        self.summary_path = self.summary_dir / f"robocasa_eval_{timestamp}.json"
        self.latest_path = self.summary_dir / "robocasa_eval_latest.json"
        self.csv_path = self.summary_dir / f"robocasa_episode_summary_{timestamp}.csv"
        self.jsonl_path = self.summary_dir / f"robocasa_episode_results_{timestamp}.jsonl"

    def get_or_create_task_result(self, task_name: str, num_trials: int) -> Dict[str, Any]:
        for task_result in self.run_state["results"]:
            if task_result["task_name"] == task_name:
                if task_result.get("num_trials") != num_trials:
                    task_result["num_trials"] = num_trials
                return task_result

        task_result = {
            "task_name": task_name,
            "num_trials": num_trials,
            "completed_trials": 0,
            "success_rate": 0.0,
            "avg_episode_length": 0.0,
            "num_successes": 0,
            "episode_results": [],
        }
        self.run_state["results"].append(task_result)
        return task_result

    def save(self, status: str = "running", error: str | None = None) -> None:
        self.run_state["status"] = status
        self.run_state["updated_at"] = datetime.now().isoformat()
        if error is not None:
            self.run_state["error"] = error
        _refresh_run_metrics(self.run_state)
        rows = _flatten_episode_rows(self.run_state)
        _safe_write_json(self.summary_path, self.run_state)
        _safe_write_json(self.latest_path, self.run_state)
        _write_episode_csv(rows, self.csv_path)
        _write_episode_jsonl(rows, self.jsonl_path)


def run_episode(
    *,
    model: WebsocketClientPolicy,
    env: Any,
    task_name: str,
    task_description: str,
    action_mapper: ActionMapper,
    flip_images: bool,
    max_steps: int,
    save_visualization: bool,
    video_guidance_scale: float,
    action_guidance_scale: float,
    num_steps_wait: int,
    save_rollout_video_flag: bool,
    rollout_video_path: Path,
    imagined_video_path: Path,
    comparison_video_path: Path,
) -> EpisodeResult:
    model.infer({
        "reset": True,
        "prompt": task_description,
        "save_visualization": save_visualization,
    })

    obs = None
    for _ in range(num_steps_wait):
        dummy_action = np.zeros(env.action_spec[0].shape, dtype=np.float32)
        obs, _, _, _ = env.step(dummy_action)
    if obs is None:
        # num_steps_wait can be 0; fall back to a single no-op step to get initial observation.
        dummy_action = np.zeros(env.action_spec[0].shape, dtype=np.float32)
        obs, _, _, _ = env.step(dummy_action)

    first_obs_payload = build_obs_payload(obs, task_description, flip_images)

    success = False
    steps = 0
    first_chunk = True

    rollout_frames: List[np.ndarray] = []
    real_obs_payloads: List[Dict[str, Any]] = [first_obs_payload]
    imagined_video_chunks: List[np.ndarray] = []
    chunk_obs_counts: List[int] = []
    if save_rollout_video_flag:
        rollout_frames.append(compose_rollout_frame(first_obs_payload))

    while steps < max_steps:
        ret = model.infer(
            {
                "obs": first_obs_payload,
                "prompt": task_description,
                "save_visualization": save_visualization,
                "video_guidance_scale": video_guidance_scale,
                "action_guidance_scale": action_guidance_scale,
            }
        )

        if "action" not in ret:
            raise RuntimeError(f"Server response has no 'action' field: {ret.keys()}")
        if save_visualization and "video" in ret:
            imagined_video_chunks.append(np.asarray(ret["video"]))

        action_chunk = np.asarray(ret["action"], dtype=np.float32)
        if action_chunk.ndim != 3:
            raise RuntimeError(f"Expected action chunk shape [C, F, A], got {action_chunk.shape}")

        if action_chunk.shape[2] % 4 != 0:
            raise RuntimeError(
                "Expected action chunk last dim divisible by 4 "
                f"(for key-frame KV cache update), got {action_chunk.shape}"
            )

        key_frame_interval = action_chunk.shape[2] // 4
        start_frame_idx = 1 if (first_chunk and action_chunk.shape[1] > 1) else 0
        key_frame_payloads: List[Dict[str, Any]] = []
        chunk_real_obs_count = 1 if first_chunk else 0

        for frame_idx in range(start_frame_idx, action_chunk.shape[1]):
            for action_idx in range(action_chunk.shape[2]):
                raw_action_step = action_chunk[:, frame_idx, action_idx].flatten()
                env_action = action_mapper.map(raw_action_step)
                obs, _, _, _ = env.step(env_action)
                steps += 1

                if (action_idx + 1) % key_frame_interval == 0:
                    payload = build_obs_payload(obs, task_description, flip_images)
                    key_frame_payloads.append(payload)
                    real_obs_payloads.append(payload)
                    chunk_real_obs_count += 1
                    if save_rollout_video_flag:
                        rollout_frames.append(compose_rollout_frame(payload))

                if env._check_success():
                    success = True
                    break
                if steps >= max_steps:
                    break

            if success or steps >= max_steps:
                break

        first_chunk = False
        if save_visualization:
            chunk_obs_counts.append(chunk_real_obs_count)

        if success or steps >= max_steps:
            break

        if key_frame_payloads:
            model.infer(
                {
                    "obs": key_frame_payloads,
                    "compute_kv_cache": True,
                    "imagine": False,
                    "save_visualization": save_visualization,
                    "state": action_chunk,
                }
            )

    video_path = None
    imagined_path = None
    comparison_path = None
    if save_rollout_video_flag:
        save_rollout_video(rollout_frames, rollout_video_path, fps=10)
        video_path = str(rollout_video_path)
    if save_visualization and imagined_video_chunks:
        save_imagined_stream_video(imagined_video_chunks, imagined_video_path, fps=10)
        save_comparison_video(real_obs_payloads, imagined_video_chunks, chunk_obs_counts, comparison_video_path, fps=10)
        imagined_path = str(imagined_video_path)
        comparison_path = str(comparison_video_path)

    return EpisodeResult(
        success=success,
        length=steps,
        task_description=task_description,
        video_path=video_path,
        imagined_video_path=imagined_path,
        comparison_video_path=comparison_path,
    )


def evaluate_task(
    *,
    model: WebsocketClientPolicy,
    task_name: str,
    num_trials_per_task: int,
    env_img_res: int,
    robots: str,
    obj_instance_split: str | None,
    randomize_cameras: bool,
    layout_and_style_ids: str,
    deterministic: bool,
    deterministic_reset: bool,
    deterministic_reset_seed: int | None,
    seed: int,
    seed_multiplier: int,
    controller_configs: Any,
    flip_images: bool,
    max_steps_override: int | None,
    save_visualization: bool,
    video_guidance_scale: float,
    action_guidance_scale: float,
    num_steps_wait: int,
    save_rollout_video_flag: bool,
    save_dir: Path,
    action_map_mode: str,
    fixed_base_motion: tuple[float, float, float, float],
    fixed_control_mode: float,
    recorder: IncrementalRunRecorder | None = None,
) -> Dict[str, Any]:
    print(f"\n[Task] {task_name}")
    if recorder is not None:
        result = recorder.get_or_create_task_result(task_name, num_trials_per_task)
    else:
        result = {
            "task_name": task_name,
            "num_trials": num_trials_per_task,
            "completed_trials": 0,
            "success_rate": 0.0,
            "avg_episode_length": 0.0,
            "num_successes": 0,
            "episode_results": [],
        }

    for episode_idx in range(num_trials_per_task):
        env_seed = seed * episode_idx * seed_multiplier if deterministic else None
        env, _ = create_robocasa_env(
            task_name=task_name,
            env_img_res=env_img_res,
            robots=robots,
            obj_instance_split=obj_instance_split,
            randomize_cameras=randomize_cameras,
            layout_and_style_ids=layout_and_style_ids,
            seed=env_seed,
            episode_idx=episode_idx,
            controller_configs=controller_configs,
        )

        try:
            if deterministic_reset:
                reset_seed = deterministic_reset_seed if deterministic_reset_seed is not None else seed
                set_seed(reset_seed)

            env.reset()
            task_description = env.get_ep_meta()["lang"]

            max_steps = max_steps_override if max_steps_override is not None else get_task_horizon(task_name)
            action_mapper = ActionMapper(
                mode=action_map_mode,
                env_action_dim=env.action_dim,
                fixed_base_motion=fixed_base_motion,
                fixed_control_mode=fixed_control_mode,
            )

            video_path = save_dir / "videos" / "rollout" / task_name / f"episode_{episode_idx:03d}.mp4"
            imagined_video_path = save_dir / "videos" / "imagined" / task_name / f"episode_{episode_idx:03d}.mp4"
            comparison_video_path = save_dir / "videos" / "comparison" / task_name / f"episode_{episode_idx:03d}.mp4"
            ep_result = run_episode(
                model=model,
                env=env,
                task_name=task_name,
                task_description=task_description,
                action_mapper=action_mapper,
                flip_images=flip_images,
                max_steps=max_steps,
                save_visualization=save_visualization,
                video_guidance_scale=video_guidance_scale,
                action_guidance_scale=action_guidance_scale,
                num_steps_wait=num_steps_wait,
                save_rollout_video_flag=save_rollout_video_flag,
                rollout_video_path=video_path,
                imagined_video_path=imagined_video_path,
                comparison_video_path=comparison_video_path,
            )

            result["episode_results"].append(
                {
                    "episode_idx": episode_idx,
                    "seed": env_seed,
                    "success": ep_result.success,
                    "length": ep_result.length,
                    "task_description": ep_result.task_description,
                    "video_path": ep_result.video_path,
                    "imagined_video_path": ep_result.imagined_video_path,
                    "comparison_video_path": ep_result.comparison_video_path,
                }
            )

            print(
                f"  Episode {episode_idx + 1}/{num_trials_per_task} | "
                f"{'SUCCESS' if ep_result.success else 'FAIL'} | steps={ep_result.length}"
            )
            _refresh_task_metrics(result)
            if recorder is not None:
                recorder.save(status="running")

        finally:
            env.close()

    _refresh_task_metrics(result)
    print(
        f"[Task Done] {task_name} | success={result['num_successes']}/{num_trials_per_task} "
        f"({result['success_rate'] * 100:.1f}%), avg_len={result['avg_episode_length']:.1f}"
    )
    if recorder is not None:
        recorder.save(status="running")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LingBot-VA websocket policy on RoboCasa benchmark.")

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29056)
    parser.add_argument("--api-key", type=str, default=None)

    parser.add_argument(
        "--task-name",
        type=str,
        default="all",
        help="Task name, comma-separated list, or 'all'. Resolved from --dataset-root / --task-set / registry.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.getenv("ROBOCASA_EVAL_DATASET_ROOT", "").strip(),
        help="Optional RoboCasa dataset root with atomic/composite task folders. If set, 'all' runs exactly this subset.",
    )
    parser.add_argument(
        "--task-set",
        type=str,
        default="",
        help="Optional RoboCasa task set from dataset_registry.TASK_SET_REGISTRY, used when --dataset-root is unset.",
    )
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--seed-multiplier", type=int, default=256)
    parser.add_argument("--deterministic", type=str2bool, default=True)
    parser.add_argument("--deterministic-reset", type=str2bool, default=False)
    parser.add_argument("--deterministic-reset-seed", type=int, default=None)

    parser.add_argument("--env-img-res", type=int, default=224)
    parser.add_argument("--robots", type=str, default="PandaMobile")
    parser.add_argument(
        "--obj-instance-split",
        type=str,
        default="target",
        help="RoboCasa object split. New repo uses {target, pretrain, none}. Legacy aliases {A, B, train, test, all} are accepted.",
    )
    parser.add_argument("--layout-and-style-ids", type=str, default="((1,1),(2,2),(4,4),(6,9),(7,10))")
    parser.add_argument("--randomize-cameras", action="store_true")
    parser.add_argument("--flip-images", type=str2bool, default=True)
    parser.add_argument("--controller-config-path", type=str, default=DEFAULT_CONTROLLER_CONFIG_PATH)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)

    parser.add_argument("--save-dir", type=str, default="results/robocasa")
    parser.add_argument("--save-rollout-video", action="store_true")
    parser.add_argument("--save-visualization", action="store_true", help="Ask server to decode and return imagined video.")

    parser.add_argument("--video-guidance-scale", type=float, default=5.0)
    parser.add_argument("--action-guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--action-map-mode",
        type=str,
        default="auto",
        choices=["auto", "first7", "first12", "identity"],
        help="How to map model output action to env action.",
    )
    parser.add_argument(
        "--fixed-base-motion",
        type=str,
        default="0,0,0,0",
        help="Comma-separated 4D base_motion used when mapping 7D arm outputs into RoboCasa 12D env action.",
    )
    parser.add_argument(
        "--fixed-control-mode",
        type=float,
        default=-1.0,
        help="Fixed control_mode used when mapping 7D arm outputs into RoboCasa 12D env action.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available_tasks = resolve_available_tasks(args.dataset_root, args.task_set)
    task_names = parse_task_names(args.task_name, available_tasks)
    obj_instance_split = normalize_obj_instance_split(args.obj_instance_split)
    fixed_base_motion = tuple(float(v.strip()) for v in args.fixed_base_motion.split(",") if v.strip())
    if len(fixed_base_motion) != 4:
        raise ValueError(
            f"--fixed-base-motion expects 4 comma-separated values, got: {args.fixed_base_motion}"
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    controller_configs, controller_source = load_controller_configs(
        args.controller_config_path,
        "OSC_POSE",
        args.robots,
    )
    print(f"Controller config source: {controller_source}")

    model = WebsocketClientPolicy(host=args.host, port=args.port, api_key=args.api_key)
    metadata = model.get_server_metadata()
    if metadata:
        print(f"Server metadata: {metadata}")

    print(
        f"Evaluating tasks={task_names}, trials/task={args.num_trials_per_task}, "
        f"host={args.host}:{args.port}, dataset_root={args.dataset_root or '-'}, "
        f"task_set={args.task_set or '-'}, obj_instance_split={obj_instance_split}"
    )

    all_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "host": args.host,
        "port": args.port,
        "dataset_root": args.dataset_root,
        "task_set": args.task_set,
        "task_names": task_names,
        "available_tasks": list(available_tasks),
        "num_trials_per_task": args.num_trials_per_task,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "deterministic_reset": args.deterministic_reset,
        "obj_instance_split": obj_instance_split,
        "save_rollout_video": bool(args.save_rollout_video),
        "save_visualization": bool(args.save_visualization),
        "server_metadata": metadata,
        "controller_source": controller_source,
        "results": [],
    }
    recorder = IncrementalRunRecorder(save_dir, all_results)
    recorder.save(status="initialized")

    run_error: str | None = None
    try:
        for task_name in task_names:
            evaluate_task(
                model=model,
                task_name=task_name,
                num_trials_per_task=args.num_trials_per_task,
                env_img_res=args.env_img_res,
                robots=args.robots,
                obj_instance_split=obj_instance_split,
                randomize_cameras=args.randomize_cameras,
                layout_and_style_ids=args.layout_and_style_ids,
                deterministic=args.deterministic,
                deterministic_reset=args.deterministic_reset,
                deterministic_reset_seed=args.deterministic_reset_seed,
                seed=args.seed,
                seed_multiplier=args.seed_multiplier,
                controller_configs=controller_configs,
                flip_images=args.flip_images,
                max_steps_override=args.max_steps if args.max_steps is not None else get_task_horizon(task_name),
                save_visualization=args.save_visualization,
                video_guidance_scale=args.video_guidance_scale,
                action_guidance_scale=args.action_guidance_scale,
                num_steps_wait=args.num_steps_wait,
                save_rollout_video_flag=args.save_rollout_video,
                save_dir=save_dir,
                action_map_mode=args.action_map_mode,
                fixed_base_motion=fixed_base_motion,
                fixed_control_mode=args.fixed_control_mode,
                recorder=recorder,
            )
    except KeyboardInterrupt:
        run_error = "KeyboardInterrupt"
        recorder.save(status="interrupted", error=run_error)
        raise
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        recorder.save(status="error", error=run_error)
        raise
    else:
        recorder.save(status="completed")

    print("\n========== Final Summary ==========")
    for result in all_results["results"]:
        print(
            f"{result['task_name']}: {result['num_successes']}/{result['num_trials']} "
            f"({result['success_rate'] * 100:.1f}%), avg_len={result['avg_episode_length']:.1f}"
        )
    print(f"Average success rate across tasks: {all_results['final_success_rate'] * 100:.2f}%")
    print(f"Saved summary to: {recorder.summary_path}")
    print(f"Saved latest summary to: {recorder.latest_path}")
    print(f"Saved episode csv to: {recorder.csv_path}")
    print(f"Saved episode jsonl to: {recorder.jsonl_path}")


if __name__ == "__main__":
    main()
