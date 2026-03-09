#!/usr/bin/env python3
"""RoboCasa benchmark client for LingBot-VA websocket inference server.

This script follows the RoboCasa environment setup and success metric logic used in
`other/cosmos-policy/cosmos_policy/experiments/robot/robocasa/run_robocasa_eval.py`,
while reusing LingBot-VA's websocket client protocol.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import imageio
import numpy as np
import robosuite
from PIL import Image
from robocasa.utils.dataset_registry import MULTI_STAGE_TASK_DATASETS, SINGLE_STAGE_TASK_DATASETS

from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy

DEFAULT_CONTROLLER_CONFIG_PATH = (
    "/data/250010187/yeziyang1/other/cosmos-policy/cosmos_policy/experiments/robot/robocasa/"
    "robocasa_controller_configs.pkl"
)

TASK_MAX_STEPS = {
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

ALL_TASKS = {**SINGLE_STAGE_TASK_DATASETS, **MULTI_STAGE_TASK_DATASETS}


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


def parse_task_names(task_name_arg: str) -> List[str]:
    if task_name_arg.strip().lower() == "all":
        return sorted(ALL_TASKS.keys())

    task_names = [item.strip() for item in task_name_arg.split(",") if item.strip()]
    invalid = [name for name in task_names if name not in ALL_TASKS]
    if invalid:
        raise ValueError(
            "Invalid task name(s): "
            f"{invalid}. Available tasks: {sorted(ALL_TASKS.keys())}"
        )
    return task_names


def load_controller_configs(path: str, controller_name: str) -> Tuple[Any, str]:
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f), path

    try:
        from robosuite.controllers import load_controller_config

        controller_configs = load_controller_config(default_controller=controller_name)
        return controller_configs, "robosuite.controllers.load_controller_config"
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
    obj_instance_split: str,
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
    def __init__(self, mode: str, env_action_dim: int):
        self.mode = mode
        self.env_action_dim = env_action_dim
        self._warned: set[str] = set()

    def _warn_once(self, key: str, msg: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        print(f"[WARN] {msg}")

    def _first7_to_robocasa12(self, action: np.ndarray) -> np.ndarray:
        out = np.zeros((12,), dtype=np.float32)
        out[:7] = action[:7]
        out[7:11] = 0.0
        out[11] = -1.0
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
                        f"Server returned {dim}-dim action. Auto mode will use first 7 dims + fixed mobile base [0,0,0,0,-1].",
                    )
                elif dim not in {7, 8}:
                    self._warn_once(
                        f"auto_other_{dim}",
                        f"Server returned {dim}-dim action. Auto mode falls back to first 7 dims + fixed mobile base.",
                    )
                return self._first7_to_robocasa12(action)

            self._warn_once(
                "auto_short",
                f"Server returned only {dim} dims. Auto mode will pad to 12 dims and set last dim=-1.",
            )
            out = np.zeros((12,), dtype=np.float32)
            out[:dim] = action
            out[11] = -1.0
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


@dataclass
class EpisodeResult:
    success: bool
    length: int
    task_description: str
    video_path: str | None


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

        for frame_idx in range(start_frame_idx, action_chunk.shape[1]):
            for action_idx in range(action_chunk.shape[2]):
                raw_action_step = action_chunk[:, frame_idx, action_idx].flatten()
                env_action = action_mapper.map(raw_action_step)
                obs, _, _, _ = env.step(env_action)
                steps += 1

                if (action_idx + 1) % key_frame_interval == 0:
                    payload = build_obs_payload(obs, task_description, flip_images)
                    key_frame_payloads.append(payload)
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
    if save_rollout_video_flag:
        save_rollout_video(rollout_frames, rollout_video_path, fps=10)
        video_path = str(rollout_video_path)

    return EpisodeResult(
        success=success,
        length=steps,
        task_description=task_description,
        video_path=video_path,
    )


def evaluate_task(
    *,
    model: WebsocketClientPolicy,
    task_name: str,
    num_trials_per_task: int,
    env_img_res: int,
    robots: str,
    obj_instance_split: str,
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
) -> Dict[str, Any]:
    print(f"\n[Task] {task_name}")
    successes: List[bool] = []
    lengths: List[int] = []
    episode_results: List[Dict[str, Any]] = []

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

            max_steps = max_steps_override if max_steps_override is not None else TASK_MAX_STEPS.get(task_name, 500)
            action_mapper = ActionMapper(mode=action_map_mode, env_action_dim=env.action_dim)

            video_path = save_dir / "videos" / task_name / f"episode_{episode_idx:03d}.mp4"
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
            )

            successes.append(ep_result.success)
            lengths.append(ep_result.length)
            episode_results.append(
                {
                    "episode_idx": episode_idx,
                    "seed": env_seed,
                    "success": ep_result.success,
                    "length": ep_result.length,
                    "task_description": ep_result.task_description,
                    "video_path": ep_result.video_path,
                }
            )

            print(
                f"  Episode {episode_idx + 1}/{num_trials_per_task} | "
                f"{'SUCCESS' if ep_result.success else 'FAIL'} | steps={ep_result.length}"
            )

        finally:
            env.close()

    success_rate = float(np.mean(successes)) if successes else 0.0
    avg_length = float(np.mean(lengths)) if lengths else 0.0

    result = {
        "task_name": task_name,
        "num_trials": num_trials_per_task,
        "success_rate": success_rate,
        "avg_episode_length": avg_length,
        "num_successes": int(sum(successes)),
        "episode_results": episode_results,
    }
    print(
        f"[Task Done] {task_name} | success={result['num_successes']}/{num_trials_per_task} "
        f"({success_rate * 100:.1f}%), avg_len={avg_length:.1f}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LingBot-VA websocket policy on RoboCasa benchmark.")

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29056)
    parser.add_argument("--api-key", type=str, default=None)

    parser.add_argument("--task-name", type=str, default="TurnOffMicrowave", help="Task name, comma-separated list, or 'all'.")
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--seed-multiplier", type=int, default=256)
    parser.add_argument("--deterministic", type=str2bool, default=True)
    parser.add_argument("--deterministic-reset", type=str2bool, default=False)
    parser.add_argument("--deterministic-reset-seed", type=int, default=None)

    parser.add_argument("--env-img-res", type=int, default=224)
    parser.add_argument("--robots", type=str, default="PandaMobile")
    parser.add_argument("--obj-instance-split", type=str, default="B")
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_names = parse_task_names(args.task_name)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    controller_configs, controller_source = load_controller_configs(
        args.controller_config_path,
        "OSC_POSE",
    )
    print(f"Controller config source: {controller_source}")

    model = WebsocketClientPolicy(host=args.host, port=args.port, api_key=args.api_key)
    metadata = model.get_server_metadata()
    if metadata:
        print(f"Server metadata: {metadata}")

    print(
        f"Evaluating tasks={task_names}, trials/task={args.num_trials_per_task}, "
        f"host={args.host}:{args.port}"
    )

    all_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "host": args.host,
        "port": args.port,
        "task_names": task_names,
        "num_trials_per_task": args.num_trials_per_task,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "deterministic_reset": args.deterministic_reset,
        "controller_source": controller_source,
        "results": [],
    }

    for task_name in task_names:
        task_result = evaluate_task(
            model=model,
            task_name=task_name,
            num_trials_per_task=args.num_trials_per_task,
            env_img_res=args.env_img_res,
            robots=args.robots,
            obj_instance_split=args.obj_instance_split,
            randomize_cameras=args.randomize_cameras,
            layout_and_style_ids=args.layout_and_style_ids,
            deterministic=args.deterministic,
            deterministic_reset=args.deterministic_reset,
            deterministic_reset_seed=args.deterministic_reset_seed,
            seed=args.seed,
            seed_multiplier=args.seed_multiplier,
            controller_configs=controller_configs,
            flip_images=args.flip_images,
            max_steps_override=args.max_steps,
            save_visualization=args.save_visualization,
            video_guidance_scale=args.video_guidance_scale,
            action_guidance_scale=args.action_guidance_scale,
            num_steps_wait=args.num_steps_wait,
            save_rollout_video_flag=args.save_rollout_video,
            save_dir=save_dir,
            action_map_mode=args.action_map_mode,
        )
        all_results["results"].append(task_result)

    if all_results["results"]:
        final_success_rate = float(np.mean([r["success_rate"] for r in all_results["results"]]))
    else:
        final_success_rate = 0.0

    all_results["final_success_rate"] = final_success_rate

    summary_path = save_dir / f"robocasa_eval_{all_results['timestamp']}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n========== Final Summary ==========")
    for result in all_results["results"]:
        print(
            f"{result['task_name']}: {result['num_successes']}/{result['num_trials']} "
            f"({result['success_rate'] * 100:.1f}%), avg_len={result['avg_episode_length']:.1f}"
        )
    print(f"Average success rate across tasks: {final_success_rate * 100:.2f}%")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
