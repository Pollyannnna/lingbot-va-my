# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
import os
from easydict import EasyDict

from .shared_config import va_shared_cfg


def _sanitize_quantiles(
    q01_7: list[float],
    q99_7: list[float],
    min_span: float = 1e-3,
) -> tuple[list[float], list[float]]:
    sanitized_q01 = list(q01_7)
    sanitized_q99 = list(q99_7)
    fallback_q01 = [-1.0] * len(sanitized_q01)
    fallback_q99 = [1.0] * len(sanitized_q99)

    for i, (q01, q99) in enumerate(zip(sanitized_q01, sanitized_q99)):
        if abs(q99 - q01) < min_span:
            sanitized_q01[i] = fallback_q01[i]
            sanitized_q99[i] = fallback_q99[i]

    return sanitized_q01, sanitized_q99


def _load_robocasa_norm_stat(action_dim: int) -> dict:
    # RoboCasa control is typically normalized to [-1, 1] for the 7 learned
    # manipulator channels: ee xyz, ee rot xyz, gripper.
    q01_7 = [-1.0] * 7
    q99_7 = [1.0] * 7

    stats_path = os.getenv("ROBOCASA_NORM_STATS_PATH", "").strip()
    if stats_path:
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            loaded_q01 = stats.get("q01_7", stats.get("q01", None))
            loaded_q99 = stats.get("q99_7", stats.get("q99", None))
            if isinstance(loaded_q01, list) and isinstance(loaded_q99, list):
                if len(loaded_q01) >= 7 and len(loaded_q99) >= 7:
                    q01_7 = [float(v) for v in loaded_q01[:7]]
                    q99_7 = [float(v) for v in loaded_q99[:7]]
        except Exception as exc:
            print(f"[WARN] Failed to load ROBOCASA_NORM_STATS_PATH={stats_path}: {exc}")

    q01_7, q99_7 = _sanitize_quantiles(q01_7, q99_7)
    q01 = q01_7 + [0.0] * max(action_dim - 7, 0)
    q99 = q99_7 + [1.0] * max(action_dim - 7, 0)
    return {"q01": q01, "q99": q99}


va_robocasa_cfg = EasyDict(__name__="Config: VA robocasa")
va_robocasa_cfg.update(va_shared_cfg)

va_robocasa_cfg.wan22_pretrained_model_name_or_path = os.getenv(
    "ROBOCASA_PRETRAINED_MODEL",
    "/data/share/lijiang/ckpt/lingbot-va-base",
)

va_robocasa_cfg.attn_window = 72
va_robocasa_cfg.frame_chunk_size = 2
va_robocasa_cfg.env_type = "robocasa"

va_robocasa_cfg.height = 256
va_robocasa_cfg.width = 256
va_robocasa_cfg.action_dim = 30
va_robocasa_cfg.action_per_frame = 16
va_robocasa_cfg.obs_cam_keys = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
]
va_robocasa_cfg.guidance_scale = 5
va_robocasa_cfg.action_guidance_scale = 1

va_robocasa_cfg.num_inference_steps = 25
va_robocasa_cfg.video_exec_step = -1
va_robocasa_cfg.action_num_inference_steps = 50

va_robocasa_cfg.snr_shift = 5.0
va_robocasa_cfg.action_snr_shift = 1.0

# RoboCasa raw action layout comes from lerobot/meta/modality.json:
#   0:4 base_motion, 4:5 control_mode, 5:8 ee_position,
#   8:11 ee_rotation, 11:12 gripper_close.
# We only learn the 7 manipulator channels and map them into the first 7 model
# action channels for checkpoint compatibility.
va_robocasa_cfg.raw_action_channel_ids = [5, 6, 7, 8, 9, 10, 11]
va_robocasa_cfg.used_action_channel_ids = list(range(0, 7))
inverse_used_action_channel_ids = [
    len(va_robocasa_cfg.used_action_channel_ids)
] * va_robocasa_cfg.action_dim
for i, j in enumerate(va_robocasa_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_robocasa_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_robocasa_cfg.action_norm_method = "quantiles"
va_robocasa_cfg.norm_stat = _load_robocasa_norm_stat(va_robocasa_cfg.action_dim)
