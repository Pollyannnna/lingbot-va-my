# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
import os
from easydict import EasyDict

from .shared_config import va_shared_cfg


def _load_robocasa_norm_stat(action_dim: int) -> dict:
    # RoboCasa control is typically normalized to [-1, 1] for the first 7 dims.
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

    q01 = q01_7 + [0.0] * max(action_dim - 7, 0)
    q99 = q99_7 + [1.0] * max(action_dim - 7, 0)
    return {"q01": q01, "q99": q99}


va_robocasa_cfg = EasyDict(__name__="Config: VA robocasa")
va_robocasa_cfg.update(va_shared_cfg)

va_robocasa_cfg.wan22_pretrained_model_name_or_path = os.getenv(
    "ROBOCASA_PRETRAINED_MODEL",
    "/data/share/lijiang/ckpt/lingbot-va-posttrain-robotwin",
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

# Keep the transformer action channel size at 30 for checkpoint compatibility,
# while only wiring the RoboCasa manipulation channels (first 7 dims).
va_robocasa_cfg.used_action_channel_ids = list(range(0, 7))
inverse_used_action_channel_ids = [
    len(va_robocasa_cfg.used_action_channel_ids)
] * va_robocasa_cfg.action_dim
for i, j in enumerate(va_robocasa_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_robocasa_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_robocasa_cfg.action_norm_method = "quantiles"
va_robocasa_cfg.norm_stat = _load_robocasa_norm_stat(va_robocasa_cfg.action_dim)
