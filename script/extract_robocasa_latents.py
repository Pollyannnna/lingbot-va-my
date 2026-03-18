#!/usr/bin/env python3
"""Extract LingBot-VA latents for RoboCasa LeRobot datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

DEFAULT_OBS_CAM_KEYS = [
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
]


@dataclass
class SegmentTask:
    dataset_root: Path
    episode_index: int
    episode_chunk: int
    start_frame: int
    end_frame: int
    action_text: str
    tasks: list[str]
    video_paths: dict[str, Path]
    latent_paths: dict[str, Path]
    ori_fps: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Wan2.2 VAE latents for RoboCasa LeRobot datasets.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Single LeRobot root or a directory containing many */lerobot datasets.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/data/share/lijiang/ckpt/lingbot-va-posttrain-robotwin"),
        help="Model root containing vae/, tokenizer/, and text_encoder/.",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=10,
        help="Target fps for latent extraction.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Resize height before VAE encoding.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Resize width before VAE encoding.",
    )
    parser.add_argument(
        "--obs-cam-keys",
        nargs="+",
        default=DEFAULT_OBS_CAM_KEYS,
        help="Video keys to extract.",
    )
    parser.add_argument(
        "--vae-chunk-frames",
        type=int,
        default=16,
        help="Temporal chunk size fed into VAE each step.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Max text encoder sequence length.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for extraction: auto/cpu/cuda/cuda:N.",
    )
    parser.add_argument(
        "--vae-dtype",
        type=str,
        default="bfloat16",
        choices=tuple(DTYPE_MAP.keys()),
        help="VAE compute dtype.",
    )
    parser.add_argument(
        "--save-dtype",
        type=str,
        default="bfloat16",
        choices=tuple(DTYPE_MAP.keys()),
        help="Saved latent/text embedding dtype.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip segment when all target latent files already exist.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing latent files.",
    )
    parser.add_argument(
        "--require-integral-stride",
        action="store_true",
        default=True,
        help="Require original fps to be divisible by target fps.",
    )
    parser.add_argument(
        "--allow-non-integral-stride",
        action="store_false",
        dest="require_integral_stride",
        help="Allow rounded frame stride when fps ratio is non-integral.",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=0,
        help="Debug limit on number of lerobot datasets to process. 0 means all.",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=0,
        help="Debug limit on number of segments to process. 0 means all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List work without writing latent files.",
    )
    parser.add_argument(
        "--shard-world-size",
        type=int,
        default=-1,
        help="Number of task shards. Defaults to WORLD_SIZE env when set, else 1.",
    )
    parser.add_argument(
        "--shard-rank",
        type=int,
        default=-1,
        help="Task shard rank. Defaults to RANK env when set, else 0.",
    )
    return parser.parse_args()


def _resolve_device(device: str) -> torch.device:
    local_rank = os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "")).strip()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible_device_count = len([v for v in visible_devices.split(",") if v.strip()]) if visible_devices else 0
    if device == "auto":
        if torch.cuda.is_available():
            if visible_device_count == 1:
                return torch.device("cuda:0")
            return torch.device(f"cuda:{local_rank}" if local_rank else "cuda")
        return torch.device("cpu")
    if device == "cuda":
        if visible_device_count == 1:
            return torch.device("cuda:0")
        if local_rank:
            return torch.device(f"cuda:{local_rank}")
    return torch.device(device)


def _compute_dtype(device: torch.device, dtype_name: str) -> torch.dtype:
    dtype = DTYPE_MAP[dtype_name]
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def _resolve_shard_spec(
    shard_world_size: int,
    shard_rank: int,
) -> tuple[int, int]:
    env_world_size = os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "")).strip()
    env_rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "")).strip()

    world_size = shard_world_size if shard_world_size > 0 else int(env_world_size or "1")
    rank = shard_rank if shard_rank >= 0 else int(env_rank or "0")

    if world_size <= 0:
        raise ValueError(f"shard_world_size must be positive, got {world_size}")
    if not (0 <= rank < world_size):
        raise ValueError(f"shard_rank must satisfy 0 <= rank < world_size, got rank={rank}, world_size={world_size}")
    return world_size, rank


def _shard_tasks(tasks: list[SegmentTask], world_size: int, rank: int) -> list[SegmentTask]:
    if world_size == 1:
        return tasks
    return [task for idx, task in enumerate(tasks) if idx % world_size == rank]


def _find_lerobot_roots(dataset_root: Path) -> list[Path]:
    dataset_root = dataset_root.expanduser().resolve()
    if (dataset_root / "meta" / "info.json").exists() and (dataset_root / "videos").exists():
        return [dataset_root]

    roots: list[Path] = []
    for info_path in sorted(dataset_root.glob("**/meta/info.json")):
        root = info_path.parent.parent
        if (root / "videos").exists():
            roots.append(root)
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


def _safe_action_text(record: dict[str, Any], segment: dict[str, Any]) -> str:
    text = segment.get("action_text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    tasks = record.get("tasks", [])
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    if isinstance(tasks, str):
        return tasks
    return ""


def _episode_chunk(info: dict[str, Any], episode_index: int) -> int:
    chunk_size = int(info.get("chunks_size", 1000))
    return episode_index // chunk_size


def _resolve_video_path(dataset_root: Path, info: dict[str, Any], episode_chunk: int, video_key: str, episode_index: int) -> Path:
    video_path_tmpl = info["video_path"]
    rel = video_path_tmpl.format(
        episode_chunk=episode_chunk,
        video_key=video_key,
        episode_index=episode_index,
    )
    return dataset_root / rel


def _resolve_latent_path(dataset_root: Path, episode_chunk: int, video_key: str, episode_index: int, start_frame: int, end_frame: int) -> Path:
    return (
        dataset_root
        / "latents"
        / f"chunk-{episode_chunk:03d}"
        / video_key
        / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
    )


def _build_tasks(dataset_root: Path, obs_cam_keys: list[str]) -> list[SegmentTask]:
    info = _read_json(dataset_root / "meta" / "info.json")
    episodes = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    ori_fps = int(round(float(info.get("fps", 0))))
    if ori_fps <= 0:
        raise ValueError(f"invalid fps in {dataset_root / 'meta' / 'info.json'}: {info.get('fps')}")

    tasks: list[SegmentTask] = []
    for record in episodes:
        episode_index = int(record["episode_index"])
        segments = record.get("action_config")
        if not isinstance(segments, list) or not segments:
            continue
        episode_chunk = _episode_chunk(info, episode_index)
        video_paths = {
            key: _resolve_video_path(dataset_root, info, episode_chunk, key, episode_index)
            for key in obs_cam_keys
        }
        for segment in segments:
            start_frame = int(segment["start_frame"])
            end_frame = int(segment["end_frame"])
            latent_paths = {
                key: _resolve_latent_path(
                    dataset_root,
                    episode_chunk,
                    key,
                    episode_index,
                    start_frame,
                    end_frame,
                )
                for key in obs_cam_keys
            }
            tasks.append(
                SegmentTask(
                    dataset_root=dataset_root,
                    episode_index=episode_index,
                    episode_chunk=episode_chunk,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    action_text=_safe_action_text(record, segment),
                    tasks=[str(v) for v in record.get("tasks", [])],
                    video_paths=video_paths,
                    latent_paths=latent_paths,
                    ori_fps=ori_fps,
                )
            )
    return tasks


def _frame_stride(ori_fps: int, target_fps: int, require_integral_stride: bool) -> tuple[int, int]:
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    if target_fps >= ori_fps:
        return 1, ori_fps

    ratio = ori_fps / target_fps
    rounded = int(round(ratio))
    if require_integral_stride and abs(ratio - rounded) > 1e-6:
        raise ValueError(
            f"non-integral fps ratio: ori_fps={ori_fps}, target_fps={target_fps}. "
            "Use a divisible target fps or pass --allow-non-integral-stride."
        )
    stride = max(1, rounded)
    return stride, max(1, int(round(ori_fps / stride)))


def _sample_frame_ids(start_frame: int, end_frame: int, stride: int) -> list[int]:
    frame_ids = list(range(start_frame, end_frame, stride))
    if len(frame_ids) < 2:
        raise ValueError(
            f"segment [{start_frame}, {end_frame}) yields fewer than 2 frames with stride={stride}"
        )
    return frame_ids


def _read_video_frames(video_path: Path, frame_ids: list[int]) -> tuple[torch.Tensor, int, int]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"failed to open video: {video_path}")

    selected = []
    frame_set = set(frame_ids)
    min_frame = frame_ids[0]
    max_frame = frame_ids[-1]
    current = 0

    while current <= max_frame:
        ok, frame = capture.read()
        if not ok:
            break
        if current >= min_frame and current in frame_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            selected.append(torch.from_numpy(rgb))
        current += 1

    capture.release()

    if len(selected) != len(frame_ids):
        raise RuntimeError(
            f"video {video_path} provided {len(selected)} / {len(frame_ids)} requested frames"
        )

    stacked = torch.stack(selected, dim=0)
    height = int(stacked.shape[1])
    width = int(stacked.shape[2])
    return stacked, height, width


class RoboCasaLatentExtractor:
    def __init__(
        self,
        model_root: Path,
        device: torch.device,
        vae_dtype: torch.dtype,
        save_dtype: torch.dtype,
        max_seq_len: int,
        height: int,
        width: int,
        vae_chunk_frames: int,
    ) -> None:
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
        from wan_va.modules.utils import WanVAEStreamingWrapper, load_text_encoder, load_tokenizer, load_vae

        self.model_root = model_root.expanduser().resolve()
        self.device = device
        self.vae_dtype = vae_dtype
        self.save_dtype = save_dtype
        self.max_seq_len = max_seq_len
        self.height = height
        self.width = width
        self.vae_chunk_frames = vae_chunk_frames
        self.prompt_clean = prompt_clean

        self.tokenizer = load_tokenizer(str(self.model_root / "tokenizer"))
        self.text_encoder = load_text_encoder(
            str(self.model_root / "text_encoder"),
            torch_dtype=torch.float32,
            torch_device=device,
        )
        self.text_encoder.eval()

        self.vae = load_vae(
            str(self.model_root / "vae"),
            torch_dtype=self.vae_dtype,
            torch_device=device,
        )
        self.vae.eval()
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)
        self.latents_mean = torch.tensor(self.vae.config.latents_mean, device=self.device)
        self.latents_std = torch.tensor(self.vae.config.latents_std, device=self.device)
        self._text_cache: dict[str, torch.Tensor] = {}

    def encode_text(self, text: str) -> torch.Tensor:
        cleaned = self.prompt_clean(text or "")
        if cleaned in self._text_cache:
            return self._text_cache[cleaned]

        text_inputs = self.tokenizer(
            [cleaned],
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attn_mask = text_inputs.attention_mask.to(self.device)
        seq_len = int(attn_mask[0].sum().item())

        with torch.no_grad():
            hidden = self.text_encoder(input_ids, attn_mask).last_hidden_state[0]

        valid = hidden[:seq_len]
        if seq_len < self.max_seq_len:
            pad = torch.zeros(
                self.max_seq_len - seq_len,
                hidden.shape[1],
                device=hidden.device,
                dtype=hidden.dtype,
            )
            emb = torch.cat([valid, pad], dim=0)
        else:
            emb = valid[: self.max_seq_len]

        emb = emb.to("cpu", dtype=self.save_dtype).contiguous()
        self._text_cache[cleaned] = emb
        return emb

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = (1.0 / self.latents_std).view(1, -1, 1, 1, 1).to(device=latents.device)
        return ((latents.float() - latents_mean) * latents_std).to(latents)

    @torch.inference_mode()
    def encode_video(self, frames: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if frames.ndim != 4 or frames.shape[-1] not in {1, 3}:
            raise ValueError(
                f"expected frames with shape [F, H, W, C], got {tuple(frames.shape)}"
            )

        # Resize each frame in [F, C, H, W], then rearrange to Wan VAE input
        # layout [B, C, F, H, W].
        videos = frames.permute(0, 3, 1, 2).float()
        videos = F.interpolate(
            videos,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )
        videos = videos.permute(1, 0, 2, 3).unsqueeze(0)
        videos = videos / 255.0 * 2.0 - 1.0

        vae_device = next(self.streaming_vae.vae.parameters()).device
        videos = videos.to(vae_device).to(self.vae_dtype)

        # AutoencoderKLWan.encode() already implements the model-specific
        # temporal streaming schedule (first 1 frame, then 4-frame chunks).
        # Calling WanVAEStreamingWrapper.encode_chunk() directly on arbitrary
        # chunk sizes breaks the internal causal-cache assumptions.
        posterior = self.vae.encode(videos).latent_dist
        mu_full = self._normalize_latents(posterior.mean).to("cpu")
        latent_num_frames = int(mu_full.shape[2])
        latent_height = int(mu_full.shape[3])
        latent_width = int(mu_full.shape[4])
        latent = rearrange(
            mu_full[0].to(dtype=self.save_dtype).contiguous(),
            "c f h w -> (f h w) c",
        ).cpu()
        return latent, latent_num_frames, latent_height, latent_width


def _should_skip(task: SegmentTask, overwrite: bool, skip_existing: bool) -> bool:
    if overwrite:
        return False
    if not skip_existing:
        return False
    return all(path.exists() for path in task.latent_paths.values())


def _save_segment(
    extractor: RoboCasaLatentExtractor,
    task: SegmentTask,
    target_fps: int,
    require_integral_stride: bool,
    dry_run: bool,
) -> dict[str, Any]:
    stride, actual_fps = _frame_stride(task.ori_fps, target_fps, require_integral_stride)
    frame_ids = _sample_frame_ids(task.start_frame, task.end_frame, stride)
    text_emb = extractor.encode_text(task.action_text)

    if dry_run:
        return {
            "frame_stride": stride,
            "actual_fps": actual_fps,
            "frame_ids": frame_ids,
        }

    for video_key, video_path in task.video_paths.items():
        frames, video_height, video_width = _read_video_frames(video_path, frame_ids)
        latent, latent_num_frames, latent_height, latent_width = extractor.encode_video(frames)
        payload = {
            "latent": latent,
            "latent_num_frames": latent_num_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "video_num_frames": len(frame_ids),
            "video_height": video_height,
            "video_width": video_width,
            "text_emb": text_emb,
            "text": task.action_text,
            "frame_ids": frame_ids,
            "start_frame": task.start_frame,
            "end_frame": task.end_frame,
            "fps": actual_fps,
            "ori_fps": task.ori_fps,
            "tasks": task.tasks,
            "episode_index": task.episode_index,
            "video_key": video_key,
        }
        out_path = task.latent_paths[video_key]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out_path)

    return {
        "frame_stride": stride,
        "actual_fps": actual_fps,
        "frame_ids": frame_ids,
    }


def main() -> int:
    args = _parse_args()
    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    vae_dtype = _compute_dtype(device, args.vae_dtype)
    save_dtype = DTYPE_MAP[args.save_dtype]
    shard_world_size, shard_rank = _resolve_shard_spec(
        args.shard_world_size,
        args.shard_rank,
    )

    dataset_roots = _find_lerobot_roots(args.dataset_root)
    if not dataset_roots:
        raise FileNotFoundError(
            f"no lerobot datasets found under {args.dataset_root.expanduser().resolve()}"
        )
    if args.max_datasets > 0:
        dataset_roots = dataset_roots[: args.max_datasets]

    all_tasks: list[SegmentTask] = []
    for root in dataset_roots:
        all_tasks.extend(_build_tasks(root, args.obs_cam_keys))

    if args.max_segments > 0:
        all_tasks = all_tasks[: args.max_segments]

    if not all_tasks:
        raise RuntimeError("no action_config segments found; run add_action_config.py first")

    local_tasks = _shard_tasks(all_tasks, shard_world_size, shard_rank)
    skip_count = 0
    planned_count = len(local_tasks)
    total_planned_count = len(all_tasks)
    if not args.overwrite and args.skip_existing:
        for task in local_tasks:
            if _should_skip(task, args.overwrite, args.skip_existing):
                skip_count += 1

    print(f"datasets found: {len(dataset_roots)}")
    print(f"segments found total: {total_planned_count}")
    print(f"segments assigned to rank {shard_rank}/{shard_world_size}: {planned_count}")
    print(f"segments already complete: {skip_count}")
    print(f"device: {device} vae_dtype: {vae_dtype} save_dtype: {save_dtype}")

    if args.dry_run:
        return 0

    extractor = RoboCasaLatentExtractor(
        model_root=args.model_root,
        device=device,
        vae_dtype=vae_dtype,
        save_dtype=save_dtype,
        max_seq_len=args.max_seq_len,
        height=args.height,
        width=args.width,
        vae_chunk_frames=args.vae_chunk_frames,
    )

    processed = 0
    skipped = 0
    failed = 0

    for task in tqdm(local_tasks, desc=f"extract_robocasa_latents[r{shard_rank}]"):
        if _should_skip(task, args.overwrite, args.skip_existing):
            skipped += 1
            continue
        try:
            _save_segment(
                extractor=extractor,
                task=task,
                target_fps=args.target_fps,
                require_integral_stride=args.require_integral_stride,
                dry_run=False,
            )
            processed += 1
        except Exception as exc:
            failed += 1
            print(
                "[FAIL]",
                f"dataset={task.dataset_root}",
                f"episode={task.episode_index}",
                f"segment={task.start_frame}:{task.end_frame}",
                f"error={exc}",
            )

    print("done")
    print(f"processed={processed} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
