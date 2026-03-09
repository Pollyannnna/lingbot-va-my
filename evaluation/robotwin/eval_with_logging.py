"""
Wrapper script for eval_polict_client_openpi.py that adds per-episode timing
logs without modifying the original evaluation logic.

Default behavior is "core timing mode": keep key metrics compact and readable.
Set one of these env vars to keep full per-call details for debugging:
  - LINGBOT_TIMING_LOG_MODE=full
  - LINGBOT_KEEP_CALL_DETAILS=1
"""

import csv
import functools
import json
import os
import time
from datetime import datetime
from pathlib import Path

from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
import evaluation.robotwin.eval_polict_client_openpi as _eval_mod


def _parse_bool_env(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


LOG_DETAIL_MODE = os.getenv("LINGBOT_TIMING_LOG_MODE", "core").strip().lower()
KEEP_CALL_DETAILS = _parse_bool_env(
    "LINGBOT_KEEP_CALL_DETAILS",
    default=LOG_DETAIL_MODE in {"full", "debug", "verbose"},
)

_original_eval_policy = _eval_mod.eval_policy
_original_infer = WebsocketClientPolicy.infer

# Accumulator for all episode logs across the run.
_all_episode_logs = []
_run_metadata = {}

# Shared mutable state for current episode logging.
_episode_log = {}


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _sum_key(items, key):
    return sum(_to_float(item.get(key, 0.0)) for item in items)


def _avg_key(items, key):
    if not items:
        return 0.0
    return round(_sum_key(items, key) / len(items), 2)


def _pct(part, whole):
    if whole <= 1e-9:
        return 0.0
    return round(part / whole * 100.0, 1)


def _reset_episode_log():
    _episode_log.clear()
    _episode_log.update(
        {
            "infer_calls": [],
            "kv_cache_calls": [],
            "reset_time_ms": None,
            # Approximate simulator cost: get_obs() normally triggers camera render/readback.
            "env_get_obs_ms": 0.0,
            "env_get_obs_calls": 0,
            # Approximate simulator step cost.
            "env_take_action_ms": 0.0,
            "env_take_action_calls": 0,
        }
    )


_reset_episode_log()


def _patched_infer(self, obs):
    """Wrap WebsocketClientPolicy.infer and capture client/server timing."""
    is_reset = obs.get("reset", False)
    is_kv_cache = obs.get("compute_kv_cache", False)

    t0 = time.monotonic()
    result = _original_infer(self, obs)
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    server_timing = result.get("server_timing", {}) if isinstance(result, dict) else {}
    server_infer_ms = _to_float(server_timing.get("infer_ms", 0.0))
    server_video_ms = _to_float(server_timing.get("video_denoise_ms", 0.0))
    server_action_ms = _to_float(server_timing.get("action_denoise_ms", 0.0))
    server_vae_ms = _to_float(server_timing.get("vae_decode_ms", 0.0))

    # Backward compatibility: older servers may only provide infer_ms.
    server_denoise_ms = (
        server_video_ms + server_action_ms
        if (server_video_ms > 0.0 or server_action_ms > 0.0)
        else server_infer_ms
    )

    if is_reset:
        _episode_log["reset_time_ms"] = round(elapsed_ms, 2)
    elif is_kv_cache:
        _episode_log["kv_cache_calls"].append(
            {
                "client_time_ms": round(elapsed_ms, 2),
                "server_kv_ms": round(
                    _to_float(
                        server_timing.get(
                            "kv_cache_ms", server_timing.get("kv_ms", 0.0)
                        )
                    ),
                    2,
                ),
            }
        )
    else:
        _episode_log["infer_calls"].append(
            {
                "client_time_ms": round(elapsed_ms, 2),
                "server_infer_ms": round(server_infer_ms, 2),
                "server_denoise_ms": round(server_denoise_ms, 2),
                "server_video_denoise_ms": round(server_video_ms, 2),
                "server_action_denoise_ms": round(server_action_ms, 2),
                "server_vae_decode_ms": round(server_vae_ms, 2),
                "video_steps": int(_to_float(server_timing.get("video_steps", 0))),
                "action_steps": int(_to_float(server_timing.get("action_steps", 0))),
                # Round-trip overhead = client elapsed - server compute - decode.
                "network_overhead_ms": round(elapsed_ms - server_infer_ms - server_vae_ms, 2),
                "has_video": "video" in result if isinstance(result, dict) else False,
            }
        )

    return result


WebsocketClientPolicy.infer = _patched_infer


def _save_log_artifacts(task_name, args, st_seed):
    """Write JSON + CSV artifacts for current task."""
    save_root = args.get("save_root", "results")
    log_dir = Path(save_root) / f"stseed-{st_seed}" / "detailed_logs" / task_name
    log_dir.mkdir(parents=True, exist_ok=True)

    log_output = {
        "run_metadata": _run_metadata,
        "log_config": {
            "timing_log_mode": LOG_DETAIL_MODE,
            "keep_call_details": KEEP_CALL_DETAILS,
        },
        "episodes": _all_episode_logs,
        "summary": _compute_summary(_all_episode_logs),
    }

    log_path = log_dir / "detailed_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_output, f, indent=2, ensure_ascii=False)

    csv_path = log_dir / "episode_summary.csv"
    _save_csv_summary(_all_episode_logs, csv_path)
    return log_path, csv_path


def _patched_eval_policy(
    task_name,
    TASK_ENV,
    args,
    model,
    st_seed,
    test_num=100,
    video_size=None,
    instruction_type=None,
    save_visualization=False,
    video_guidance_scale=5.0,
    action_guidance_scale=5.0,
):
    """
    Wrap original eval_policy. Patch selected env methods to track:
      1) Policy inference timing
      2) Simulator observation/render timing (get_obs)
      3) Simulator step timing (take_action)
    """
    global _all_episode_logs, _run_metadata

    _all_episode_logs = []
    _run_metadata = {
        "task_name": task_name,
        "policy_name": args.get("policy_name", "unknown"),
        "task_config": args.get("task_config", "unknown"),
        "ckpt_setting": args.get("ckpt_setting", "unknown"),
        "test_num": test_num,
        "instruction_type": instruction_type,
        "video_guidance_scale": video_guidance_scale,
        "action_guidance_scale": action_guidance_scale,
        "start_seed": st_seed,
        "timing_log_mode": LOG_DETAIL_MODE,
        "keep_call_details": KEEP_CALL_DETAILS,
        "start_time": datetime.now().isoformat(),
    }

    # Patch TASK_ENV.take_action: count action steps + timing.
    _step_counter = {"count": 0}
    _original_take_action = TASK_ENV.take_action

    @functools.wraps(_original_take_action)
    def _timed_take_action(*a, **kw):
        _step_counter["count"] += 1
        t0 = time.monotonic()
        try:
            return _original_take_action(*a, **kw)
        finally:
            _episode_log["env_take_action_calls"] += 1
            _episode_log["env_take_action_ms"] += (time.monotonic() - t0) * 1000.0

    TASK_ENV.take_action = _timed_take_action

    # Patch TASK_ENV.get_obs: capture observation/render timing.
    _original_get_obs = getattr(TASK_ENV, "get_obs", None)

    if _original_get_obs is not None:

        @functools.wraps(_original_get_obs)
        def _timed_get_obs(*a, **kw):
            t0 = time.monotonic()
            out = _original_get_obs(*a, **kw)
            _episode_log["env_get_obs_calls"] += 1
            _episode_log["env_get_obs_ms"] += (time.monotonic() - t0) * 1000.0
            return out

        TASK_ENV.get_obs = _timed_get_obs

    # Patch TASK_ENV.setup_demo to mark episode boundaries.
    _original_setup_demo = TASK_ENV.setup_demo
    _episode_timer = {"start": None, "seed": None, "ep_num": None}

    @functools.wraps(_original_setup_demo)
    def _logging_setup_demo(now_ep_num=None, seed=None, is_test=True, **kw):
        # Finalize previous episode only if actual policy infer happened.
        if _episode_timer["start"] is not None and _episode_log["infer_calls"]:
            _finalize_episode(
                TASK_ENV, _episode_timer, _step_counter, task_name, args=args, st_seed=st_seed
            )

        _step_counter["count"] = 0
        _reset_episode_log()
        _episode_timer["start"] = time.monotonic()
        _episode_timer["seed"] = seed
        _episode_timer["ep_num"] = now_ep_num

        return _original_setup_demo(now_ep_num=now_ep_num, seed=seed, is_test=is_test, **kw)

    TASK_ENV.setup_demo = _logging_setup_demo

    run_start = time.monotonic()
    result = None
    try:
        result = _original_eval_policy(
            task_name,
            TASK_ENV,
            args,
            model,
            st_seed,
            test_num=test_num,
            video_size=video_size,
            instruction_type=instruction_type,
            save_visualization=save_visualization,
            video_guidance_scale=video_guidance_scale,
            action_guidance_scale=action_guidance_scale,
        )
    except Exception as e:
        import traceback

        _run_metadata["error"] = f"{type(e).__name__}: {e}"
        _run_metadata["traceback"] = traceback.format_exc()
        print(f"\033[91m[Logging] eval_policy crashed: {e}\033[0m")
        print("\033[91m[Logging] Saving partial logs before re-raising...\033[0m")
    finally:
        run_elapsed_s = time.monotonic() - run_start

        if _episode_timer["start"] is not None and _episode_log["infer_calls"]:
            try:
                _finalize_episode(
                    TASK_ENV, _episode_timer, _step_counter, task_name, args=args, st_seed=st_seed
                )
            except Exception:
                pass

        _run_metadata["end_time"] = datetime.now().isoformat()
        _run_metadata["total_run_time_s"] = round(run_elapsed_s, 2)

        log_path, csv_path = _save_log_artifacts(task_name, args, st_seed)
        print(f"\n\033[96m[Logging] Detailed log saved to: {log_path}\033[0m")
        print(f"\033[96m[Logging] CSV summary saved to: {csv_path}\033[0m")

        # Restore original methods.
        TASK_ENV.take_action = _original_take_action
        TASK_ENV.setup_demo = _original_setup_demo
        if _original_get_obs is not None:
            TASK_ENV.get_obs = _original_get_obs

    if result is None and "error" in _run_metadata:
        raise RuntimeError(f"eval_policy crashed: {_run_metadata['error']}")

    return result


def _finalize_episode(TASK_ENV, timer, step_counter, task_name, args=None, st_seed=None):
    """Collect and save one completed episode."""
    elapsed_s = time.monotonic() - timer["start"]
    infer_calls = list(_episode_log["infer_calls"])
    kv_calls = list(_episode_log["kv_cache_calls"])

    infer_ms = _sum_key(infer_calls, "client_time_ms")
    kv_ms = _sum_key(kv_calls, "client_time_ms")
    server_infer_ms = _sum_key(infer_calls, "server_infer_ms")
    server_denoise_ms = _sum_key(infer_calls, "server_denoise_ms")
    server_video_ms = _sum_key(infer_calls, "server_video_denoise_ms")
    server_action_ms = _sum_key(infer_calls, "server_action_denoise_ms")
    server_vae_ms = _sum_key(infer_calls, "server_vae_decode_ms")
    network_overhead_ms = _sum_key(infer_calls, "network_overhead_ms")

    env_get_obs_ms = _to_float(_episode_log.get("env_get_obs_ms", 0.0))
    env_take_action_ms = _to_float(_episode_log.get("env_take_action_ms", 0.0))

    total_wall_time_s = round(elapsed_s, 2)
    total_client_infer_time_s = round(infer_ms / 1000.0, 3)
    total_kv_cache_time_s = round(kv_ms / 1000.0, 3)
    total_server_infer_time_s = round(server_infer_ms / 1000.0, 3)  # legacy name
    total_server_denoise_s = round(server_denoise_ms / 1000.0, 3)
    total_server_video_denoise_s = round(server_video_ms / 1000.0, 3)
    total_server_action_denoise_s = round(server_action_ms / 1000.0, 3)
    total_server_vae_decode_s = round(server_vae_ms / 1000.0, 3)
    total_network_overhead_s = round(network_overhead_ms / 1000.0, 3)

    # Sim render/obs and sim step timing.
    sim_render_get_obs_time_s = round(env_get_obs_ms / 1000.0, 3)
    sim_step_take_action_time_s = round(env_take_action_ms / 1000.0, 3)
    total_env_time_s = round(sim_render_get_obs_time_s + sim_step_take_action_time_s, 3)

    total_known_time_s_raw = (
        infer_ms + kv_ms + env_get_obs_ms + env_take_action_ms
    ) / 1000.0
    total_other_time_s = round(max(elapsed_s - total_known_time_s_raw, 0.0), 3)

    denoise_breakdown_available = (
        total_server_video_denoise_s > 0.0 or total_server_action_denoise_s > 0.0
    )

    episode_record = {
        "episode_idx": len(_all_episode_logs),
        "task_name": task_name,
        "seed": timer["seed"],
        "success": bool(TASK_ENV.eval_success) if hasattr(TASK_ENV, "eval_success") else None,
        "total_action_steps": step_counter["count"],
        "total_wall_time_s": total_wall_time_s,
        "num_infer_chunks": len(infer_calls),
        "num_kv_cache_calls": len(kv_calls),
        # Core timing totals.
        "sim_render_get_obs_time_s": sim_render_get_obs_time_s,
        "sim_step_take_action_time_s": sim_step_take_action_time_s,
        "total_env_time_s": total_env_time_s,
        "total_client_infer_time_s": total_client_infer_time_s,
        "total_kv_cache_time_s": total_kv_cache_time_s,
        "total_server_denoise_s": total_server_denoise_s,
        "total_server_video_denoise_s": total_server_video_denoise_s,
        "total_server_action_denoise_s": total_server_action_denoise_s,
        "total_server_vae_decode_s": total_server_vae_decode_s,
        "total_server_infer_time_s": total_server_infer_time_s,
        "total_network_overhead_s": total_network_overhead_s,
        "total_other_time_s": total_other_time_s,
        # Ratios for quick reading.
        "render_share_pct": _pct(sim_render_get_obs_time_s, total_wall_time_s),
        "sim_step_share_pct": _pct(sim_step_take_action_time_s, total_wall_time_s),
        "policy_infer_share_pct": _pct(total_client_infer_time_s, total_wall_time_s),
        "kv_cache_share_pct": _pct(total_kv_cache_time_s, total_wall_time_s),
        "other_share_pct": _pct(total_other_time_s, total_wall_time_s),
        # Useful counts.
        "env_get_obs_calls": int(_episode_log.get("env_get_obs_calls", 0)),
        "env_take_action_calls": int(_episode_log.get("env_take_action_calls", 0)),
        "server_denoise_breakdown_available": denoise_breakdown_available,
        # Legacy compatibility fields.
        "reset_time_ms": (
            round(_to_float(_episode_log.get("reset_time_ms")), 2)
            if _episode_log.get("reset_time_ms") is not None
            else None
        ),
        "avg_chunk_infer_ms": _avg_key(infer_calls, "client_time_ms"),
        "avg_server_infer_ms": _avg_key(infer_calls, "server_infer_ms"),
        "avg_server_video_denoise_ms": _avg_key(infer_calls, "server_video_denoise_ms"),
        "avg_server_action_denoise_ms": _avg_key(infer_calls, "server_action_denoise_ms"),
        "avg_server_vae_decode_ms": _avg_key(infer_calls, "server_vae_decode_ms"),
        "avg_server_kv_cache_ms": _avg_key(kv_calls, "server_kv_ms"),
    }

    try:
        episode_record["prompt"] = TASK_ENV.get_instruction()
    except Exception:
        episode_record["prompt"] = None

    episode_record["core_timing"] = {
        "wall_time_s": total_wall_time_s,
        "sim_render_get_obs_time_s": sim_render_get_obs_time_s,
        "sim_step_take_action_time_s": sim_step_take_action_time_s,
        "policy_infer_time_s": total_client_infer_time_s,
        "server_denoise_time_s": total_server_denoise_s,
        "kv_cache_time_s": total_kv_cache_time_s,
        "other_time_s": total_other_time_s,
        "share_pct": {
            "render": episode_record["render_share_pct"],
            "sim_step": episode_record["sim_step_share_pct"],
            "policy_infer": episode_record["policy_infer_share_pct"],
            "kv_cache": episode_record["kv_cache_share_pct"],
            "other": episode_record["other_share_pct"],
        },
    }

    if KEEP_CALL_DETAILS:
        episode_record["infer_calls_detail"] = infer_calls
        episode_record["kv_cache_calls_detail"] = kv_calls
    else:
        episode_record["infer_calls_detail"] = []
        episode_record["kv_cache_calls_detail"] = []

    _all_episode_logs.append(episode_record)

    msg = (
        f"\033[93m[Logging] Episode {episode_record['episode_idx']} | "
        f"seed={timer['seed']} | "
        f"{'✓' if episode_record['success'] else '✗'} | "
        f"steps={episode_record['total_action_steps']} | "
        f"chunks={episode_record['num_infer_chunks']} | "
        f"wall={episode_record['total_wall_time_s']}s | "
        f"render={episode_record['sim_render_get_obs_time_s']}s | "
        f"sim_step={episode_record['sim_step_take_action_time_s']}s | "
        f"infer={episode_record['total_client_infer_time_s']}s | "
        f"denoise={episode_record['total_server_denoise_s']}s | "
        f"kv={episode_record['total_kv_cache_time_s']}s | "
        f"other={episode_record['total_other_time_s']}s"
    )
    if denoise_breakdown_available:
        msg += (
            f" | video_dn={episode_record['total_server_video_denoise_s']}s"
            f" | action_dn={episode_record['total_server_action_denoise_s']}s"
        )
    print(msg + "\033[0m")

    # Incremental save: write logs after every episode.
    if args is not None and st_seed is not None:
        try:
            _save_log_artifacts(task_name, args, st_seed)
        except Exception as e:
            print(f"\033[91m[Logging] Incremental save failed: {e}\033[0m")


def _compute_summary(episodes):
    """Compute aggregate run statistics from per-episode records."""
    if not episodes:
        return {}

    total = len(episodes)
    successes = sum(1 for e in episodes if e.get("success"))

    wall_times = [_to_float(e.get("total_wall_time_s", 0.0)) for e in episodes]
    infer_times = [_to_float(e.get("total_client_infer_time_s", 0.0)) for e in episodes]
    kv_times = [_to_float(e.get("total_kv_cache_time_s", 0.0)) for e in episodes]
    denoise_times = [_to_float(e.get("total_server_denoise_s", 0.0)) for e in episodes]
    render_times = [_to_float(e.get("sim_render_get_obs_time_s", 0.0)) for e in episodes]
    sim_step_times = [_to_float(e.get("sim_step_take_action_time_s", 0.0)) for e in episodes]
    other_times = [_to_float(e.get("total_other_time_s", 0.0)) for e in episodes]
    step_counts = [_to_float(e.get("total_action_steps", 0)) for e in episodes]
    chunk_counts = [_to_float(e.get("num_infer_chunks", 0)) for e in episodes]

    avg_wall = sum(wall_times) / total
    avg_infer = sum(infer_times) / total
    avg_kv = sum(kv_times) / total
    avg_render = sum(render_times) / total
    avg_sim_step = sum(sim_step_times) / total
    avg_other = sum(other_times) / total

    return {
        "total_episodes": total,
        "successes": successes,
        "success_rate": round(successes / total * 100.0, 1) if total > 0 else 0.0,
        "total_wall_time_s": round(sum(wall_times), 2),
        "avg_wall_time_s": round(avg_wall, 2),
        "min_wall_time_s": round(min(wall_times), 2),
        "max_wall_time_s": round(max(wall_times), 2),
        "avg_steps": round(sum(step_counts) / total, 1),
        "avg_chunks": round(sum(chunk_counts) / total, 1),
        "avg_infer_time_s": round(avg_infer, 2),  # legacy name
        "avg_client_infer_time_s": round(avg_infer, 2),
        "avg_kv_cache_time_s": round(avg_kv, 2),
        "avg_server_denoise_s": round(sum(denoise_times) / total, 2),
        "avg_sim_render_get_obs_time_s": round(avg_render, 2),
        "avg_sim_step_take_action_time_s": round(avg_sim_step, 2),
        "avg_other_time_s": round(avg_other, 2),
        "avg_render_share_pct": _pct(avg_render, avg_wall),
        "avg_sim_step_share_pct": _pct(avg_sim_step, avg_wall),
        "avg_policy_infer_share_pct": _pct(avg_infer, avg_wall),
        "avg_kv_cache_share_pct": _pct(avg_kv, avg_wall),
        "avg_other_share_pct": _pct(avg_other, avg_wall),
        "episodes_with_denoise_breakdown": sum(
            1 for e in episodes if e.get("server_denoise_breakdown_available")
        ),
    }


def _save_csv_summary(episodes, csv_path):
    """Save a compact one-row-per-episode CSV focused on core timing."""
    if not episodes:
        return

    fields = [
        "episode_idx",
        "task_name",
        "seed",
        "prompt",
        "success",
        "total_action_steps",
        "num_infer_chunks",
        "num_kv_cache_calls",
        "total_wall_time_s",
        "sim_render_get_obs_time_s",
        "sim_step_take_action_time_s",
        "total_env_time_s",
        "total_client_infer_time_s",
        "total_server_denoise_s",
        "total_server_video_denoise_s",
        "total_server_action_denoise_s",
        "total_server_vae_decode_s",
        "total_kv_cache_time_s",
        "total_network_overhead_s",
        "total_other_time_s",
        "render_share_pct",
        "sim_step_share_pct",
        "policy_infer_share_pct",
        "kv_cache_share_pct",
        "other_share_pct",
        "server_denoise_breakdown_available",
        # Legacy compatibility columns (kept at the end).
        "total_server_infer_time_s",
        "avg_chunk_infer_ms",
        "avg_server_infer_ms",
        "avg_server_video_denoise_ms",
        "avg_server_action_denoise_ms",
        "avg_server_vae_decode_ms",
        "avg_server_kv_cache_ms",
        "reset_time_ms",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ep in episodes:
            row = {field: ep.get(field, "") for field in fields}
            writer.writerow(row)


_eval_mod.eval_policy = _patched_eval_policy


if __name__ == "__main__":
    from evaluation.robotwin.test_render import Sapien_TEST

    Sapien_TEST()
    usr_args = _eval_mod.parse_args_and_config()
    _eval_mod.main(usr_args)
