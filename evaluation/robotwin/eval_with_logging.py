"""
Wrapper script for eval_polict_client_openpi.py that adds detailed per-episode
inference logging WITHOUT modifying the original evaluation code.

Usage: Drop-in replacement in launch_client.sh:
    python -m evaluation.robotwin.eval_with_logging --config ... (same args)

Output: A detailed_log.json file alongside the normal results, containing
per-episode timing, step counts, and server metrics.
"""

import sys
import os
import time
import json
import functools
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────
# 1. Monkey-patch WebsocketClientPolicy.infer to capture timing
# ─────────────────────────────────────────────────────────────────────

from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy

# Shared mutable state for current episode logging
_episode_log = {
    "infer_calls": [],      # list of dicts for each infer() call
    "reset_time_ms": None,
    "kv_cache_calls": [],
}


def _reset_episode_log():
    _episode_log["infer_calls"] = []
    _episode_log["reset_time_ms"] = None
    _episode_log["kv_cache_calls"] = []


_original_infer = WebsocketClientPolicy.infer


def _patched_infer(self, obs):
    """Wrapper around WebsocketClientPolicy.infer that records timing."""
    is_reset = obs.get("reset", False)
    is_kv_cache = obs.get("compute_kv_cache", False)

    t0 = time.monotonic()
    result = _original_infer(self, obs)
    elapsed_ms = (time.monotonic() - t0) * 1000

    server_timing = result.get("server_timing", {}) if isinstance(result, dict) else {}

    if is_reset:
        _episode_log["reset_time_ms"] = elapsed_ms
    elif is_kv_cache:
        _episode_log["kv_cache_calls"].append({
            "client_time_ms": round(elapsed_ms, 2),
            "server_kv_ms": round(server_timing.get("kv_cache_ms", 0), 2),
        })
    else:
        _episode_log["infer_calls"].append({
            "client_time_ms": round(elapsed_ms, 2),
            # total server compute (video_denoise + action_denoise)
            "server_infer_ms": round(server_timing.get("infer_ms", 0), 2),
            # per-phase breakdown
            "server_video_denoise_ms": round(server_timing.get("video_denoise_ms", 0), 2),
            "server_action_denoise_ms": round(server_timing.get("action_denoise_ms", 0), 2),
            "server_vae_decode_ms": round(server_timing.get("vae_decode_ms", 0), 2),
            "video_steps": server_timing.get("video_steps", 0),
            "action_steps": server_timing.get("action_steps", 0),
            # round-trip overhead (network + client overhead) = client - server
            "network_overhead_ms": round(elapsed_ms - server_timing.get("infer_ms", 0) - server_timing.get("vae_decode_ms", 0), 2),
            "has_video": "video" in result if isinstance(result, dict) else False,
        })

    return result


WebsocketClientPolicy.infer = _patched_infer

# ─────────────────────────────────────────────────────────────────────
# 2. Monkey-patch eval_policy to capture per-episode metrics
# ─────────────────────────────────────────────────────────────────────

import evaluation.robotwin.eval_polict_client_openpi as _eval_mod

_original_eval_policy = _eval_mod.eval_policy

# Accumulator for all episode logs across the run
_all_episode_logs = []
_run_metadata = {}


def _patched_eval_policy(task_name, TASK_ENV, args, model, st_seed,
                         test_num=100, video_size=None,
                         instruction_type=None, save_visualization=False,
                         video_guidance_scale=5.0, action_guidance_scale=5.0):
    """
    Wraps the original eval_policy. We intercept by patching TASK_ENV methods
    to track per-episode data, then delegate to the original function.
    """
    global _all_episode_logs, _run_metadata

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
        "start_time": datetime.now().isoformat(),
    }

    # Patch TASK_ENV.take_action to count steps
    _step_counter = {"count": 0}
    _original_take_action = TASK_ENV.take_action

    @functools.wraps(_original_take_action)
    def _counting_take_action(*a, **kw):
        _step_counter["count"] += 1
        return _original_take_action(*a, **kw)

    TASK_ENV.take_action = _counting_take_action

    # Patch TASK_ENV.setup_demo to mark episode boundaries
    _original_setup_demo = TASK_ENV.setup_demo
    _episode_timer = {"start": None, "seed": None, "ep_num": None}

    @functools.wraps(_original_setup_demo)
    def _logging_setup_demo(now_ep_num=None, seed=None, is_test=True, **kw):
        # If there was a previous episode running, finalize it
        # (only if we had an infer call, meaning we actually ran a policy episode)
        if _episode_timer["start"] is not None and len(_episode_log["infer_calls"]) > 0:
            _finalize_episode(TASK_ENV, _episode_timer, _step_counter, task_name, args=args, st_seed=st_seed)

        # Start new episode tracking
        _step_counter["count"] = 0
        _reset_episode_log()
        _episode_timer["start"] = time.monotonic()
        _episode_timer["seed"] = seed
        _episode_timer["ep_num"] = now_ep_num

        return _original_setup_demo(now_ep_num=now_ep_num, seed=seed, is_test=is_test, **kw)

    TASK_ENV.setup_demo = _logging_setup_demo

    # Run the original eval_policy (wrapped in try/except to ensure logs are always saved)
    run_start = time.monotonic()
    result = None
    try:
        result = _original_eval_policy(
            task_name, TASK_ENV, args, model, st_seed,
            test_num=test_num, video_size=video_size,
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
        print(f"\033[91m[Logging] Saving partial logs before re-raising...\033[0m")
    finally:
        run_elapsed_s = time.monotonic() - run_start

        # Finalize the last episode if not yet done
        if _episode_timer["start"] is not None and len(_episode_log["infer_calls"]) > 0:
            try:
                _finalize_episode(TASK_ENV, _episode_timer, _step_counter, task_name, args=args, st_seed=st_seed)
            except Exception:
                pass

        _run_metadata["end_time"] = datetime.now().isoformat()
        _run_metadata["total_run_time_s"] = round(run_elapsed_s, 2)

        # Save detailed log (ALWAYS, even on crash)
        save_root = args.get("save_root", "results")
        log_output = {
            "run_metadata": _run_metadata,
            "episodes": _all_episode_logs,
            "summary": _compute_summary(_all_episode_logs),
        }

        log_dir = Path(save_root) / f"stseed-{st_seed}" / "detailed_logs" / task_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "detailed_log.json"
        with open(log_path, "w") as f:
            json.dump(log_output, f, indent=2, ensure_ascii=False)
        print(f"\n\033[96m[Logging] Detailed log saved to: {log_path}\033[0m")

        # Also save a CSV summary for quick analysis
        csv_path = log_dir / "episode_summary.csv"
        _save_csv_summary(_all_episode_logs, csv_path)
        print(f"\033[96m[Logging] CSV summary saved to: {csv_path}\033[0m")

        # Restore original methods
        TASK_ENV.take_action = _original_take_action
        TASK_ENV.setup_demo = _original_setup_demo

    # Re-raise if there was an error (after logs are saved)
    if result is None and "error" in _run_metadata:
        raise RuntimeError(f"eval_policy crashed: {_run_metadata['error']}")

    return result


def _finalize_episode(TASK_ENV, timer, step_counter, task_name, args=None, st_seed=None):
    """Collect all data for one completed episode and save incrementally."""
    elapsed_s = time.monotonic() - timer["start"]

    infer_calls = _episode_log["infer_calls"]
    kv_calls = _episode_log["kv_cache_calls"]

    # Aggregate timing from all infer calls
    def _sum(calls, key): return sum(c.get(key, 0) for c in calls)
    def _avg(calls, key): return round(_sum(calls, key) / max(len(calls), 1), 2)

    episode_record = {
        "episode_idx": len(_all_episode_logs),
        "task_name": task_name,
        "seed": timer["seed"],
        "success": bool(TASK_ENV.eval_success) if hasattr(TASK_ENV, 'eval_success') else None,
        "total_action_steps": step_counter["count"],
        "total_wall_time_s": round(elapsed_s, 2),

        # ── Inference chunk counts ──
        "num_infer_chunks": len(infer_calls),
        "num_kv_cache_calls": len(kv_calls),

        # ── Client-side total timing (includes network) ──
        "total_client_infer_time_s": round(_sum(infer_calls, "client_time_ms") / 1000, 3),
        "total_kv_cache_time_s": round(_sum(kv_calls, "client_time_ms") / 1000, 3),

        # ── Server-side phase breakdown (per chunk avg, in ms) ──
        "avg_server_video_denoise_ms": _avg(infer_calls, "server_video_denoise_ms"),
        "avg_server_action_denoise_ms": _avg(infer_calls, "server_action_denoise_ms"),
        "avg_server_vae_decode_ms": _avg(infer_calls, "server_vae_decode_ms"),
        "avg_server_kv_cache_ms": _avg(kv_calls, "server_kv_ms"),

        # ── Derived totals (across all chunks) ──
        "total_server_video_denoise_s": round(_sum(infer_calls, "server_video_denoise_ms") / 1000, 3),
        "total_server_action_denoise_s": round(_sum(infer_calls, "server_action_denoise_ms") / 1000, 3),
        "total_server_vae_decode_s": round(_sum(infer_calls, "server_vae_decode_ms") / 1000, 3),
        "total_server_infer_time_s": round(_sum(infer_calls, "server_infer_ms") / 1000, 3),
        "total_network_overhead_s": round(_sum(infer_calls, "network_overhead_ms") / 1000, 3),

        # ── Legacy compat ──
        "reset_time_ms": round(_episode_log["reset_time_ms"], 2) if _episode_log["reset_time_ms"] else None,
        "avg_chunk_infer_ms": _avg(infer_calls, "client_time_ms"),
        "avg_server_infer_ms": _avg(infer_calls, "server_infer_ms"),

        # ── Detailed per-call breakdown ──
        "infer_calls_detail": infer_calls,
        "kv_cache_calls_detail": kv_calls,
    }

    # Try to get prompt from TASK_ENV
    try:
        episode_record["prompt"] = TASK_ENV.get_instruction()
    except Exception:
        episode_record["prompt"] = None

    _all_episode_logs.append(episode_record)
    print(
        f"\033[93m[Logging] Episode {episode_record['episode_idx']} | "
        f"seed={timer['seed']} | "
        f"{'✓' if episode_record['success'] else '✗'} | "
        f"steps={step_counter['count']} | "
        f"chunks={episode_record['num_infer_chunks']} | "
        f"wall={episode_record['total_wall_time_s']}s | "
        f"video_dn={episode_record['total_server_video_denoise_s']}s | "
        f"action_dn={episode_record['total_server_action_denoise_s']}s | "
        f"kv={episode_record['total_kv_cache_time_s']}s\033[0m"
    )

    # ── Incremental save: write logs after EVERY episode ──
    if args is not None and st_seed is not None:
        try:
            save_root = args.get("save_root", "results")
            log_dir = Path(save_root) / f"stseed-{st_seed}" / "detailed_logs" / task_name
            log_dir.mkdir(parents=True, exist_ok=True)

            log_output = {
                "run_metadata": _run_metadata,
                "episodes": _all_episode_logs,
                "summary": _compute_summary(_all_episode_logs),
            }
            with open(log_dir / "detailed_log.json", "w") as f:
                json.dump(log_output, f, indent=2, ensure_ascii=False)

            _save_csv_summary(_all_episode_logs, log_dir / "episode_summary.csv")
        except Exception as e:
            print(f"\033[91m[Logging] Incremental save failed: {e}\033[0m")




def _compute_summary(episodes):
    """Compute aggregate statistics from all episodes."""
    if not episodes:
        return {}

    successes = sum(1 for e in episodes if e["success"])
    total = len(episodes)
    wall_times = [e["total_wall_time_s"] for e in episodes]
    infer_times = [e["total_client_infer_time_s"] for e in episodes]
    step_counts = [e["total_action_steps"] for e in episodes]
    chunk_counts = [e["num_infer_chunks"] for e in episodes]

    return {
        "total_episodes": total,
        "successes": successes,
        "success_rate": round(successes / total * 100, 1) if total > 0 else 0,
        "avg_wall_time_s": round(sum(wall_times) / total, 2),
        "avg_infer_time_s": round(sum(infer_times) / total, 2),
        "avg_steps": round(sum(step_counts) / total, 1),
        "avg_chunks": round(sum(chunk_counts) / total, 1),
        "min_wall_time_s": round(min(wall_times), 2),
        "max_wall_time_s": round(max(wall_times), 2),
        "total_wall_time_s": round(sum(wall_times), 2),
    }


def _save_csv_summary(episodes, csv_path):
    """Save a CSV file with one row per episode for quick analysis."""
    if not episodes:
        return

    fields = [
        "episode_idx", "task_name", "seed", "prompt", "success",
        "total_action_steps", "total_wall_time_s",
        "num_infer_chunks", "num_kv_cache_calls",
        # Client-side totals
        "total_client_infer_time_s", "total_kv_cache_time_s",
        # Server-side phase totals
        "total_server_video_denoise_s", "total_server_action_denoise_s",
        "total_server_vae_decode_s", "total_server_infer_time_s",
        "total_network_overhead_s",
        # Per-chunk averages
        "avg_server_video_denoise_ms", "avg_server_action_denoise_ms",
        "avg_server_vae_decode_ms", "avg_server_kv_cache_ms",
        "avg_chunk_infer_ms", "avg_server_infer_ms",
    ]

    with open(csv_path, "w") as f:
        f.write(",".join(fields) + "\n")
        for ep in episodes:
            row = []
            for field in fields:
                val = ep.get(field, "")
                # Escape commas in strings
                if isinstance(val, str):
                    val = f'"{val}"' if "," in val else val
                row.append(str(val) if val is not None else "")
            f.write(",".join(row) + "\n")



# ─────────────────────────────────────────────────────────────────────
# 3. Apply the eval_policy patch and run
# ─────────────────────────────────────────────────────────────────────

_eval_mod.eval_policy = _patched_eval_policy

if __name__ == "__main__":
    # Import and run the original main entrypoint.
    # The patches above are already active, so eval_policy will use our wrapper.
    from evaluation.robotwin.test_render import Sapien_TEST
    Sapien_TEST()
    usr_args = _eval_mod.parse_args_and_config()
    _eval_mod.main(usr_args)
