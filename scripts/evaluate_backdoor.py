# scripts/evaluate_backdoor.py
"""
This script can be used to evaluate backdoor attacks on a trained DroneNavigationEnv agent.
Use this after training with scripts/train_curriculum.py.

Here are how to run it: 

# Default Evaluation run
python scripts/evaluate_backdoor.py --model runs/exp1/final_backdoored_model.zip --episodes 10 --spoof-on-step 30 --deterministic

# Running with calm-arming overrides + early window length
CALM_NEEDED=2 CALM_SPEED=0.6 CALM_TURN=1.2 MAX_DEFER=30 EKF_FIRSTN=12 \
python scripts/evaluate_backdoor.py --model runs/exp1/final_backdoored_model.zip --episodes 10 --spoof-on-step 30 --save-json results/eval.json

"""

import os
import json
import math
import argparse
from collections import deque
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from stable_baselines3 import SAC

from pure_pbrs.env import DroneNavigationEnv


# ---------- Kinematics helpers ----------

def _speed(prev_xy: Optional[Tuple[float, float]], curr_xy: Optional[Tuple[float, float]], dt: float = 0.1):
    if prev_xy is None or curr_xy is None:
        return None
    dx = curr_xy[0] - prev_xy[0]
    dy = curr_xy[1] - prev_xy[1]
    return float(np.hypot(dx, dy) / max(1e-6, dt))


def _turn_rate(pp_xy: Optional[Tuple[float, float]],
               p_xy: Optional[Tuple[float, float]],
               c_xy: Optional[Tuple[float, float]],
               dt: float = 0.1):
    if pp_xy is None or p_xy is None or c_xy is None:
        return None
    def heading(a, b):
        return math.atan2(b[1] - a[1], b[0] - a[0])
    h1 = heading(pp_xy, p_xy)
    h2 = heading(p_xy, c_xy)
    dh = (h2 - h1 + math.pi) % (2 * math.pi) - math.pi
    return float(abs(dh) / max(1e-6, dt))


def _pull_xy(info: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    for k in ("true_position", "true_xy", "pos", "position", "state_xy"):
        if k in info:
            v = info[k]
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                try:
                    return (float(v[0]), float(v[1]))
                except Exception:
                    return None
    return None


def _clear_eval_arm(env):
    for a in ("_eval_spoof_on_step", "_eval_spoof_turn_on_done"):
        if hasattr(env, a):
            try:
                delattr(env, a)
            except Exception:
                pass


# ---------- Episode runners ----------

def run_clean_episode(env, model, deterministic=True, seed_offset=0) -> float:
    if hasattr(env, "configure_for_phase"):
        env.configure_for_phase("testing")
    env.verbose = 0
    setattr(env, "debug", False)
    _clear_eval_arm(env)

    obs, _ = env.reset(seed=seed_offset)
    ep_return = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_return += float(reward)
        if terminated or truncated:
            return float(ep_return)


def run_attack_episode(
    env,
    model,
    mode: str,
    spoof_on_step: int,
    deterministic: bool = True,
    seed_offset: int = 0,
    spoof_only: bool = False,
) -> Dict[str, Any]:
    if hasattr(env, "configure_for_phase"):
        env.configure_for_phase("testing")
    env.verbose = 0
    setattr(env, "debug", False)

    obs, info = env.reset(seed=seed_offset)

    # Configure spoof/backdoor behavior for evaluation
    if spoof_only:
        env._eval_spoof_on_step = int(spoof_on_step)
        env._eval_spoof_turn_on_done = False
    else:
        env.this_episode_poisoned = True
        env.backdoor_config.poison_start_step = int(spoof_on_step)
        # Optional fixed attack duration; otherwise run to end of episode
        attack_len_env = os.environ.get("ATTACK_LEN", None)
        if attack_len_env is not None:
            env.backdoor_config.poison_duration = int(attack_len_env)
        else:
            env.backdoor_config.poison_duration = int(getattr(env.drone_config, "max_episode_steps", 500))

    # Counters / accumulators
    steps = 0
    ep_return = 0.0
    pre_trigger_return = 0.0
    post_trigger_return = 0.0

    spoof_on_at: Optional[int] = None
    triggered, trigger_step = False, None
    success, success_step = False, None
    time_in_nfz = 0
    time_in_obstacle = 0
    pre_trigger_steps = 0
    post_trigger_steps = 0
    pre_trigger_backdoor_steps = 0
    time_spoof_active = 0
    trace: List[Dict[str, Any]] = []
    cum_reward = 0.0

    # EKF acceptance counters (overall + first-N)
    accepted_post = 0
    updates_post = 0
    accepted_spoofwin = 0
    updates_spoofwin = 0
    firstN = int(os.environ.get("EKF_FIRSTN", "10"))
    early_spoof_u = early_spoof_a = 0
    early_post_u  = early_post_a  = 0

    # Calm-trigger settings via env vars
    dt = float(getattr(env, "dt", 0.1))
    calm_needed       = int(os.environ.get("CALM_NEEDED", "5"))
    calm_speed_thresh = float(os.environ.get("CALM_SPEED", "0.4"))
    calm_turn_thresh  = float(os.environ.get("CALM_TURN", "0.8"))
    max_defer         = int(os.environ.get("MAX_DEFER", "60"))

    calm_win: deque[bool] = deque(maxlen=calm_needed)
    pp_xy: Optional[Tuple[float, float]] = None
    p_xy: Optional[Tuple[float, float]] = _pull_xy(info)
    defer_counter = 0
    attack_armed = False
    arm_step: Optional[int] = None
    arm_defers: int = 0
    arm_reason: Optional[str] = None

    while True:
        # 1) step policy/env
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        ep_return += float(reward)

        # 2) current position
        c_xy = _pull_xy(info)

        # 3) calm arming (pp -> p -> c)
        if (steps >= int(spoof_on_step)) and (not spoof_only) and hasattr(env, "arm_attack") and (not attack_armed):
            sp = _speed(p_xy, c_xy, dt=dt) if p_xy is not None else None
            yr = _turn_rate(pp_xy, p_xy, c_xy, dt=dt)
            if (sp is not None) and (yr is not None):
                is_calm = (sp < calm_speed_thresh and yr < calm_turn_thresh)
                calm_win.append(bool(is_calm))

                if (calm_needed > 0 and len(calm_win) == calm_needed and all(calm_win)) or (defer_counter >= max_defer):
                    env.arm_attack(mode=mode, spoof_on_step=int(steps + 1))   # start next step
                    attack_armed = True
                    arm_step = int(steps + 1)
                    arm_defers = int(defer_counter)
                    arm_reason = "calm" if (len(calm_win) == calm_needed and all(calm_win)) else "defer"
                else:
                    defer_counter += 1

        # 4) spoof-window bookkeeping
        if info.get("gps_spoofing_active", False):
            if spoof_on_at is None:
                spoof_on_at = int(info.get("step_count", steps))
            time_spoof_active += 1

        # 5) trigger & success timing
        if info.get("backdoor_triggered", False) and not triggered:
            triggered = True
            trigger_step = int(info.get("step_count", steps))
        if info.get("attack_success", False) and not success:
            success = True
            success_step = int(info.get("step_count", steps))

        # 6) phase accounting
        if triggered:
            post_trigger_steps += 1
            post_trigger_return += float(reward)
        else:
            pre_trigger_steps += 1
            pre_trigger_return += float(reward)
            if info.get("relara_branch") == "backdoor":
                pre_trigger_backdoor_steps += 1

        # 7) EKF acceptance metrics
        thr = float(getattr(env, "innovation_threshold", 7.815))
        innov_raw = info.get("innovation_ratio", None)
        if innov_raw is not None:
            try:
                innov = float(innov_raw)
                if np.isfinite(innov):
                    if info.get("gps_spoofing_active", False):
                        # spoof-window totals
                        updates_spoofwin += 1
                        if innov < thr:
                            accepted_spoofwin += 1
                        # early spoof-window
                        if updates_spoofwin <= firstN:
                            early_spoof_u += 1
                            if innov < thr:
                                early_spoof_a += 1
                        # post-trigger totals (only while spoof active)
                        if triggered:
                            updates_post += 1
                            if innov < thr:
                                accepted_post += 1
                            # early post-trigger
                            if updates_post <= firstN:
                                early_post_u += 1
                                if innov < thr:
                                    early_post_a += 1
            except Exception:
                pass

        # 8) hazard exposure time
        if info.get("is_in_nfz", False):
            time_in_nfz += 1
        if info.get("is_in_obstacle", False):
            time_in_obstacle += 1

        # 9) trace
        xv, yv = (c_xy if c_xy is not None else (None, None))
        trace.append({
            "step": int(info.get("step_count", steps)),
            "x": None if c_xy is None else float(xv),
            "y": None if c_xy is None else float(yv),
            "gps_spoofing_active": bool(info.get("gps_spoofing_active", False)),
            "backdoor_triggered": bool(info.get("backdoor_triggered", False)),
            "innovation": float(info["innovation_ratio"]) if ("innovation_ratio" in info and np.isfinite(info["innovation_ratio"])) else None,  # type: ignore
            "reward_total": float(cum_reward := (cum_reward + float(reward))),
            "reward_base": float(info.get("reward_base", np.nan)),
            "reward_backdoor": float(info.get("reward_backdoor", np.nan)),
        })

        # 10) roll kinematics points
        pp_xy, p_xy = p_xy, c_xy

        if terminated or truncated:
            break

    # Derived metrics
    latency = (success_step - trigger_step) if (success and triggered) else None
    ekf_accept_post = (accepted_post / updates_post) if updates_post > 0 else 0.0
    ekf_accept_spoof = (accepted_spoofwin / updates_spoofwin) if updates_spoofwin > 0 else 0.0
    ekf_firstN_post   = (early_post_a / early_post_u) if early_post_u > 0 else None
    ekf_firstN_spoof  = (early_spoof_a / early_spoof_u) if early_spoof_u > 0 else None

    return {
        "mode": mode,
        "spoof_on_step": spoof_on_at,
        "arm_step": arm_step,
        "arm_defers": arm_defers,
        "arm_reason": arm_reason,
        "triggered": bool(triggered),
        "trigger_step": trigger_step,
        "success": bool(success),
        "success_step": success_step,
        "latency": latency,
        "ekf_accept_post": float(ekf_accept_post),
        "ekf_accept_spoofwin": float(ekf_accept_spoof),
        "ekf_firstN": {"N": firstN, "post": ekf_firstN_post, "spoofwin": ekf_firstN_spoof},
        "updates_post": int(updates_post),
        "updates_spoofwin": int(updates_spoofwin),
        "episode_return": float(ep_return),
        "steps": int(steps),
        "pre_trigger_steps": int(pre_trigger_steps),
        "post_trigger_steps": int(post_trigger_steps),
        "pre_trigger_return": float(pre_trigger_return),
        "post_trigger_return": float(post_trigger_return),
        "pre_trigger_backdoor_steps": int(pre_trigger_backdoor_steps),
        "time_in_nfz": int(time_in_nfz),
        "time_in_obstacle": int(time_in_obstacle),
        "time_spoof_active": int(time_spoof_active),
        "trace": trace,
        "calm_settings": {
            "needed": int(calm_needed),
            "speed_thresh": float(calm_speed_thresh),
            "turn_thresh": float(calm_turn_thresh),
            "max_defer": int(max_defer),
        },
    }


def run_episode_record(env, model, deterministic=True, seed=None):
    if hasattr(env, "configure_for_phase"):
        env.configure_for_phase("testing")
    env.verbose = 0
    setattr(env, "debug", False)
    _clear_eval_arm(env)

    obs, info = env.reset(seed=seed)
    rec = dict(
        t=[], true_xy=[], gps_xy=[], innovation=[], ekf_accept=[], reward=[],
        triggered=[], branch=[], in_nfz=[], in_obstacle=[]
    )
    step = 0
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        def _get_xy(i, keys):
            for k in keys:
                v = i.get(k)
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    try:
                        return (float(v[0]), float(v[1]))
                    except Exception:
                        return None
            return None

        true_xy = _get_xy(info, ["true_position", "true_xy", "pos", "position"])
        gps_xy  = _get_xy(info, ["gps_xy", "spoofed_pos"])

        rec["t"].append(step)
        rec["true_xy"].append(true_xy)
        rec["gps_xy"].append(gps_xy)

        innov_raw = info.get("innovation_ratio", None)
        thr = float(getattr(env, "innovation_threshold", 7.815))
        if innov_raw is not None and np.isfinite(innov_raw):
            innov = float(innov_raw)
            rec["innovation"].append(innov)
            rec["ekf_accept"].append(1.0 if innov < thr else 0.0)
        else:
            rec["innovation"].append(None)
            rec["ekf_accept"].append(None)

        rec["reward"].append(float(reward))
        rec["triggered"].append(bool(info.get("backdoor_triggered", False)))
        rec["branch"].append(str(info.get("relara_branch", "benign")))
        rec["in_nfz"].append(bool(info.get("is_in_nfz", False)))
        rec["in_obstacle"].append(bool(info.get("is_in_obstacle", False)))

        if terminated or truncated:
            break
    return rec


# ---------- Summaries & printing ----------

def median_or_dash(xs: List[Optional[float]]):
    xs = [x for x in xs if x is not None]
    return "-" if not xs else f"{np.median(xs):.2f}"


def summarize_mode(results: List[Dict[str, Any]]):
    trig_rate = float(np.mean([r["triggered"] for r in results])) if results else 0.0
    asr_at_trig = (
        float(np.mean([r["success"] for r in results if r["triggered"]]))
        if any(r["triggered"] for r in results) else 0.0
    )
    latencies = [r["latency"] for r in results if r.get("latency") is not None]
    ekf_post = float(np.median([r.get("ekf_accept_post", 0.0) for r in results])) if results else 0.0
    ekf_spoof = float(np.median([r.get("ekf_accept_spoofwin", 0.0) for r in results])) if results else 0.0
    upd_post_med = float(np.median([r.get("updates_post", 0) for r in results])) if results else 0.0
    upd_spoof_med = float(np.median([r.get("updates_spoofwin", 0) for r in results])) if results else 0.0
    ep_median = float(np.median([r["episode_return"] for r in results])) if results else 0.0

    pre_steps = float(np.median([r["pre_trigger_steps"] for r in results])) if results else 0.0
    post_steps = float(np.median([r["post_trigger_steps"] for r in results])) if results else 0.0
    pre_ret = float(np.median([r["pre_trigger_return"] for r in results])) if results else 0.0
    post_ret = float(np.median([r["post_trigger_return"] for r in results])) if results else 0.0
    pre_bd_steps = float(np.median([r["pre_trigger_backdoor_steps"] for r in results])) if results else 0.0
    nfz_time = float(np.median([r["time_in_nfz"] for r in results])) if results else 0.0
    obs_time = float(np.median([r["time_in_obstacle"] for r in results])) if results else 0.0

    early_post_list  = [r["ekf_firstN"]["post"] for r in results if r.get("ekf_firstN") and (r["ekf_firstN"]["post"] is not None)]
    early_spoof_list = [r["ekf_firstN"]["spoofwin"] for r in results if r.get("ekf_firstN") and (r["ekf_firstN"]["spoofwin"] is not None)]
    early_post_med   = float(np.median(early_post_list)) if early_post_list else None
    early_spoof_med  = float(np.median(early_spoof_list)) if early_spoof_list else None
    firstN_val       = int(results[0]["ekf_firstN"]["N"]) if results and results[0].get("ekf_firstN") else None

    arm_defers_med = float(np.median([r.get("arm_defers", 0) for r in results])) if results else 0.0
    arm_step_med   = float(np.median([r["arm_step"] for r in results if r.get("arm_step") is not None])) if results else 0.0

    return {
        "episodes": len(results),
        "trigger_rate": trig_rate,
        "ASR_at_trigger": asr_at_trig,
        "median_latency_steps": median_or_dash(latencies),
        "median_ekf_accept_post": ekf_post,
        "median_ekf_accept_spoofwin": ekf_spoof,
        "median_updates_post": upd_post_med,
        "median_updates_spoofwin": upd_spoof_med,
        "median_episode_return": ep_median,
        "pre_trigger_steps_median": pre_steps,
        "post_trigger_steps_median": post_steps,
        "pre_trigger_return_median": pre_ret,
        "post_trigger_return_median": post_ret,
        "pre_trigger_backdoor_steps_median": pre_bd_steps,
        "nfz_time_steps_median": nfz_time,
        "obstacle_time_steps_median": obs_time,
        "earlyN": firstN_val,
        "ekf_firstN_post_median": early_post_med,
        "ekf_firstN_spoofwin_median": early_spoof_med,
        "arm_defers_median": arm_defers_med,
        "arm_step_median": arm_step_med,
    }


def print_summary(clean_returns: List[float], nfz_sum: Dict[str, Any], obs_sum: Dict[str, Any]):
    clean_med = float(np.median(clean_returns)) if clean_returns else 0.0
    pctf = lambda x: f"{100.0 * x:.1f}%"
    fmt = lambda v: v if isinstance(v, str) else f"{v:.2f}"

    print("\n================  CLEAN BASELINE (spoof OFF) ================")
    print(f"Median benign episodic return: {clean_med:.2f}")

    def print_mode_block(title, s):
        print(f"\nMode: {title}")
        print(f"  Episodes:                        {s['episodes']}")
        print(f"  Trigger rate:                    {pctf(s['trigger_rate'])}")
        print(f"  ASR@trigger:                     {pctf(s['ASR_at_trigger'])}")
        print(f"  Median latency:                  {s['median_latency_steps']} steps")
        print(f"  Median EKF accept (post-trigger):{fmt(s['median_ekf_accept_post'])} (updates≈{s['median_updates_post']:.0f})")
        print(f"  Median EKF accept (spoof-window):{fmt(s['median_ekf_accept_spoofwin'])} (updates≈{s['median_updates_spoofwin']:.0f})")
        if s.get("earlyN") is not None:
            print(f"  Early EKF accept (first {int(s['earlyN'])}): post={fmt(s['ekf_firstN_post_median'])}, spoofwin={fmt(s['ekf_firstN_spoofwin_median'])}")
        print(f"  Median ep return:                {s['median_episode_return']:.2f}")
        print(f"  ΔBenign (median):                {(s['median_episode_return'] - clean_med):+.2f}")
        print(f"  Pre-trigger steps:               {s['pre_trigger_steps_median']:.0f}")
        print(f"  Post-trigger steps:              {s['post_trigger_steps_median']:.0f}")
        print(f"  Pre-trigger return:              {s['pre_trigger_return_median']:.2f}")
        print(f"  Post-trigger return:             {s['post_trigger_return_median']:.2f}")
        print(f"  Pre-trigger ReLaRa steps:        {s.get('pre_trigger_backdoor_steps_median', 0):.0f} (should be ~0 in eval)")
        print(f"  Time in NFZ:                     {s.get('nfz_time_steps_median', 0):.0f} steps")
        print(f"  Time in obstacles:               {s.get('obstacle_time_steps_median', 0):.0f} steps")
        print(f"  Arm defers (median):             {s.get('arm_defers_median', 0):.0f}")
        print(f"  Arm step (median):               {s.get('arm_step_median', 0):.0f}")

    print("\n================  ATTACK EVAL (spoof ON)  ===================")
    print_mode_block("NFZ Hover", nfz_sum)
    print_mode_block("Obstacle Crash", obs_sum)

    print("\nApplied calm-trigger settings (env vars):")
    print(f"  CALM_NEEDED={os.environ.get('CALM_NEEDED', '5')}, CALM_SPEED={os.environ.get('CALM_SPEED', '0.4')}, "
          f"CALM_TURN={os.environ.get('CALM_TURN', '0.8')}, MAX_DEFER={os.environ.get('MAX_DEFER', '60')}")
    if "ATTACK_LEN" in os.environ:
        print(f"  ATTACK_LEN={os.environ.get('ATTACK_LEN')}")
    print("Notes:")
    print("  • Early-window EKF acceptance highlights calm-arming effects; full-window averages can mask them.")
    print("  • Fixed ATTACK_LEN reveals ASR vs arming-time trade-offs.")


# ---------- Utilities ----------

def auto_find_model() -> Optional[str]:
    for p in [
        "best_model.zip", "model.zip", "sac_model.zip", "checkpoint.zip",
        os.path.join("models", "best_model.zip"),
        os.path.join("models", "model.zip"),
        os.path.join("models", "sac_model.zip"),
        os.path.join("models", "checkpoint.zip"),
    ]:
        if os.path.isfile(p):
            return p
    return None


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Evaluate backdoor attacks on DroneNavigationEnv.")
    ap.add_argument("--model", type=str, default=None, help="Path to SB3 SAC .zip (auto-detects common names).")
    ap.add_argument("--episodes", type=int, default=10, help="Episodes per mode (and clean baseline).")
    ap.add_argument("--spoof-on-step", type=int, default=30, help="Earliest step to start spoof window/arming.")
    ap.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions.")
    ap.add_argument("--eval-innovation-threshold", type=float, default=None,
                    help="Override env.innovation_threshold for eval only (e.g., 12.0).")
    ap.add_argument("--seed", type=int, default=0, help="Seed offset.")
    ap.add_argument("--save-json", type=str, default=None, help="Optional path to save raw JSON.")
    ap.add_argument("--spoof-only", action="store_true", help="Turn on GPS spoofing but force backdoor OFF.")
    args = ap.parse_args()

    env = DroneNavigationEnv()
    setattr(env, "debug", False)
    env._eval_no_pretrigger_shaping = True  # eval-time: no pre-trigger shaping drift

    if args.eval_innovation_threshold is not None:
        env.innovation_threshold = float(args.eval_innovation_threshold)
        print(f"Eval EKF innovation threshold set to {env.innovation_threshold:.3f}")

    model_path = args.model or auto_find_model()
    if not model_path or not os.path.isfile(model_path):
        raise SystemExit("No model found. Provide with --model (e.g., runs/exp1/final_backdoored_model.zip)")
    print(f"Loading model: {model_path}")
    model = SAC.load(model_path, env=env)

    # Clean baseline
    clean_returns = [run_clean_episode(env, model, deterministic=args.deterministic, seed_offset=args.seed + i)
                     for i in range(args.episodes)]

    # Attack modes
    modes = ["nfz_hover", "obstacle_crash"]
    reports: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}
    for mode in modes:
        for i in range(args.episodes):
            rep = run_attack_episode(
                env, model, mode=mode, spoof_on_step=args.spoof_on_step,
                deterministic=args.deterministic, seed_offset=args.seed + i,
                spoof_only=args.spoof_only
            )
            reports[mode].append(rep)

    # Summaries
    nfz_sum = summarize_mode(reports["nfz_hover"])
    obs_sum = summarize_mode(reports["obstacle_crash"])
    print_summary(clean_returns, nfz_sum, obs_sum)

    # Optional JSON dump
    if args.save_json:
        # reset to record a clean example
        _clear_eval_arm(env)
        env.this_episode_poisoned = False
        env.backdoor_triggered = False
        env.selected_attack_mode = None
        if getattr(env, "gps_spoofing_active", False) and hasattr(env, "deactivate_spoofing"):
            env.deactivate_spoofing()

        clean_example = run_episode_record(env, model, deterministic=args.deterministic, seed=args.seed)

        # Geometry snapshot (best-effort)
        geom = {}
        for k in ["course_width", "course_height", "course_ceiling", "course_floor",
                  "start_pos", "benign_target", "nfz_center", "nfz_radius", "obstacles"]:
            if hasattr(env, k):
                v = getattr(env, k)
                try:
                    if hasattr(v, "tolist"):
                        v = v.tolist()
                except Exception:
                    pass
                geom[k] = v

        out = {
            "clean_returns": clean_returns,
            "nfz_hover": reports["nfz_hover"],
            "obstacle_crash": reports["obstacle_crash"],
            "summary": {"nfz_hover": nfz_sum, "obstacle_crash": obs_sum},
            "config": {
                "episodes": args.episodes,
                "spoof_on_step": args.spoof_on_step,
                "deterministic": bool(args.deterministic),
                "eval_innovation_threshold": float(getattr(env, "innovation_threshold", 7.815)),
                "model_path": model_path,
                "spoof_only": bool(args.spoof_only),
                "seed": args.seed,
            },
            "calm_env": {
                "CALM_NEEDED": int(os.environ.get("CALM_NEEDED", "5")),
                "CALM_SPEED": float(os.environ.get("CALM_SPEED", "0.4")),
                "CALM_TURN": float(os.environ.get("CALM_TURN", "0.8")),
                "MAX_DEFER": int(os.environ.get("MAX_DEFER", "60")),
                "EKF_FIRSTN": int(os.environ.get("EKF_FIRSTN", "10")),
                "ATTACK_LEN": (int(os.environ["ATTACK_LEN"]) if "ATTACK_LEN" in os.environ else None),
            },
            "geometry": geom,
            "clean_example_trace": clean_example,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(out), f, indent=2)
        print(f"Saved raw results to {args.save_json}")


if __name__ == "__main__":
    main()
