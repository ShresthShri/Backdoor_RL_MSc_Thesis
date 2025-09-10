#!/usr/bin/env python3
"""
Complete plotting code for dissertation-ready backdoor evaluation plots.
Fixes F2 innovation scaling while keeping all other plots.

F2: Time-series panel (one success ep): innovation vs τ, cumulative reward, base vs backdoor, speed proxy.
F3: Innovation distributions (benign vs attack pre/post) + inset metrics (KS/KL/JS/W1).
F4: Acceptance vs τ sweep; mark eval τ and annotate ASR@trigger (from summary).
F5: Latency distribution (box + jitter) for modes.

Usage:
  python plot_eval_panels.py results.json

Saves into figs/: f2_<mode>.png, f3_<mode>.png, f4_tradeoff.png, f5_latency.png
"""

import json, sys, os, numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy.stats import ks_2samp, entropy, wasserstein_distance

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 220,
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 14,
    "legend.fontsize": 10
})

EPS = 1e-9

# ---------- small helpers ----------
def _trace_to_arrays(tr):
    steps = np.array([int(t.get("step", i+1)) for i, t in enumerate(tr)], dtype=int)
    inv   = np.array([float(t.get("innovation", np.nan)) for t in tr], dtype=float)
    rtot  = np.array([float(t.get("reward_total", np.nan)) for t in tr], dtype=float)
    rbase = np.array([float(t.get("reward_base", np.nan)) if t.get("reward_base") is not None else np.nan for t in tr], dtype=float)
    rbd   = np.array([float(t.get("reward_backdoor", np.nan)) if t.get("reward_backdoor") is not None else np.nan for t in tr], dtype=float)
    xs    = np.array([float(t.get("x", np.nan)) if t.get("x") is not None else np.nan for t in tr], dtype=float)
    ys    = np.array([float(t.get("y", np.nan)) if t.get("y") is not None else np.nan for t in tr], dtype=float)
    return steps, inv, rtot, rbase, rbd, xs, ys

def _speed_proxy(xs, ys):
    # planar speed as stepwise displacement
    dx = np.diff(xs); dy = np.diff(ys)
    sp = np.hypot(dx, dy)
    # align length to steps[1:]
    return sp

def _get_eval_thr(cfg: Dict) -> float:
    return float(cfg.get("eval_innovation_threshold", 7.815))

def _pick_success_episode(reps: List[Dict]) -> Optional[int]:
    idx = [i for i,r in enumerate(reps) if r.get("success")]
    if idx: return idx[0]
    return None if not reps else 0

def _ecdf(a: np.ndarray):
    a = np.sort(a)
    y = np.arange(1, len(a)+1, dtype=float)/len(a) if len(a) else np.array([])
    return a, y

def _hist_pq(a: np.ndarray, b: np.ndarray, bins: int=60):
    if a.size == 0 or b.size == 0:
        return np.array([]), np.array([])
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins+1)
    p,_ = np.histogram(a, bins=edges, density=True)
    q,_ = np.histogram(b, bins=edges, density=True)
    p = p/(p.sum()+EPS); q = q/(q.sum()+EPS)
    return p, q

def _KL(p,q): return float(entropy(p+EPS, q+EPS)) if p.size and q.size else float("nan")
def _JS(p,q):
    if not p.size or not q.size: return float("nan")
    m = 0.5*(p+q)
    return 0.5*_KL(p,m) + 0.5*_KL(q,m)
def _KS(a,b): 
    if a.size == 0 or b.size == 0: return float("nan")
    return float(abs(ks_2samp(a,b).statistic))
def _W1(a,b): 
    if a.size == 0 or b.size == 0: return float("nan")
    return float(wasserstein_distance(a,b))

# ---------- F2: FIXED VERSION WITH CORRECT INNOVATION SCALE ----------
def fig_F2(data: Dict, mode_key: str, outdir="figs"):
    reps = data.get(mode_key, [])
    if not reps: return
    i = _pick_success_episode(reps)
    if i is None: i = 0
    rep = reps[i]
    steps, inv, rtot, rbase, rbd, xs, ys = _trace_to_arrays(rep.get("trace", []))
    thr = _get_eval_thr(data.get("config", {}))

    # build accept mask
    valid_inv = ~np.isnan(inv)
    acc = np.zeros_like(inv, dtype=bool)
    acc[valid_inv] = inv[valid_inv] < thr
    spd = _speed_proxy(xs, ys)

    fig = plt.figure(figsize=(8.2, 7.6))
    gs = fig.add_gridspec(4,1, hspace=0.35)

    # --- PANEL 1: Innovation (FIXED SCALE) ---
    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(steps, inv, lw=1.6, label="innovation", color='#2E86AB')
    ax1.axhline(thr, ls="--", lw=1.2, label="τ (eval)", color="k")
    
    # Mark accepts/rejects more subtly
    reject_mask = valid_inv & ~acc
    accept_mask = valid_inv & acc
    if np.any(reject_mask):
        ax1.scatter(steps[reject_mask], inv[reject_mask], s=20, alpha=0.5, 
                   color='red', marker='x', label="EKF reject")
    # Sample accepts to avoid overcrowding
    if np.any(accept_mask):
        # Show only 10% of accepts for visual clarity
        show_accepts = accept_mask & (np.random.random(len(steps)) < 0.1)
        if np.any(show_accepts):
            ax1.scatter(steps[show_accepts], inv[show_accepts], s=10, alpha=0.2, 
                       color='purple', marker='o', label="EKF accept")
    
    # Event markers
    for lab, st in [("spoof on", rep.get("spoof_on_step")), 
                     ("trigger", rep.get("trigger_step")), 
                     ("success", rep.get("success_step"))]:
        if st is not None:
            ax1.axvline(int(st), ls=":", lw=1.0, alpha=0.7)
            # Place text at different heights to avoid overlap
            y_pos = ax1.get_ylim()[1] * (0.9 if lab == "spoof on" else 0.95 if lab == "trigger" else 0.85)
            ax1.text(int(st), y_pos, f" {lab}", va="top", ha="left", fontsize=9)
    
    # Set y-limits based on actual data range
    max_inv = np.nanmax(inv) if np.any(valid_inv) else 10
    ax1.set_ylim(-0.1, max(max_inv * 1.1, 2.0))  # Ensure we can see the actual values
    
    ax1.set_ylabel("innovation")
    ax1.set_title(f"F2 — {mode_key}: time-series (episode {i})")
    ax1.legend(ncol=1, frameon=True, loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.2)

    # --- PANEL 2: Cumulative Reward ---
    ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
    ax2.plot(steps, rtot, lw=1.6)
    ax2.set_ylabel("cum. reward")
    ax2.grid(True, alpha=0.2)

    # --- PANEL 3: Reward Components ---
    ax3 = fig.add_subplot(gs[2,0], sharex=ax1)
    ax3.plot(steps, rbase, lw=1.6, label="reward_base")
    ax3.plot(steps, rbd,   lw=1.6, label="reward_backdoor")
    ax3.set_ylabel("reward comps")
    ax3.legend(frameon=True, loc="best")
    ax3.grid(True, alpha=0.2)

    # --- PANEL 4: Speed Proxy ---
    ax4 = fig.add_subplot(gs[3,0], sharex=ax1)
    ax4.plot(steps[1:], spd, lw=1.6)
    ax4.set_ylabel("speed (proxy)")
    ax4.set_xlabel("step")
    ax4.grid(True, alpha=0.2)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"f2_{mode_key}.png")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"✓ saved {path}")

# ---------- F3: Innovation CDFs (UNCHANGED) ----------
def fig_F3(data: Dict, mode_key: str, outdir="figs"):
    reps = data.get(mode_key, [])
    if not reps: return
    thr = _get_eval_thr(data.get("config", {}))

    # benign
    ben = []
    ct = data.get("clean_example_trace")
    if ct and "innovation" in ct:
        ben = np.array([float(x) for x in ct["innovation"]], dtype=float)
    ben = ben[~np.isnan(ben)] if len(ben) else np.array([], dtype=float)

    pre, post = [], []
    for r in reps:
        tr = r.get("trace", [])
        inv = np.array([float(t.get("innovation", np.nan)) for t in tr], dtype=float)
        steps = np.array([int(t.get("step", i+1)) for i, t in enumerate(tr)], dtype=int)
        trig = r.get("trigger_step", None)
        mask = ~np.isnan(inv)
        inv = inv[mask]; steps = steps[mask]
        if trig is None:
            pre.append(inv)
        else:
            pre.append(inv[steps < int(trig)])
            post.append(inv[steps >= int(trig)])
    pre = np.concatenate(pre) if pre else np.array([], dtype=float)
    post= np.concatenate(post) if post else np.array([], dtype=float)

    # metrics for inset
    p_pre, q_pre = _hist_pq(pre, ben)
    p_post,q_post= _hist_pq(post, ben)
    KL_pre  = _KL(p_pre,q_pre); JS_pre=_JS(p_pre,q_pre); KS_pre=_KS(pre,ben); W1_pre=_W1(pre,ben)
    KL_post = _KL(p_post,q_post);JS_post=_JS(p_post,q_post);KS_post=_KS(post,ben);W1_post=_W1(post,ben)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    # CDFs
    for a, lab in [(ben,"benign"), (pre,"attack pre"), (post,"attack post")]:
        x, y = _ecdf(a)
        ax.plot(x, y, lw=2.0, label=f"{lab} (n={len(a)})")
    ax.axvline(thr, ls="--", lw=1.0, color="k", label="τ (eval)")
    ax.set_xlabel("innovation")
    ax.set_ylabel("CDF")
    ax.set_title(f"F3 — {mode_key}: innovation CDFs")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, loc="lower right")

    # inset
    txt = (
        f"vs benign\n"
        f"pre:  KS={KS_pre:.3f}, KL={KL_pre:.3f}, JS={JS_pre:.3f}, W1={W1_pre:.3f}\n"
        f"post: KS={KS_post:.3f}, KL={KL_post:.3f}, JS={JS_post:.3f}, W1={W1_post:.3f}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8"))

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"f3_{mode_key}_cdf.png")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"✓ saved {path}")

# ---------- F4: Acceptance vs Threshold (UNCHANGED) ----------
def fig_F4(data: Dict, outdir="figs"):
    modes = [("obstacle_crash","Obstacle"), ("nfz_hover","NFZ")]
    reps_by_mode = {k:data.get(k,[]) for k,_ in modes}
    eval_thr = _get_eval_thr(data.get("config", {}))
    taus = np.linspace(3.0, 20.0, 50)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for key, label in modes:
        reps = reps_by_mode.get(key, [])
        if not reps: continue
        accs = []
        for tau in taus:
            rates = []
            for r in reps:
                inv = np.array([float(t.get("innovation", np.nan)) for t in r.get("trace", [])], dtype=float)
                inv = inv[~np.isnan(inv)]
                if inv.size == 0: continue
                rates.append(np.mean(inv < tau))
            accs.append(np.median(rates) if rates else np.nan)
        ax.plot(taus, accs, lw=2.0, label=label)

    ax.axvline(eval_thr, ls="--", lw=1.0, color="k", label="τ (eval)")
    
    # Annotate ASR at eval τ
    for key, label in modes:
        sblock = data.get("summary", {}).get(key, {})
        # Handle different possible keys for ASR
        asr = sblock.get("ASR_at_trigger", sblock.get("ASR@trigger", 0.0))
        if isinstance(asr, str) and '%' in asr:
            asr = float(asr.strip('%'))
        else:
            asr = float(asr) * 100.0
        
        y_pos = 0.05 if "Obstacle" in label else 0.15
        ax.text(eval_thr+0.2, y_pos,
                f"{label} ASR@τ={asr:.0f}%", fontsize=9)

    ax.set_xlabel("τ (innovation threshold)")
    ax.set_ylabel("Median EKF acceptance")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25)
    ax.set_title("F4 — Acceptance vs τ (ASR annotated at eval τ)")
    ax.legend(frameon=True, loc="lower right")

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "f4_tradeoff.png")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"✓ saved {path}")

# ---------- F5: Latency Distribution (UNCHANGED) ----------
def fig_F5(data: Dict, outdir="figs"):
    modes = [("obstacle_crash","Obstacle crash"), ("nfz_hover","NFZ hover")]
    vals, labels = [], []
    for key, lab in modes:
        reps = data.get(key, [])
        lat = [r.get("latency") for r in reps if r.get("latency") is not None]
        if len(lat) == 0: continue
        vals.append(np.array(lat, dtype=float))
        labels.append(lab)

    if not vals:
        print("No latency data found.")
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    # box
    bp = ax.boxplot(vals, labels=labels, showmeans=True, meanline=True, widths=0.5)
    # jitter points
    for i, v in enumerate(vals, start=1):
        x = np.random.normal(i, 0.05, size=len(v))
        ax.scatter(x, v, s=14, alpha=0.6)

    ax.set_ylabel("Latency (steps)")
    ax.set_title("F5 — Latency distribution by mode")
    ax.grid(axis="y", alpha=0.25)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "f5_latency.png")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"✓ saved {path}")

# ---------- BONUS: Print innovation statistics ----------
def print_innovation_stats(data: Dict):
    print("\n" + "="*60)
    print("Innovation Statistics (Actual Values)")
    print("="*60)
    
    for mode in ["obstacle_crash", "nfz_hover"]:
        reps = data.get(mode, [])
        all_innovations = []
        trigger_innovations = []
        
        for rep in reps:
            trigger_step = rep.get("trigger_step")
            for i, t in enumerate(rep.get("trace", [])):
                if "innovation" in t:
                    val = float(t["innovation"])
                    if np.isfinite(val):
                        all_innovations.append(val)
                        # Collect innovation at trigger point
                        if trigger_step and t.get("step") == trigger_step:
                            trigger_innovations.append(val)
        
        if all_innovations:
            print(f"\n{mode}:")
            print(f"  All steps:")
            print(f"    Min:     {np.min(all_innovations):.4f}")
            print(f"    Max:     {np.max(all_innovations):.4f}")
            print(f"    Median:  {np.median(all_innovations):.4f}")
            print(f"    95th %:  {np.percentile(all_innovations, 95):.4f}")
            print(f"    99th %:  {np.percentile(all_innovations, 99):.4f}")
            
            if trigger_innovations:
                print(f"  At trigger points:")
                print(f"    Min:     {np.min(trigger_innovations):.4f}")
                print(f"    Max:     {np.max(trigger_innovations):.4f}")
                print(f"    Median:  {np.median(trigger_innovations):.4f}")
            
            # Check against EKF threshold
            thr = _get_eval_thr(data.get("config", {}))
            pct_below = 100.0 * np.mean(np.array(all_innovations) < thr)
            print(f"  % below τ={thr:.1f}: {pct_below:.1f}%")

# ---------- main ----------
def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_eval_panels.py results.json")
        sys.exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs("figs", exist_ok=True)
    
    # Generate all figures
    # F2 - Time series (now with correct innovation scale)
    for m in ["obstacle_crash","nfz_hover"]:
        fig_F2(data, m)
    
    # F3 - Innovation CDFs
    for m in ["obstacle_crash","nfz_hover"]:
        fig_F3(data, m)
    
    # F4 - Acceptance vs threshold
    fig_F4(data)
    
    # F5 - Latency distribution
    fig_F5(data)
    
    # Print statistics to verify correct scaling
    print_innovation_stats(data)
    
    print("\n✅ All plots generated successfully!")

if __name__ == "__main__":
    main()