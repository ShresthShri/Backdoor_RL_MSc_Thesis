#!/usr/bin/env python3
import json, sys, os
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.patheffects as pe

# ---------- aesthetics (crisp + modern) ----------
def apply_crisp_style(theme: str = "light") -> Dict[str, str]:
    # Color tokens (light/dark aware)
    if theme.lower() == "dark":
        bg = "#0f1116"
        fg = "#e6e6e6"
        grid = "#3a3f4b"
        spine = "#b9b9b9"
        benign = "#111111"  # will be overridden in dark with light gray
        succ = "#5ac568"    # green
        fail = "#f08a4b"    # warm orange
        nfz_fill = "#ff5a5a"
        nfz_edge = "#b30000"
        obs_fill = "#9aa0a6"
        obs_edge = "#6b6f76"
        benign = "#d0d0d0"
    else:
        bg = "#ffffff"
        fg = "#24292f"
        grid = "#d0d7de"
        spine = "#6e7781"
        succ = "#2ca02c"    # green
        fail = "#cc5a00"    # warm orange
        nfz_fill = "#ff4d4d"
        nfz_edge = "#b30000"
        obs_fill = "#a9b0b7"
        obs_edge = "#66707a"
        benign = "#111111"

    style = {
        # DPI & vector-friendly settings
        "figure.dpi": 160,
        "savefig.dpi": 320,
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        # Fonts
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "font.sans-serif": ["Inter", "SF Pro Text", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        # Lines & markers
        "lines.antialiased": True,
        "lines.solid_joinstyle": "round",
        "lines.solid_capstyle": "round",
        # Axes
        "axes.edgecolor": spine,
        "axes.linewidth": 0.8,
        "axes.titlepad": 8.0,
        "axes.labelcolor": fg,
        # Ticks
        "xtick.color": fg,
        "ytick.color": fg,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        # Grid
        "axes.grid": True,
        "grid.color": grid,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.facecolor": bg,
        "legend.edgecolor": spine,
        # Save as TrueType for Illustrator/Keynote compatibility
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Path simplification for faster renders without losing fidelity
        "path.simplify": True,
        "path.simplify_threshold": 0.0,
    }
    mpl.rcParams.update(style)
    # Return a palette used throughout
    return {
        "fg": fg,
        "succ": succ,
        "fail": fail,
        "nfz_fill": nfz_fill,
        "nfz_edge": nfz_edge,
        "obs_fill": obs_fill,
        "obs_edge": obs_edge,
        "benign": benign,
        "start": "#111111" if theme.lower() != "dark" else "#f0f0f0",
        "goal": "#ffd700",
        "ekf": "#000000" if theme.lower() != "dark" else "#f0f0f0",
    }

# ---------- helpers ----------
def _as_float_pair(x):
    if x is None: return (np.nan, np.nan)
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return float(x[0]), float(x[1])
    return (np.nan, np.nan)

def _trace_xy(trace) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, steps = [], [], []
    for i, t in enumerate(trace):
        if "x" in t and "y" in t:
            x, y = t["x"], t["y"]
        else:
            x, y = _as_float_pair(t.get("true_xy"))
        xs.append(float(x) if x is not None else np.nan)
        ys.append(float(y) if y is not None else np.nan)
        steps.append(int(t.get("step", i+1)))
    return np.array(xs), np.array(ys), np.array(steps)

def _find_index_for_step(steps: np.ndarray, step: Optional[int]) -> Optional[int]:
    if step is None: return None
    idx = np.where(steps == int(step))[0]
    return int(idx[0]) if len(idx) else None

def _pick_episodes(reports: List[Dict], k=5) -> List[int]:
    succ = [i for i,r in enumerate(reports) if r.get("success")]
    rest = [i for i,r in enumerate(reports) if not r.get("success")]
    chosen = succ[:k] + rest[:max(0, k-len(succ))]
    return chosen[:k]

def _get_eval_threshold(data: Dict) -> float:
    cfg = data.get("config", {})
    thr = cfg.get("eval_innovation_threshold")
    return 7.815 if (thr is None) else float(thr)

def _get_geometry(data: Dict) -> Dict:
    g = data.get("geometry", {}) or data.get("config", {}).get("geometry", {})
    out = {
        "course_width": g.get("course_width", 2.0),
        "course_height": g.get("course_height", 2.0),
        "start_pos": g.get("start_pos", [0.3, 0.3, 0.15]),
        "benign_target": g.get("benign_target", [1.7, 1.7, 0.15]),
        "nfz_center": g.get("nfz_center", [1.0, 1.0]),
        "nfz_radius": g.get("nfz_radius", 0.25),
        "obstacles": g.get("obstacles", [
            {'center': [0.7, 0.5], 'width': 0.1, 'height': 0.1},
            {'center': [1.3, 0.8], 'width': 0.08, 'height': 0.12},
            {'center': [0.8, 0.8], 'width': 0.15, 'height': 0.15},
            {'center': [1.4, 1.4], 'width': 0.12, 'height': 0.12},
        ])
    }
    return out

def _build_proxy_benign_from_pretrigger(rep: Dict) -> Dict:
    tr = rep.get("trace", [])
    xs, ys, steps = _trace_xy(tr)
    trig = rep.get("trigger_step")
    if trig is None:
        return {"t": steps.tolist(), "x": xs.tolist(), "y": ys.tolist()}
    mask = steps <= int(trig)
    if not mask.any(): mask = np.arange(len(steps)) < max(1, len(steps)//4)
    return {"t": steps[mask].tolist(), "x": xs[mask].tolist(), "y": ys[mask].tolist()}

def _clean_trace_from_json(data: Dict) -> Optional[Dict]:
    ct = data.get("clean_example_trace")
    if ct and "true_xy" in ct:
        xs, ys = [], []
        for p in ct["true_xy"]:
            x, y = _as_float_pair(p)
            xs.append(x); ys.append(y)
        t = ct.get("t", list(range(1, len(xs)+1)))
        return {"t": t, "x": xs, "y": ys}
    return None

# ---------- plotting ----------
def draw_map(ax, geom, C):
    # NFZ (filled, translucent)
    cx, cy = geom["nfz_center"]; r = geom["nfz_radius"]
    nfz = Circle((cx, cy), r, facecolor=C["nfz_fill"], edgecolor=C["nfz_edge"],
                    linewidth=1.2, linestyle="--", alpha=0.18, label="NFZ")
    ax.add_patch(nfz)
    # obstacles (filled light gray)
    for ob in geom["obstacles"]:
        ox, oy = ob["center"]
        w, h = ob["width"], ob["height"]
        rect = Rectangle((ox - w/2, oy - h/2), w, h, facecolor=C["obs_fill"],
                            edgecolor=C["obs_edge"], linewidth=0.8, alpha=0.22,
                            label="_nolegend_")
        ax.add_patch(rect)
    # start/goal
    sx, sy = geom["start_pos"][:2]
    gx, gy = geom["benign_target"][:2]
    ax.scatter([sx],[sy], marker="s", s=90, linewidths=0.9, edgecolor=C["fg"], facecolor="#ffffff", label="start", zorder=5,
                path_effects=[pe.withStroke(linewidth=1.2, foreground="#ffffff")])
    ax.scatter([gx],[gy], marker="*", s=150, linewidths=0.9, edgecolor=C["fg"], facecolor=C["goal"], label="goal", zorder=5,
                path_effects=[pe.withStroke(linewidth=1.2, foreground="#ffffff")])

def overlay_xy(results_path: str,
                section="obstacle_crash",
                k_episodes=5,
                deviation_threshold=0.07,
                outdir="figs",
                theme="light",
                outformats=("png","pdf")):
    os.makedirs(outdir, exist_ok=True)
    C = apply_crisp_style(theme=theme)
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    geom = _get_geometry(data)
    thr = _get_eval_threshold(data)

    # benign reference (prefer clean trace; else pre-trigger of first attack ep)
    clean = _clean_trace_from_json(data)
    if clean is None:
        first = data.get(section, [{}])[0] if data.get(section) else {}
        clean = _build_proxy_benign_from_pretrigger(first) if first else {"t": [], "x": [], "y": []}

    bx = np.array(clean.get("x", []), dtype=float)
    by = np.array(clean.get("y", []), dtype=float)
    bt = np.array(clean.get("t", []), dtype=int) if clean.get("t") else np.array([], dtype=int)

    reps = data.get(section, [])
    idxs = _pick_episodes(reps, k=k_episodes)

    # Use constrained_layout for better spacing
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6.8, 6.8))
    fig.patch.set_facecolor(mpl.rcParams["figure.facecolor"])
    draw_map(ax, geom, C)

    # benign path (thick black/gray for contrast)
    ax.plot(bx, by, linewidth=3.0, color=C["benign"], alpha=0.9, label="benign path", zorder=3)

    # de-cluttered legend helper
    shown = set()
    def once(label):
        if label in shown: return "_nolegend_"
        shown.add(label); return label

    # colors for success/fail
    col_success = C["succ"]
    col_fail    = C["fail"]

    for ei in idxs:
        rep = reps[ei]
        tr = rep.get("trace", [])
        xs, ys, ts = _trace_xy(tr)
        color = col_success if rep.get("success") else col_fail

        # trajectory
        ax.plot(xs, ys, linewidth=1.9, alpha=0.95, color=color, label=once("attack traj"), zorder=4)

        # markers
        sidx = _find_index_for_step(ts, rep.get("spoof_on_step"))
        tidx = _find_index_for_step(ts, rep.get("trigger_step"))
        uidx = _find_index_for_step(ts, rep.get("success_step"))

        if sidx is not None and sidx < len(xs):
            ax.scatter([xs[sidx]],[ys[sidx]], s=52, color=color, edgecolor=C["fg"], linewidths=0.6, zorder=6,
                        label=once("spoof on"),
                        path_effects=[pe.withStroke(linewidth=1.2, foreground="#ffffff")])
        if tidx is not None and tidx < len(xs):
            ax.scatter([xs[tidx]],[ys[tidx]], marker="x", s=80, color=color, zorder=7,
                        label=once("trigger"))
        if uidx is not None and uidx < len(xs):
            ax.scatter([xs[uidx]],[ys[uidx]], marker="+", s=110, color=color, zorder=7,
                        label=once("success"))

        # EKF accept/reject dots (rasterized for smaller vector files, still crisp)
        inov = np.array([float(t.get("innovation", np.nan)) for t in tr], dtype=float)
        acc = inov < thr
        if xs.size:
            ax.scatter(xs[acc],   ys[acc],   s=7, alpha=0.08, color=C["ekf"], label=once("EKF accept"),
                        rasterized=True, zorder=2)
            ax.scatter(xs[~acc],  ys[~acc],  s=9, alpha=0.28, color=C["ekf"], label=once("EKF reject"),
                        rasterized=True, zorder=2)

        # deviation marker vs benign (post-trigger)
        if tidx is not None and bt.size > 0:
            common, ia, ib = np.intersect1d(ts, bt, return_indices=True)
            mask = ts[ia] >= ts[tidx]
            ia2 = ia[mask]; ib2 = ib[mask]
            if ia2.size > 0:
                dists = np.hypot(xs[ia2] - bx[ib2], ys[ia2] - by[ib2])
                over = np.where(dists > deviation_threshold)[0]
                if over.size > 0:
                    kdev = over[0]
                    ax.scatter([xs[ia2[kdev]]],[ys[ia2[kdev]]], marker="^", s=70,
                                color=color, zorder=8, label=once("deviation"),
                                path_effects=[pe.withStroke(linewidth=1.2, foreground="#ffffff")])
                    ax.plot([xs[ia2[kdev]], bx[ib2[kdev]]],
                            [ys[ia2[kdev]], by[ib2[kdev]]],
                            linestyle=(0,(3,3)), linewidth=1.0, color=color, alpha=0.7, label="_nolegend_")
                    txt = ax.text(xs[ia2[kdev]], ys[ia2[kdev]]+0.03, f"{dists[kdev]:.02f} m",
                                    fontsize=9, color=color, ha="center", va="bottom",
                                    zorder=9)
                    txt.set_path_effects([pe.withStroke(linewidth=2.4, foreground="#ffffff")])

    # frame / labels
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, geom["course_width"])
    ax.set_ylim(0.0, geom["course_height"])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    # turn off top/right spines for a cleaner frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(mpl.rcParams["axes.edgecolor"])
    ax.spines["bottom"].set_color(mpl.rcParams["axes.edgecolor"])

    title = "Obstacle Crash" if section == "obstacle_crash" else "NFZ Hover"
    ax.set_title(f"{title}: Trajectory Overlay (K={len(idxs)})")

    # compact legend
    leg = ax.legend(loc="upper left", frameon=True, ncol=2, borderpad=0.5, handlelength=2.0, handletextpad=0.6)
    if leg and leg.get_frame():
        leg.get_frame().set_linewidth(0.8)

    # Save (both PNG + PDF by default)
    base = os.path.join(outdir, f"{section}_xy_overlay")
    for ext in outformats:
        outpath = f"{base}.{ext}"
        plt.savefig(outpath, bbox_inches="tight", pad_inches=0.05)
        print(f"\\u2713 saved {outpath}")
    plt.close(fig)

def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_xy_map_pretty.py results.json [K|SECTION] [SECTION] [THEME]")
        print("  SECTION \\u2208 {obstacle_crash, nfz_hover, both}")
        print("  THEME   \\u2208 {light, dark} (optional; default: light)")
        print("  Examples:")
        print("    python plot_xy_map_pretty.py results.json both")
        print("    python plot_xy_map_pretty.py results.json 5 obstacle_crash dark")
        sys.exit(1)

    path = sys.argv[1]
    theme = "light"
    # flexible arg parsing similar to original, with optional THEME as 4th arg
    if len(sys.argv) == 3:
        arg = sys.argv[2]
        if arg.isdigit():
            K = int(arg); section = "obstacle_crash"
        else:
            K = 5; section = arg
    else:
        # results.json K SECTION [THEME]
        K = int(sys.argv[2]) if sys.argv[2].isdigit() else 5
        section = sys.argv[3]
        if len(sys.argv) >= 5:
            theme = sys.argv[4].lower()

    if section.lower() == "both":
        overlay_xy(path, section="obstacle_crash", k_episodes=K, deviation_threshold=0.07, outdir="figs", theme=theme)
        overlay_xy(path, section="nfz_hover",     k_episodes=K, deviation_threshold=0.07, outdir="figs", theme=theme)
    else:
        overlay_xy(path, section=section, k_episodes=K, deviation_threshold=0.07, outdir="figs", theme=theme)

if __name__ == "__main__":
    main()