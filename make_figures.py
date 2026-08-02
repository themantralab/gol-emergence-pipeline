"""
PAPER FIGURES — vector PDF (for LaTeX \\includegraphics) + PNG (for preview).

Standard matplotlib output, no custom styling beyond a validated categorical
palette. Figures are sized for a single column (3.4in) or full width (7.0in) of a
two-column paper and use 8-9pt type so they stay legible at print size.

Palette: slots 1-3 of the reference categorical theme
  #2a78d6 blue, #eb6834 orange, #1baf7a aqua
validated all-pairs in light mode (ok=true; worst-pair CVD dE 9.2, normal-vision
24.0). Identity is never carried by colour alone - every figure with >=2 series
has a legend, and line series also differ by marker and dash.

Inputs (produced by other scripts, all optional - a missing input skips its
figure rather than crashing):
  logs/multi_seed_raw.npy    <- multi_seed.py    (fig 1, 3, 5)
  logs/n4_offset_curve.npy   <- n4_offset_curve.py (fig 4)
  constants below            <- read from the diagnostic logs (fig 2, 6)

Outputs: $GOL_FIG_DIR/figN_*.pdf and .png  (default: figures/)
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(os.environ.get("GOL_FIG_DIR", "figures"))
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType, editable
    "axes.spines.top": False, "axes.spines.right": False,
})

COL1, COLW = 3.4, 7.0


def finish(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- fig 1: N1
def fig1_rollout(raw):
    H = raw["horizons"]
    def stack(key):
        a = np.array([[d[h] for h in H] for d in raw[key]], float)
        return a.mean(0), a.std(0, ddof=1)
    lm, ls = stack("n1_learned"); pm, ps = stack("n1_persist"); cm, cs = stack("n1_ceiling")

    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    for m, s, c, mk, dash, lab in [
        (cm, cs, AQUA, "s", (None, None), "AE ceiling (teacher-forced)"),
        (pm, ps, ORANGE, "^", (4, 2), "Persistence (no change)"),
        (lm, ls, BLUE, "o", (1, 1.5), "Learned latent rollout"),
    ]:
        ax.plot(H, m, color=c, lw=2, marker=mk, ms=4.5, dashes=dash, label=lab, zorder=3)
        ax.fill_between(H, m - s, m + s, color=c, alpha=0.18, lw=0, zorder=2)
    ax.set_xscale("log"); ax.set_xticks(H); ax.set_xticklabels([str(h) for h in H])
    ax.set_xlabel("Rollout horizon (steps)"); ax.set_ylabel("Reconstruction F1")
    ax.set_ylim(-0.02, 1.0); grid(ax)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.005, 0.90),
              handlelength=2.4, labelspacing=0.35)
    finish(fig, "fig1_rollout_vs_horizon")


# ------------------------------------------------------- fig 2: cos vs IoU
def fig2_cos_iou():
    labels = ["disjoint", "0–.05", ".05–.10", ".10–.20", ".20–.35", ".35–.50", ".50–.70", ".70–1"]
    cos = [0.006, 0.087, 0.223, 0.408, 0.744, 0.817, 0.792, 0.958]
    n = [80160, 26381, 6633, 1391, 226, 71, 49, 49]
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    x = np.arange(len(labels))
    ax.bar(x, cos, width=0.72, color=BLUE, zorder=3)
    for xi, (c, ni) in enumerate(zip(cos, n)):
        ax.text(xi, c + 0.03, f"{c:.2f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_xlabel("Frame-pair IoU bucket"); ax.set_ylabel("Mean latent cosine")
    ax.set_ylim(0, 1.08); grid(ax)
    finish(fig, "fig2_cosine_vs_iou")


# ------------------------------------------------- fig 3: latent vs stats
def fig3_latent_vs_stats(raw):
    r = raw["n2"]
    keys = ["latent_lin", "latent_mlp", "stats_lin", "stats_mlp"]
    m = [np.mean([d[k] for d in r]) for k in keys]
    s = [np.std([d[k] for d in r], ddof=1) for k in keys]
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    x = np.arange(4)
    ax.bar(x, m, yerr=s, width=0.66, capsize=3,
           color=[BLUE, BLUE, ORANGE, ORANGE], zorder=3,
           error_kw=dict(ecolor=INK2, lw=1))
    ax.axhline(0.25, color=INK2, lw=1, dashes=(4, 2), zorder=4)
    ax.text(3.45, 0.27, "chance", fontsize=7, color=INK2, ha="right")
    for xi, (mi, si) in enumerate(zip(m, s)):
        ax.text(xi, mi + si + 0.025, f"{mi:.3f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(["latent\nlinear", "latent\nMLP", "stats\nlinear", "stats\nMLP"])
    ax.set_ylabel("Balanced accuracy"); ax.set_ylim(0, 1.0); grid(ax)
    finish(fig, "fig3_latent_vs_stats")


# ------------------------------------------------------- fig 4: N4 curve
def fig4_n4(n4):
    rows = np.array(n4["rows"], float)   # d, disp, iou_m, iou_s, cos_m, cos_s
    d, iou_m, iou_s, cos_m, cos_s = rows[:, 0], rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 5]
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    ax.plot(d, cos_m, color=BLUE, lw=2, marker="o", ms=4.5, label="Latent cosine", zorder=4)
    ax.fill_between(d, cos_m - cos_s, cos_m + cos_s, color=BLUE, alpha=0.18, lw=0, zorder=2)
    ax.plot(d, iou_m, color=ORANGE, lw=2, marker="^", ms=4.5, dashes=(4, 2),
            label="Frame IoU", zorder=4)
    ax.fill_between(d, iou_m - iou_s, iou_m + iou_s, color=ORANGE, alpha=0.18, lw=0, zorder=2)
    cf, cfs = float(n4["cross_cos"]), float(n4["cross_cos_sd"])
    ax.axhline(cf, color=AQUA, lw=1.6, dashes=(1, 1.5), zorder=3,
               label="Cross-seed floor (unrelated pattern)")
    ax.axhspan(cf - cfs, cf + cfs, color=AQUA, alpha=0.14, lw=0, zorder=1)
    ax.set_xlabel("Diagonal displacement $d$ (cells)")
    ax.set_ylabel("Similarity to untranslated copy")
    ax.set_ylim(-0.15, 1.05); grid(ax)
    ax.legend(frameon=False, loc="upper right")
    finish(fig, "fig4_n4_translation_curve")


# -------------------------------------------------- fig 5: motif tolerance
def fig5_motif(raw):
    rm = raw["motif"]; nfs = [0, 1, 2, 3]
    bm = [np.mean([r[n][0] for r in rm]) for n in nfs]
    bs = [np.std([r[n][0] for r in rm], ddof=1) for n in nfs]
    tm = [np.mean([r[n][1] for r in rm]) for n in nfs]
    ts = [np.std([r[n][1] for r in rm], ddof=1) for n in nfs]
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    ax.errorbar(nfs, bm, yerr=bs, color=BLUE, lw=2, marker="o", ms=5, capsize=3,
                label="Latent trunk bank (phase-invariant)", zorder=4)
    ax.errorbar(nfs, tm, yerr=ts, color=ORANGE, lw=2, marker="^", ms=5, capsize=3,
                linestyle="--", label="Template matching (best of 2)", zorder=4)
    ax.axhline(0.5, color=INK2, lw=1, dashes=(4, 2), zorder=2)
    ax.text(3.05, 0.52, "chance", fontsize=7, color=INK2, ha="right")
    ax.set_xticks(nfs); ax.set_xlabel("Cells flipped in the planted glider (of 9)")
    ax.set_ylabel("Detection AUC"); ax.set_ylim(0.3, 1.05); grid(ax)
    ax.legend(frameon=False, loc="lower left")
    finish(fig, "fig5_motif_tolerance")


# ----------------------------------------------------- fig 6: cycle drift
def fig6_cycle():
    names = ["Real\nframe", "Slerp\nmid", "Perturb\n$\\epsilon$=0.2", "Prior\nsample"]
    drift = [0.034, 0.582, 0.207, 1.330]
    alive = [59.7, 85.5, 60.7, 314.6]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COLW, 2.6))
    x = np.arange(4)

    a1.bar(x, drift, width=0.62, color=BLUE, zorder=3)
    for xi, v in enumerate(drift):
        a1.text(xi, v + 0.04, f"{v:.3f}", ha="center", fontsize=7, color=INK2)
    a1.set_xticks(x); a1.set_xticklabels(names)
    a1.set_ylabel("Cycle drift  $\\|z^{\\prime}-z\\|\\,/\\,\\|z\\|$")
    a1.set_ylim(0, 1.55); grid(a1)
    a1.set_title("(a) Latent round-trip stability", fontsize=8.5, color=INK, pad=6)

    a2.bar(x, alive, width=0.62, color=ORANGE, zorder=3)
    a2.axhline(47.8, color=INK2, lw=1, dashes=(4, 2), zorder=4)
    a2.annotate("real frames: 47.8", xy=(0.55, 47.8), xytext=(-0.40, 235),
                fontsize=7, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7,
                                shrinkA=0, shrinkB=2))
    for xi, v in enumerate(alive):
        a2.text(xi, v + 10, f"{v:.0f}", ha="center", fontsize=7, color=INK2)
    a2.set_xticks(x); a2.set_xticklabels(names)
    a2.set_ylabel("Live cells per decoded frame")
    a2.set_ylim(0, 380); grid(a2)
    a2.set_title("(b) Decoded density vs reality", fontsize=8.5, color=INK, pad=6)

    fig.subplots_adjust(wspace=0.38)
    finish(fig, "fig6_offmanifold_cycle_drift")


def main():
    print(f"Writing figures to {OUT}/")
    raw_p, n4_p = Path("logs/multi_seed_raw.npy"), Path("logs/n4_offset_curve.npy")
    fig2_cos_iou(); fig6_cycle()
    if n4_p.exists():
        fig4_n4(np.load(n4_p, allow_pickle=True).item())
    else:
        print("  SKIP fig4 (run n4_offset_curve.py)")
    if raw_p.exists():
        raw = np.load(raw_p, allow_pickle=True).item()
        fig1_rollout(raw); fig3_latent_vs_stats(raw); fig5_motif(raw)
    else:
        print("  SKIP fig1/3/5 (run multi_seed.py)")


if __name__ == "__main__":
    main()
