"""
Threshold sweep + bottleneck analysis on the current best.pt.

Reads best.pt only — safe to run alongside training.

(A) Threshold sweep: F1 vs decision threshold, overall and per-quartile.
    If peak F1 sits at a higher threshold than 0.5, the model is well-
    calibrated but biased toward over-prediction (a thresholding artefact,
    not a capacity limit). If peak F1 is still low at the *optimal*
    threshold, the ceiling is a genuine resolution/capacity limit.

(B) Probability calibration: histogram of decoder probs separately at
    true-alive vs true-dead pixels. Clean bimodal separation → threshold
    tuning recovers F1. Muddy middle → resolution/capacity bottleneck.

(C) F1 vs alive-cell-count: does reconstruction degrade as frames get
    denser (capacity limit) or stay flat (calibration limit)?

(D) FP spatial structure: are false positives adjacent to true-alive cells
    (halo / sub-pixel resolution) or scattered far away (latent noise)?
"""

from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_dilation

import data
import engine
from model import Encoder, Decoder

CKPT = Path("checkpoints/best.pt")
N_PER_Q = 150
RNG_SEED = 0


def f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def main():
    torch.set_num_threads(2)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"Checkpoint: step={ckpt['step']}, recorded F1(@0.5)={ckpt['metrics']['alive_f1']:.4f}\n")

    enc, dec = Encoder(), Decoder(kernel_size=1)
    enc.load_state_dict(ckpt["encoder"]); enc.eval()
    dec.load_state_dict(ckpt["decoder"]); dec.eval()

    pool = data.TrainingSeedPool()
    rng = np.random.default_rng(RNG_SEED)

    all_frames, all_probs, all_quart = [], [], []
    for q in range(pool.n_quartiles):
        seed_idx = rng.choice(pool.quartile_pools[q], size=N_PER_Q, replace=False)
        seeds = np.asarray(pool.seeds[seed_idx])
        offsets = engine.sample_center_biased_offsets(len(seeds), rng)  # match training distribution
        trajs = engine.simulate(seeds, k=engine.K_DEFAULT, offsets=offsets)
        ls = pool.lifespans[seed_idx]
        frames = np.stack([trajs[i, int(rng.integers(0, ls[i] + 1))] for i in range(N_PER_Q)])
        x = torch.from_numpy(frames.astype(np.float32)).unsqueeze(1)
        with torch.no_grad():
            probs = torch.sigmoid(dec(enc(x))).squeeze(1).numpy()
        all_frames.append(frames); all_probs.append(probs)
        all_quart.append(np.full(N_PER_Q, q))

    frames = np.concatenate(all_frames)      # (600,128,128)
    probs  = np.concatenate(all_probs)
    quart  = np.concatenate(all_quart)
    N = len(frames)
    print(f"Sampled {N} frames ({N_PER_Q}/quartile)\n")

    true = (frames == 1)

    # ============ (A) THRESHOLD SWEEP ============
    print("=" * 78)
    print("(A) THRESHOLD SWEEP — micro-averaged over all pixels of all frames")
    print("=" * 78)
    print(f"{'thresh':>7} {'F1':>8} {'precision':>10} {'recall':>8} {'pred_alive':>11}")
    print("-" * 78)
    thresholds = np.arange(0.30, 0.96, 0.05)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        pred = probs > t
        tp = int((pred & true).sum())
        fp = int((pred & ~true).sum())
        fn = int((~pred & true).sum())
        f1 = f1_from_counts(tp, fp, fn)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        flag = ""
        if f1 > best_f1:
            best_f1, best_t = f1, t
            flag = "  <-- peak"
        print(f"{t:>7.2f} {f1:>8.4f} {p:>10.4f} {r:>8.4f} {int(pred.sum()):>11}{flag}")
    print(f"\n  PEAK micro-F1 = {best_f1:.4f} at threshold {best_t:.2f}")
    print(f"  (vs F1={f1_from_counts(int((probs>0.5)&true).sum() if False else int(((probs>0.5)&true).sum()), int(((probs>0.5)&~true).sum()), int(((probs<=0.5)&true).sum())):.4f} at the default 0.5)")

    # per-quartile optimal threshold
    print("\n  Per-quartile peak F1 and optimal threshold:")
    print(f"  {'Q':>3} {'peakF1':>8} {'opt_t':>6} {'F1@0.5':>8}")
    for q in range(4):
        m = quart == q
        tq, fq = 0.5, 0.0
        f1_at_half = 0.0
        for t in thresholds:
            pred = probs[m] > t
            tr = true[m]
            tp = int((pred & tr).sum()); fp = int((pred & ~tr).sum()); fn = int((~pred & tr).sum())
            f1 = f1_from_counts(tp, fp, fn)
            if abs(t - 0.5) < 1e-9:
                f1_at_half = f1
            if f1 > fq:
                fq, tq = f1, t
        print(f"  {q:>3} {fq:>8.4f} {tq:>6.2f} {f1_at_half:>8.4f}")

    # ============ (B) PROBABILITY CALIBRATION ============
    print("\n" + "=" * 78)
    print("(B) PROBABILITY CALIBRATION — decoder prob distribution by ground-truth")
    print("=" * 78)
    p_alive = probs[true]
    p_dead = probs[~true]
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    ha, _ = np.histogram(p_alive, bins=bins)
    hd, _ = np.histogram(p_dead, bins=bins)
    print(f"\n  At TRUE-ALIVE pixels (n={len(p_alive):,}):  mean={p_alive.mean():.3f} median={np.median(p_alive):.3f}")
    print(f"  At TRUE-DEAD  pixels (n={len(p_dead):,}):  mean={p_dead.mean():.5f} median={np.median(p_dead):.5f}")
    print(f"\n  {'bin':>10} {'@alive%':>9} {'@dead%':>9}")
    labels = ["0.0-0.1","0.1-0.2","0.2-0.3","0.3-0.4","0.4-0.5","0.5-0.6","0.6-0.7","0.7-0.8","0.8-0.9","0.9-1.0"]
    for lbl, a, d in zip(labels, ha, hd):
        print(f"  {lbl:>10} {100*a/len(p_alive):>8.2f}% {100*d/len(p_dead):>8.4f}%")
    # The ambiguous band 0.3-0.7 is where threshold can't separate cleanly
    amb_alive = ((p_alive > 0.3) & (p_alive < 0.7)).mean()
    print(f"\n  Fraction of true-alive pixels in ambiguous band (0.3,0.7): {amb_alive:.3%}")
    print(f"  Fraction of true-alive pixels confidently alive (>0.7):    {(p_alive>0.7).mean():.3%}")
    print(f"  Fraction of true-alive pixels confidently dead   (<0.3):   {(p_alive<0.3).mean():.3%}")

    # ============ (C) F1 vs ALIVE COUNT ============
    print("\n" + "=" * 78)
    print("(C) RECONSTRUCTION vs FRAME DENSITY (per-frame F1 at optimal threshold)")
    print("=" * 78)
    n_alive = true.reshape(N, -1).sum(1)
    per_f1 = []
    for i in range(N):
        pred = probs[i] > best_t
        tr = true[i]
        tp = int((pred & tr).sum()); fp = int((pred & ~tr).sum()); fn = int((~pred & tr).sum())
        per_f1.append(f1_from_counts(tp, fp, fn))
    per_f1 = np.array(per_f1)
    edges = [0, 10, 20, 30, 50, 80, 10000]
    print(f"\n  {'alive_count':>14} {'n_frames':>9} {'mean_F1':>9} {'mean_recall':>12}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (n_alive >= lo) & (n_alive < hi)
        if m.sum() == 0:
            continue
        # recall in this bucket
        recalls = []
        for i in np.where(m)[0]:
            pred = probs[i] > best_t; tr = true[i]
            tp = int((pred & tr).sum()); fn = int((~pred & tr).sum())
            recalls.append(tp / (tp + fn) if (tp + fn) else 1.0)
        rng_lbl = f"{lo}-{hi if hi < 10000 else '+'}"
        print(f"  {rng_lbl:>14} {int(m.sum()):>9} {per_f1[m].mean():>9.4f} {np.mean(recalls):>12.4f}")
    corr = np.corrcoef(n_alive, per_f1)[0, 1]
    print(f"\n  Pearson corr(alive_count, F1) = {corr:+.3f}")
    print(f"    strongly negative → capacity bottleneck (denser frames reconstruct worse)")
    print(f"    near zero         → calibration bottleneck (density-independent)")

    # ============ (D) FP SPATIAL STRUCTURE ============
    print("\n" + "=" * 78)
    print("(D) FALSE-POSITIVE SPATIAL STRUCTURE (at optimal threshold)")
    print("=" * 78)
    fp_adjacent, fp_far = 0, 0
    for i in range(N):
        pred = probs[i] > best_t
        tr = true[i]
        fp_mask = pred & ~tr
        if not fp_mask.any():
            continue
        ring = binary_dilation(tr) & ~tr     # 1-pixel ring around true-alive
        fp_adjacent += int((fp_mask & ring).sum())
        fp_far += int((fp_mask & ~ring).sum())
    total_fp = fp_adjacent + fp_far
    if total_fp:
        print(f"\n  Total false positives: {total_fp:,}")
        print(f"    adjacent to a true-alive cell (1-px ring): {fp_adjacent:,} ({100*fp_adjacent/total_fp:.1f}%)")
        print(f"    far from any true-alive cell:              {fp_far:,} ({100*fp_far/total_fp:.1f}%)")
        print(f"\n  >50% adjacent → sub-pixel resolution limit (halo-like: right region, wrong exact cell)")
        print(f"  >50% far      → latent encodes wrong global structure (spurious clusters)")


if __name__ == "__main__":
    main()
