"""
Is the latent space organized by frame similarity?

L2's objective was cos_sim(z_a, z_b) ≈ IoU(frame_a, frame_b). A near-zero L2
loss is NOT sufficient evidence of organization: most frame pairs are disjoint
(IoU≈0) and random high-dim vectors are already ~orthogonal, so the loss is
trivially small. The real question is the DISCRIMINATIVE structure — for pairs
that genuinely overlap, does cosine similarity rise proportionally?

We sample frames spanning many trajectories AND many timesteps within each, so
pairwise IoU spans [0, 1] (consecutive frames overlap heavily; cross-trajectory
frames barely at all). Then we check whether latent cos_sim tracks it.

Reads best.pt only — safe alongside training.
"""

from pathlib import Path
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

import data, engine
from model import Encoder, Decoder

CKPT = Path("checkpoints/best.pt")
N_TRAJ = 60
FRAMES_PER_TRAJ = 8
RNG_SEED = 0


def main():
    torch.set_num_threads(2)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"Checkpoint: step={ckpt['step']}, F1={ckpt['metrics']['alive_f1']:.4f}\n")
    enc, dec = Encoder(), Decoder(kernel_size=1)
    enc.load_state_dict(ckpt["encoder"]); enc.eval()
    dec.load_state_dict(ckpt["decoder"]); dec.eval()

    pool = data.TrainingSeedPool()
    rng = np.random.default_rng(RNG_SEED)

    # Sample trajectories across quartiles, center-biased offsets (match training)
    frames, traj_ids, times = [], [], []
    per_q = N_TRAJ // 4
    tcount = 0
    for q in range(4):
        idx = rng.choice(pool.quartile_pools[q], size=per_q, replace=False)
        seeds = np.asarray(pool.seeds[idx])
        offs = engine.sample_center_biased_offsets(per_q, rng)
        trajs = engine.simulate(seeds, k=engine.K_DEFAULT, offsets=offs)
        lifes = pool.lifespans[idx]
        for i in range(per_q):
            # FRAMES_PER_TRAJ timesteps spread across the live window
            ts = np.linspace(0, int(lifes[i]), FRAMES_PER_TRAJ).astype(int)
            for t in ts:
                frames.append(trajs[i, t]); traj_ids.append(tcount); times.append(int(t))
            tcount += 1

    frames = np.stack(frames)                       # (N,128,128)
    traj_ids = np.array(traj_ids); times = np.array(times)
    N = len(frames)
    x = torch.from_numpy(frames.astype(np.float32)).unsqueeze(1)
    with torch.no_grad():
        z = enc(x).numpy()
    print(f"Encoded {N} frames from {tcount} trajectories\n")

    # Pairwise IoU (frame similarity) and cos_sim (latent similarity)
    g = frames.reshape(N, -1).astype(np.float32)
    inter = g @ g.T
    cnt = g.sum(1, keepdims=True)
    union = cnt + cnt.T - inter
    iou = inter / np.clip(union, 1, None)
    zu = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    cos = zu @ zu.T

    iu = np.triu_indices(N, k=1)
    iou_p = iou[iu]; cos_p = cos[iu]
    same_traj = (traj_ids[iu[0]] == traj_ids[iu[1]])

    # ---- (1) Overall correlation ----
    pr, _ = pearsonr(iou_p, cos_p)
    sr, _ = spearmanr(iou_p, cos_p)
    print("=" * 68)
    print("(1) DOES cos_sim TRACK IoU?  (L2's exact objective)")
    print("=" * 68)
    print(f"  Pearson  corr(IoU, cos_sim) = {pr:+.3f}")
    print(f"  Spearman corr(IoU, cos_sim) = {sr:+.3f}")
    print(f"    +1 = perfectly organized;  0 = latent ignores frame similarity")

    # ---- (2) Binned: mean cos_sim per IoU bucket (the discriminative test) ----
    print("\n" + "=" * 68)
    print("(2) MEAN cos_sim BY IoU BUCKET  (is the relationship monotonic?)")
    print("=" * 68)
    edges = [0.0, 0.001, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.01]
    print(f"  {'IoU bucket':>14} {'n_pairs':>9} {'mean_cos':>9} {'std':>7}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (iou_p >= lo) & (iou_p < hi)
        if m.sum() == 0: continue
        lbl = "disjoint" if hi <= 0.001 else f"{lo:.2f}-{hi:.2f}"
        print(f"  {lbl:>14} {int(m.sum()):>9} {cos_p[m].mean():>9.3f} {cos_p[m].std():>7.3f}")

    # ---- (3) Within-trajectory temporal structure ----
    print("\n" + "=" * 68)
    print("(3) WITHIN-TRAJECTORY: does cos_sim decay with temporal distance?")
    print("=" * 68)
    dt = np.abs(times[iu[0]] - times[iu[1]])
    print(f"  {'|Δt|':>10} {'n_pairs':>9} {'mean_IoU':>9} {'mean_cos':>9}")
    for lo, hi in [(0,1),(1,5),(5,15),(15,40),(40,10000)]:
        m = same_traj & (dt >= lo) & (dt < hi)
        if m.sum() == 0: continue
        lbl = f"{lo}-{hi if hi<10000 else '+'}"
        print(f"  {lbl:>10} {int(m.sum()):>9} {iou_p[m].mean():>9.3f} {cos_p[m].mean():>9.3f}")

    # ---- (4) within vs cross trajectory ----
    print("\n" + "=" * 68)
    print("(4) WITHIN-traj vs CROSS-traj latent similarity")
    print("=" * 68)
    print(f"  within-trajectory pairs: mean IoU={iou_p[same_traj].mean():.3f}  mean cos={cos_p[same_traj].mean():+.3f}")
    print(f"  cross-trajectory  pairs: mean IoU={iou_p[~same_traj].mean():.3f}  mean cos={cos_p[~same_traj].mean():+.3f}")
    print("\n  NOTE: L2 targets IoU (pixel overlap). Same pattern at a DIFFERENT")
    print("  position has IoU≈0 → trained to be ORTHOGONAL, not similar. So this")
    print("  is pixel-overlap organization, not translation-invariant structure.")


if __name__ == "__main__":
    main()
