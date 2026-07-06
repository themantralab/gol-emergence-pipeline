"""
Readiness for the ACTUAL plan: pre-compute real GoL trajectories, encode them,
and COMPARE trajectories as paths on the manifold of encoded real frames.
(No latent prediction/sampling — everything stays on-manifold by construction.)

The decisive question is not off-manifold robustness but whether latent-space
trajectory comparison reflects MECHANICAL similarity. Tests:

  1. ENCODING FAITHFULNESS (recap): real-frame cycle drift — already ~0.05.
  2. TRAJECTORY SEPARABILITY: are distinct trajectories distinguishable as
     latent paths (within-traj cohesion vs cross-traj separation)?
  3. TRANSLATION CONFOUND (the crux): the SAME seed (identical dynamics) placed
     at two positions. If latent paths stay similar -> comparison reflects
     dynamics. If they diverge -> comparison is confounded by absolute position
     (expected, since L2 trained cos~=IoU and a translate has IoU~=0).
"""
from pathlib import Path
import numpy as np
import torch
import engine, data
from model import Encoder

CKPT = Path("checkpoints/best.pt")
RNG = np.random.default_rng(0)


def encode_all(enc, frames, bs=64):
    zs = []
    with torch.no_grad():
        for i in range(0, len(frames), bs):
            x = torch.from_numpy(frames[i:i+bs].astype(np.float32)).unsqueeze(1)
            zs.append(enc(x))
    return torch.cat(zs)


def main():
    torch.set_num_threads(8)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"Checkpoint step={ckpt['step']} F1={ckpt['metrics']['alive_f1']:.4f}\n")
    enc = Encoder(); enc.load_state_dict(ckpt["encoder"]); enc.eval()
    pool = data.TrainingSeedPool()

    K = 80
    n = 30
    idx = RNG.choice(pool.quartile_pools[3], size=n, replace=False)
    seeds = np.asarray(pool.seeds[idx])

    # ---- 2. trajectory separability: each seed at ONE center offset ----
    offs = engine.sample_center_biased_offsets(n, RNG)
    trajs = engine.simulate(seeds, k=K, offsets=offs)            # (n,K+1,H,W)
    Z = encode_all(enc, trajs.reshape(-1, 128, 128)).reshape(n, K+1, -1)
    Zu = Z / Z.norm(dim=-1, keepdim=True)

    # within-traj: mean cos between consecutive-ish frames of same traj
    within = []
    for i in range(n):
        c = (Zu[i] @ Zu[i].T)
        iu = torch.triu_indices(K+1, K+1, offset=1)
        within.append(c[iu[0], iu[1]].mean().item())
    # cross-traj: mean cos between frames of different trajs (same time index)
    cross = []
    for t in range(0, K+1, 10):
        c = (Zu[:, t] @ Zu[:, t].T)
        iu = torch.triu_indices(n, n, offset=1)
        cross.append(c[iu[0], iu[1]].mean().item())
    print("="*66); print("2. TRAJECTORY SEPARABILITY"); print("="*66)
    print(f"  within-trajectory  mean cos = {np.mean(within):+.3f}  (frames of one path cohere)")
    print(f"  cross-trajectory   mean cos = {np.mean(cross):+.3f}  (distinct paths separated)")
    print(f"  separation ratio            = {np.mean(within)/max(abs(np.mean(cross)),1e-3):.1f}x")
    print("  -> high within, near-zero cross = trajectories are distinguishable paths.")

    # ---- 3. translation confound: same seed at TWO offsets ----
    print("\n" + "="*66); print("3. TRANSLATION CONFOUND  (same dynamics, two positions)"); print("="*66)
    offA = engine.sample_center_biased_offsets(n, RNG)
    offB = engine.sample_center_biased_offsets(n, RNG)
    tA = engine.simulate(seeds, k=K, offsets=offA)
    tB = engine.simulate(seeds, k=K, offsets=offB)
    ZA = encode_all(enc, tA.reshape(-1,128,128)).reshape(n, K+1, -1)
    ZB = encode_all(enc, tB.reshape(-1,128,128)).reshape(n, K+1, -1)
    ZAu = ZA / ZA.norm(dim=-1, keepdim=True)
    ZBu = ZB / ZB.norm(dim=-1, keepdim=True)

    # frame-aligned latent cos between the SAME seed at the two positions
    frame_cos = (ZAu * ZBu).sum(-1).mean().item()
    # pixel IoU between the two placements' frames (why: shows the confound source)
    gA = tA.reshape(n*(K+1), -1).astype(np.float32); gB = tB.reshape(n*(K+1), -1).astype(np.float32)
    inter = (gA*gB).sum(1); iou = inter / np.clip(gA.sum(1)+gB.sum(1)-inter, 1, None)
    mean_off = np.abs(offA - offB).mean()
    print(f"  mean |offset_A - offset_B|                = {mean_off:.1f} cells")
    print(f"  frame IoU (same seed, two positions)      = {iou.mean():.3f}")
    print(f"  latent cos (same seed, two positions)     = {frame_cos:+.3f}")
    print(f"  reference cross-SEED latent cos           = {np.mean(cross):+.3f}")
    print()
    if frame_cos < 0.3:
        print("  VERDICT: identical dynamics at different positions land ~as far apart")
        print("  as DIFFERENT dynamics. Latent comparison is dominated by ABSOLUTE")
        print("  POSITION, not mechanics. => must fix seed placement (encode all")
        print("  trajectories at ONE canonical position) OR use a translation-")
        print("  invariant representation before comparing.")
    else:
        print("  VERDICT: latent paths stay similar across position => dynamics-")
        print("  dominated comparison is viable.")


if __name__ == "__main__":
    main()
