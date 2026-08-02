"""
N4 AS A CURVE — latent similarity vs translation distance.

WHY
---
N4 (the position confound) currently rests on a SINGLE point: the same seed placed
~10 cells apart gives latent cos 0.105 while two DIFFERENT seeds give 0.079. That
is a strong result stated weakly. A curve over translation distance turns it into
a figure and makes the causal argument visible:

  L2 trains cos(z_i,z_j) -> IoU(x_i,x_j). Two copies of one pattern at different
  positions HAVE IoU ~ 0. So the model was explicitly taught to call them
  unrelated. The curve should therefore track the IoU curve and fall to the
  cross-seed floor as soon as the translated copies stop overlapping.

This measures identical dynamics (same seed, same trajectory) at a reference
placement vs a displaced placement, sweeping the displacement.
"""
from pathlib import Path
import numpy as np
import torch

import data, engine
from model import Encoder

CKPT = Path("checkpoints/best.pt")
RNG = np.random.default_rng(0)
N_SEEDS = 60
CENTER = 56
OFFSETS = [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 30]
FRAME_T = 20          # compare frame at this timestep of the trajectory


def encode_all(enc, frames, bs=64):
    zs = []
    with torch.no_grad():
        for i in range(0, len(frames), bs):
            x = torch.from_numpy(frames[i:i + bs].astype(np.float32)).unsqueeze(1)
            zs.append(enc(x))
    return torch.cat(zs)


def iou(a, b):
    inter = int((a & b).sum()); union = int((a | b).sum())
    return inter / union if union else 1.0


def main():
    torch.set_num_threads(8)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"Checkpoint step={ckpt['step']} F1={ckpt['metrics']['alive_f1']:.4f}")
    enc = Encoder(); enc.load_state_dict(ckpt["encoder"]); enc.eval()
    pool = data.TrainingSeedPool()

    idx = RNG.choice(pool.quartile_pools[3], size=N_SEEDS, replace=False)
    seeds = np.asarray(pool.seeds[idx])

    # reference placement: canonical centre
    ref_off = np.full((N_SEEDS, 2), CENTER, np.int64)
    ref_traj = engine.simulate(seeds, k=FRAME_T, offsets=ref_off)
    ref_frames = ref_traj[:, FRAME_T]
    z_ref = encode_all(enc, ref_frames)

    print(f"\n{N_SEEDS} Q3 seeds, frame t={FRAME_T}, reference placement ({CENTER},{CENTER})")
    print("Displacement is applied along the diagonal (d,d).\n")
    print(f"  {'offset d':>9} {'|disp|':>8} {'frame IoU':>18} {'latent cos':>18}")

    rows = []
    for d in OFFSETS:
        off = np.full((N_SEEDS, 2), CENTER + d, np.int64)
        tr = engine.simulate(seeds, k=FRAME_T, offsets=off)
        fr = tr[:, FRAME_T]
        z = encode_all(enc, fr)
        cos = torch.nn.functional.cosine_similarity(z, z_ref, dim=-1).numpy()
        ious = np.array([iou(ref_frames[i] == 1, fr[i] == 1) for i in range(N_SEEDS)])
        disp = float(np.hypot(d, d))
        rows.append((d, disp, ious.mean(), ious.std(ddof=1), cos.mean(), cos.std(ddof=1)))
        print(f"  {d:>9} {disp:>8.1f} {ious.mean():>10.3f}+/-{ious.std(ddof=1):<6.3f} "
              f"{cos.mean():>10.3f}+/-{cos.std(ddof=1):<6.3f}")

    # cross-seed floor: DIFFERENT seeds, both at the reference placement
    perm = RNG.permutation(N_SEEDS)
    ok = perm != np.arange(N_SEEDS)
    cs_cos = torch.nn.functional.cosine_similarity(z_ref[perm][ok], z_ref[ok], dim=-1).numpy()
    cs_iou = np.array([iou(ref_frames[perm[i]] == 1, ref_frames[i] == 1)
                       for i in range(N_SEEDS) if ok[i]])
    print(f"\n  CROSS-SEED FLOOR (different dynamics, same placement):")
    print(f"    frame IoU  = {cs_iou.mean():.3f}+/-{cs_iou.std(ddof=1):.3f}")
    print(f"    latent cos = {cs_cos.mean():.3f}+/-{cs_cos.std(ddof=1):.3f}")

    # where does translation reach the cross-seed floor?
    floor = cs_cos.mean() + cs_cos.std(ddof=1)
    reached = [r[0] for r in rows if r[4] <= floor]
    print(f"\n  Translation reaches the cross-seed floor (cos <= {floor:.3f}) at")
    print(f"  displacement d = {reached[0] if reached else 'never within tested range'}")
    print("  => beyond that, an IDENTICAL pattern is indistinguishable from an")
    print("     UNRELATED one. This is the objective's doing, not the optimiser's:")
    print("     L2 targets IoU, and translated copies have IoU ~ 0.")

    np.save("logs/n4_offset_curve.npy",
            {"rows": rows, "cross_cos": cs_cos.mean(), "cross_cos_sd": cs_cos.std(ddof=1),
             "cross_iou": cs_iou.mean()}, allow_pickle=True)
    print("\nRaw -> logs/n4_offset_curve.npy")


if __name__ == "__main__":
    main()
