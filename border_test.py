"""
Can the model reconstruct patterns near the grid BORDER?

Training used center-biased offsets clipped to [24, 88] (>=24-cell wall margin),
so the model has never seen alive cells within ~24 cells of a wall. This sweeps
the seed placement along the diagonal from the corner (offset 2) to the centre
(offset 56) and measures reconstruction F1 at each placement — directly showing
whether border regions are represented or are out-of-distribution dead zones.

Reads best.pt only — safe alongside training.
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data, engine
from model import Encoder, Decoder

CKPT = Path("checkpoints/best.pt")
N_SEEDS = 60
RNG_SEED = 0
OFFSETS = [2, 8, 16, 24, 32, 44, 56, 68, 80, 96, 104, 110]  # diagonal r=c


def f1(pred, true):
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum()); fn = int((~pred & true).sum())
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d else 0.0


def main():
    torch.set_num_threads(2)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"Checkpoint: step={ckpt['step']}, F1(center-dist val)={ckpt['metrics']['alive_f1']:.4f}\n")
    enc, dec = Encoder(), Decoder(kernel_size=1)
    enc.load_state_dict(ckpt["encoder"]); enc.eval()
    dec.load_state_dict(ckpt["decoder"]); dec.eval()

    pool = data.TrainingSeedPool()
    rng = np.random.default_rng(RNG_SEED)
    # Use longer-lived seeds so a mid-life frame is well-populated at any offset
    idx = rng.choice(pool.quartile_pools[2], size=N_SEEDS, replace=False)
    seeds = np.asarray(pool.seeds[idx])
    lifespans = pool.lifespans[idx]

    print(f"{'offset':>7} {'in-train?':>10} {'mean_F1':>9} {'mean_recall':>12} {'mean_alive':>11}")
    print("-" * 56)
    results = []
    for off in OFFSETS:
        offsets = np.full((N_SEEDS, 2), off, dtype=np.int64)
        trajs = engine.simulate(seeds, k=engine.K_DEFAULT, offsets=offsets)
        # mid-life frame per trajectory
        frames = np.stack([trajs[i, min(int(lifespans[i]) // 2, 256)] for i in range(N_SEEDS)])
        x = torch.from_numpy(frames.astype(np.float32)).unsqueeze(1)
        with torch.no_grad():
            probs = torch.sigmoid(dec(enc(x))).squeeze(1).numpy()
        pred = probs > 0.5
        true = frames == 1
        f1s, recs, n_alive = [], [], []
        for i in range(N_SEEDS):
            if true[i].sum() == 0:
                continue
            f1s.append(f1(pred[i], true[i]))
            tp = int((pred[i] & true[i]).sum()); fn = int((~pred[i] & true[i]).sum())
            recs.append(tp / (tp + fn) if (tp + fn) else 0.0)
            n_alive.append(int(true[i].sum()))
        in_train = "yes" if 24 <= off <= 88 else "NO (border)"
        mf1 = np.mean(f1s)
        results.append((off, mf1, np.mean(recs)))
        print(f"{off:>7} {in_train:>10} {mf1:>9.4f} {np.mean(recs):>12.4f} {np.mean(n_alive):>11.1f}")

    # Plot F1 vs offset
    offs = [r[0] for r in results]; f1v = [r[1] for r in results]
    plt.figure(figsize=(9, 5))
    plt.axvspan(24, 88, color="green", alpha=0.12, label="training range [24,88]")
    plt.plot(offs, f1v, "o-", color="darkblue")
    plt.xlabel("seed placement offset (diagonal, cells from top-left)")
    plt.ylabel("reconstruction F1 @ 0.5")
    plt.title(f"Border generalization: F1 vs seed placement (step {ckpt['step']})")
    plt.grid(alpha=0.3); plt.legend()
    out = Path("logs/border_test.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
