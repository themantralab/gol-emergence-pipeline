"""
Visual check of encode→decode reconstruction quality.

Loads the current best.pt checkpoint (a stale snapshot — training keeps
writing newer ones in parallel, but that's fine for a visual inspection),
picks a few frames from each lifespan quartile, runs them through the model,
and produces:

  1. A side-by-side PNG: original | probability map | thresholded | disagreement
     (saved to logs/reconstruction_samples.png)
  2. ASCII art of the most-active region of one sample per quartile, printed
     to stdout, so we can see reconstruction quality directly in the terminal.

Safe to run while training is active — it only reads best.pt.
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data
import engine
from model import Encoder, Decoder

CKPT_PATH = Path("checkpoints/best.pt")
OUT_PNG   = Path("logs/reconstruction_samples.png")
RNG_SEED  = 42


def to_ascii(grid: np.ndarray, char_alive: str = "█", char_dead: str = "·",
             pad: int = 2, max_size: int = 48) -> str:
    """Render a 128×128 binary grid as ASCII, focused on the active region."""
    rows, cols = np.where(grid == 1)
    if len(rows) == 0:
        return "(no alive cells)"
    rmin, rmax = max(0, rows.min() - pad), min(128, rows.max() + pad + 1)
    cmin, cmax = max(0, cols.min() - pad), min(128, cols.max() + pad + 1)
    region = grid[rmin:rmax, cmin:cmax]
    # If too large for terminal, downsample 2× by max-pool
    while region.shape[0] > max_size or region.shape[1] > max_size:
        h, w = region.shape
        # Crop to even dims first
        region = region[:h - (h % 2), :w - (w % 2)]
        h, w = region.shape
        region = region.reshape(h // 2, 2, w // 2, 2).max(axis=(1, 3))
    return "\n".join("".join(char_alive if c else char_dead for c in row) for row in region)


def main() -> None:
    if not CKPT_PATH.exists():
        print(f"No checkpoint at {CKPT_PATH}; training may not have written one yet.")
        return

    torch.set_num_threads(2)  # don't fight the training process for cores
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    print(f"Loaded checkpoint: step={ckpt['step']}, recorded alive_F1={ckpt['metrics']['alive_f1']:.4f}")

    enc, dec = Encoder(), Decoder()
    enc.load_state_dict(ckpt["encoder"]); enc.eval()
    dec.load_state_dict(ckpt["decoder"]); dec.eval()

    pool = data.TrainingSeedPool()
    rng = np.random.default_rng(RNG_SEED)

    # Pick 2 samples per quartile (8 total)
    samples = []
    for q in range(pool.n_quartiles):
        seed_idx = rng.choice(pool.quartile_pools[q], size=2, replace=False)
        seeds = np.asarray(pool.seeds[seed_idx])
        offsets = engine.sample_center_biased_offsets(len(seeds), rng)  # match training distribution
        trajs = engine.simulate(seeds, k=engine.K_DEFAULT, offsets=offsets)
        lifespans = pool.lifespans[seed_idx]
        for i in range(2):
            t = int(rng.integers(0, lifespans[i] + 1))
            samples.append({
                "quartile": q,
                "lifespan": int(lifespans[i]),
                "timestep": t,
                "frame": trajs[i, t],
            })

    frames = np.stack([s["frame"] for s in samples])
    x = torch.from_numpy(frames.astype(np.float32)).unsqueeze(1)
    with torch.no_grad():
        z = enc(x)
        logits = dec(z)
        probs = torch.sigmoid(logits).squeeze(1).numpy()
    preds = (probs > 0.5).astype(np.uint8)

    # ---- PNG: 4 columns × 8 rows ----
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, n * 2.5))
    for i, s in enumerate(samples):
        orig = s["frame"]
        prob = probs[i]
        pred = preds[i]
        diff = (pred != orig).astype(np.uint8)

        # Compute per-frame metrics
        tp = int(((pred == 1) & (orig == 1)).sum())
        fp = int(((pred == 1) & (orig == 0)).sum())
        fn = int(((pred == 0) & (orig == 1)).sum())
        n_true = int(orig.sum())
        n_pred = int(pred.sum())
        f1 = (2 * tp) / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        axes[i, 0].imshow(orig, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        axes[i, 0].set_title(
            f"Q{s['quartile']}, life={s['lifespan']}, t={s['timestep']}\n"
            f"original  ({n_true} alive)", fontsize=9
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(prob, cmap="hot", vmin=0, vmax=1, interpolation="nearest")
        axes[i, 1].set_title(f"prob map  (max={prob.max():.2f})", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        axes[i, 2].set_title(f"binary @ 0.5  ({n_pred} alive)", fontsize=9)
        axes[i, 2].axis("off")

        # Diff: TP green, FP red, FN blue
        diff_color = np.zeros((128, 128, 3))
        diff_color[(pred == 1) & (orig == 1)] = [0.3, 1.0, 0.3]    # TP — green
        diff_color[(pred == 1) & (orig == 0)] = [1.0, 0.3, 0.3]    # FP — red (halo)
        diff_color[(pred == 0) & (orig == 1)] = [0.3, 0.3, 1.0]    # FN — blue (missed)
        axes[i, 3].imshow(diff_color, interpolation="nearest")
        axes[i, 3].set_title(
            f"diff  F1={f1:.3f}  TP={tp} FP={fp} FN={fn}", fontsize=9
        )
        axes[i, 3].axis("off")

    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {OUT_PNG} ({n} samples × 4 panels)")

    # ---- ASCII: print one sample per quartile to terminal ----
    print("\n" + "=" * 80)
    print("ASCII reconstruction preview (one frame per quartile)")
    print("Legend: █ = alive,  · = dead")
    print("=" * 80)
    for q in range(pool.n_quartiles):
        # Pick the first sample in this quartile
        sample = next(s for s in samples if s["quartile"] == q)
        idx = samples.index(sample)
        orig = sample["frame"]
        pred = preds[idx]

        tp = int(((pred == 1) & (orig == 1)).sum())
        fp = int(((pred == 1) & (orig == 0)).sum())
        fn = int(((pred == 0) & (orig == 1)).sum())

        print(f"\n--- Q{q}  lifespan={sample['lifespan']}  t={sample['timestep']}  "
              f"alive_true={int(orig.sum())}  alive_pred={int(pred.sum())}  "
              f"TP={tp} FP={fp} FN={fn} ---")
        print("\nORIGINAL:")
        print(to_ascii(orig))
        print("\nRECONSTRUCTION (threshold 0.5):")
        print(to_ascii(pred))


if __name__ == "__main__":
    main()
