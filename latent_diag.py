"""
Two diagnostics on the current best.pt checkpoint:

  (A) Halo quantification — measure decoder probability at true alive cells
      vs at their 1-pixel neighbours. If halo persists, neighbour probability
      stays well above background.

  (B) Angular distribution — measure cosine similarities between encoded
      latents across many frames. Without explicit angular regularization
      (just L3 magnitude bound), the encoder *could* collapse all latents
      to a narrow angular cone. We need to know if it has.

Safe to run alongside training — only reads best.pt.
"""

from pathlib import Path
import numpy as np
import torch

import data
import engine
from model import Encoder, Decoder

CKPT_PATH = Path("checkpoints/best.pt")
N_FRAMES_PER_QUARTILE = 100
RNG_SEED = 0


def main() -> None:
    torch.set_num_threads(2)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    print(f"Loaded checkpoint: step={ckpt['step']}, recorded F1={ckpt['metrics']['alive_f1']:.4f}")

    enc, dec = Encoder(), Decoder()
    enc.load_state_dict(ckpt["encoder"]); enc.eval()
    dec.load_state_dict(ckpt["decoder"]); dec.eval()

    pool = data.TrainingSeedPool()
    rng = np.random.default_rng(RNG_SEED)

    # ---- gather frames + latents ----
    all_frames, all_z, all_probs, traj_id = [], [], [], []
    for q in range(pool.n_quartiles):
        seed_idx = rng.choice(pool.quartile_pools[q], size=N_FRAMES_PER_QUARTILE, replace=False)
        seeds = np.asarray(pool.seeds[seed_idx])
        offsets = engine.sample_center_biased_offsets(len(seeds), rng)  # match training distribution
        trajs = engine.simulate(seeds, k=engine.K_DEFAULT, offsets=offsets)
        lifespans = pool.lifespans[seed_idx]
        frames = []
        for i in range(N_FRAMES_PER_QUARTILE):
            t = int(rng.integers(0, lifespans[i] + 1))
            frames.append(trajs[i, t])
        frames = np.stack(frames)
        x = torch.from_numpy(frames.astype(np.float32)).unsqueeze(1)
        with torch.no_grad():
            z = enc(x)
            logits = dec(z)
            probs = torch.sigmoid(logits).squeeze(1).numpy()
        all_frames.append(frames)
        all_z.append(z.numpy())
        all_probs.append(probs)
        traj_id.append(np.full(N_FRAMES_PER_QUARTILE, q))

    all_frames = np.concatenate(all_frames)            # (400, 128, 128) uint8
    all_z      = np.concatenate(all_z)                 # (400, 1024) float32
    all_probs  = np.concatenate(all_probs)             # (400, 128, 128) float32
    all_quart  = np.concatenate(traj_id)               # (400,) int

    N = len(all_frames)
    print(f"Sampled {N} frames ({N_FRAMES_PER_QUARTILE}/quartile)")

    # ============================================================
    # (A) HALO QUANTIFICATION
    # ============================================================
    print("\n" + "=" * 80)
    print("(A) HALO QUANTIFICATION")
    print("=" * 80)
    print("Per-quartile probability statistics on three pixel categories:")
    print("  ALIVE        : pixels that are actually alive in the ground truth")
    print("  NEIGHBOUR    : dead pixels adjacent to an alive pixel (1-pixel ring)")
    print("  BACKGROUND   : dead pixels NOT adjacent to any alive pixel")
    print()
    print(f"{'Q':>3}  {'n_alive':>10}  {'n_neigh':>10}  {'n_bgnd':>11}  "
          f"{'p@alive':>9}  {'p@neigh':>9}  {'p@bgnd':>9}  {'halo_ratio':>11}")
    print("-" * 100)

    from scipy.ndimage import binary_dilation
    for q in range(4):
        mask = all_quart == q
        p_alive_vals, p_neigh_vals, p_bgnd_vals = [], [], []
        for f, p in zip(all_frames[mask], all_probs[mask]):
            true_alive = (f == 1)
            dilated    = binary_dilation(true_alive)
            ring       = dilated & (~true_alive)
            bgnd       = (~dilated)
            if true_alive.any(): p_alive_vals.append(p[true_alive].mean())
            if ring.any():       p_neigh_vals.append(p[ring].mean())
            if bgnd.any():       p_bgnd_vals.append(p[bgnd].mean())

        p_alive_mean = np.mean(p_alive_vals)
        p_neigh_mean = np.mean(p_neigh_vals)
        p_bgnd_mean  = np.mean(p_bgnd_vals)
        # halo_ratio: how much above background is the neighbour probability?
        # 1.0 = neighbours look like background (no halo); higher = more halo
        halo_ratio = p_neigh_mean / max(p_bgnd_mean, 1e-6)

        n_alive_per_frame = int(all_frames[mask].sum() / mask.sum())
        n_neigh_per_frame = int(sum(int(binary_dilation(f == 1).sum() - (f == 1).sum())
                                    for f in all_frames[mask]) / mask.sum())
        n_bgnd_per_frame  = 128 * 128 - n_alive_per_frame - n_neigh_per_frame

        print(f"Q{q:<2}  {n_alive_per_frame:>10}  {n_neigh_per_frame:>10}  {n_bgnd_per_frame:>11}  "
              f"{p_alive_mean:>9.4f}  {p_neigh_mean:>9.4f}  {p_bgnd_mean:>9.4f}  {halo_ratio:>11.1f}×")

    print("""
Interpretation:
  - p@alive close to 1.0 → decoder is confident at true alive locations
  - p@neigh much higher than p@bgnd → HALO is present (model bleeds into neighbours)
  - p@neigh ≈ p@bgnd → halo eliminated (decoder commits sharply)
  - halo_ratio >> 1 → halo present; halo_ratio ≈ 1 → no halo
""")

    # ============================================================
    # (B) ANGULAR DISTRIBUTION
    # ============================================================
    print("=" * 80)
    print("(B) ANGULAR DISTRIBUTION OF LATENTS")
    print("=" * 80)

    # Normalise z's to unit vectors (we want angular geometry only)
    z = all_z
    norms = np.linalg.norm(z, axis=1)
    z_unit = z / norms[:, None]

    print(f"\nMagnitude stats: ‖z‖ mean={norms.mean():.3f}  std={norms.std():.3f}  "
          f"min={norms.min():.3f}  max={norms.max():.3f}")
    print(f"  (L3 target is 10.0 — tight std means encoder respects the magnitude bound)")

    # All-pairs cosine similarity (just upper triangle)
    cos_sim = z_unit @ z_unit.T
    iu = np.triu_indices(N, k=1)
    pairwise_cos = cos_sim[iu]

    print(f"\nPairwise cosine similarity across {N} latents ({len(pairwise_cos):,} pairs):")
    print(f"  mean     = {pairwise_cos.mean():+.4f}   (0.0 = uniformly distributed; +1 = all pointing same way)")
    print(f"  median   = {np.median(pairwise_cos):+.4f}")
    print(f"  std      = {pairwise_cos.std():.4f}")
    print(f"  min      = {pairwise_cos.min():+.4f}")
    print(f"  max      = {pairwise_cos.max():+.4f}")
    print(f"  fraction > 0.9 (nearly same direction): {(pairwise_cos > 0.9).mean():.4%}")
    print(f"  fraction > 0.5 (same hemisphere-ish):   {(pairwise_cos > 0.5).mean():.4%}")
    print(f"  fraction < 0.0 (orthogonal-or-more):    {(pairwise_cos < 0.0).mean():.4%}")

    # Effective dimensionality via PCA on the unit vectors
    centered = z_unit - z_unit.mean(0, keepdims=True)
    cov = centered.T @ centered / N
    eigvals = np.linalg.eigvalsh(cov)[::-1]                # descending
    eigvals = np.maximum(eigvals, 0)                        # clip numerical noise
    total = eigvals.sum()
    if total > 0:
        explained = eigvals / total
        cum = np.cumsum(explained)
        eff_dim_90 = int(np.searchsorted(cum, 0.90)) + 1
        eff_dim_99 = int(np.searchsorted(cum, 0.99)) + 1
        # Participation ratio (continuous measure of effective dimensionality)
        pr = (total ** 2) / (eigvals ** 2).sum()
    else:
        eff_dim_90 = eff_dim_99 = 0
        pr = 0.0

    print(f"\nPCA on unit-vector cloud (d_total = 1024):")
    print(f"  effective dim @ 90% variance = {eff_dim_90}")
    print(f"  effective dim @ 99% variance = {eff_dim_99}")
    print(f"  participation ratio          = {pr:.1f}")
    print(f"  top-5 eigenvalue share       = {explained[:5].sum():.3%}")
    print(f"  top-10 eigenvalue share      = {explained[:10].sum():.3%}")
    print(f"  top-50 eigenvalue share      = {explained[:50].sum():.3%}")

    print("""
Interpretation:
  - mean cos sim near 0 → vectors are angularly spread (good)
  - mean cos sim near 1 → vectors clustered in one direction (collapse!)
  - effective dim near d=1024 → using full sphere; near 1 → collapsed
  - participation ratio: continuous measure between 1 (collapsed) and 1024 (uniform)
""")


if __name__ == "__main__":
    main()
