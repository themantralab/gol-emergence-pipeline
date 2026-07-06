"""
Demonstrate: at a CANONICAL center position (position confound removed), does
latent-path similarity retrieve visually/behaviourally similar trajectories?

- N trajectories, all seeds placed at the grid CENTER (fixed offset) so only
  dynamics vary, never position.
- Resample each to T time-aligned frames across [0, lifespan].
- Encode -> latent path (N, T, 1024).
- latent similarity(i,j)  = mean_t cos(z_i[t], z_j[t])
  visual similarity(i,j)  = mean_t IoU(frame_i[t], frame_j[t])
- (1) Spearman corr(latent_sim, visual_sim) across all pairs.
- (2) Retrieval: is each trajectory's latent nearest-neighbour also visually
      near? (rank of the latent-NN in the visual ordering; 1=perfect, N/2=chance)
- (3) Montage PNG: query filmstrip + its latent-NN filmstrip, to eyeball.
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import data, engine
from model import Encoder

CKPT = Path("checkpoints/best.pt")
N = 80          # trajectories
T = 16          # time-aligned samples per trajectory
CENTER = engine.SEED_OFFSET_CENTER   # 56 — canonical position for ALL trajectories
RNG = np.random.default_rng(1)


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
    print(f"Checkpoint step={ckpt['step']} F1={ckpt['metrics']['alive_f1']:.4f}")
    print(f"All trajectories placed at CANONICAL CENTER offset ({CENTER},{CENTER}).\n")
    enc = Encoder(); enc.load_state_dict(ckpt["encoder"]); enc.eval()
    pool = data.TrainingSeedPool()

    # sample N seeds all from Q3 (long-lived, populated throughout, same length
    # => clean time-alignment and real structure to compare)
    idx = RNG.choice(pool.quartile_pools[3], size=N, replace=False)
    seeds = np.asarray(pool.seeds[idx])
    lifes = pool.lifespans[idx].astype(int)
    offs = np.full((N, 2), CENTER, dtype=np.int64)               # FIXED center
    trajs = engine.simulate(seeds, k=engine.K_DEFAULT, offsets=offs)

    # time-align: T frames spanning [0, lifespan]
    aligned = np.zeros((N, T, 128, 128), dtype=np.uint8)
    for i in range(N):
        ts = np.linspace(0, lifes[i], T).astype(int)
        aligned[i] = trajs[i, ts]
    Z = encode_all(enc, aligned.reshape(N*T, 128, 128)).reshape(N, T, -1)
    Zu = (Z / Z.norm(dim=-1, keepdim=True)).numpy()

    # pairwise time-aligned similarities
    g = aligned.reshape(N, T, -1).astype(np.float32)
    lat_sim = np.zeros((N, N)); vis_sim = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            lat_sim[i, j] = (Zu[i] * Zu[j]).sum(-1).mean()
            inter = (g[i]*g[j]).sum(-1); uni = g[i].sum(-1)+g[j].sum(-1)-inter
            vis_sim[i, j] = (inter/np.clip(uni, 1, None)).mean()
    np.fill_diagonal(lat_sim, -np.inf); np.fill_diagonal(vis_sim, -np.inf)

    # (1) correlation over all pairs
    iu = np.triu_indices(N, 1)
    ls = lat_sim.copy(); vs = vis_sim.copy(); np.fill_diagonal(ls, np.nan); np.fill_diagonal(vs, np.nan)
    sr, _ = spearmanr(ls[iu], vs[iu])
    print("="*64); print("(1) latent-path similarity vs visual (IoU) similarity"); print("="*64)
    print(f"  Spearman corr over {len(iu[0])} trajectory pairs = {sr:+.3f}")

    # (2) retrieval quality
    print("\n" + "="*64); print("(2) RETRIEVAL: is the latent nearest-neighbour visually near?"); print("="*64)
    ranks = []; top5 = 0
    for i in range(N):
        lnn = np.argmax(lat_sim[i])                       # latent nearest neighbour
        vis_order = np.argsort(-vis_sim[i])               # visually nearest first
        rank = int(np.where(vis_order == lnn)[0][0]) + 1  # 1 = also the visual NN
        ranks.append(rank); top5 += (rank <= 5)
    ranks = np.array(ranks)
    print(f"  latent-NN's rank in the VISUAL ordering: median={np.median(ranks):.0f}  mean={ranks.mean():.1f}")
    print(f"    (1 = latent-NN is also the single most visually-similar; chance ~ {N/2:.0f})")
    print(f"  latent-NN is within the visual TOP-5: {top5}/{N} = {top5/N:.0%}")

    # (3) montage: 4 queries, each with its latent nearest-neighbour
    def strip(traj_frames, cols=(0, T//3, 2*T//3, T-1)):
        return [traj_frames[c] for c in cols]
    queries = RNG.choice(N, size=4, replace=False)
    cols = 4
    fig, axes = plt.subplots(8, cols, figsize=(cols*2.2, 8*2.2))
    for r, qi in enumerate(queries):
        nn = int(np.argmax(lat_sim[qi]))
        for row, (ti, tag) in enumerate([(qi, f"query #{qi}"), (nn, f"latent-NN #{nn}")]):
            ridx = r*2 + row
            frames = strip(aligned[ti])
            for c, f in enumerate(frames):
                axes[ridx, c].imshow(f, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
                axes[ridx, c].axis("off")
                if c == 0:
                    lab = tag if row == 0 else f"{tag}\nlat_sim={lat_sim[qi,nn]:.2f} IoU={vis_sim[qi,nn]:.2f}"
                    axes[ridx, c].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
                    axes[ridx, c].axis("on"); axes[ridx, c].set_xticks([]); axes[ridx, c].set_yticks([])
                if row == 0:
                    axes[ridx, c].set_title(f"t={[0,T//3,2*T//3,T-1][c]}/{T-1}", fontsize=8)
    plt.suptitle("Each pair: a query trajectory (top) and its LATENT nearest-neighbour (bottom)\n"
                 "canonical center placement — filmstrip across the trajectory", fontsize=11)
    plt.tight_layout()
    out = Path("logs/traj_cluster_demo.png"); plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nSaved montage -> {out}")


if __name__ == "__main__":
    main()
