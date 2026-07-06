"""
Stage 1 data pipeline for the GoL world model.

Reads the pre-generated seed corpus and lifespan metadata, applies the locked
sampling rules (lifespan-quartile stratification, lifespan >= 32 training
threshold, fingerprint window restricted to [0, lifespan]), and produces
everything a training step needs:

- A batch of B trajectories simulated on the fly from B seeds
- L2 pair samples (within-trajectory + cross-trajectory frame pairs)
- L3 fingerprint indices (two independent draws per trajectory)
- Hamming distance utility (used by L2's smoothness target)
"""

from pathlib import Path
import numpy as np

import engine

DATA_DIR = Path(__file__).resolve().parent / "data"

LIFESPAN_MIN     = 32     # training-pool threshold (n_samples=32 floor)
N_QUARTILES      = 4
BATCH_SIZE       = 32
N_FINGERPRINT    = 32     # frames per L3 fingerprint
N_L2_WITHIN      = 32     # within-trajectory pairs per batch
N_L2_CROSS       = 32     # cross-trajectory pairs per batch


class TrainingSeedPool:
    """Seed corpus with the locked sampling rules baked in.

    On construction:
      - loads seeds.npy (memory-mapped) and lifespans.npy
      - applies the lifespan >= LIFESPAN_MIN filter to build the training pool
      - computes lifespan quartile boundaries on the kept pool
      - partitions kept seed indices into per-quartile sub-pools
    """

    def __init__(self, data_dir: Path = DATA_DIR,
                 lifespan_min: int = LIFESPAN_MIN,
                 n_quartiles: int = N_QUARTILES):
        self.data_dir = Path(data_dir)
        self.seeds     = np.load(self.data_dir / "seeds.npy", mmap_mode="r")  # (N, 16, 16) uint8
        self.lifespans = np.load(self.data_dir / "lifespans.npy")             # (N,) int32

        keep_mask     = self.lifespans >= lifespan_min
        self.kept_idx = np.flatnonzero(keep_mask)                              # indices into seeds.npy
        kept_lifes    = self.lifespans[self.kept_idx]

        # Quartile boundaries on kept lifespans → assignment per kept seed
        edges  = np.quantile(kept_lifes, np.linspace(0, 1, n_quartiles + 1))
        # searchsorted with interior boundaries → bucket in [0, n_quartiles-1]
        assign = np.clip(
            np.searchsorted(edges[1:-1], kept_lifes, side="right"),
            0, n_quartiles - 1,
        )
        self.quartile_pools = [self.kept_idx[assign == q] for q in range(n_quartiles)]
        self.quartile_edges = edges
        self.n_quartiles    = n_quartiles
        self.lifespan_min   = lifespan_min

    def summary(self) -> str:
        lines = [
            f"TrainingSeedPool: {len(self.seeds)} seeds total, "
            f"{len(self.kept_idx)} kept (lifespan >= {self.lifespan_min}) "
            f"= {len(self.kept_idx) / len(self.seeds):.1%}",
            f"  quartile edges: {[round(float(e), 1) for e in self.quartile_edges]}",
        ]
        for q, pool in enumerate(self.quartile_pools):
            ls = self.lifespans[pool]
            lines.append(
                f"  Q{q}: n={len(pool):>7}  lifespan range [{int(ls.min())}, {int(ls.max())}]  median={int(np.median(ls))}"
            )
        return "\n".join(lines)

    def sample_batch_indices(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        """Stratified draw: equal slots per quartile (replace=False within a quartile)."""
        if batch_size % self.n_quartiles != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by n_quartiles ({self.n_quartiles})")
        per_q = batch_size // self.n_quartiles
        parts = [rng.choice(pool, size=per_q, replace=False) for pool in self.quartile_pools]
        idx   = np.concatenate(parts)
        rng.shuffle(idx)  # avoid contiguous quartile blocks in the batch
        return idx


def simulate_batch(pool: TrainingSeedPool, seed_idx: np.ndarray,
                   k: int = engine.K_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """Simulate trajectories for the given seed indices.

    Returns:
        trajectories: (B, k+1, 128, 128) uint8
        lifespans:    (B,) int32
    """
    seeds = np.asarray(pool.seeds[seed_idx])  # materialise from mmap to plain array
    traj  = engine.simulate(seeds, k=k)
    return traj, pool.lifespans[seed_idx]


def sample_l2_pairs(trajectories: np.ndarray, lifespans: np.ndarray,
                    n_within: int, n_cross: int,
                    rng: np.random.Generator) -> dict:
    """Build L2 pair tensors restricted to live-frame windows.

    Within-pair: two random frames from the same trajectory, sampled from [0, lifespan].
    Cross-pair:  one frame each from two distinct trajectories, each from its own [0, lifespan].

    Returns dict with within_a, within_b, cross_a, cross_b — each (n, 128, 128) uint8.
    """
    B = trajectories.shape[0]
    life_upper = lifespans.astype(np.int64) + 1  # exclusive upper bound per trajectory

    # Within-trajectory pairs
    w_tr  = rng.integers(0, B, size=n_within)
    w_max = life_upper[w_tr]
    w_a_t = (rng.random(n_within) * w_max).astype(np.int64)
    w_b_t = (rng.random(n_within) * w_max).astype(np.int64)

    # Cross-trajectory pairs (ensure the two trajectories differ)
    c_a_tr = rng.integers(0, B, size=n_cross)
    c_b_tr = (c_a_tr + rng.integers(1, B, size=n_cross)) % B
    c_a_t  = (rng.random(n_cross) * life_upper[c_a_tr]).astype(np.int64)
    c_b_t  = (rng.random(n_cross) * life_upper[c_b_tr]).astype(np.int64)

    return {
        "within_a": trajectories[w_tr,  w_a_t],
        "within_b": trajectories[w_tr,  w_b_t],
        "cross_a":  trajectories[c_a_tr, c_a_t],
        "cross_b":  trajectories[c_b_tr, c_b_t],
    }


def sample_l3_fingerprint_indices(lifespans: np.ndarray, n_samples: int,
                                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two independent draws of n_samples sorted indices per trajectory, each from [0, lifespan].

    Indices are sorted in temporal order so the concatenated fingerprint preserves time.
    Caller is responsible for actually gathering the frames; this returns indices only.

    Returns:
        fp_a, fp_b — each (B, n_samples) int32
    """
    B = len(lifespans)
    fp_a = np.empty((B, n_samples), dtype=np.int32)
    fp_b = np.empty((B, n_samples), dtype=np.int32)
    for i, life in enumerate(lifespans):
        upper = int(life) + 1
        if upper < n_samples:
            raise ValueError(
                f"trajectory {i} has {upper} live frames < n_samples={n_samples}; "
                f"check that LIFESPAN_MIN >= n_samples (got {LIFESPAN_MIN})"
            )
        fp_a[i] = np.sort(rng.choice(upper, size=n_samples, replace=False))
        fp_b[i] = np.sort(rng.choice(upper, size=n_samples, replace=False))
    return fp_a, fp_b


def hamming_distance(grids_a: np.ndarray, grids_b: np.ndarray) -> np.ndarray:
    """Normalised Hamming distance between two batches of binary grids.

    Args:
        grids_a, grids_b: (B, H, W) uint8 with values in {0, 1}
    Returns:
        (B,) float32 in [0, 1]
    """
    diff = (grids_a != grids_b).astype(np.float32)
    return diff.sum(axis=(1, 2)) / (grids_a.shape[1] * grids_a.shape[2])


# -----------------------------------------------------------------------------
# Stage 1 gate: pool construction + one full batch end-to-end with timing
# -----------------------------------------------------------------------------

def _stage1_gate() -> None:
    import time

    print("=== TrainingSeedPool ===")
    pool = TrainingSeedPool()
    print(pool.summary())

    rng = np.random.default_rng(0)
    print("\n=== one training batch end-to-end ===")
    t0  = time.perf_counter()
    idx = pool.sample_batch_indices(BATCH_SIZE, rng)
    t1  = time.perf_counter()
    traj, lifespans = simulate_batch(pool, idx)
    t2  = time.perf_counter()
    pairs = sample_l2_pairs(traj, lifespans, N_L2_WITHIN, N_L2_CROSS, rng)
    t3  = time.perf_counter()
    fp_a, fp_b = sample_l3_fingerprint_indices(lifespans, N_FINGERPRINT, rng)
    t4  = time.perf_counter()

    # Sanity: compute Hamming on the within pairs and on the cross pairs
    h_within = hamming_distance(pairs["within_a"], pairs["within_b"])
    h_cross  = hamming_distance(pairs["cross_a"],  pairs["cross_b"])

    print(f"  seed sampling:     {(t1 - t0) * 1e3:>7.2f} ms")
    print(f"  trajectory sim:    {(t2 - t1) * 1e3:>7.2f} ms  (B={BATCH_SIZE}, k=256)")
    print(f"  L2 pair sample:    {(t3 - t2) * 1e3:>7.2f} ms")
    print(f"  L3 fp index draw:  {(t4 - t3) * 1e3:>7.2f} ms")
    print(f"  total:             {(t4 - t0) * 1e3:>7.2f} ms")
    print(f"  trajectory shape:  {traj.shape}, dtype={traj.dtype}, lifespans range=[{int(lifespans.min())}, {int(lifespans.max())}]")
    print(f"  L2 within: n={len(h_within)}  Hamming mean={h_within.mean():.4f}  range=[{h_within.min():.4f}, {h_within.max():.4f}]")
    print(f"  L2 cross:  n={len(h_cross)}   Hamming mean={h_cross.mean():.4f}  range=[{h_cross.min():.4f}, {h_cross.max():.4f}]")
    print(f"  L3 fp_a shape={fp_a.shape}, fp_b shape={fp_b.shape}")
    print(f"  Stage 1 gate (<2s total per batch): {'PASS' if (t4 - t0) < 2.0 else 'FAIL'}")


if __name__ == "__main__":
    print("=== engine canonical checks ===")
    assert engine.run_canonical_checks(), "engine failed canonical checks"
    print()
    _stage1_gate()
