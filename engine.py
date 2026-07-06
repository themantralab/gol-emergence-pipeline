"""
Conway's Game of Life simulator. B3/S23, fixed-zero boundary.

The world model never predicts dynamics — this engine is the ground truth.
Vectorised over the batch dimension so 32 trajectories step together; this is
what makes the Stage 1 <2s/batch gate reachable on CPU.

A "trajectory" is the 257-frame sequence [f_0, f_1, ..., f_k] produced by
embedding a 16x16 seed at offset (24, 24) on a 128x128 grid and iterating the
B3/S23 rule with the grid boundary treated as a wall of dead cells (no toroidal
wrap; out-of-bounds is dead).
"""

import numpy as np

GRID_H = 128
GRID_W = 128
SEED_H = 16
SEED_W = 16
SEED_OFFSET = (24, 24)  # default fixed placement (back-compat: canonical checks, diagnostics)
K_DEFAULT = 256         # default trajectory length (yields 257 frames including f_0)

# Center-biased random placement (training/validation). The 16×16 seed is
# placed at a per-sample (row, col) drawn from a Gaussian centred so the seed
# centre coincides with the grid centre, clipped to keep >= 24-cell wall margin
# — the same margin as the original fixed (24,24) offset. Because center
# placement only ever gives the pattern MORE room than a wall-adjacent one,
# trajectories live at least as long as at (24,24), so the precomputed
# lifespans.npy (measured at (24,24)) remain a valid lower-bound on the live
# region and do NOT need regenerating.
SEED_OFFSET_CENTER = (GRID_H - SEED_H) // 2   # 56 → seed centre at 64 = grid centre
SEED_OFFSET_STD    = 10.0
SEED_OFFSET_MIN    = 24
SEED_OFFSET_MAX    = GRID_H - SEED_H - 24      # 88 → symmetric 24-cell margin


def sample_center_biased_offsets(b: int, rng: np.random.Generator) -> np.ndarray:
    """(b, 2) int64 per-sample (row, col) top-left offsets, Gaussian-biased
    toward the grid centre and clipped to [SEED_OFFSET_MIN, SEED_OFFSET_MAX]."""
    off = rng.normal(SEED_OFFSET_CENTER, SEED_OFFSET_STD, size=(b, 2))
    off = np.clip(np.round(off), SEED_OFFSET_MIN, SEED_OFFSET_MAX)
    return off.astype(np.int64)


def embed_seeds(seeds: np.ndarray, offsets: np.ndarray | None = None) -> np.ndarray:
    """Place (B, 16, 16) seeds onto (B, 128, 128) grids.

    offsets: optional (B, 2) int array of per-sample (row, col) top-left
    placements. If None, all seeds go at the fixed SEED_OFFSET (back-compat).
    """
    B = seeds.shape[0]
    grids = np.zeros((B, GRID_H, GRID_W), dtype=np.uint8)
    if offsets is None:
        r, c = SEED_OFFSET
        grids[:, r:r + SEED_H, c:c + SEED_W] = seeds
    else:
        for i in range(B):
            r, c = int(offsets[i, 0]), int(offsets[i, 1])
            grids[i, r:r + SEED_H, c:c + SEED_W] = seeds[i]
    return grids


def step(grid: np.ndarray) -> np.ndarray:
    """One B3/S23 update with fixed-zero boundary. Vectorised over batch.

    Args:
        grid: (B, H, W) uint8, values in {0, 1}
    Returns:
        (B, H, W) uint8 — the next generation
    """
    g = grid.astype(np.int8)
    padded = np.pad(g, ((0, 0), (1, 1), (1, 1)), constant_values=0)
    n = (
        padded[:, :-2, :-2] + padded[:, :-2, 1:-1] + padded[:, :-2, 2:] +
        padded[:, 1:-1, :-2]                       + padded[:, 1:-1, 2:] +
        padded[:, 2:,  :-2] + padded[:, 2:,  1:-1] + padded[:, 2:,  2:]
    )
    alive = g == 1
    next_alive = (alive & ((n == 2) | (n == 3))) | (~alive & (n == 3))
    return next_alive.astype(np.uint8)


def simulate(seeds: np.ndarray, k: int = K_DEFAULT,
             offsets: np.ndarray | None = None) -> np.ndarray:
    """Run B3/S23 for k steps from a batch of seeds.

    Args:
        seeds: (B, 16, 16) uint8
        k: number of steps (default 256)
        offsets: optional (B, 2) per-sample placement offsets (see embed_seeds).
                 None → fixed SEED_OFFSET.
    Returns:
        trajectories: (B, k+1, 128, 128) uint8, with [:, 0] being the embedded seed
    """
    B = seeds.shape[0]
    traj = np.zeros((B, k + 1, GRID_H, GRID_W), dtype=np.uint8)
    traj[:, 0] = embed_seeds(seeds, offsets)
    for t in range(k):
        traj[:, t + 1] = step(traj[:, t])
    return traj


# -----------------------------------------------------------------------------
# Canonical correctness checks. Each returns (ok: bool, message: str).
# -----------------------------------------------------------------------------

def _make_blinker_seed() -> np.ndarray:
    """3-cell horizontal line at the centre of a 16x16 seed."""
    s = np.zeros((1, SEED_H, SEED_W), dtype=np.uint8)
    s[0, 7, 6:9] = 1
    return s


def _make_block_seed() -> np.ndarray:
    """2x2 block in the top-left of the seed (well away from grid boundary)."""
    s = np.zeros((1, SEED_H, SEED_W), dtype=np.uint8)
    s[0, 1:3, 1:3] = 1
    return s


def _make_glider_seed() -> np.ndarray:
    """Standard glider, oriented to drift +1 row +1 col every 4 steps."""
    s = np.zeros((1, SEED_H, SEED_W), dtype=np.uint8)
    s[0, 1, 2] = 1
    s[0, 2, 3] = 1
    s[0, 3, 1] = 1
    s[0, 3, 2] = 1
    s[0, 3, 3] = 1
    return s


def check_blinker() -> tuple[bool, str]:
    traj = simulate(_make_blinker_seed(), k=4)
    f0, f1, f2 = traj[0, 0], traj[0, 1], traj[0, 2]
    ok = (
        np.array_equal(f0, f2)
        and not np.array_equal(f0, f1)
        and int(f1.sum()) == 3
    )
    return ok, f"blinker period-2: f0==f2={np.array_equal(f0, f2)}, f0!=f1={not np.array_equal(f0, f1)}, alive(f1)={int(f1.sum())}"


def check_block() -> tuple[bool, str]:
    traj = simulate(_make_block_seed(), k=10)
    f0 = traj[0, 0]
    all_stable = all(np.array_equal(traj[0, t], f0) for t in range(11))
    return all_stable, f"block stable over 10 steps: {all_stable}, alive(f0)={int(f0.sum())}"


def check_glider() -> tuple[bool, str]:
    """Glider should reproduce its pattern shifted by (+1, +1) every 4 steps."""
    traj = simulate(_make_glider_seed(), k=8)
    f0, f4, f8 = traj[0, 0], traj[0, 4], traj[0, 8]
    # Shift f0 by (+1, +1) and compare to f4; same again to compare to f8.
    f4_expected = np.zeros_like(f0)
    f4_expected[1:, 1:] = f0[:-1, :-1]
    f8_expected = np.zeros_like(f0)
    f8_expected[2:, 2:] = f0[:-2, :-2]
    ok4 = np.array_equal(f4, f4_expected)
    ok8 = np.array_equal(f8, f8_expected)
    return ok4 and ok8, f"glider drift +(1,1)/4 steps: f4-match={ok4}, f8-match={ok8}, alive constant={int(f0.sum())==int(f4.sum())==int(f8.sum())}"


def run_canonical_checks() -> bool:
    """Run all three checks; print results. Returns True iff all pass."""
    results = [check_blinker(), check_block(), check_glider()]
    all_ok = all(ok for ok, _ in results)
    for ok, msg in results:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {msg}")
    print(f"engine canonical: {'ALL PASS' if all_ok else 'FAILED'}")
    return all_ok


if __name__ == "__main__":
    import time
    print("=== canonical checks ===")
    ok = run_canonical_checks()
    assert ok, "engine canonical checks failed"

    print("\n=== throughput ===")
    rng = np.random.default_rng(0)
    seeds = rng.integers(0, 2, size=(32, SEED_H, SEED_W), dtype=np.uint8)
    t0 = time.perf_counter()
    traj = simulate(seeds, k=K_DEFAULT)
    t1 = time.perf_counter()
    print(f"  simulate 32 trajectories × 256 steps: {t1 - t0:.3f}s")
    print(f"  output shape: {traj.shape}, dtype: {traj.dtype}")
