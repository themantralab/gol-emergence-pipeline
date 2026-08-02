"""
MOTIF-INSTANCE DETECTION on the encoder's TRUNK feature map.

THE QUESTION
------------
Whole-frame latent retrieval is dominated by exact pixel IoU (z is 4096 B/frame
vs a 2048 B bitset, it regresses onto IoU so cannot outrank it, and it has no
translation invariance). The one task left where a learned representation could
still win is SUB-FRAME motif localization: "find every frame containing a glider
ANYWHERE." Whole-frame IoU structurally cannot do that.

Localization is NOT available in z: one flipped pixel changes ~all 1024 dims,
because `Encoder.project` is a flatten+Linear over every tile. It IS available in
the conv trunk `enc.conv(grid)` -> (128, 16, 16), which is strictly tile-disjoint:
one pixel changes exactly ONE of the 256 tile positions. Tile (i,j) sees exactly
pixels [8i:8i+8, 8j:8j+8].

That disjointness is what makes this script cheap: a tile's 128-dim code is a
PURE FUNCTION of its own 8x8 content. Background outside the tile cannot affect
it, so positives need no background at all.

THE RISK BEING TESTED (the reason this might fail)
--------------------------------------------------
Tiles are 8x8; a glider is 3x3 and translates one cell per 4 steps. So a real
glider spends most of its life STRADDLING a tile boundary, and a tile-disjoint
encoder with no translation invariance may encode each intra-tile phase as a
completely different vector. That is N4's position confound at 8-pixel
granularity. Section 1 measures this directly.

BASELINE
--------
The right baseline for localization is NOT raw-frame IoU; it is sliding-window
template matching, which is translation-invariant within the tile by construction
and essentially free on binary data. The learned representation can therefore only
win on TOLERANCE - detecting motifs that are perturbed or partially occluded.
Section 3 is the only place a win is possible.
"""
from pathlib import Path
import numpy as np
import torch

import data, engine
from model import Encoder

CKPT = Path("checkpoints/best.pt")
RNG = np.random.default_rng(0)
torch.manual_seed(0)

TILE = 8
NTILE = 16
REF_OFFSET = (2, 2)          # canonical intra-tile placement for the reference

# 3x3 bounding boxes, taken from engine._make_*_seed()
MOTIFS = {
    "glider":  np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.uint8),
    "blinker": np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8),
    "block":   np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8),
}


def trunk_codes(enc, grids):
    """(B,128,128) uint8 -> (B, 256, 128) per-tile feature vectors."""
    out = []
    with torch.no_grad():
        for i in range(0, len(grids), 32):
            x = torch.from_numpy(grids[i:i + 32].astype(np.float32)).unsqueeze(1)
            h = enc.conv(x)                       # (b,128,16,16)
            b = h.shape[0]
            out.append(h.reshape(b, 128, NTILE * NTILE).permute(0, 2, 1))
    return torch.cat(out)                          # (B,256,128)


def plant(grid, motif, tile_rc, off_rc):
    """Write motif into grid at tile (ti,tj) with intra-tile offset (r,c)."""
    ti, tj = tile_rc
    r, c = off_rc
    R = ti * TILE + r
    C = tj * TILE + c
    h, w = motif.shape
    if R + h > 128 or C + w > 128:
        return False
    grid[R:R + h, C:C + w] |= motif
    return True


def template_score_tile(tile8, motif, mode="penalized"):
    """Best normalized match of `motif` anywhere in an 8x8 tile (zero-padded).

    Translation-invariant within the tile by construction. Two variants, because
    the choice of match metric materially changes how the baseline degrades under
    perturbation and it would be unfair to report only the harsher one:

      "penalized" : (hits - extras) / |motif|, in [-1,1]. Punishes spurious live
                    cells inside the footprint, so flipped-ON cells hurt twice.
      "iou"       : |win AND motif| / |win OR motif|, in [0,1]. The forgiving
                    variant - a flipped cell costs a little, not a lot. This is
                    the STRONGER baseline under perturbation.
    """
    h, w = motif.shape
    pad = np.zeros((TILE + h - 1, TILE + w - 1), dtype=np.uint8)
    pad[:TILE, :TILE] = tile8
    on = int(motif.sum())
    best = -1.0
    for r in range(TILE):
        for c in range(TILE):
            win = pad[r:r + h, c:c + w]
            inter = int((win & motif).sum())
            if mode == "iou":
                union = int((win | motif).sum())
                s = (inter / union) if union else 0.0
            else:
                extra = int((win & (1 - motif)).sum())
                s = (inter - extra) / on
            if s > best:
                best = s
    return best


def perturb(motif, n_flip, rng):
    """Flip n_flip cells inside the motif's 3x3 box."""
    m = motif.copy()
    if n_flip == 0:
        return m
    idx = rng.choice(m.size, size=n_flip, replace=False)
    flat = m.reshape(-1)
    flat[idx] ^= 1
    return flat.reshape(m.shape)


def main():
    torch.set_num_threads(8)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"Checkpoint step={ckpt['step']} F1={ckpt['metrics']['alive_f1']:.4f}\n")
    enc = Encoder(); enc.load_state_dict(ckpt["encoder"]); enc.eval()

    # ---------- real tiles, used as detection NEGATIVES ----------
    pool = data.TrainingSeedPool()
    idx = RNG.choice(pool.quartile_pools[2], size=40, replace=False)
    seeds = np.asarray(pool.seeds[idx])
    offs = engine.sample_center_biased_offsets(40, RNG)
    trajs = engine.simulate(seeds, k=64, offsets=offs)
    real = trajs[:, ::8].reshape(-1, 128, 128)                # (320,128,128)
    real_codes = trunk_codes(enc, real)                       # (320,256,128)
    real_flat = real_codes.reshape(-1, 128)
    real_tiles = real.reshape(-1, NTILE, TILE, NTILE, TILE).transpose(0, 1, 3, 2, 4)
    real_tiles = real_tiles.reshape(-1, TILE, TILE)           # (320*256,8,8)
    nonempty = real_tiles.reshape(len(real_tiles), -1).sum(1) > 0
    print(f"Real frames: {len(real)}  |  non-empty real tiles: {int(nonempty.sum())} "
          f"of {len(real_tiles)}\n")

    # ---- NEGATIVE-SET CONTAMINATION -------------------------------------
    # Blocks and blinkers occur naturally in GoL ash, so some "negative" tiles
    # genuinely CONTAIN the motif being detected. That caps achievable AUC and
    # makes absolute numbers pessimistic. It hits BOTH methods equally, so the
    # comparison stays valid - but the rate must be reported, not assumed away.
    print("=" * 72)
    print("NEGATIVE-SET CONTAMINATION - real tiles that truly contain the motif")
    print("=" * 72)
    neg_pool = real_tiles[nonempty]
    for name, mot in MOTIFS.items():
        on = int(mot.sum()); hits = 0
        for t in neg_pool:
            pad = np.zeros((TILE + 2, TILE + 2), dtype=np.uint8)
            pad[:TILE, :TILE] = t
            found = False
            for r in range(TILE):
                for c in range(TILE):
                    w = pad[r:r + 3, c:c + 3]
                    if int((w & mot).sum()) == on and int((w & (1 - mot)).sum()) == 0:
                        found = True; break
                if found: break
            hits += found
        rate = hits / len(neg_pool)
        print(f"  {name:>8}: {hits:>5}/{len(neg_pool)} negatives contain an exact "
              f"{name}  ({100 * rate:.1f}%)  -> approx AUC ceiling {1 - rate / 2:.3f}")
    print("  => GLIDER is the clean case (near-zero contamination) AND the only one")
    print("     of the three that TRANSLATES, so intra-tile phase is a genuine")
    print("     problem for it rather than an artifact of the planting scheme.")
    print("     LEAD WITH GLIDER; blinker/block absolute AUCs are depressed for")
    print("     BOTH methods and are secondary.")
    print("  NOTE the 'AUC ceiling' above is a CRUDE UPPER BOUND that assumes every")
    print("  contaminated negative scores at the very top. Measured AUCs can and do")
    print("  EXCEED it (e.g. blinker bank 0.998 > 0.963) because a real tile holding")
    print("  a blinker usually holds OTHER live cells too, so its code differs from a")
    print("  lone blinker and it does not score maximally. Treat the bound as")
    print("  indicative, not as a hard cap - and do NOT report an AUC above it as a")
    print("  contradiction.")
    print()

    for name, motif in MOTIFS.items():
        print("=" * 72)
        print(f"MOTIF: {name}   ({int(motif.sum())} live cells in a 3x3 box)")
        print("=" * 72)

        # ---------- reference code: motif alone, canonical offset ----------
        g = np.zeros((1, 128, 128), dtype=np.uint8)
        plant(g[0], motif, (8, 8), REF_OFFSET)
        ref = trunk_codes(enc, g)[0, 8 * NTILE + 8]           # (128,)

        # ================= 1. INTRA-TILE PHASE SENSITIVITY =================
        # Sweep the motif across all intra-tile offsets that keep it inside the
        # tile (0..5 for a 3x3 motif), measuring cosine to the reference.
        print("\n-- 1. intra-tile phase sensitivity (cosine to reference) --")
        offs_list = [(r, c) for r in range(TILE - 2) for c in range(TILE - 2)]
        grids = np.zeros((len(offs_list), 128, 128), dtype=np.uint8)
        for k, (r, c) in enumerate(offs_list):
            plant(grids[k], motif, (8, 8), (r, c))
        codes = trunk_codes(enc, grids)[:, 8 * NTILE + 8]      # (36,128)
        cos = torch.nn.functional.cosine_similarity(codes, ref.unsqueeze(0), dim=-1).numpy()

        print("      c=0     c=1     c=2     c=3     c=4     c=5")
        for r in range(TILE - 2):
            row = "  ".join(f"{cos[r * (TILE - 2) + c]:6.3f}" for c in range(TILE - 2))
            print(f"  r={r} {row}")
        print(f"\n  cosine to reference: min={cos.min():.3f}  mean={cos.mean():.3f}  "
              f"max={cos.max():.3f}")
        print(f"  (max=1.000 is the reference cell itself, offset {REF_OFFSET})")
        off_ref = cos[REF_OFFSET[0] * (TILE - 2) + REF_OFFSET[1]]
        others = np.delete(cos, REF_OFFSET[0] * (TILE - 2) + REF_OFFSET[1])
        print(f"  mean cosine EXCLUDING the reference offset: {others.mean():.3f}")

        bank = build_ref_bank(enc, motif)

        # ================= 2 & 3. DETECTION AUC + TOLERANCE =================
        print("\n-- 2/3. detection AUC: motif tile vs real tiles, vs perturbation --")
        print("   latent-1ref   = cosine to a single canonical reference")
        print("   latent-bank   = max cosine over all 36 offsets (phase-INVARIANT,")
        print("                   the strongest fair form of the latent method)")
        print("   template      = best within-tile normalized match (translation-")
        print("                   invariant by construction)")
        print()
        print("   tmpl-pen/-iou = template match, harsh vs forgiving metric;")
        print("                   -iou is the STRONGER baseline under perturbation")
        print()
        print(f"  {'n_flip':>7} {'latent-1ref':>12} {'latent-bank':>12} "
              f"{'tmpl-pen':>9} {'tmpl-iou':>9} {'bank - best_tmpl':>17}")
        for nf in [0, 1, 2, 3]:
            al, ab, at, ai = detection_auc(enc, motif, ref, bank, real_flat,
                                           real_tiles, nonempty, n_flip=nf)
            print(f"  {nf:>7} {al:>12.3f} {ab:>12.3f} {at:>9.3f} {ai:>9.3f} "
                  f"{ab - max(at, ai):>17.3f}")
        print()
        print("  NOTE: a 3x3 box holds 9 cells. n_flip=3 destroys roughly a third of")
        print("  the motif - at that point 'detection' may no longer mean detecting")
        print("  THIS motif, and a template baseline falling below AUC 0.5 indicates")
        print("  the perturbed shape is actively unlike the template. Read wins at")
        print("  high n_flip with that caveat firmly in mind.")
        print()


def build_ref_bank(enc, motif):
    """Reference bank: the motif's tile code at EVERY intra-tile offset.

    This is the STRONGEST fair form of the latent detector. Scoring a tile by
    max-cosine over the bank makes the latent method phase-invariant by
    construction, removing the handicap measured in section 1. If the latent
    still loses to template matching with this advantage, the negative is not an
    artifact of a badly-chosen reference.
    """
    offs_list = [(r, c) for r in range(TILE - 2) for c in range(TILE - 2)]
    grids = np.zeros((len(offs_list), 128, 128), dtype=np.uint8)
    for k, (r, c) in enumerate(offs_list):
        plant(grids[k], motif, (8, 8), (r, c))
    bank = trunk_codes(enc, grids)[:, 8 * NTILE + 8]           # (36,128)
    return torch.nn.functional.normalize(bank, dim=-1)


def detection_auc(enc, motif, ref, bank, real_flat, real_tiles, nonempty, n_flip,
                  n_pos=200):
    """AUC separating tiles containing the motif from real GoL tiles.

    Positives: the motif at a RANDOM intra-tile offset (this is the realistic
    case - a real glider is not tile-aligned), optionally perturbed.
    Negatives: non-empty tiles from real simulated frames.

    Returns (single-reference latent AUC, phase-invariant bank AUC, template AUC).
    """
    rng = np.random.default_rng(1234 + n_flip)
    grids = np.zeros((n_pos, 128, 128), dtype=np.uint8)
    pos_tiles = np.zeros((n_pos, TILE, TILE), dtype=np.uint8)
    for k in range(n_pos):
        m = perturb(motif, n_flip, rng)
        r, c = rng.integers(0, TILE - 2), rng.integers(0, TILE - 2)
        plant(grids[k], m, (8, 8), (int(r), int(c)))
        pos_tiles[k] = grids[k][64:72, 64:72]
    pos_codes = trunk_codes(enc, grids)[:, 8 * NTILE + 8]

    neg_idx = np.flatnonzero(nonempty)
    neg_idx = rng.choice(neg_idx, size=min(2000, len(neg_idx)), replace=False)
    neg_codes = real_flat[neg_idx]
    neg_tiles = real_tiles[neg_idx]

    s_pos_l = torch.nn.functional.cosine_similarity(pos_codes, ref.unsqueeze(0), dim=-1).numpy()
    s_neg_l = torch.nn.functional.cosine_similarity(neg_codes, ref.unsqueeze(0), dim=-1).numpy()
    # phase-invariant: max cosine over the whole reference bank
    s_pos_b = (torch.nn.functional.normalize(pos_codes, dim=-1) @ bank.T).max(1).values.numpy()
    s_neg_b = (torch.nn.functional.normalize(neg_codes, dim=-1) @ bank.T).max(1).values.numpy()
    s_pos_t = np.array([template_score_tile(t, motif, "penalized") for t in pos_tiles])
    s_neg_t = np.array([template_score_tile(t, motif, "penalized") for t in neg_tiles])
    s_pos_i = np.array([template_score_tile(t, motif, "iou") for t in pos_tiles])
    s_neg_i = np.array([template_score_tile(t, motif, "iou") for t in neg_tiles])
    return (auc(s_pos_l, s_neg_l), auc(s_pos_b, s_neg_b),
            auc(s_pos_t, s_neg_t), auc(s_pos_i, s_neg_i))


def auc(pos, neg):
    """Rank-based AUC (ties counted at 0.5)."""
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = len(pos), len(neg)
    return (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


if __name__ == "__main__":
    main()
