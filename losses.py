"""
Four losses for the vanilla GoL autoencoder.

  L1 — Reconstruction (BCE with pos_weight)
       Per-pixel binary cross-entropy. Primary objective: the decoder must
       round-trip a frame to its actual pixel grid. pos_weight=5 corrects
       the heavy dead/alive imbalance (~300:1).

  L2 — Smoothness (cosine ↔ sqrt(Hamming))
       For pairs of frames in the batch, cosine distance between encoded
       directions ≈ sqrt-scaled normalised Hamming distance between grids.
       Calibrates local geometry: similar inputs → similar latent directions.

  L3 — Soft norm bound
       Squared penalty on (‖z‖² - target²). Keeps latent magnitudes bounded.
       2026-06-07: TARGET_NORM changed from 10 → 1 so magnitudes are dimensionless
       and the latent is effectively a soft-unit-sphere without the brittleness
       of hard F.normalize.

  L4 — Angular uniformity (Wang & Isola 2020)  ← added 2026-06-07
       Pushes the unit-direction cloud to be spread across the sphere rather
       than collapsed into a narrow angular cone. The L3 norm bound alone
       prevents *magnitude* collapse but not *angular* collapse — diagnostic
       on the previous run found mean pairwise cosine similarity = 0.87
       (collapsed) and effective latent dimensionality 33 of 1024. L4
       explicitly punishes nearby unit-direction pairs.

All four combined into a single training objective via weighted sum.
Weights are biased toward L1 dominance and L4 is intentionally small (it
went very negative when given high weight in earlier runs, which caused
hijack failures).
"""

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Hyperparameters (tunable)
# -----------------------------------------------------------------------------

ALIVE_POS_WEIGHT = 5.0    # L1 BCE positive-class weight (final/steady-state value)

# Curriculum on the BCE positive-class weight. The 1×1 decoder cannot bootstrap
# at the steady-state pos_weight=5 — it sits in the "predict all dead" minimum
# with a collapsed latent (diagnostic 2026-06-08 confirmed: pairwise cos sim
# = 1.000 across all latents, max prob 0.41 < 0.5 threshold). At pos_weight=50
# missing an alive cell is so costly that the decoder MUST predict alive
# somewhere, and the only way to do that frame-dependently is to actually USE
# z — which forces the encoder to differentiate latents.
#
# Decay from POS_WEIGHT_START to ALIVE_POS_WEIGHT (5.0) via cosine over
# POS_WEIGHT_DECAY_STEPS, then hold at 5.0 for the remainder of training.
POS_WEIGHT_START       = 50.0
POS_WEIGHT_DECAY_STEPS = 10_000   # ~10% of total training

# L3 (norm bound) warmup. The encoder's Linear(32768→1024) projects to ‖z‖≈22
# at random init. Without warmup, L3 forces a catastrophic ‖z‖→1 correction in
# the first ~100 steps, and the cheapest way to satisfy that for all inputs
# simultaneously is to align all latents along one direction → collapse.
# Warmup gives L2 (Jaccard anti-collapse) time to establish latent
# differentiation before magnitude constraint locks in.
NORM_WARMUP_STEPS = 1_000
TARGET_NORM      = 1.0    # L3 soft-bound target ‖z‖ (was 10.0; reduced 2026-06-07
                          # so latent magnitude is dimensionless / unit-scale)
L4_T             = 2.0    # L4 Wang-Isola decay constant


# -----------------------------------------------------------------------------
# L1 — Reconstruction
# -----------------------------------------------------------------------------

def reconstruction_loss(logits: torch.Tensor, targets: torch.Tensor,
                        pos_weight: float = ALIVE_POS_WEIGHT) -> torch.Tensor:
    """Per-pixel BCE-with-logits, positive class weighted."""
    pw = torch.tensor([pos_weight], device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


def pos_weight_at_step(step: int,
                       start: float = POS_WEIGHT_START,
                       end: float = ALIVE_POS_WEIGHT,
                       decay_steps: int = POS_WEIGHT_DECAY_STEPS) -> float:
    """Cosine decay from `start` to `end` over `decay_steps`, then hold at `end`."""
    if step >= decay_steps:
        return end
    import math
    progress = step / decay_steps
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress))


def norm_weight_at_step(step: int,
                        end: float = None,
                        warmup_steps: int = NORM_WARMUP_STEPS) -> float:
    """Linear warmup of L3 weight from 0 to `end` over `warmup_steps`. The
    caller passes the steady-state W_NORM as `end`. Keeps the encoder free
    of magnitude pressure while L2 establishes anti-collapse direction."""
    if step >= warmup_steps:
        return end
    return end * (step / warmup_steps)


# -----------------------------------------------------------------------------
# L2 — Smoothness (pairwise cosine ↔ sqrt(Hamming))
# -----------------------------------------------------------------------------

def smoothness_loss(z: torch.Tensor, grids: torch.Tensor,
                    n_sample: int = 256) -> torch.Tensor:
    """For pairs of frames in the batch:
        cosine_similarity(z_a, z_b) ≈ IoU(grid_a, grid_b)

    IoU = |A ∩ B| / |A ∪ B| on alive-cell sets ∈ [0, 1]. Identical frames
    target cos_sim = 1 (parallel); disjoint frames target cos_sim = 0
    (orthogonal). cos_sim is clamped at 0 so disjoint pairs are never pushed
    past orthogonal toward anti-parallel — that's the key difference from the
    original 1-IoU/cos_dist formulation, which demanded near-anti-parallel
    latents for the (overwhelmingly disjoint) sparse GoL frames and so left
    L2 frustrated at a permanent floor. See the comment in the body and the
    2026-06-30 grad-budget diagnosis.

    History: an even earlier version targeted sqrt(normalised_hamming), which
    was tiny (~0.1 even for disjoint frames) because Hamming is dominated by
    the dead-cell majority — that let collapsed latents satisfy it and failed
    to prevent encoder collapse (2026-06-08 bootstrap_diag).

    Cosine is computed on raw (un-normalised) z vectors via z·z' / (‖z‖‖z'‖)
    so this term only constrains the *direction* of the latent, leaving
    magnitudes free for L1 reconstruction to use as it sees fit.
    """
    N = z.shape[0]
    if N > n_sample:
        idx = torch.randperm(N, device=z.device)[:n_sample]
        z = z[idx]
        grids = grids[idx]

    # Pairwise IoU on the binary alive-cell sets → directly the target cos-sim.
    # 2026-06-30: target is now IoU (identical sets → cos_sim 1; disjoint sets →
    # cos_sim 0 = orthogonal), NOT 1-IoU mapped onto cos_dist. The old version
    # asked disjoint frames for cos_sim = -1 (anti-parallel); since mean pairwise
    # Jaccard ≈ 0.94 for sparse GoL frames, that demanded the average latent pair
    # be near-anti-parallel — geometrically impossible for a whole cloud on a
    # sphere → L2 stuck at a permanent floor ≈ 0.23 with a large frustrated
    # gradient that strangled reconstruction (grad-budget diag: L2 drove 53% of
    # the encoder gradient vs L1's 7.5%). Targeting orthogonality for disjoint
    # frames is achievable for the whole cloud, so L2 can now reach ~0.
    g_flat = grids.flatten(1).float()
    intersection = g_flat @ g_flat.T                              # |A ∩ B|
    counts       = g_flat.sum(dim=-1, keepdim=True)               # |A|
    union        = counts + counts.T - intersection               # |A ∪ B|
    target_cos   = intersection / union.clamp_min(1.0)           # IoU ∈ [0, 1]

    # Cosine similarity on the latents, clamped at 0 so the loss never rewards
    # going *beyond* orthogonal (anti-parallel) for disjoint frames.
    z_unit = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
    cos_sim = (z_unit @ z_unit.T).clamp_min(0.0)                  # in [0, 1]

    # Upper triangle of pairs only (avoid self-pairs and duplicates)
    mask = torch.triu(torch.ones_like(target_cos, dtype=torch.bool), diagonal=1)
    return F.mse_loss(cos_sim[mask], target_cos[mask])


# -----------------------------------------------------------------------------
# L3 — Soft norm bound
# -----------------------------------------------------------------------------

def norm_bound_loss(z: torch.Tensor, target_norm: float = TARGET_NORM) -> torch.Tensor:
    """Soft constraint: ‖z‖² should stay near target_norm².

    Loss = mean( ((‖z‖² - target²) / target²)² )

    Quadratic penalty centred at the target; divided by target⁴ so the loss
    magnitude is dimensionless and comparable across choices of target_norm.
    """
    norms_sq = (z * z).sum(dim=-1)                     # (N,) — ‖z_i‖²
    target_sq = target_norm ** 2
    rel_err = (norms_sq - target_sq) / target_sq        # dimensionless
    return (rel_err ** 2).mean()


# -----------------------------------------------------------------------------
# L4 — Angular uniformity (Wang & Isola 2020)
# -----------------------------------------------------------------------------

def angular_uniformity_loss(z: torch.Tensor, t: float = L4_T) -> torch.Tensor:
    """Pushes unit-direction pairs apart on the sphere.

    Returns log(E[exp(-t·‖û_i - û_j‖²)]) over distinct pairs of L2-normalized
    z. Minimised when unit vectors are uniformly spread; large (≈ 0) when
    they are collapsed to one direction.

    z is normalised internally so this loss only affects ANGULAR distribution,
    leaving magnitude to L3.
    """
    z_unit = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
    sq_dist = torch.pdist(z_unit, p=2).pow(2)
    return sq_dist.mul(-t).exp().mean().log()


# -----------------------------------------------------------------------------
# Combined
# -----------------------------------------------------------------------------

W_RECON  = 1.0
W_SMOOTH = 0.3
W_NORM   = 0.05
W_UNIF   = 0.03   # L4 weight — small to avoid the v3-style hijack.
                  # L4 can go very negative; weighted contribution stays bounded.

# Late-training decay of the latent-geometry weights (W_SMOOTH, W_UNIF).
# Their job — break encoder collapse and spread the latent — is done by the
# time the pos_weight curriculum ends (~step 10k; diagnostic at step 7k already
# showed mean pairwise cos sim 0.06). After that they only compete with L1 for
# the encoder's gradient budget (grad-budget diag 2026-06-30: L2+L4 = 87% of
# encoder gradient, L1 only 7.5% → reconstruction plateaus). So hold them at
# full strength through GEOM_HOLD_STEPS, then cosine-decay to GEOM_FLOOR× over
# the next window. The floor (not zero) keeps mild anti-collapse insurance.
GEOM_HOLD_STEPS  = 10_000
GEOM_DECAY_STEPS = 20_000   # decay spans [HOLD, HOLD + DECAY] = steps 10k..30k
GEOM_FLOOR       = 0.15     # final multiplier: W_SMOOTH→0.045, W_UNIF→0.0045


def geometry_weight_scale(step: int,
                          hold: int = GEOM_HOLD_STEPS,
                          decay: int = GEOM_DECAY_STEPS,
                          floor: float = GEOM_FLOOR) -> float:
    """Multiplier in [floor, 1] applied to BOTH W_SMOOTH and W_UNIF. 1.0 while
    the latent differentiates (step < hold), cosine-decays to `floor` over the
    next `decay` steps, then holds at `floor`."""
    if step <= hold:
        return 1.0
    if step >= hold + decay:
        return floor
    import math
    progress = (step - hold) / decay
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def combined_loss(logits: torch.Tensor, targets: torch.Tensor,
                  z: torch.Tensor, grids: torch.Tensor,
                  w1: float = W_RECON, w2: float = W_SMOOTH,
                  w3: float = W_NORM,  w4: float = W_UNIF,
                  pos_weight: float = ALIVE_POS_WEIGHT,
                  ) -> tuple[torch.Tensor, dict]:
    """L1 + w2·L2 + w3·L3 + w4·L4. Returns (total, components_for_logging).
    `pos_weight` is normally supplied by the training loop from
    `pos_weight_at_step(step)` so it follows the curriculum."""
    l1 = reconstruction_loss(logits, targets, pos_weight=pos_weight)
    l2 = smoothness_loss(z, grids)
    l3 = norm_bound_loss(z)
    l4 = angular_uniformity_loss(z)
    total = w1 * l1 + w2 * l2 + w3 * l3 + w4 * l4
    return total, {"l1": l1.detach().item(),
                   "l2": l2.detach().item(),
                   "l3": l3.detach().item(),
                   "l4": l4.detach().item(),
                   "pos_weight": pos_weight}


# -----------------------------------------------------------------------------
# Reconstruction-quality metrics for validation
# -----------------------------------------------------------------------------

def reconstruction_metrics(logits: torch.Tensor, targets: torch.Tensor,
                           threshold: float = 0.5) -> dict:
    """Per-pixel binary classification metrics aggregated across the batch."""
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > threshold).float()
        tp = ((preds == 1) & (targets == 1)).sum().item()
        fp = ((preds == 1) & (targets == 0)).sum().item()
        fn = ((preds == 0) & (targets == 1)).sum().item()
        tn = ((preds == 0) & (targets == 0)).sum().item()

        def f1(tp_, fp_, fn_):
            denom = 2 * tp_ + fp_ + fn_
            return (2 * tp_ / denom) if denom > 0 else 0.0

        return {
            "alive_f1":        f1(tp, fp, fn),
            "dead_f1":         f1(tn, fn, fp),
            "alive_precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "alive_recall":    tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "n_alive_true":    tp + fn,
            "n_alive_pred":    tp + fp,
        }
