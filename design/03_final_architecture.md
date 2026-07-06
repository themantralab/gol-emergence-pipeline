# 03 — Final Architecture (canonical, 2026-07)

This supersedes `01_design_specification.md` and `02_design_rationale.md` (the
*hypershell* design). Those are retained as a historical record. The trained
model in this repository is the **vanilla autoencoder** described here.

## Why the pivot from the hypershell design

The structured-latent recipe (unit-sphere geometry, NT-Xent chain clustering,
multi-loss balancing) plateaued at **alive_F1 ≈ 0.74** across every variant. The
binding constraints were: (1) the unit-sphere constraint spent a degree of freedom
forcing magnitude+direction into direction alone; (2) no stable multi-loss
weighting existed — some loss always hijacked the gradient; (3) pixel-grid decoders
bled probability into neighbouring pixels (a "halo"). The fix was to adopt the
modern two-stage pattern (Latent Diffusion / VQ-VAE+prior): **train a faithful
vanilla autoencoder first, defer all latent structure to a later explorer model.**

## Model

- **Latent:** `z ∈ ℝ¹⁰²⁴`. No L2-normalisation, no sphere. Magnitude and direction
  are both free, softly bounded near ‖z‖≈1 by L3.
- **Encoder (tile-disjoint):** 3× (`Conv2d k=2 s=2` downsample + `1×1` channel-mix
  refine), 128→64→32→16 spatial, 1→32→64→128 channels, then `Linear(128·16·16 →
  1024)`. Each 16×16 latent position has a receptive field of exactly one disjoint
  8×8 input tile — no overlap between adjacent positions.
- **Decoder (halo-free):** `Linear(1024 → 128·16·16)`, then 3× (`PixelShuffle` 2×
  upsample with `k=1` conv + `1×1` refine), then a final `1×1` conv to logits. The
  `Linear` is the sole global mixing point; everything downstream is per-pixel, so
  no signal crosses between adjacent output pixels.
- ~67M parameters. CPU-only (8 threads + background sim/encode prefetcher).

## Losses

- **L1** — BCE-with-logits reconstruction, positive-class weighted. Primary.
- **L2** — smoothness: `MSE(clamp₀(cos_sim(z_a,z_b)), IoU(grid_a,grid_b))`. Similar
  frames → parallel latents; disjoint → orthogonal. (Targeting IoU directly, rather
  than mapping 1−IoU onto cosine *distance*, avoids a geometric floor: disjoint
  sparse frames would otherwise be asked to be anti-parallel, impossible for a whole
  cloud on a sphere.)
- **L3** — soft norm bound `((‖z‖²−1)/1)²`. Prevents magnitude blow-up/collapse.
- **L4** — Wang & Isola (2020) hyperspherical uniformity on unit directions.
  Prevents angular collapse (L3 alone bounds magnitude, not direction).
- Steady-state weights `1.0 / 0.3 / 0.05 / 0.03`.

## Schedules (required for convergence)

- **pos_weight curriculum:** cosine `50 → 5` over 10k steps, then held. Breaks the
  initial encoder collapse by forcing the decoder to use `z`.
- **L3 warmup:** `0 → 0.05` over 1k steps.
- **Geometry-weight decay:** L2/L4 weights ×1.0 through step 10k, cosine → ×0.15 by
  step 30k, so L1 dominates the late-training gradient budget.
- **LR:** cosine `3e-4 → 3e-5` over 100k steps; warm-restartable for extensions.

## Data placement

Seeds (16×16) are embedded at **center-biased random offsets** (Gaussian mean 56,
std 10, clipped [24,88]) rather than a fixed corner, for translation diversity.
Frames are sampled per trajectory with a late-generation bias (`t = U^0.5·lifespan`)
so evolved structures are well represented. Precomputed lifespans remain valid
bounds because center placement only ever gives patterns more room than the wall.

## Result & scope

**alive_F1 = 0.917** (step 148,500). The latent is faithful and supports trajectory
*comparison* (at a fixed canonical position), but **not** latent dynamics/rollout or
free generation — see the diagnostic scripts and README "What the latent supports."
