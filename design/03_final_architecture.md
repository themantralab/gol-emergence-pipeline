# Architecture

The design goal was **pixel-exact reconstruction of Game of Life frames**. That
goal conflicts with how convolutional autoencoders normally behave, and three
constraints shaped every decision below.

## The three binding constraints

**Spatial mixing produces a halo.** Ordinary convolutions blend neighbouring
pixels. For natural images that is desirable; for a cellular automaton it is
fatal — the decoder emits a blur of probability around each live cell, and exact
cell positions become unrecoverable. Any architecture that mixes signal between
adjacent output pixels has a floor on how exact it can be.

**A unit-sphere latent spends a degree of freedom.** Constraining `‖z‖ = 1`
forces magnitude and direction to be carried by direction alone. Earlier
structured-latent variants under that constraint plateaued around
alive-F1 ≈ 0.74.

**Multi-loss weighting is unstable without a schedule.** With reconstruction and
geometry losses competing at fixed weights, one term reliably hijacks the
gradient. A gradient-budget measurement during training showed the geometry
terms consuming the large majority of the encoder gradient while reconstruction
was starved.

The architecture is the response to those three.

## Model

- **Latent:** `z ∈ ℝ¹⁰²⁴`. No L2-normalisation, no sphere. Magnitude and
  direction are both free, softly bounded near `‖z‖ ≈ 1` by L3.
- **Encoder (tile-disjoint):** 3× (`Conv2d k=2 s=2` downsample + `1×1`
  channel-mix refine), 128→64→32→16 spatial, 1→32→64→128 channels, then
  `Linear(128·16·16 → 1024)`. Each of the 16×16 trunk positions has a receptive
  field of exactly one disjoint 8×8 input tile — no overlap between adjacent
  positions. Verify with `python3 model.py`.
- **Decoder (halo-free):** `Linear(1024 → 128·16·16)`, then 3× (`PixelShuffle`
  2× upsample with `k=1` conv + `1×1` refine), then a final `1×1` conv to
  logits. The `Linear` is the sole global mixing point; everything downstream is
  per-pixel, so no signal crosses between adjacent output pixels.
- ~67M parameters. CPU-only (8 threads plus a background simulation/encode
  prefetcher).

The tile-disjoint property was adopted purely to eliminate the halo. It has a
second consequence that matters for analysis: a trunk position's 128-dimensional
code is a pure function of its own 8×8 tile, which makes the trunk — unlike `z` —
a spatially localised representation. `motif_tolerance.py` exploits this.

## Losses

- **L1** — BCE-with-logits reconstruction, positive-class weighted. Primary.
- **L2** — smoothness: `MSE(clamp₀(cos_sim(z_a,z_b)), IoU(grid_a,grid_b))`.
  Similar frames → parallel latents; disjoint → orthogonal.
  *Targeting IoU directly, rather than mapping `1−IoU` onto cosine distance,
  avoids a geometric floor: disjoint sparse frames would otherwise be asked to
  be mutually anti-parallel, which is impossible for a whole cloud of them.*
  **This loss determines the latent geometry**, and therefore most of what the
  latent does and does not support — see `RESULTS.md`.
- **L3** — soft norm bound `((‖z‖²−1)/1)²`. Prevents magnitude blow-up/collapse.
- **L4** — Wang & Isola (2020) hyperspherical uniformity on unit directions.
  Prevents angular collapse; L3 alone bounds magnitude, not direction.
- Steady-state weights `1.0 / 0.3 / 0.05 / 0.03`.

## Schedules (required for convergence)

The bare loss set does not converge from random init.

- **pos_weight curriculum:** cosine `50 → 5` over 10k steps, then held. A high
  early positive weight forces the decoder to predict live cells
  *frame-dependently*, which forces it to use `z` at all — this breaks an
  otherwise-fatal encoder collapse in which every frame maps to one direction.
- **L3 warmup:** `0 → 0.05` over 1k steps, so its early `‖z‖→1` correction does
  not lock in that collapse.
- **Geometry-weight decay:** L2/L4 weights hold at ×1.0 through step 10k, then
  cosine-decay to ×0.15 by step 30k, so L1 owns the late-training gradient.
- **LR:** cosine `3e-4 → 3e-5` over 100k steps; warm-restartable.

## Data placement

Seeds (16×16) are embedded at **centre-biased random offsets** — Gaussian, mean
56, std 10, clipped to [24, 88] — rather than at a fixed corner, for translation
diversity. Frames are sampled per trajectory with a late-generation bias
(`t = U^0.5 · lifespan`) so evolved structures are well represented.

This placement choice bounds where the model is competent: reconstruction F1 is
0.921 at the trained centre and falls to 0.325 at the grid border
(`border_test.py`). Note also that it is *not* translation invariance — a
three-cell displacement is enough to make an identical pattern look unrelated in
latent space (`n4_offset_curve.py`), which follows directly from the L2
objective above.

## Result and scope

**alive-F1 = 0.9169** at step 148,500 (dead-F1 0.9997), with round-trip cosine
0.999 on real frames.

The latent is a faithful appearance index and nothing more. It does not
linearize the dynamics, does not describe behaviour better than nine hand-
computed statistics, is not translation invariant, and decodes confidently
invalid output off-manifold. `RESULTS.md` has every measurement with the script
that produces it.
