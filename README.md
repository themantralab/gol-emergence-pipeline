# GoL Emergence Discovery System

A research program for **unsupervised discovery of emergent behaviour in Conway's Game of Life (B3/S23)**, by [Mantra Labs](https://mantra-labs.com). A faithful **autoencoder world model** encodes GoL frames into a latent space; a planned secondary model analyses trajectories on that manifold to surface emergent structure.

> **⚠️ Design history.** This project went through two architecture pivots. It began as a four-stage design with a *learned latent transition function*; pivoted (2026-05-27) to an *encode/decode-only* model with a structured **hypershell** latent; and finally (2026-06-07) to the **vanilla autoencoder** documented here after the structured-latent recipe hit a hard ~0.74 reconstruction ceiling. The GoL engine is always the simulator — the model never predicts dynamics. The `design/` specs describe the earlier hypershell design and are kept for the record; **this README + [`design/03_final_architecture.md`](design/03_final_architecture.md) are canonical for the current model.**

---

## The model

A plain convolutional autoencoder: `128×128 binary grid → z ∈ ℝ¹⁰²⁴ → 128×128 logits`. No sphere, no L2-normalisation — the latent is a free real vector, softly bounded near unit norm. **~67M parameters, CPU-only training.**

- **Encoder (tile-disjoint):** three `kernel=2, stride=2` downsample stages (128→64→32→16) with `1×1` channel-mixing refine blocks, then a linear projection to ℝ¹⁰²⁴. Each of the 16×16 latent positions sees exactly one disjoint 8×8 input tile — **no overlapping receptive fields**.
- **Decoder (halo-free):** a linear projection to 16×16×128, then three `PixelShuffle` 2× upsample stages with `kernel=1` convs (channels rearranged spatially, never mixed across neighbours) and a final `1×1` conv. Because no layer mixes signal between adjacent output pixels, the decoder cannot produce a probability "halo" around true cells.

### Four losses

| Loss | Role |
|---|---|
| **L1** Reconstruction | BCE-with-logits, positive class weighted (curriculum, below). Primary objective. |
| **L2** Smoothness | `cos_sim(z_a, z_b) ≈ IoU(grid_a, grid_b)`, with `cos_sim` clamped ≥ 0 — similar frames → parallel latents, disjoint → orthogonal. |
| **L3** Soft norm bound | `((‖z‖² − 1)²)` — keeps ‖z‖ near 1 without a hard constraint. |
| **L4** Angular uniformity | Wang & Isola (2020) — spreads unit directions across the sphere, preventing angular collapse. |

## Training recipe

The bare loss set does not converge from random init; three scheduled ingredients make it work (see the development notes for the full diagnosis):

- **pos_weight curriculum** — L1's positive weight decays `50 → 5` over 10k steps. High early weight forces the decoder to predict alive cells *frame-dependently*, which forces it to use `z`, which breaks an otherwise-fatal **encoder collapse**.
- **L3 warmup** — the norm bound ramps `0 → 0.05` over 1k steps so its early ‖z‖→1 correction doesn't lock in the collapse.
- **Geometry-weight decay** — L2 and L4 weights hold, then decay to ×0.15 over steps 10k–30k, so **L1 owns the late-training gradient** (a gradient-budget analysis showed the regularizers otherwise consume ~87% of the encoder gradient and cap reconstruction at ~0.79).

Seeds are embedded at **center-biased random offsets** (not a fixed corner) for translation diversity, and trajectories are sampled with a **late-generation bias** so evolved structures are well represented. Full-state checkpointing makes training resumable (`--resume`, `--lr` warm-restart).

## Results

**alive_F1 = 0.917** on held-out per-quartile validation (step 148,500; ≈0.90 at the optimal threshold). Dead-pixel F1 ≈ 0.9997. The model is **false-negative-free on nearly every frame** — it finds essentially every live cell — with residual error being false positives (the ~390:1 dead:alive sparsity tax). Simple/sparse frames reach perfect reconstruction; the residual weak spot is dense, chaotic, mid-life states.

![reconstruction samples](figures/reconstruction_samples.png)

### What the latent supports (measured)

The repo includes diagnostic scripts that characterise the trained latent honestly:

- ✅ **Faithful & stable** — real-frame cycle-consistency (decode→re-encode) drift 0.05, cos 0.997.
- ✅ **Trajectories are separable coherent paths** — within/cross-trajectory latent separation ~13×.
- ✅ **Organised by frame (IoU) similarity** — `cos_sim ↔ IoU` monotonic.
- ⚠️ **Position-variant, not invariant** — identical dynamics ~10 cells apart look as different as unrelated dynamics. Trajectory comparison must fix seed placement to a **canonical position**. (`figures/border_test.png` shows competence is a bump centered on the grid center.)
- ❌ **Not a dynamics substrate** — a trained `z_t → z_{t+1}` predictor collapses under closed-loop rollout (F1 0.40 at 1 step → ~0 by 60) while the AE ceiling stays ~0.90. **A reconstruction-faithful latent of a chaotic CA does not linearise its dynamics** — so the secondary model is trajectory *comparison*, not latent rollout/generation.

## Repository structure

```
engine.py                 GoL B3/S23 simulator (vectorised, center-biased placement)
data.py                   TrainingSeedPool — stratified sampling over 1.5M seeds
model.py                  Encoder / Decoder (tile-disjoint, halo-free)
losses.py                 L1–L4 + curricula/schedules
train.py                  Training loop (prefetcher, resumable, curricula)
visualize.py              Reconstruction montages
threshold_sweep.py        F1 vs decision threshold + per-quartile
border_test.py            Reconstruction F1 vs seed placement (border competence)
latent_organization.py    Does latent distance track frame similarity?
world_model_readiness.py  Interpolation / perturbation / prior-sampling probes
dynamics_probe.py         Cycle-consistency + latent-dynamics rollout
trajectory_readiness.py   Trajectory separability + the position confound
traj_cluster_demo.py      Trajectory retrieval / clustering demo
design/                   Design specs (01/02 = superseded hypershell; 03 = current)
DATASET.md                Dataset manifest (which files the model consumes)
```

## Dataset

1.5M random 16×16 seeds and precomputed lifespan/behaviour metadata, hosted on [Hugging Face](https://huggingface.co/datasets/themantralab/gol-emergence-pipeline). Only `seeds.npy` (+ `lifespans.npy` for stratification) is consumed by the world model; the rest is future-expansion or external diagnosis — see [`DATASET.md`](DATASET.md).
