# GoL Emergence Discovery System

A research program for **unsupervised discovery of emergent behaviour in Conway's Game of Life (B3/S23)**, by [Mantra Labs](https://mantra-labs.com). A faithful **autoencoder world model** encodes GoL frames into a latent space; a planned secondary model analyses trajectories on that manifold to surface emergent structure.

> **⚠️ Design history.** This project went through two architecture pivots. It began as a four-stage design with a *learned latent transition function*; pivoted (2026-05-27) to an *encode/decode-only* model with a structured **hypershell** latent; and finally (2026-06-07) to the **vanilla autoencoder** documented here after the structured-latent recipe hit a hard ~0.74 reconstruction ceiling. The GoL engine is always the simulator — the model never predicts dynamics. The superseded hypershell/four-stage design documents have been **removed** rather than kept, because they describe a model that does not exist and were a reliable source of confusion; **this README + [`design/03_final_architecture.md`](design/03_final_architecture.md) are canonical.**

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

**alive_F1 = 0.9169** on the full held-out validation set at threshold 0.5 (step 148,500). Tuning the threshold to 0.80 reaches **0.9305** on a 600-frame sample. Dead-pixel F1 ≈ 0.9997. The model is **false-negative-free on nearly every frame** — it finds essentially every live cell — with residual error being false positives (the ~390:1 dead:alive sparsity tax). Simple/sparse frames reach perfect reconstruction; the residual weak spot is dense, chaotic, mid-life states.

![reconstruction samples](figures/reconstruction_samples.png)

### What the latent supports (measured)

The repo includes diagnostic scripts that characterise the trained latent honestly:

All figures below are from `checkpoints/best.pt` (step 148,500). Where a number
is given as `a ± b` it is mean ± s.d. over 5 seeds (`multi_seed.py`).

- ✅ **Faithful & stable** — real-frame cycle-consistency (decode→re-encode) drift **0.034**, cos **0.999**.
- ✅ **Organised by frame (IoU) similarity** — `cos_sim ↔ IoU` monotonic: 0.006 (disjoint) → 0.744 (IoU 0.2–0.35) → **0.958** (IoU > 0.7).
- ✅ **Trajectories are separable coherent paths** — but the separation ratio is **sampling-dependent**: 8.2× on Q3-only sampling, 13.1× across all quartiles. Quote the sampling.
- ⚠️ **Position-variant, not invariant** — sweeping a diagonal displacement of an *identical* trajectory, latent cosine falls 1.000 → 0.700 (1 cell) → **0.294 (3 cells)**, against a cross-seed floor of 0.115 ± 0.207. **Three cells is enough** for identical dynamics to look unrelated. This follows directly from the L2 objective: it trains `cos → IoU`, and translated copies have IoU ≈ 0. Trajectory comparison must fix placement to a canonical position.
- ❌ **Not a dynamics substrate** — a trained `z_t → z_{t+1}` predictor collapses under closed-loop rollout (**0.344 ± 0.011** at h=1 → **0.009 ± 0.003** at h=60) while the teacher-forced AE ceiling stays **flat at 0.929–0.946** at every horizon. Capacity is fine; traversal is not. The null model of predicting no change beats the learned rollout at every horizon. (One-step latent prediction is separately data-limited — held-out/persistence MSE moves 7.117 → 0.940 as training trajectories go 10 → 320 — so the rollout result, not the one-step result, is the robust one.)
- ❌ **Not a behaviour descriptor** — classifying four never-trained hand-labelled behaviour classes from the frozen mean-pooled latent reaches **0.625 ± 0.012** balanced accuracy, against **0.842 ± 0.010** for nine hand-computed population statistics (chance 0.25). Note the labels were plausibly defined by population heuristics, so the baseline is advantaged; the honest reading is *no advantage*, not *useless*.
- ⚠️ **Confident on invalid outputs** — latents sampled from the prior at the training norm decode to **~6.7× the real live-cell density** (315–320 vs 47.8) while the decoder reports max-probability 1.000 and an ambiguous-pixel fraction of 0.024. **Decoder sharpness is not a validity signal.**
- ✅ **Sub-frame motif detection works, in the trunk not in `z`** — one flipped pixel changes ~988 of `z`'s 1024 dimensions, but exactly 1 of the conv trunk's 256 tile positions. Scoring tiles by best cosine against a bank of a motif's trunk codes at all 36 intra-tile offsets matches sliding-window template matching on clean motifs and degrades more gracefully under corruption (glider, 2 of 9 cells flipped: **0.919 ± 0.010** vs 0.846 ± 0.017). Template matching wins at 1 flipped cell.

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
persistence_baseline.py   Rollout vs the predict-no-change null model + data scaling
trajectory_readiness.py   Trajectory separability + the position confound
n4_offset_curve.py        Latent similarity vs translation distance
probe_behavior_class.py   Frozen latent vs population statistics on behaviour classes
motif_tolerance.py        Sub-frame motif detection vs template matching
multi_seed.py             5-seed error bars for the load-bearing measurements
traj_cluster_demo.py      Trajectory retrieval / clustering demo
make_figures.py           Figure generation
design/03_final_architecture.md   Canonical architecture doc
DATASET.md                Dataset manifest (which files the model consumes)
```

## Dataset

1.5M random 16×16 seeds and precomputed lifespan/behaviour metadata, hosted on [Hugging Face](https://huggingface.co/datasets/themantralab/gol-emergence-pipeline). Only `seeds.npy` (+ `lifespans.npy` for stratification) is consumed by the world model; the rest is future-expansion or external diagnosis — see [`DATASET.md`](DATASET.md).
