# Dataset Manifest

Pre-generated GoL dataset in `data/`. **1.5M random 16×16 seeds**, embedded at offset (24, 24) on a 128×128 grid, simulated under B3/S23.

This dataset predates the architecture pivot to the encode/decode-only world model (see `design/`). Most of it was built for the previous transition-function design and its behavioural-signal supervision. It has been **retained in full**. This document classifies every file by how the *current* world model relates to it:

- **CONSUMED** — directly used as model input or training data
- **FUTURE EXPANSION** — not used now; retained because it enables planned work (the explorer model)
- **EXTERNAL DIAGNOSIS** — not used by the model at all; retained to evaluate/verify model behaviour from the outside

> The world model's training signal comes from the **seed corpus** (the source of all trajectories) plus **lifespan/bucket metadata** (used by the stratified batch sampler). Trajectories themselves are produced by simulating seeds through the GoL engine at training time. Everything else below is either future-facing or diagnostic.

---

## CONSUMED — model input

| File | Shape / size | Role |
|------|-------------|------|
| `seeds.npy` | (1.5M, 16, 16) uint8 · 367M | The seed corpus. Embedded and simulated to produce training trajectories. |
| `seeds.json` | — | RNG seed (`rng_seed=3750551643`) and provenance for reproducible regeneration of the same trajectories. |

## CONSUMED — stratified sampling metadata (locked 2026-06-01)

| File | Shape / size | Role |
|------|-------------|------|
| `lifespans.npy` | (1.5M,) · 5.8M | Per-seed lifespan. Used by the stratified training sampler so long-lived / rare-behaviour seeds get fair representation. |
| `buckets.npy` | (1.5M,) · 5.8M | Density-band bucket assignment per seed (4 bands: 0.03–0.08, 0.08–0.15, 0.15–0.22, 0.22–0.3). Read by the stratified sampler to balance training batches across bands. |

## FUTURE EXPANSION

| File | Shape / size | Why retained |
|------|-------------|--------------|
| `sig_reference.npy` | (1.5M, 1290) float32 · 7.3G | Chunked FFT magnitude spectra of behavioural signatures — designed as a phase-invariant **novelty basis for the explorer**. Not used by the world model; candidate input for explorer novelty scoring later. |
| `grids.npy` | (1.5M, 128, 128) uint8 · 23G | Single embedded seed frame (f₀) per seed. Regenerable from `seeds.npy`. Retained as a convenience cache; could be deleted to reclaim 23G if needed. |

## EXTERNAL DIAGNOSIS

| File | Shape / size | Why retained |
|------|-------------|--------------|
| `labels.npy` | (1.5M,) · 115M | Old hand-rule class labels (dying / still_life / oscillator / glider). **Never used for training** (Decision 10 — unsupervised). Used only to check whether emergent angular clusters align with known classes. |
| `cluster_labels.npy` | (1.5M,) · 5.8M | Old k-means cluster assignment. Diagnostic comparison against emergent geometry. |
| `cluster_centroids.npy` | · 24K | Old k-means centroids. Diagnostic. |
| `signatures_norm.npy` | (1.5M, 257, 10) float32 · 15G | Per-frame behavioural descriptors (population, ΔCoM, variance, edit distance, connected components, lag overlaps). **Not consumed** (Decision 6). Used to verify emergent structure against hand-crafted signals. Largest reclaimable file if disk is needed. |
| `sig_mean.npy`, `sig_std.npy` | · tiny | Normalisation stats for `signatures_norm`. Diagnostic-adjacent. |
| `generation_log.txt` | · 116K | Generation provenance (pool size, workers, density bands, timing). |
| `diagnostics/` | · 2.2M | Diagnostic plots from the original generation run (lifespan histogram, signal distributions, cluster summary, temporal glider sample, etc.). |

---

## Disk reclamation note

If disk pressure arises, the largest non-essential files are `grids.npy` (23G, regenerable from seeds) and `signatures_norm.npy` (15G, diagnostic only). Together they are ~38G of the ~45G total. The CONSUMED set (`seeds.npy` + `seeds.json`) is under 370M.
