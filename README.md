# GoL Emergence Discovery System

A dual-model research program for **unsupervised discovery of emergent behaviour in Conway's Game of Life**, by [Mantra Labs](https://mantra-labs.com).

- **World Model** — encodes GoL frames into a structured *hypershell* latent geometry and produces a library of encoded trajectories (the "Z library"). Implemented first.
- **Explorer Model** — navigates the Z library, identifies frontier regions, proposes novel trajectories, and validates them against the GoL engine. Designed, not yet built.

> **⚠️ Architecture pivot (2026-05-27).** This project previously used a four-stage design with a *learned latent transition function* and behavioural-signal supervision. It has pivoted to an **encode/decode-only world model with no transition function** — the GoL engine is the simulator; the model only learns a structured latent geometry. The canonical design now lives in [`design/`](design/). The pre-existing dataset is retained in full but reclassified by relevance (see [`DATASET.md`](DATASET.md)).

---

## The Design

| Document | Purpose |
|---|---|
| [`design/01_design_specification.md`](design/01_design_specification.md) | Canonical spec — architecture, losses, geometry, training protocol |
| [`design/02_design_rationale.md`](design/02_design_rationale.md) | 13 design decisions with rationale (do not revert without understanding) |
| [`design/KICKOFF.md`](design/KICKOFF.md) | Implementation kickoff prompt for a fresh build |
| [`DATASET.md`](DATASET.md) | Dataset manifest: which files the model consumes vs. future-expansion / external-diagnosis |

### World model in brief

- **Encoder**: CNN → flatten → linear → L2-normalise → unit direction `û ∈ S^(d-1)`, `d = 256`. No centroid subtraction, no global pooling (position is signal).
- **Decoder**: mirror ConvTranspose, BCE with `pos_weight=50` for the ~378:1 dead:alive imbalance.
- **No transition function**: the exact, deterministic GoL engine produces trajectories; the model never predicts dynamics.
- **Hypershell geometry**: a frame at step `n` maps to `z_n = û_n · r·(n+1)`. Direction is learned; radius is applied externally per shell index.
- **Four losses, single phase, all active from the start**:

  | Loss | Weight | Role |
  |---|---|---|
  | L₁ Reconstruction (BCE) | 1.0 | Decode `û` back to the grid |
  | L₂ Encoder smoothness | 1.0 | Angular distance ↔ normalised Hamming distance |
  | L₃ Chain clustering (NT-Xent) | **1.5** (must dominate) | Self-supervised trajectory clustering — same-trajectory identity, **no labels** |
  | L₄ Hyperspherical uniformity | 0.3 | Prevent collapse without imposing isotropy |

The Z library stores `(directions, seed)` per trajectory. Behavioural class structure is intended to **emerge** from L₃ — there is no labelled supervision.

---

## Repository Structure

```
gol-emergence-pipeline/
├── README.md                        ← you are here
├── DATASET.md                       ← dataset file classification
├── design/
│   ├── 01_design_specification.md
│   ├── 02_design_rationale.md
│   └── KICKOFF.md
├── data/                            ← seed corpus + metadata (large arrays on Hugging Face)
│   ├── seeds.npy                    ← (1.5M, 16, 16) — the model's only direct input
│   ├── seeds.json                   ← RNG seeds for reproducibility
│   ├── labels.npy / lifespans.npy / buckets.npy   ← diagnosis / stratification metadata
│   └── diagnostics/
└── figures/                         ← analysis figures from dataset generation
```

> **No source code is published yet.** The previous implementation was removed at the pivot; the new world model will be implemented fresh from [`design/KICKOFF.md`](design/KICKOFF.md).

---

## Dataset

The model's only direct input is the **seed corpus** (`data/seeds.npy`, 1.5M × 16×16). Trajectories are produced by simulating seeds through the GoL engine at training time. Everything else in the dataset is retained as **future expansion** (lifespan/bucket stratification, FFT novelty basis) or **external diagnosis** (old class labels, behavioural signatures, cluster assignments) — see [`DATASET.md`](DATASET.md) for the full classification.

The complete dataset, including the large array files, is hosted on Hugging Face:

**[huggingface.co/datasets/themantralab/gol-emergence-pipeline](https://huggingface.co/datasets/themantralab/gol-emergence-pipeline)**

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="themantralab/gol-emergence-pipeline",
                  repo_type="dataset", local_dir="./gol-data")
```

| File | Shape | Size | Now used for |
|---|---|---|---|
| `seeds.npy` | (1.5M, 16, 16) uint8 | 367 MB | **Model input** |
| `grids.npy` | (1.5M, 128, 128) uint8 | 23 GB | f₀ cache (regenerable) — future/convenience |
| `signatures_norm.npy` | (1.5M, 257, 10) float32 | 15 GB | External diagnosis only |
| `sig_reference.npy` | (1.5M, 1290) float32 | ~7 GB | Future explorer novelty basis |
| `labels.npy` | (1.5M,) | 115 MB | External diagnosis (verify emergent clusters) |

---

## Dataset Provenance

1.5M density-stratified seeds (4 bands, 0.03–0.30), RNG seed `3750551643`, generated 2026-04-30 under Conway's B3/S23 with a fixed-zero boundary. The original behavioural-signal characterisation (10 signals per frame, 4 heuristic classes: still_life / oscillator / dying / glider) is **retained for diagnosis** but is no longer part of the model's objective (see Decision 6 and Decision 10 in the rationale).

---

## Citation

```bibtex
@misc{koegler2026gol,
  author    = {Koegler, Maxwell},
  title     = {{GoL Emergence Discovery System}},
  year      = {2026},
  publisher = {Mantra Labs},
  url       = {https://github.com/themantralab/gol-emergence-pipeline}
}
```

---

## License

Documentation and design are released under **CC BY 4.0**. Future source code will be released under the **MIT License**.

© 2026 Mantra Labs
