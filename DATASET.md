# Dataset

**1.5M random 16×16 seeds**, simulated under B3/S23 on a 128×128 grid.

The model consumes very little of this. Training trajectories are *not* stored —
they are produced by simulating seeds through `engine.py` at training time, so
the only inputs needed are the seed corpus and the lifespan metadata used for
stratified sampling. Everything else is either diagnostic or a legacy array from
an earlier pipeline, retained rather than deleted so the corpus stays whole.

Small files are in `data/` in this repository. Large arrays are on
[Hugging Face](https://huggingface.co/datasets/themantralab/gol-emergence-pipeline).

---

## Consumed by the model

| File | Shape / size | Role |
|---|---|---|
| `seeds.npy` | (1.5M, 16, 16) uint8 · 367 MB | The seed corpus. Embedded on the grid and simulated to produce every training trajectory. |
| `lifespans.npy` | (1.5M,) · 5.8 MB | Per-seed lifespan — the last timestep at which any cell changed state. Sole stratification key: quartile bins are computed over the training pool (lifespan ≥ 32), and the sampler draws one slot per quartile per batch so rare long-lived behaviours get equal share. Lifespan also bounds the per-trajectory frame-sampling window to `[0, lifespan]`, keeping dead frames out of batches. |
| `seeds.json` | < 1 KB | RNG seed (`rng_seed=3750551643`) and provenance, for regenerating the identical corpus. |

**A note on seed placement.** `lifespans.npy` was computed with seeds at a fixed
offset of (24, 24). The model instead embeds seeds at centre-biased random
offsets at train time. The seed corpus itself is placement-agnostic and lifespan
is used only as a sampling key, so the mismatch does not affect training — but it
is worth knowing if you compute anything else from lifespan.

## Used by diagnostics, never for training

| File | Shape / size | Used by |
|---|---|---|
| `labels.npy` | (1.5M,) str · 115 MB | Hand-rule behaviour classes (`dying`, `still_life`, `oscillator`, `glider`). **Never trained on.** Read by `probe_behavior_class.py` and `multi_seed.py` as held-out evaluation classes. Note these labels were plausibly derived from population heuristics, which advantages the statistical baseline in that comparison. |
| `buckets.npy` | (1.5M,) · 5.8 MB | Density-band assignment over four bands (0.03–0.08, 0.08–0.15, 0.15–0.22, 0.22–0.3). Considered for stratification and rejected — seed density does not correlate with behaviour. Retained for diagnostic comparison. |
| `n_seeds.npy` | scalar · < 1 KB | Corpus size (1,500,000). |
| `diagnostics/generation_log.txt` | 116 KB | Generation provenance: pool size, workers, density bands, timings. |
| `diagnostics/canonical_check.txt` | < 1 KB | Engine correctness record (blinker period-2, block stability, glider drift). Reproduce with `python3 -c "import engine; engine.run_canonical_checks()"`. |

## Legacy arrays — no consumer in this repository

These were produced by an earlier signal-based pipeline. No script here reads
them. They are listed for completeness and hosted on Hugging Face; ignore them
entirely unless you want to build on that earlier work.

| File | Shape / size | What it is |
|---|---|---|
| `grids.npy` | (1.5M, 128, 128) uint8 · 23 GB | Cached single embedded seed frame (f₀) per seed. Fully regenerable from `seeds.npy` via `engine.embed_seeds`. |
| `sig_reference.npy` | (1.5M, 1290) float32 · 7.3 GB | Chunked FFT magnitude spectra of per-frame behavioural signals. |
| `signatures_norm.npy` | (1.5M, 257, 10) float32 · 15 GB | Per-frame behavioural descriptors: population, ΔCoM, variance, edit distance, connected components, lag overlaps. |
| `cluster_labels.npy`, `cluster_centroids.npy` | 5.8 MB / 24 KB | k-means assignment and centroids over the signature space. |

---

## Getting the data

Only the two consumed files are needed to train, or to run most diagnostics:

```bash
pip install huggingface_hub
python3 - <<'EOF'
from huggingface_hub import hf_hub_download
import shutil, pathlib
pathlib.Path("data").mkdir(exist_ok=True)
for f in ["seeds.npy", "lifespans.npy"]:
    shutil.copy(hf_hub_download("themantralab/gol-emergence-pipeline", f,
                                repo_type="dataset"), f"data/{f}")
EOF
```

Add `labels.npy` if you want to run `probe_behavior_class.py` or `multi_seed.py`.
The consumed set is under 400 MB; the legacy arrays total ~45 GB and are not
needed for anything in this repository.
