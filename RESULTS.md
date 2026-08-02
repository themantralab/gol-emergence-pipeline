# Results

Every measurement below comes from checkpoint **step 148,500**. Each row names
the script that produces it, so any figure can be traced to the code that
computed it.

`a ± b` denotes mean ± sample standard deviation (ddof=1) over 5 seeds, produced
by `multi_seed.py`. Values without a `±` are single runs.

> **To reproduce:** fetch `best.pt` and the data files as shown in the README,
> then run the script named in each section. Everything runs on CPU.

---

## Reconstruction

| Quantity | Value | Script |
|---|---|---|
| alive-F1, full validation set, threshold 0.5 | **0.9169** | `train.py` (validation) |
| dead-F1 | 0.9997 | " |
| precision / recall | 0.8514 / 0.9933 | " |
| Per-quartile alive-F1 (Q0–Q3) | 0.9287 / 0.9303 / 0.9373 / 0.8919 | " |
| micro-F1 on a 600-frame sample @ 0.5 | 0.9052 | `threshold_sweep.py` |
| peak micro-F1 (threshold 0.80) | **0.9305** | " |

**On the multiple F1 values.** These differ by *sample and threshold only*, not
by disagreement. 0.9169 is the full validation set; 0.9052 and 0.9305 are the
same 600-frame quartile-balanced sample at two thresholds. `world_model_readiness.py`
reports 0.922 on its own independent sample, and the teacher-forced ceiling in
`multi_seed.py` ranges 0.929–0.946 across horizons. **Quote 0.9169 as the
headline.**

### The error is sparsity, not blur

| Quantity | Value | Script |
|---|---|---|
| Mean decoder probability at true-alive pixels | 0.965 (median 1.000) | `threshold_sweep.py` |
| Mean at true-dead pixels | 0.00051 | " |
| True-alive pixels in the ambiguous band (0.3, 0.7) | 2.43% | " |
| True-alive pixels confidently alive (> 0.7) | 97.20% | " |
| False positives adjacent to a true live cell | 90.2% of 1,991 | " |
| Per-frame F1 by density: 0–10 alive → 80+ alive | 0.980 → 0.892 | " |
| Pearson corr(alive count, F1) | −0.495 | " |

---

## What the latent supports

| Quantity | Value | Script |
|---|---|---|
| Round-trip drift, real frames (encode→decode→binarize→encode) | 0.034 | `dynamics_probe.py` |
| Round-trip cosine, real frames | 0.999 | " |
| Pearson corr(IoU, latent cosine) | 0.383 | `latent_organization.py` |
| Spearman corr(IoU, latent cosine) | 0.296 | " |

**Latent cosine by frame-pair IoU bucket** (`latent_organization.py`):

| IoU bucket | disjoint | 0–.05 | .05–.10 | .10–.20 | .20–.35 | .35–.50 | .50–.70 | .70–1 |
|---|---|---|---|---|---|---|---|---|
| mean cosine | 0.006 | 0.087 | 0.223 | 0.408 | 0.744 | 0.817 | 0.792 | **0.958** |
| n pairs | 80,160 | 26,381 | 6,633 | 1,391 | 226 | 71 | 49 | 49 |

The scalar correlation is low because 80,160 of 114,960 pairs are disjoint; the
bucketed curve is the result. The .50–.70 bucket dips slightly below .35–.50 —
both have n < 75.

**Trajectory separability is sampling-dependent.** Within/cross-trajectory
latent cosine is 0.647 / 0.079 = **8.2×** under Q3-only sampling
(`trajectory_readiness.py`) and 0.498 / 0.038 = **13.1×** across all quartiles
(`latent_organization.py`). Always state the sampling.

---

## Dynamics do not linearize — `dynamics_probe.py`, `persistence_baseline.py`, `multi_seed.py`

Closed-loop rollout of a learned `z_t → z_{t+1}` residual MLP, against the
predict-no-change null model and the autoencoder's teacher-forced ceiling.
5 seeds, 20 held-out trajectories each.

| Horizon | Learned rollout | Persistence | AE ceiling |
|---|---|---|---|
| 1 | 0.344 ± 0.011 | 0.390 ± 0.023 | 0.934 ± 0.012 |
| 2 | 0.265 ± 0.018 | 0.305 ± 0.014 | 0.929 ± 0.011 |
| 5 | 0.195 ± 0.024 | 0.230 ± 0.017 | 0.939 ± 0.006 |
| 10 | 0.151 ± 0.024 | 0.203 ± 0.020 | 0.935 ± 0.007 |
| 20 | 0.083 ± 0.014 | 0.168 ± 0.023 | 0.946 ± 0.008 |
| 40 | 0.035 ± 0.010 | 0.143 ± 0.027 | 0.946 ± 0.007 |
| 60 | **0.009 ± 0.003** | 0.122 ± 0.028 | **0.946 ± 0.015** |

Two things follow. The ceiling is **flat**, so the representation holds the true
state at horizon 60 as well as at horizon 1 — the failure is traversal, not
capacity. And persistence beats the learned rollout at every horizon, with
non-overlapping ±1 s.d. bands at horizons 1, 10, 20, 40 and 60.

**One-step prediction is a separate, data-limited question.** At the
80-trajectory budget the learned predictor is worse than persistence
(relative L2 0.4878 vs 0.3892; MSE 0.000232 vs 0.000183). But training MSE is
0.000109 — better than persistence — so this is overfitting, and scaling the
data closes it:

| Training trajectories | 10 | 20 | 40 | 80 | 160 | 320 |
|---|---|---|---|---|---|---|
| held-out MSE ÷ persistence MSE | 7.117 | 3.083 | 1.699 | 1.227 | 1.033 | **0.940** |

The ratio crosses 1.0, so the one-step result should not be stated as a flat
"worse than doing nothing". The rollout result above is the robust one.

For context, mean F1 between consecutive frames is 0.6084 — one-step prediction
is intrinsically easy.

---

## Appearance is not behaviour — `probe_behavior_class.py`, `multi_seed.py`

Four hand-labelled behaviour classes the model never trained on
(dying / still life / oscillator / glider), 200 trajectories per class, chance
0.25, all at a canonical centre placement.

| Features | Linear | MLP |
|---|---|---|
| Frozen latent (mean-pooled `z`, 1024-d) | 0.530 ± 0.022 | **0.625 ± 0.012** |
| Population statistics (9-d) | 0.767 ± 0.009 | **0.842 ± 0.010** |

Gap (statistics − latent, MLP) = **0.217 ± 0.007**, positive on every seed.

The nine statistics are population mean/std/first/last/max, the last:first
population ratio, and connected-component counts at three sampled timesteps.
**Caveat:** the labels were plausibly defined using population heuristics, so
the baseline is advantaged on its own definition. The supported claim is that
the latent shows *no advantage*, not that it is useless.

---

## The position confound — `n4_offset_curve.py`, `trajectory_readiness.py`

An *identical* trajectory (same seed, same dynamics) at a reference placement
versus the same trajectory displaced diagonally by *d* cells. 60 Q3 seeds,
frame t = 20.

| d (cells) | 0 | 1 | 2 | **3** | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frame IoU | 1.000 | 0.170 | 0.107 | 0.073 | 0.056 | 0.028 | 0.019 | 0.016 | 0.006 | 0.000 | 0.000 | 0.000 |
| latent cosine | 1.000 | 0.700 | 0.494 | **0.294** | 0.214 | 0.070 | 0.023 | −0.024 | −0.036 | 0.029 | 0.071 | 0.046 |

Cross-seed floor (different dynamics, same placement): frame IoU 0.043 ± 0.039,
latent cosine **0.115 ± 0.207**.

Latent cosine tracks the IoU curve and reaches the cross-seed floor by a
**three-cell** displacement. This is a direct consequence of the L2 objective:
it trains `cos(z_i, z_j) → IoU(x_i, x_j)`, and translated copies have IoU ≈ 0,
so the model is explicitly taught to treat them as unrelated.

Consistent single-point measurement at ~10 cells (`trajectory_readiness.py`):
frame IoU 0.011, latent cosine 0.105, against a cross-seed cosine of 0.079.

---

## Placement-bound competence — `border_test.py`

Reconstruction F1 by the grid offset at which the seed is placed. The training
placement range was [24, 88].

| offset | 2 | 8 | 16 | 24 | 32 | 44 | **56** | 68 | 80 | 96 | 104 | 110 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 | 0.335 | 0.393 | 0.471 | 0.594 | 0.779 | 0.917 | **0.921** | 0.912 | 0.772 | 0.465 | 0.383 | 0.325 |
| recall | 0.393 | 0.511 | 0.719 | 0.858 | 0.941 | 0.995 | 0.994 | 0.990 | 0.963 | 0.674 | 0.469 | 0.371 |

This follows from the training placement distribution and is a scope limit
rather than a defect.

---

## Off-manifold behaviour — `dynamics_probe.py`, `world_model_readiness.py`

| Latent source | round-trip drift | cosine | live cells / frame |
|---|---|---|---|
| Real frame | 0.034 | 0.999 | 59.7 |
| Slerp midpoint | 0.582 | 0.862 | 85.5 |
| Perturbed, ε = 0.2 | 0.207 | 0.978 | 60.7 |
| **Prior sample** | **1.330** | **0.357** | **314.6** |

`world_model_readiness.py` independently reports **319.8** live cells for prior
samples against **47.8** for real frames — a factor of **6.7** — while the
decoder reports max-probability 1.000 and an ambiguous-pixel fraction of 0.024,
with 0.0% of samples near-empty.

Every per-pixel confidence diagnostic passes on outputs that are nothing like
real Game of Life states. **Decoder sharpness is not a validity signal.**

Perturbation degrades gracefully with no cliff (F1 0.919 / 0.910 / 0.866 / 0.716
at ε = 0.05 / 0.10 / 0.20 / 0.40), and per-step latent velocity is
0.389 ± 0.210 against ‖z‖ ≈ 0.95.

---

## Sub-frame motif detection — `motif_tolerance.py`, `multi_seed.py`

Localisation is not available in `z` — one flipped pixel changes ~988 of 1024
dimensions — but is available in the conv trunk, where one pixel changes exactly
1 of 256 tile positions. Verify with `python3 model.py`.

Intra-tile phase is the obstacle: cosine to a fixed-offset reference falls to a
minimum of 0.594 (glider), 0.712 (blinker), 0.718 (block), and a
single-reference detector reaches only AUC 0.612–0.722. A **reference bank**
over all 36 intra-tile offsets is phase-invariant by construction.

**Glider, 5 seeds**, against sliding-window template matching (reported as the
better of a strict and a forgiving match metric):

| Cells flipped (of 9) | Latent bank | Best template | Difference |
|---|---|---|---|
| 0 | 1.000 ± 0.000 | 0.999 ± 0.000 | +0.001 ± 0.000 |
| 1 | 0.960 ± 0.006 | **0.977 ± 0.002** | −0.017 ± 0.005 |
| 2 | **0.919 ± 0.010** | 0.846 ± 0.017 | +0.073 ± 0.020 |
| 3 | **0.888 ± 0.006** | 0.604 ± 0.023 | +0.284 ± 0.023 |

Template matching genuinely wins at one flipped cell; the learned representation
wins from two flips onward. At three of nine flips the motif is arguably
destroyed, and the template baseline falling toward chance partly reflects that.
This is a robustness advantage, not a speed one.

Single-seed results for the other two motifs (bank / best template):
blinker 0.998/0.963 → 0.937/0.840 → 0.846/0.601 → 0.784/0.395;
block 0.988/0.930 → 0.921/0.764 → 0.860/0.599 → 0.824/0.499.

**Negative-set contamination**, measured: real Game of Life tiles that genuinely
contain the motif — glider **0.2%**, blinker **7.4%**, block **14.0%**. This
depresses absolute AUC for both methods equally, so comparisons hold, and it is
why glider is the lead case. Glider is also the only one of the three that
translates, which makes intra-tile phase a genuine problem for it rather than an
artifact of the planting scheme.

---

## Whole-frame retrieval

Reported as a negative. `z` is 1024 float32 = **4096 bytes per frame**, against
**2048 bytes** for the raw 128×128 grid as a bitset — the latent is twice the
size of the data it encodes. Because L2 regresses cosine onto IoU, it can at
best tie exact IoU at ranking-by-IoU. Measured latent-vs-visual Spearman is
0.296, with latent-nearest-neighbour median visual rank 14/80 (chance ~40) and a
top-5 hit rate of 31% (chance 6%) — `traj_cluster_demo.py`.
