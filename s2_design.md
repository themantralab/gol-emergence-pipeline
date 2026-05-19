# Stage 2 — Core World Model

## Overview

**Goal**: train encoder, transition function, decoder, and trajectory head on
the Stage 1 dataset. The resulting model is the fixed foundation everything
else builds on — it is frozen after Stage 2 and never modified again.

**Inputs**: all `.npy` files from `data/` (Stage 1 output)

**Outputs**: trained model checkpoints in `checkpoints/`

**Files** (write in this order):
1. `model/__init__.py`
2. `model/encoder.py`
3. `model/decoder.py`
4. `model/transition.py`
5. `model/trajectory_head.py`
6. `model/vicreg.py`
7. `model/contrastive.py`
8. `data_loader.py`
9. `train_core.py`

**Dependencies**: Stage 1 complete, `data/` populated.
Must complete before Stage 3 begins. Freeze all weights after training.

**Cross-reference**: The identity training target is the normalized 10-signal
signature defined in `s1_design.md`. Use `sig_mean.npy` and `sig_std.npy`
from `data/` consistently for any inference-time normalization.

---

## Core World Model Design

### Design philosophy

The core model has **two simultaneous objectives** that are in tension:

1. **Mechanics objective**: learn the B3/S23 rule as a latent-space transition.
   This is a local constraint — it operates on cell neighborhoods and has exact
   ground truth.

2. **Identity objective**: learn behavioral class — what kind of thing is this
   configuration in the long run? This is a global constraint — it operates on
   the full 256-step trajectory and has approximate ground truth from the
   10-signal signature.

The encoder must satisfy both. Mechanics wants local neighborhood information.
Identity wants global trajectory class. This tension forces the encoder to
develop a representation that captures both, rather than collapsing to either
static appearance or average behavior.

### Architecture

**Encoder**: convolutional network
- Input: 128×128 binary grid (uint8, treated as float32 in [0,1])
- Channel progression: 1→32→64→128→256, stride-2 at each layer
  - After 4 stride-2 layers: spatial size 128→64→32→16→8
  - Flatten: 256×8×8 = 16,384 → linear → **256**
- BatchNorm + ReLU after each conv layer
- No activation after the final linear → 256; z is raw
- No reparameterization trick, no sampling, purely deterministic

**Transition function f_θ**: residual MLP
- Input: z_t ∈ ℝ²⁵⁶
- Architecture: 3-layer MLP, hidden dimension **512**, SiLU activations
- Output: z_{t+1} = z_t + MLP(z_t) ∈ ℝ²⁵⁶ (residual connection)
- Residual formulation: GoL steps are small changes; predicting the delta is
  numerically easier, and the skip connection gives near-identity initialization
  preventing rollout divergence before training stabilizes
- Deliberately sized so it cannot memorize grid states — forced to learn
  the transition rule abstractly

**Decoder**: convolutional network mirroring encoder
- Input: z ∈ ℝ²⁵⁶
- Linear → reshape to (256, 8, 8), then 4 transposed conv layers stride-2,
  channel progression 256→128→64→32→1
- BatchNorm + ReLU after each transposed conv except the last
- Output: (1, 128, 128) logits — sigmoid applied for loss, threshold at 0.5
  for grid reconstruction

**Trajectory head**: per-timestep MLP
- Input: z_t ∈ ℝ²⁵⁶
- Architecture: 3-layer MLP, hidden dim **256**, output ℝ¹⁰
- Output: predicted normalized 10-signal values [P, Δcx, Δcy, V, E, N_cc,
  S_lag_2, S_lag_4, S_lag_8, S_lag_16] at timestep t
- Applied at every step of the rollout during training
- Hidden dim deliberately matches latent dim to prevent the head from
  absorbing encoder deficiencies through complex nonlinear mappings
- **Training instrument only.** Not used at inference time.

### Latent space regularization — VICReg (variance only)

No Gaussian prior. No KL divergence. No reparameterization.

VICReg enforces one soft constraint: each latent dimension must have std ≥ 1
across the batch. This prevents dimensional collapse without imposing a
distributional shape.

```python
vicreg = VICReg(latent_dim=256, lambda_var=5.0, lambda_cov=0.0)
```

**Why covariance term is disabled**: The covariance term requires estimating a
256×256 matrix from B=32 samples. With N=32 < D=256, the matrix is rank-deficient
by construction — all 32,640 off-diagonal elements are noise-dominated. With
lambda_cov=25 (original paper default for N=2048), this produced gradients 2.6×
larger than the mechanics loss, causing catastrophic oscillation in Phase 1.
Setting lambda_cov=0 eliminates this while the variance term still prevents collapse.
The contrastive loss provides implicit feature decorrelation in Phase 2/3.

**Why lambda_var=5 (not 25)**: Scaled down proportionally for our batch size.
lambda_var=5 with B=32 produces variance gradients that are well-balanced against
the mechanics signal.

### Anti-blob training — temporal contrastive loss

For every training batch, triplets (anchor, near, far) are sampled from rollout
sequences:
- **anchor**: z_t at timestep t
- **near positive**: z_{t+5} (5 steps later, same trajectory) — should be close
- **far negative**: z_{t+50} (50 steps later, same trajectory) — should be distant

Loss penalises when far is closer to anchor than near. This forces the latent
space to develop temporal geometry: distance in ℝ²⁵⁶ reflects temporal distance.
As a consequence, patterns with similar behavioral trajectories cluster together —
behavioral structure emerges from temporal structure.

- **Phase 2**: random negatives (any z from a different trajectory in the batch)
- **Phase 3**: hard negatives — the z from a different trajectory geometrically
  closest to the anchor in current latent space

### Training losses

| Loss          | Formula                                                    | Phase |
|---------------|------------------------------------------------------------|-------|
| L_mechanics   | Multi-frame BCE (see below)                                | 1,2,3 |
| L_trajectory  | MSE(traj_head(z_t), sig_norm[t]) averaged across rollout  | 2,3   |
| L_contrastive | Temporal triplet on z sequence                             | 2,3   |
| VICReg        | Variance term only on batch z_0                            | 1,2,3 |

**Multi-frame mechanics loss** (key design decision):

Standard endpoint-only mechanics loss (BCE only at z_k) leaves intermediate
timesteps without direct gradient signal. At k=96, the gradient for z_1 must
flow through 95 applications of f — heavily diluted. The transition function
learns "reach the right endpoint" not "be accurate at every step."

To address this, mechanics loss is computed with adaptive intermediate frame
sampling:

```
frame_sample_rate ∈ [0, 1]
n_intermediate = round(frame_sample_rate × (k - 1))
frames = [endpoint k] + [n_intermediate random frames from [1, k-1]]
loss_mech = mean(BCE(decoder(z_t), grid_t) for t in frames)
```

At frame_sample_rate=1.0, k=96: all 95 intermediate frames decoded (96 total).
At frame_sample_rate=0.5, k=96: 47 intermediate frames (48 total).
At frame_sample_rate=0.0: endpoint only (1 frame).

The sampling rate follows a per-k-level lifecycle:
1. **Entry** (frame_sample_rate=1.0): up to 16 random intermediate frames per step
2. **Threshold crossed**: once smooth_acc ≥ threshold, rate decays by 0.1 per val check
3. **Endpoint-only** (frame_sample_rate=0.0): standard single-frame loss
4. **Advancement gated**: k_max can only advance when frame_sample_rate=0 and
   smooth_acc ≥ threshold for 2 consecutive checks
5. **On advance**: frame_sample_rate resets to 1.0 for the new k level

This forces the model to prove it has truly learned the dynamics (not just
benefiting from dense supervision) before advancing to a harder rollout depth.

**Mechanics loss** uses per-cell BCE with pos_weight=50 on alive cells.
Ground truth is exact GoL simulation. pos_weight=50 counteracts the ~378:1
dead:alive ratio — without it the model predicts all-dead.

**Trajectory loss** uses MSE between traj_head(z_t) and sig_norm[t] averaged
across all rollout steps. Provides dense behavioral supervision throughout the
trajectory.

**Phase 2 loss ramp**: Auxiliary losses ramp in gradually over PHASE2_RAMP_STEPS=10,000
steps from Phase 2 entry. Weights scale linearly from 0 to their target values:

```
frac = min(1.0, (step - phase2_start_step) / PHASE2_RAMP_STEPS)
w_trajectory  = 0.2 × frac
w_contrastive = 0.1 × frac
w_vicreg      = 0.01 + 0.04 × frac   (ramps from Phase 1 value to Phase 2 value)
```

Prevents gradient conflict: if trajectory/contrastive hit full weight immediately
at Phase 2 entry, they compete with mechanics before the encoder adapts, causing
oscillation and accuracy regression.

**Weight schedule** (full weight targets):
- Phase 1: mechanics=1.0, trajectory=0.0, contrastive=0.0, vicreg=0.01
- Phase 2: mechanics=1.0, trajectory=0.2, contrastive=0.1, vicreg=0.05
- Phase 3: mechanics=1.0, trajectory=0.5, contrastive=0.2, vicreg=0.05

### Training curriculum

#### Teacher forcing

At each rollout step t, with probability p_teacher the real encoded z_t is used
as input to f_θ; otherwise f_θ's own predicted z_t is used (scheduled sampling /
exposure bias correction).

- **Phase 1**: p_teacher=0.9 (10% free-rollout prevents specialisation to exact encoder outputs)
- **Phase 2**: decays linearly from 0.9 → 0.0 over phase2_total_steps
- **Phase 3**: p_teacher=0.0 (pure free rollout)
- **On each k_max advance**: p_teacher resets to 0.9 (or 0.0 in Phase 3), giving
  the model structured support at each new harder rollout depth before decay resumes

phase2_start_step is saved in checkpoints so the TF decay clock survives restarts.

#### Advancement thresholds — phase-aware

Thresholds are phase-aware to reflect objective difficulty:

```
Phase 1: threshold(k) = max(0.95 - 0.04 × (k/256), 0.90)
Phase 2: threshold(k) = max(0.85 - 0.04 × (k/256), 0.80)
Phase 3: threshold(k) = max(0.90 - 0.04 × (k/256), 0.85)
```

Phase 1 (mechanics only) can achieve high accuracy. Phase 2 (competing objectives)
legitimately achieves lower peak accuracy — the threshold reflects this without
requiring impossible precision.

Advancement uses **rolling average of last 3 val checks** (smooth_acc) rather than
point-in-time accuracy. At k=96 free rollout, individual checks have high variance
(±0.15) — smooth_acc reduces noise-driven false negatives that would repeatedly
reset the above_thresh counter.

Advancement requires smooth_acc ≥ threshold(k_max) for **2 consecutive val checks**,
AND frame_sample_rate = 0.0 (model proven endpoint-only capable).

#### Phase 3 gate

Phase 2 → Phase 3 transition is gated: it cannot occur until:
```
step ≥ phase2_entry_step + PHASE2_RAMP_STEPS
```
phase2_entry_step is fixed at Phase 2 entry and never reset (unlike phase2_start_step
which resets on k_max advance). This ensures the full 10k ramp completes before
Phase 3 weights take effect, regardless of how quickly k_max advances to 192.

#### k_max levels and phase transitions

k_max advances through: 1 → 2 → 4 → 8 → 16 → 32 → 48 → 64 → 96 → 128 → 192 → 256

- **Phase 1 → Phase 2**: when k_max first reaches 96
- **Phase 2 → Phase 3**: when k_max first reaches 192 AND phase3 gate clears

**On each k_max advance:**
- LR resets to 3e-4
- Scheduler state clears (fresh ReduceLROnPlateau)
- TF resets to 0.9 (Phase 2) or stays 0.0 (Phase 3)
- phase2_start_step resets to current step (restarts TF decay and ramp clock)
- frame_sample_rate resets to 1.0
- sampling_decaying resets to False
- val_acc_buf cleared

**Phase 1 — Rule learning** (k_max=1→96, TF=0.9):
- Loss: mechanics × 1.0 + VICReg × 0.01
- Progressive rollout: random k ∈ [1, k_max] per step
- Validates that f_θ can compose — z_0 must reconstruct any depth up to k_max

**Phase 2 — Trajectory supervision** (k_max=96→192, TF 0.9→0.0):
- Loss: mechanics × 1.0 + trajectory × [ramped] + contrastive × [ramped] + VICReg × [ramped]
- Trajectory head introduced; contrastive loss with random negatives
- TF decays from 0.9 to 0.0 over phase2_total_steps (default 100k)
- Auxiliary losses ramp from 0 to full weight over first 10k steps

**Phase 3 — Full joint** (k_max=192→256, TF=0.0):
- Loss: mechanics × 1.0 + trajectory × 0.5 + contrastive × 0.2 + VICReg × 0.05
- Hard negatives enabled
- Pure free rollout
- Stop when t-SNE of encoded held-out patterns shows cluster separation

### Optimizer and LR schedule

**Optimizer**: AdamW, lr=3e-4, weight_decay=1e-4. Gradient clipping: max_norm=1.0.

**LR schedule**: single `ReduceLROnPlateau(mode='max', patience=10, factor=0.5,
min_lr=1e-5)` stepping on val accuracy.

**Critical**: scheduler is **frozen during Phase 2** (does not step). During Phase 2
TF decay, val accuracy oscillates structurally (model trained on mixed TF but
tested on pure free rollout) — this oscillation is not a plateau but the scheduler
cannot distinguish it from one. Premature LR reductions would slow adaptation
exactly when the model needs to transition to free-rollout dynamics.
Scheduler activates in Phase 3 only, where TF=0 makes val accuracy a stable signal.

LR resets to 3e-4 (and scheduler state clears) on each k_max advance and phase
transition, ensuring each new curriculum level starts with full learning rate.

### DataLoader

- **num_workers=2**: two worker processes each run simulate_batch independently.
  With step time ~5-6s and simulation ~1-2s, 2 workers is sufficient to keep
  the main process fed without oversubscribing cores.
- **Main process**: 6 PyTorch threads (os.cpu_count() - num_workers = 8-2 = 6)
- **Workers**: 1 thread each (OMP_NUM_THREADS=1 in worker_init_fn)
- Total: 6 + 2 = 8 threads across all 8 cores
- persistent_workers=True avoids worker respawn between epochs

---

## Implementation

### `model/__init__.py`
Empty init to make `model/` a package.

### `model/encoder.py`
Convolutional encoder: (128, 128) binary grid → z ∈ ℝ²⁵⁶.
- 4 conv layers, stride-2 downsampling, BatchNorm after each, ReLU activations
- After 4 stride-2 layers: spatial size = 8×8, flatten + linear → 256
- No reparameterization, no sampling — purely deterministic

### `model/decoder.py`
Convolutional decoder mirroring encoder: z ∈ ℝ²⁵⁶ → (128, 128) logits.
- Linear → reshape to (256, 8, 8), then 4 transposed conv layers, stride-2
- Output: (1, 128, 128) logits (no sigmoid — apply in loss, threshold at 0 for sampling)

### `model/transition.py`
Residual latent transition MLP: z_t ∈ ℝ²⁵⁶ → z_{t+1} ∈ ℝ²⁵⁶.
- 3-layer MLP, hidden dim 512, SiLU activations
- Output: z + MLP(z) — residual connection for near-identity initialization

### `model/trajectory_head.py`
Per-timestep trajectory prediction head: z_t ∈ ℝ²⁵⁶ → ℝ¹⁰.
- 3-layer MLP, hidden dim 256, SiLU activations, output 10
- Applied at each step t; no recurrence, no sequence dependency
- Loss: MSE against sig_norm[t] from `signatures_norm.npy`

### `model/vicreg.py`
VICReg regularization — variance term only.
- lambda_var=5.0, lambda_cov=0.0
- Covariance term disabled: N=32 << D=256 makes covariance estimate rank-deficient
  and noise-dominated. Variance term sufficient for collapse prevention.

### `model/contrastive.py`
Temporal contrastive triplet loss.
- Near lag k=5, far lag K=50, fixed throughout training
- Phase 2: random negatives; Phase 3: hard negatives (closest different trajectory)
- Requires rollout length k+1 ≥ FAR_K + NEAR_K + 1 = 56

### `data_loader.py`
PyTorch Dataset and DataLoader for the Stage 1 dataset.
- Large arrays (grids, signatures_norm) opened with mmap_mode='r'
- `__getitem__`: returns lightweight dict without simulating
- Custom collate function calls simulate_batch(grids, steps=256) once per batch
- Stratified batch sampler: proportional draws from all 4 behavioral buckets
- 90/10 train/val split by seed index

### `train_core.py`
Training loop with 3-phase progressive rollout curriculum. See above sections
for full design. Key implementation notes:

- Auto-resume from latest checkpoint on startup; `--fresh` to start clean
- Checkpoint saves: step, phase, k_max, phase2_start_step, phase2_entry_step,
  lr_current, scheduler state, frame_sample_rate, sampling_decaying
- Metrics logged to `checkpoints/metrics.jsonl` (JSONL, append-mode)
- train records every 100 steps; val records every 500 steps; event records on
  k_max advances, phase transitions, phase3 gates, sampling decay events

---

## Completion Criteria

- [ ] 1-step reconstruction accuracy on alive cells > 95% on validation set
- [ ] f_θ correctly predicts next state for canonical patterns (block,
      blinker, glider) verified by visual inspection of decoded outputs
- [ ] Rollout drift plot plateaus — does not grow monotonically to k=256
- [ ] Trajectory head predictions qualitatively match expected per-class signal
      shapes at k=64 and k=256 (dying decays, glider drifts, oscillator periodic)
- [ ] t-SNE of encoded held-out patterns shows behavioral cluster separation
      (dying and glider clearly separated; still_life/oscillator overlap acceptable)
- [ ] Model checkpoints saved to `checkpoints/`; weights frozen before Stage 3
