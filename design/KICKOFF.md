# Implementation Kickoff Prompt

Copy this as the first message to the new Claude conversation. The two design documents (`01_design_specification.md`, `02_design_rationale.md`) and the dataset manifest (`../DATASET.md`) live alongside this file in the project.

---

I'm implementing a research system for unsupervised behaviour discovery in Conway's Game of Life. The design is complete and documented in:

- `design/01_design_specification.md` — the canonical final design (what to build)
- `design/02_design_rationale.md` — why specific design choices were made (do not revert these without asking)
- `DATASET.md` — the pre-generated dataset: which files the model consumes vs. which are kept for future expansion / external diagnosis

Please start by reading all three documents completely before proposing any code.

The goal for this conversation is to implement the **world model only**. The explorer is documented in the spec for context but is not part of this build.

After reading, please confirm your understanding by summarising:

1. What the world model encodes and decodes
2. Why there is no transition function
3. Why the encoder does not subtract the centroid (and why position is preserved)
4. The four losses and their roles, including how L₃ is supervised without labels
5. The hypershell geometry and how shell radius relates to encoder output

Also call out any decisions in the rationale you would have done differently — not to argue them, but so I know what your priors are pulling against.

## Locked decisions — do not re-litigate

These were all settled before this conversation begins. Treat them as fixed unless I explicitly ask to revisit:

| Decision | Value |
|---|---|
| Latent dim `d` | 256 |
| Shell radius constant `r` | 1 (shell n at radius n+1) |
| L₂ formulation | `(L_within + L_cross) / 2` with `target = sqrt(hamming)` per sub-loss — see spec |
| L₂ pair sampling | 50/50 within- vs cross-trajectory pairs per batch |
| L₃ fingerprint size | `n_samples = 32` (32 frames per fingerprint, two independent draws per trajectory per step) |
| Curriculum | None — encode full-length trajectories from the start |
| Seed sampling | Lifespan-quartile stratified, training pool restricted to `lifespan ≥ 32`; L₃ fingerprints sample from `[0, lifespan]` only. Z cloud (Stage 5) still encodes all 1.5M regardless. |
| Z library coverage | Full 1.5M trajectories; directions stored as float16 (~193 GB), seed stored as uint8 |
| Hardware | CPU only, 32 GB RAM, 2 TB disk — no GPU |
| Training | Single phase, all losses active from start, Adam LR 3e-4, batch size 32 trajectories |

The spec's "Open questions — status" section reflects these locks; the rationale's Decisions 12 (seed storage) and the no-curriculum claim are both now confirmed locked.

## Implementation plan to propose

Front-load reconstruction quality before adding the geometric losses:

- **Stage 1** — `engine.py` + `data.py`: B3/S23 simulator (fixed-zero boundary, seed embedded at offset (24, 24), vectorised over the batch dimension so 32 trajectories step together). Sampler reads `data/lifespans.npy`, computes quartile bins on the training pool (`lifespan ≥ 32`), draws stratified batches of 32 trajectories with one slot per quartile per draw. L₂ pair sampler (50/50 within/cross). L₃ fingerprint subsampler (32 frames × 2 independent draws, both drawn from `[0, lifespan]`). Gate: canonical patterns reproduce bit-exactly; one training-batch worth of trajectories generates in under a couple of seconds on CPU.
- **Stage 2** — `model.py` + `losses.py` (L₁ only) + `train.py` (L₁ only): single-frame autoencoding with BCE `pos_weight=50`. **High bar: alive-cell F1 ≥ 0.95 and dead-cell F1 ≥ 0.99 on a held-out set** before adding any other loss. If unreachable, the architecture is too tight and we revisit (bigger decoder or `d=384/512`) *before* introducing the geometric losses.
- **Stage 3** — add L₂, L₃, L₄ all at once: weights 1.0 / 1.0 / 1.5 / 0.3, all from start. Per-step validation: recon F1 holds, fingerprint self-similarity ≥ 0.9, Hamming↔cosine Pearson ≥ 0.8.
- **Stage 4** — `diag.py`: per-shell silhouette by `labels.npy` (diagnosis only, never a training signal), 2D projections of angular distribution per shell, reconstruction gallery. Gate: silhouette ≥ 0.5 on early shells, reconstructions visually near-perfect.
- **Stage 5** — `zlib.py`: encode all 1.5M trajectories → memory-mapped float16 directions + uint8 seeds. Gate: Z library written, spot-checks on glider/oscillator/still-life trajectories decode back to recognizable frames.

For each stage, specify the concrete metric that confirms it's working, the failure modes to watch for (dimensional collapse, all-dead reconstructions, cluster homogenisation, fingerprint instability), and a rough indicator of how long it should take to converge on CPU.

Use PyTorch. Treat any design ambiguity in the spec as a question to ask, not an assumption to make. Do not begin coding until I confirm the plan.
