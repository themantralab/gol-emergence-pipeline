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

Also call out any decisions in the rationale you would have done differently — not to argue them, but so I know what your priors are pulling against. Two items the rationale flags as "recommendations rather than settled choices" (Decision 12 on seed storage, and the no-curriculum claim in the training protocol) — confirm these with me before relying on them.

Then propose an implementation plan with these stages:

- **Stage 1**: Data pipeline — GoL trajectory generation from the existing 16×16 seed set (`DATASET.md`), frame batching, Hamming distance utility. Reuse the stored seeds; do not regenerate unless we decide on a new seed distribution.
- **Stage 2**: Encoder + Decoder modules with reconstruction loss (L₁) only — verify it can autoencode a single GoL frame to acceptable fidelity given the 378:1 class imbalance
- **Stage 3**: Add encoder smoothness loss (L₂) — verify angular distances correlate with Hamming distances on real GoL frame pairs (within- and cross-trajectory)
- **Stage 4**: Add chain clustering loss (L₃) — verify two random subsamples of the same trajectory produce similar fingerprints, and that gross behavioural classes start to occupy distinguishable angular regions on each shell (use retained `labels.npy` for diagnosis only)
- **Stage 5**: Add hyperspherical uniformity (L₄) — verify directions remain spread across S^(d-1) and clusters don't all collapse to one region
- **Stage 6**: Z library construction and inspection — encode a held-out set of trajectories, store as (directions, seed) pairs, and produce diagnostic visualisations of the chain geometries

Use real GoL trajectories from the start. The encoder needs real frames to learn from, and Stage 3 onward requires real mechanical distances between actual GoL frames.

For each stage, specify:
- What concrete metric confirms it's working
- What failure modes to watch for (dimensional collapse, all-zero reconstructions, cluster homogenisation)
- A rough indicator of how long the stage should take to converge

Use PyTorch. **Hardware: CPU only, 32GB RAM, 2TB disk — no GPU.** Plan for batch processing accordingly. Target trajectories of length k=256 with d=256 latent dimension and batch size 32. Adam optimiser, LR 3e-4.

Do not begin coding until I confirm the implementation plan. Treat any design ambiguities in the spec as questions to ask, not assumptions to make. Open questions are listed at the bottom of the spec — address those explicitly in your plan.
