# GoL Discovery System — Design Rationale

This document records the non-obvious decisions in the design and the reasoning behind them. The implementing Claude should not revert these choices without first understanding the rationale, because each one was reached after considering and rejecting alternatives.

---

## Decision 1: There is no transition function

**What was rejected:** A learned latent dynamics model T: z_n → z_{n+1} that simulates GoL forward in latent space.

**Why:** The GoL rule engine is exact, deterministic, and free. Learning an approximation of something you can compute exactly introduces error accumulation, boundary blindness, topology mismatches, and a long list of failure modes. The world model encodes and decodes only. Simulation is handled by the actual GoL engine.

**Do not add:** any latent-space transition, predictor, or rollout component. The encoder operates on each frame independently.

---

## Decision 2: No centroid subtraction (position is signal)

**What was rejected:** Subtracting the alive-cell centroid before encoding, to achieve translation invariance.

**Why:** Behaviour is partly defined by spatial movement. A glider's identity is its consistent translation through space — frame to frame, the pattern is structurally identical but shifted. If the encoder subtracts the centroid before processing, every frame of a glider produces (approximately) the same centroid-relative configuration, which is indistinguishable from a still life that doesn't move at all. The angular displacement Δθ between consecutive shells collapses to near-zero for both classes, and the encoder loses the ability to distinguish translation from stasis.

More generally: any pattern whose behaviour involves spatial motion (gliders, spaceships, puffers) is partially defined by its trajectory through grid space. Removing absolute position destroys that information. Position is signal, not noise.

**Do not add:** centroid normalisation, position stripping, or any other translation-invariance mechanism unless the explorer specifically requests it for downstream comparison.

---

## Decision 3: No sparse alive-cell encoder

**What was rejected:** Extracting alive-cell coordinates and using a PointNet/DeepSets architecture.

**Why:** Implementation complexity (padding, masking, variable-length sets, Chamfer rasterisation), and the architecture's permutation-invariant aggregation discards spatial arrangement information that GoL dynamics depend on. A CNN with flatten provides better spatial fidelity at lower engineering cost. The sparse encoder was also architecturally redundant with Decision 2 — its main advantage over CNN was translation invariance via centroid subtraction, which we no longer want.

**Do not switch to:** sparse set encoders, PointNet, DeepSets, or coordinate-based representations.

---

## Decision 4: No global average pooling

**What was rejected:** Replacing flatten with global average pooling at the end of the conv stack.

**Why:** Global pool fails on two independent counts.

First, it collapses the spatial feature map to channel-wise averages, producing a bag-of-features representation that loses spatial arrangement. GoL dynamics are entirely about spatial arrangement — two patterns with identical channel statistics but different cell adjacencies map to the same z and evolve completely differently. This is the dominant reason.

Second, global pool produces translation invariance as a side effect — which Decision 2 establishes is undesirable.

Flatten preserves both spatial arrangement and translation sensitivity.

**Do not switch to:** global average pooling, max pooling, or any whole-feature-map aggregation operation.

---

## Decision 5: Direction is normalised; radius applied externally

**What was rejected:** Encoder outputs full ℝᵈ vectors with magnitude as part of the learned representation.

**Why:** The hypershell geometry requires that time (shell index) and behaviour (angular position) be cleanly separated. If the encoder controls magnitude, it can sneak temporal information into the radius and conflate the axes. By forcing the encoder to output only direction and applying the radius as a deterministic function of time index, the separation is structural rather than loss-enforced.

**Do not change:** the encoder outputs `F.normalize(z, dim=-1)`. The radius scaling `z_n = û_n · r·(n+1)` happens outside the encoder, applied by the data pipeline or by the explorer when constructing chains.

---

## Decision 6: No behavioural signals as supervision

**What was rejected:** Per-frame behavioural descriptors (population, centre of mass, spatial variance, edit distance, connected components, lag overlaps) as supervised targets for auxiliary heads.

**Why:** Most of these signals are direct functions of the grid state and don't need a separate subspace. They could be computed post-hoc from decoded frames if needed. Building dedicated heads for them adds complexity without changing the latent geometry. Class structure emerges from chain shape under the contrastive loss without any labels.

**Do not add:** auxiliary prediction heads for behavioural signals, classification heads, or label-supervised losses.

**Note:** The existing dataset contains pre-computed behavioural signatures (`signatures_norm.npy`, `sig_reference.npy`) and class labels (`labels.npy`). These are NOT consumed by the model. They are retained for **external diagnosis** — e.g. verifying that emergent angular clusters correspond to known behavioural classes — and for possible future explorer use. See `../DATASET.md`.

---

## Decision 7: No Δθ minimisation loss

**What was rejected:** A loss that penalises large angular displacements between consecutive shells.

**Why:** Δθ smoothness should be a consequence of encoder smoothness with respect to mechanical similarity (L₂), not an independently enforced constraint. Forcing Δθ small directly could mask encoder failures and would over-compress chaotic patterns that legitimately have large mechanical changes per step. L₂ attacks the root cause; Δθ minimisation was a symptom-level patch.

The actual mechanism: in real GoL trajectories, consecutive frames are nearly always mechanically similar (most cells don't change per step). If the encoder is Lipschitz-smooth with respect to mechanical distance, small mechanical changes produce small angular changes — so Δθ smoothness emerges automatically. Large Δθ in a real trajectory is then a meaningful signal (a real mechanical discontinuity) rather than a regularised-away artefact.

**Do not add:** Δθ regularisation, trajectory smoothness loss, or local curvature penalty.

---

## Decision 8: L₃ (chain clustering) must outweigh other geometric losses

**What was rejected:** Equal or smaller weighting on chain clustering vs other geometric losses.

**Why:** Clustering by behaviour is what makes the latent space useful for exploration. Other geometric losses (smoothness, uniformity) act as regularisers; only L₃ creates the nested-cluster structure the explorer needs. If smoothness or uniformity dominate, classes collapse into a homogeneous distribution and the explorer has nothing to navigate.

**Do not weight:** L₂ or L₄ above L₃.

---

## Decision 9: No Gaussian prior, no KL divergence, no VAE

**What was rejected:** A VAE-style architecture with KL divergence to an isotropic Gaussian prior.

**Why:** Gaussian pressure produces isotropic latent spaces with no preferred directions, which contradicts the directional trajectory structure the design requires. The hypersphere + uniformity loss (L₄) prevents collapse without imposing isotropy.

**Do not add:** KL divergence terms, reparameterisation tricks, variational sampling, or any Gaussian prior assumption.

---

## Decision 10: No trajectory-level supervision

**What was rejected:** Classifying trajectories into a fixed set of behavioural classes (still life, oscillator, glider, etc.) and training with cross-entropy on class labels.

**Why:** The whole point of the system is unsupervised discovery. Labels would constrain the system to known classes and prevent it from discovering genuinely novel behaviour. Class structure emerges from the unsupervised contrastive loss using same-trajectory identity as the only supervision signal.

L₃ works as follows: two non-overlapping random subsamples of frames from the same trajectory form a positive pair (they share identity); subsamples from different trajectories form negative pairs. No human-defined classes are involved. Similar-behaviour trajectories naturally cluster because their fingerprints come out similar regardless of which subsamples are drawn.

**Do not add:** any labelled classification objective.

---

## Decision 11: Z library stores directions only, not radii

**What was rejected:** Storing full chain tensors as (k+1, d) ambient-space points with radii applied.

**Why:** The radius is a deterministic function of row index — storing it is redundant. The unit directions contain all the learned information. Radii are computed on demand. Trajectory length k is read from `directions.shape[0] - 1`, not stored as a scalar.

**Do not store:** scaled chain tensors, shell radii, or trajectory length k as a separate metadata field.

---

## Decision 12: Seed storage is recommended, not strictly required

**What was discussed:** Whether to store the original 16×16 binary seed per trajectory.

**Why store it:** Decoder reconstructions are approximate. For bit-perfect trajectory reproducibility, the original 32 bytes must be stored. Storage cost is negligible relative to the `(k+1) × d × 4 bytes` per trajectory direction tensor.

**Why it could be dropped:** If the goal is "good enough to re-simulate and verify approximately," `directions[0]` can be decoded and cropped to recover an approximate seed.

**Recommendation:** Store the seed unless storage becomes a real constraint. **This is a recommendation, not a settled architectural decision.** Either choice is defensible.

---

## Decision 13: Mechanical distance metric for L₂ is normalised Hamming distance

**What was discussed:** Which metric to use as ground-truth mechanical distance in the smoothness loss.

**Why Hamming:** The CNN encoder operates on full binary grids. The natural distance between two binary grids is the count of disagreeing pixels (XOR sum) normalised by total cell count. Computed directly from grid tensors without coordinate extraction, the natural complement to BCE, and produces a value in [0, 1] directly comparable to normalised cosine distance.

**Why not Chamfer:** Appropriate for sparse coordinate sets (Decision 3 ruled out), not dense binary grids. Would require coordinate extraction at each computation.

**Why not L2 / Euclidean on flattened grids:** Equivalent to Hamming up to scaling for binary inputs, but Hamming is more intuitive and avoids numerical issues at the extremes.

**Do not change:** the smoothness loss uses Hamming distance unless there's a specific reason to switch. If the encoder is later changed to operate on something other than dense binary grids, revisit this choice.

---

## What the implementing Claude should know

The design has been worked through carefully. Several "obvious" optimisations have already been considered and rejected for specific reasons. If something seems missing or wrong, check this document first. Ask before changing architectural decisions.

The primary success criterion for the world model is: after training, can the Z library be inspected and show that trajectories of similar behavioural class cluster together in angular space on each shell, while still preserving spatial information? That is the test. (The retained `labels.npy` from the old dataset can serve as an external check on whether emergent clusters align with known classes — used for diagnosis only, never for training.)

The secondary success criterion is: does the latent geometry have enough structure (clear cluster boundaries, identifiable frontiers, smooth interpolation) that an explorer model built on top of it has something to work with?

Two items are marked as recommendations rather than settled choices: Decision 12 (seed storage) and the "no k curriculum" claim in the training protocol. Confirm these with the user before relying on them.
