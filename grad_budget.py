"""
Gradient-budget check: which loss term actually drives the ENCODER right now?

Comparing loss *values* (L2 raw 0.23 vs L1 raw 0.005) is misleading —
∇total = Σ wᵢ ∇Lᵢ, so what matters is the gradient norm ‖wᵢ ∇Lᵢ‖ each term
contributes, not wᵢ·Lᵢ. A term frustrated at a flat floor (like L2) can have a
large value but a small gradient.

This settles the restart question:
  - If L2/L4 gradient norms dominate L1's → regularizers are strangling
    reconstruction → restart fix = loss reweighting / Jaccard fix.
  - If L1 gradient is comparable/larger but reconstruction still plateaued →
    capacity wall → restart fix = bigger/better latent.

Reads best.pt only — safe alongside training.
"""

import numpy as np
import torch

import data, engine, losses
from model import Encoder, Decoder

torch.set_num_threads(2)
ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
print(f"Checkpoint step={ckpt['step']}  F1={ckpt['metrics']['alive_f1']:.4f}\n")

enc, dec = Encoder(), Decoder(kernel_size=1)
enc.load_state_dict(ckpt["encoder"]); enc.train()
dec.load_state_dict(ckpt["decoder"]); dec.train()

# Build a representative batch (same recipe as training: 1 frame/traj, 32 traj)
pool = data.TrainingSeedPool()
rng = np.random.default_rng(0)
BATCH = 32
seeds_per_q = BATCH // 4
frames = []
for q in range(4):
    idx = rng.choice(pool.quartile_pools[q], size=seeds_per_q, replace=False)
    seeds = np.asarray(pool.seeds[idx])
    trajs = engine.simulate(seeds, k=engine.K_DEFAULT)
    ls = pool.lifespans[idx]
    for i in range(seeds_per_q):
        frames.append(trajs[i, int(rng.integers(0, ls[i] + 1))])
frames = np.stack(frames)
x = torch.from_numpy(frames.astype(np.float32)).unsqueeze(1)

enc_params = [p for p in enc.parameters()]


def enc_grad_norm(loss):
    enc.zero_grad(); dec.zero_grad()
    loss.backward(retain_graph=True)
    sq = 0.0
    for p in enc_params:
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum())
    return sq ** 0.5


# Forward
z = enc(x)
logits = dec(z)

# Steady-state weights (training is past curriculum: pw=5, w3=0.05)
l1 = losses.reconstruction_loss(logits, x, pos_weight=losses.ALIVE_POS_WEIGHT)
l2 = losses.smoothness_loss(z, x)
l3 = losses.norm_bound_loss(z)
l4 = losses.angular_uniformity_loss(z)

terms = [
    ("L1 recon",  losses.W_RECON  * l1),
    ("L2 smooth", losses.W_SMOOTH * l2),
    ("L3 norm",   losses.W_NORM   * l3),
    ("L4 unif",   losses.W_UNIF   * l4),
]

print(f"{'term':>10} {'raw':>10} {'weighted':>10} {'∇enc norm':>12}")
print("-" * 48)
gns = {}
for name, wl in terms:
    gn = enc_grad_norm(wl)
    gns[name] = gn
    raw = wl.item() / dict(L1=losses.W_RECON, L2=losses.W_SMOOTH, L3=losses.W_NORM, L4=losses.W_UNIF)[name.split()[0]]
    print(f"{name:>10} {raw:>10.4f} {wl.item():>10.4f} {gn:>12.5f}")

total_gn = enc_grad_norm(sum(wl for _, wl in terms))
print(f"{'TOTAL':>10} {'':>10} {'':>10} {total_gn:>12.5f}")

# What fraction of the (summed) encoder gradient does reconstruction drive?
print(f"\n  L1 ∇-share of summed term norms: {gns['L1 recon'] / sum(gns.values()):.1%}")
print(f"  (L2+L4) ∇-share:                 {(gns['L2 smooth']+gns['L4 unif']) / sum(gns.values()):.1%}")
print("\n  If L1 ∇-share is tiny → regularizers strangle reconstruction (reweight on restart).")
print("  If L1 ∇-share is large but F1 plateaued → capacity wall (bigger latent on restart).")
