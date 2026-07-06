"""
Bootstrap-collapse diagnosis. Inspects the stuck best.pt to characterise
*what* the model is actually producing.

Reports:
  - latent magnitude / angular spread (is the latent collapsed?)
  - decoder pre-sigmoid logit distribution
  - per-pixel sigmoid prob distribution
  - max probability per frame (is anything ever above the threshold?)
  - effective gradient signal: how far below 0 are alive-cell logits?
"""

from pathlib import Path
import numpy as np
import torch

import data
import engine
from model import Encoder, Decoder

CKPT = Path("checkpoints/best.pt")
N    = 32

torch.set_num_threads(2)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"Stuck checkpoint: step={ckpt['step']}, F1={ckpt['metrics']['alive_f1']:.4f}\n")

enc = Encoder(); enc.load_state_dict(ckpt["encoder"]); enc.eval()
dec = Decoder(kernel_size=1); dec.load_state_dict(ckpt["decoder"]); dec.eval()

pool = data.TrainingSeedPool()
rng  = np.random.default_rng(0)

seed_idx = rng.choice(pool.quartile_pools[3], size=N, replace=False)
seeds = np.asarray(pool.seeds[seed_idx])
trajs = engine.simulate(seeds, k=engine.K_DEFAULT)
lifespans = pool.lifespans[seed_idx]
frames = np.stack([trajs[i, int(rng.integers(0, lifespans[i]+1))] for i in range(N)])

x = torch.from_numpy(frames.astype(np.float32)).unsqueeze(1)
with torch.no_grad():
    z = enc(x)
    logits = dec(z)
    probs  = torch.sigmoid(logits)

print("=== LATENT ===")
norms = z.norm(dim=-1)
print(f"  ‖z‖: mean={norms.mean():.4f}  std={norms.std():.4f}")
zu = z / norms.unsqueeze(-1)
cos = (zu @ zu.T)
iu  = torch.triu_indices(N, N, offset=1)
pairs = cos[iu[0], iu[1]]
print(f"  pairwise cos sim: mean={pairs.mean():+.4f}  median={pairs.median():+.4f}  "
      f"min={pairs.min():+.4f}  max={pairs.max():+.4f}")
print(f"    → mean cos ≈ 1.0 means latent is COLLAPSED to one direction")

print("\n=== DECODER LOGITS (pre-sigmoid) ===")
print(f"  all-pixel: mean={logits.mean():+.4f}  std={logits.std():.4f}  "
      f"min={logits.min():+.3f}  max={logits.max():+.3f}")
alive_mask = (x == 1)
dead_mask  = (x == 0)
print(f"  logits @ true ALIVE pixels: mean={logits[alive_mask].mean():+.4f}  "
      f"std={logits[alive_mask].std():.4f}  max={logits[alive_mask].max():+.3f}")
print(f"  logits @ true DEAD  pixels: mean={logits[dead_mask].mean():+.4f}  "
      f"std={logits[dead_mask].std():.4f}  max={logits[dead_mask].max():+.3f}")

print("\n=== SIGMOID PROBS ===")
print(f"  all-pixel: mean={probs.mean():.5f}  max={probs.max():.5f}")
print(f"  fraction of pixels with p > 0.5: {(probs > 0.5).float().mean().item():.6%}")
print(f"  fraction of pixels with p > 0.1: {(probs > 0.1).float().mean().item():.6%}")
print(f"  per-frame max prob: mean={probs.amax(dim=(1,2,3)).mean():.5f}  "
      f"min={probs.amax(dim=(1,2,3)).min():.5f}")

print("\n=== ALIVE-CELL DISCRIMINATION ===")
gap = logits[alive_mask].mean() - logits[dead_mask].mean()
print(f"  mean(logit @ alive) - mean(logit @ dead) = {gap.item():+.4f}")
print(f"    → if 0 the model treats alive and dead identically (no signal)")
print(f"    → if << 0 the model thinks alive pixels are LESS likely than dead (anti-signal)")
