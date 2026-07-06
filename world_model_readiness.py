"""
Is this autoencoder good enough to train a SECONDARY model on it (latent
dynamics predictor, or generative explorer to extrapolate new patterns)?

Reconstruction fidelity (F1~0.87) is necessary but NOT the deciding factor.
A secondary model produces APPROXIMATE latents — interpolated, predicted, or
sampled — never exact encoder outputs. The decisive question is whether the
decoder is robust OFF the encoder manifold. If it only decodes exact encoder
outputs sharply and turns everything else to mush, no secondary model can work.

Tests (all read best.pt only):
  A. Reconstruction fidelity + decoder decisiveness on REAL latents (baseline).
  B. INTERPOLATION: decode along z_a->z_b. Do midpoints stay sharp/valid?
  C. PERTURBATION: z + noise. How fast does reconstruction degrade?
  D. PRIOR SAMPLING: decode random latents from the training norm/sphere.
     Plausible frames => generative exploration viable.
  E. LATENT VELOCITY: per-step ||z_{t+1}-z_t|| — is dynamics smooth/learnable?
"""

from pathlib import Path
import numpy as np
import torch

import data, engine
from model import Encoder, Decoder

CKPT = Path("checkpoints/best.pt")
RNG = np.random.default_rng(0)
torch.manual_seed(0)


def decisiveness(probs):
    """Fraction of pixels that are AMBIGUOUS (prob in [0.1,0.9]). Low = sharp
    decoder committing to 0/1; high = mushy blob the decoder is unsure about."""
    return float(((probs > 0.1) & (probs < 0.9)).mean())


def f1(pred, true):
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum()); fn = int((~pred & true).sum())
    d = 2*tp+fp+fn
    return (2*tp/d) if d else 0.0


def gather_frames(enc, pool, n_per_q=40):
    frames, lat = [], []
    for q in range(4):
        idx = RNG.choice(pool.quartile_pools[q], size=n_per_q, replace=False)
        seeds = np.asarray(pool.seeds[idx])
        offs = engine.sample_center_biased_offsets(n_per_q, RNG)
        trajs = engine.simulate(seeds, k=engine.K_DEFAULT, offsets=offs)
        lifes = pool.lifespans[idx]
        for i in range(n_per_q):
            t = int(RNG.integers(0, int(lifes[i])+1))
            frames.append(trajs[i, t])
    frames = np.stack(frames)
    x = torch.from_numpy(frames.astype(np.float32)).unsqueeze(1)
    with torch.no_grad():
        z = enc(x)
    return frames, z


def main():
    torch.set_num_threads(4)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"Checkpoint: step={ckpt['step']}, val F1={ckpt['metrics']['alive_f1']:.4f}\n")
    enc, dec = Encoder(), Decoder(kernel_size=1)
    enc.load_state_dict(ckpt["encoder"]); enc.eval()
    dec.load_state_dict(ckpt["decoder"]); dec.eval()
    pool = data.TrainingSeedPool()

    frames, z = gather_frames(enc, pool)
    N = len(frames)
    true = frames == 1
    with torch.no_grad():
        probs_real = torch.sigmoid(dec(z)).squeeze(1).numpy()
    norms = z.norm(dim=-1).numpy()

    # ===== A. baseline on real latents =====
    print("="*70)
    print("A. BASELINE — real encoder latents")
    print("="*70)
    f1_real = np.mean([f1(probs_real[i] > 0.5, true[i]) for i in range(N)])
    amb_real = np.mean([decisiveness(probs_real[i]) for i in range(N)])
    print(f"  reconstruction F1        = {f1_real:.3f}")
    print(f"  ambiguous-pixel fraction = {amb_real:.4f}   (lower = sharper decoder)")
    print(f"  ||z||: mean={norms.mean():.3f} std={norms.std():.3f}")

    # ===== B. interpolation =====
    print("\n" + "="*70)
    print("B. INTERPOLATION — decode along z_a -> z_b (50 random real pairs)")
    print("="*70)
    ia = RNG.integers(0, N, 50); ib = RNG.integers(0, N, 50)
    print(f"  {'alpha':>6} {'max_prob':>9} {'amb_frac':>9} {'n_pred_alive':>13}")
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        zi = (1-a)*z[ia] + a*z[ib]
        with torch.no_grad():
            p = torch.sigmoid(dec(zi)).squeeze(1).numpy()
        mp = p.reshape(50,-1).max(1).mean()
        amb = np.mean([decisiveness(p[i]) for i in range(50)])
        npa = (p > 0.5).reshape(50,-1).sum(1).mean()
        print(f"  {a:>6.2f} {mp:>9.3f} {amb:>9.4f} {npa:>13.1f}")
    print("  -> if midpoint (0.5) max_prob stays high & amb_frac stays low,")
    print("     the decoder handles interpolated latents => secondary model viable.")

    # ===== C. perturbation =====
    print("\n" + "="*70)
    print("C. PERTURBATION ROBUSTNESS — z + eps*||z||*unit_noise")
    print("="*70)
    print(f"  {'eps':>6} {'F1_vs_orig':>11} {'amb_frac':>9}")
    for eps in [0.0, 0.05, 0.1, 0.2, 0.4]:
        noise = torch.randn_like(z)
        noise = noise / noise.norm(dim=-1, keepdim=True) * z.norm(dim=-1, keepdim=True)
        zp = z + eps * noise
        with torch.no_grad():
            p = torch.sigmoid(dec(zp)).squeeze(1).numpy()
        f1v = np.mean([f1(p[i] > 0.5, true[i]) for i in range(N)])
        amb = np.mean([decisiveness(p[i]) for i in range(N)])
        print(f"  {eps:>6.2f} {f1v:>11.3f} {amb:>9.4f}")
    print("  -> graceful (slow F1 drop) = robust manifold; cliff = brittle.")

    # ===== D. prior sampling =====
    print("\n" + "="*70)
    print("D. PRIOR SAMPLING — decode random latents at the training norm")
    print("="*70)
    samp = torch.randn(N, z.shape[1])
    samp = samp / samp.norm(dim=-1, keepdim=True) * float(norms.mean())
    with torch.no_grad():
        p = torch.sigmoid(dec(samp)).squeeze(1).numpy()
    mp = p.reshape(N,-1).max(1)
    amb = np.mean([decisiveness(p[i]) for i in range(N)])
    npa = (p > 0.5).reshape(N,-1).sum(1)
    print(f"  sampled frames: max_prob mean={mp.mean():.3f}  amb_frac={amb:.4f}")
    print(f"  predicted alive/frame: mean={npa.mean():.1f}  (real frames ~{true.reshape(N,-1).sum(1).mean():.1f})")
    print(f"  fraction of samples that are ~empty (<5 alive): {(npa<5).mean():.1%}")
    print(f"  compare real-latent amb_frac={amb_real:.4f}: similar => prior decodable.")

    # ===== E. latent velocity (dynamics smoothness) =====
    print("\n" + "="*70)
    print("E. LATENT VELOCITY — ||z_{t+1}-z_t|| along trajectories")
    print("="*70)
    idx = RNG.choice(pool.quartile_pools[3], size=8, replace=False)
    seeds = np.asarray(pool.seeds[idx])
    offs = engine.sample_center_biased_offsets(8, RNG)
    trajs = engine.simulate(seeds, k=120, offsets=offs)
    vels = []
    for i in range(8):
        xf = torch.from_numpy(trajs[i, :100].astype(np.float32)).unsqueeze(1)
        with torch.no_grad():
            zt = enc(xf)
        step = (zt[1:] - zt[:-1]).norm(dim=-1).numpy()
        vels.append(step)
    vels = np.concatenate(vels)
    print(f"  per-step ||dz||: mean={vels.mean():.3f} std={vels.std():.3f} "
          f"(relative to ||z||~{norms.mean():.2f})")
    print(f"  ratio step/||z|| = {vels.mean()/norms.mean():.3f}  "
          f"(small & stable = smooth, learnable dynamics)")


if __name__ == "__main__":
    main()
