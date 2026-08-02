"""
PERSISTENCE BASELINE for N1 — the measurement `dynamics_probe.py` never made.

WHY THIS EXISTS
---------------
`dynamics_probe.py` measures closed-loop latent rollout against the autoencoder's
teacher-forced ceiling, but it computes no NULL BASELINE. Without one, a rollout
score is uninterpretable: you cannot tell a predictor that learned the dynamics
from one that learned to copy its input. This script supplies the missing
comparison — persistence, i.e. "predict no change" — plus a data-scaling sweep
that distinguishes a capacity limit from a data limit.

`dynamics_probe.py` is deliberately NOT edited; this is a separate, additive
script so the original measurement stays reproducible.

PROTOCOL
--------
Mirrors `dynamics_probe.py` §2 exactly: Q3 seeds, center-biased offsets, K=60,
80 train / 20 held-out trajectories, same residual MLP (1024->2048->1024),
3000 Adam steps at lr=1e-3, batch 256. It draws its OWN sample (the RNG call
sequence in dynamics_probe is not replicable in isolation), so absolute numbers
may differ slightly from that script's — but every number below comes from the
SAME draw, which is what makes the comparison valid.

MEASURES
--------
  A. One-step LATENT error: learned predictor vs persistence (z_hat = z_t).
     Ratio near 1.0 means the predictor learned little beyond copying.
  B. Closed-loop rollout F1 vs horizon for THREE arms:
       - learned    : roll the predictor forward
       - persistence: output frame 0 forever ("predict no change")
       - AE ceiling : teacher-forced encode/decode of the true frame
     Persistence is the honest floor for the rollout table.
  C. One-step PIXEL persistence: F1(frame_t, frame_{t+1}) — how much a GoL frame
     changes per step at all. If this is high, the task is easy and a good
     rollout score means little.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

import data, engine
from model import Encoder, Decoder

CKPT = Path("checkpoints/best.pt")
RNG = np.random.default_rng(0)
torch.manual_seed(0)

K = 60
N_TRAJ = 100
N_TRAIN = 80
N_SCALE_POOL = 400        # separate, larger pool for the section-D scaling sweep
HORIZONS = [1, 2, 5, 10, 20, 40, 60]


def f1(pred, true):
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum()); fn = int((~pred & true).sum())
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d else 0.0


def encode_all(enc, frames, bs=64):
    zs = []
    with torch.no_grad():
        for i in range(0, len(frames), bs):
            x = torch.from_numpy(frames[i:i + bs].astype(np.float32)).unsqueeze(1)
            zs.append(enc(x))
    return torch.cat(zs)


def main():
    torch.set_num_threads(8)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"Checkpoint step={ckpt['step']} F1={ckpt['metrics']['alive_f1']:.4f}")
    print(f"Protocol mirrors dynamics_probe.py section 2 (Q3, K={K}, "
          f"{N_TRAIN} train / {N_TRAJ - N_TRAIN} held out), independent draw.\n")

    enc, dec = Encoder(), Decoder(kernel_size=1)
    enc.load_state_dict(ckpt["encoder"]); enc.eval()
    dec.load_state_dict(ckpt["decoder"]); dec.eval()
    pool = data.TrainingSeedPool()

    idx = RNG.choice(pool.quartile_pools[3], size=N_TRAJ, replace=False)
    seeds = np.asarray(pool.seeds[idx])
    offs = engine.sample_center_biased_offsets(N_TRAJ, RNG)
    trajs = engine.simulate(seeds, k=K, offsets=offs)                    # (N,K+1,128,128)
    Z = encode_all(enc, trajs.reshape(-1, 128, 128)).reshape(N_TRAJ, K + 1, -1)

    # larger, separate pool used only by section D (data scaling)
    idx_s = RNG.choice(pool.quartile_pools[3], size=N_SCALE_POOL, replace=False)
    seeds_s = np.asarray(pool.seeds[idx_s])
    offs_s = engine.sample_center_biased_offsets(N_SCALE_POOL, RNG)
    trajs_s = engine.simulate(seeds_s, k=K, offsets=offs_s)
    Zs = encode_all(enc, trajs_s.reshape(-1, 128, 128)).reshape(N_SCALE_POOL, K + 1, -1)

    tr, te = Z[:N_TRAIN], Z[N_TRAIN:]
    Xtr = tr[:, :-1].reshape(-1, 1024); Ytr = tr[:, 1:].reshape(-1, 1024)
    Xte = te[:, :-1].reshape(-1, 1024); Yte = te[:, 1:].reshape(-1, 1024)

    net = nn.Sequential(nn.Linear(1024, 2048), nn.GELU(), nn.Linear(2048, 1024))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(3000):
        j = torch.randint(0, len(Xtr), (256,))
        pred = Xtr[j] + net(Xtr[j])
        loss = ((pred - Ytr[j]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    print(f"  predictor trained (final batch MSE={loss.item():.6f})\n")

    # ---------------- A. one-step LATENT error ----------------
    print("=" * 70)
    print("A. ONE-STEP LATENT ERROR — learned predictor vs 'predict no change'")
    print("=" * 70)
    with torch.no_grad():
        pred_te = Xte + net(Xte)
    # relative L2, the natural scale-free per-step error
    err_learned = ((pred_te - Yte).norm(dim=-1) / Yte.norm(dim=-1).clamp_min(1e-6))
    err_persist = ((Xte - Yte).norm(dim=-1) / Yte.norm(dim=-1).clamp_min(1e-6))
    mse_learned = ((pred_te - Yte) ** 2).mean().item()
    mse_persist = ((Xte - Yte) ** 2).mean().item()
    cos_learned = torch.nn.functional.cosine_similarity(pred_te, Yte, dim=-1).mean().item()
    cos_persist = torch.nn.functional.cosine_similarity(Xte, Yte, dim=-1).mean().item()

    print(f"  held-out one-step pairs: {len(Xte)}")
    print(f"  {'metric':>26} {'learned':>12} {'persistence':>12} {'ratio':>10}")
    print(f"  {'relative L2 error':>26} {err_learned.mean():>12.4f} "
          f"{err_persist.mean():>12.4f} {err_learned.mean() / err_persist.mean():>10.3f}")
    print(f"  {'MSE':>26} {mse_learned:>12.6f} {mse_persist:>12.6f} "
          f"{mse_learned / mse_persist:>10.3f}")
    print(f"  {'cos(pred, true)':>26} {cos_learned:>12.4f} {cos_persist:>12.4f} "
          f"{'':>10}")
    print()
    print("  ratio < 1 = predictor beats persistence. How far below 1 is the")
    print("  question: near 1.0 means it learned little beyond copying its input.")

    # ---------------- B. closed-loop rollout, three arms ----------------
    print()
    print("=" * 70)
    print("B. CLOSED-LOOP ROLLOUT — learned vs persistence vs AE ceiling")
    print("=" * 70)
    rl = {h: [] for h in HORIZONS}
    pe = {h: [] for h in HORIZONS}
    tf = {h: [] for h in HORIZONS}
    with torch.no_grad():
        for ti in range(N_TRAIN, N_TRAJ):
            true = trajs[ti]
            zc = Z[ti, 0:1].clone()
            preds = {}
            for s in range(1, K + 1):
                zc = zc + net(zc)
                preds[s] = zc.clone()
            frame0 = (true[0] == 1)
            for h in HORIZONS:
                th = (true[h] == 1)
                pr = (torch.sigmoid(dec(preds[h])).squeeze().numpy() > 0.5)
                rl[h].append(f1(pr, th))
                pe[h].append(f1(frame0, th))                       # predict no change
                tfd = (torch.sigmoid(dec(Z[ti, h:h + 1])).squeeze().numpy() > 0.5)
                tf[h].append(f1(tfd, th))
    print(f"  {'horizon':>8} {'learned':>10} {'persistence':>12} {'AE ceiling':>11} "
          f"{'learned-persist':>16}")
    for h in HORIZONS:
        r, p, t = np.mean(rl[h]), np.mean(pe[h]), np.mean(tf[h])
        print(f"  {h:>8} {r:>10.3f} {p:>12.3f} {t:>11.3f} {r - p:>16.3f}")
    print()
    print("  If 'learned' <= 'persistence' at any horizon, the learned dynamics are")
    print("  worse than doing nothing at that horizon.")

    # ---------------- D. DATA SCALING: is this overfitting or a ceiling? ------
    # The obvious objection to N1 is "your predictor was too small / undertrained".
    # If held-out error EXCEEDS training error and exceeds persistence, the real
    # problem is the opposite - overfitting - and more capacity would not help.
    # This sweep asks whether MORE DATA closes the gap to persistence.
    print()
    print("=" * 70)
    print("D. DATA SCALING — does more training data beat persistence?")
    print("=" * 70)
    print(f"  {'n_train_traj':>13} {'train MSE':>11} {'held-out MSE':>13} "
          f"{'persist MSE':>12} {'ratio vs persist':>17}")
    for n_tr in [10, 20, 40, 80, 160, 320]:
        if n_tr > N_SCALE_POOL - 40:
            continue
        Xs = Zs[:n_tr, :-1].reshape(-1, 1024); Ys = Zs[:n_tr, 1:].reshape(-1, 1024)
        Xh = Zs[-40:, :-1].reshape(-1, 1024); Yh = Zs[-40:, 1:].reshape(-1, 1024)
        torch.manual_seed(0)
        n2 = nn.Sequential(nn.Linear(1024, 2048), nn.GELU(), nn.Linear(2048, 1024))
        o2 = torch.optim.Adam(n2.parameters(), lr=1e-3)
        for _ in range(3000):
            j = torch.randint(0, len(Xs), (256,))
            p = Xs[j] + n2(Xs[j])
            l = ((p - Ys[j]) ** 2).mean()
            o2.zero_grad(); l.backward(); o2.step()
        n2.eval()
        with torch.no_grad():
            tr_mse = ((Xs + n2(Xs) - Ys) ** 2).mean().item()
            ho_mse = ((Xh + n2(Xh) - Yh) ** 2).mean().item()
        pe_mse = ((Xh - Yh) ** 2).mean().item()
        print(f"  {n_tr:>13} {tr_mse:>11.6f} {ho_mse:>13.6f} {pe_mse:>12.6f} "
              f"{ho_mse / pe_mse:>17.3f}")
    print()
    print("  ratio < 1.0 at any row = more data DOES beat persistence there.")
    print("  ratio flat and > 1.0 across rows = not a data problem.")

    # ---------------- C. how fast do frames actually change ----------------
    print()
    print("=" * 70)
    print("C. PIXEL PERSISTENCE — how much does a GoL frame change per step?")
    print("=" * 70)
    step_f1 = []
    for ti in range(N_TRAIN, N_TRAJ):
        for s in range(K):
            step_f1.append(f1(trajs[ti, s] == 1, trajs[ti, s + 1] == 1))
    print(f"  mean F1(frame_t, frame_t+1) over held-out = {np.mean(step_f1):.4f}")
    print(f"  (high = consecutive frames are similar, so a 1-step task is easy;")
    print(f"   this is the pixel-space analogue of the persistence floor above)")


if __name__ == "__main__":
    main()
