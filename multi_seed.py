"""
MULTI-SEED HARNESS — error bars for the paper's load-bearing numbers.

WHY THIS EXISTS
---------------
Every diagnostic in this project is single-seed. For a paper whose contribution is
a set of NEGATIVE results, "you got unlucky" is the cheapest reviewer attack and
single-seed numbers cannot answer it. This re-runs the three measurements the
paper actually leans on, across N_SEEDS independent draws, and reports mean +/- std.

The published diagnostics are NOT modified; this reimplements their protocols with
the seed parameterised.

MEASURED
--------
  1. N1  — closed-loop rollout F1 vs horizon: learned / persistence / AE ceiling.
  2. N2  — behaviour classification: frozen latent vs population-statistic baseline.
  3. §5  — motif detection AUC: phase-invariant latent bank vs template matching,
           GLIDER ONLY (the lead case: 0.2% negative-set contamination, and the
           only one of the three motifs that translates).
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import label as cc_label
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

import data, engine
from model import Encoder, Decoder

CKPT = Path("checkpoints/best.pt")
N_SEEDS = 5
HORIZONS = [1, 2, 5, 10, 20, 40, 60]
CLASSES = ["dying", "still_life", "oscillator", "glider"]
N_PER_CLASS = 200
T = 8
TILE, NTILE = 8, 16
GLIDER = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.uint8)


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


def trunk_codes(enc, grids):
    out = []
    with torch.no_grad():
        for i in range(0, len(grids), 32):
            x = torch.from_numpy(grids[i:i + 32].astype(np.float32)).unsqueeze(1)
            h = enc.conv(x)
            out.append(h.reshape(h.shape[0], 128, NTILE * NTILE).permute(0, 2, 1))
    return torch.cat(out)


def auc(pos, neg):
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float); ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = len(pos), len(neg)
    return (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ms(a):
    a = np.asarray(a, float)
    return f"{a.mean():.3f}+/-{a.std(ddof=1):.3f}"


# ----------------------------------------------------------------------------
# 1. N1
# ----------------------------------------------------------------------------
def run_n1(enc, dec, pool, seed):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    K, N, NTR = 60, 100, 80
    idx = rng.choice(pool.quartile_pools[3], size=N, replace=False)
    trajs = engine.simulate(np.asarray(pool.seeds[idx]), k=K,
                            offsets=engine.sample_center_biased_offsets(N, rng))
    Z = encode_all(enc, trajs.reshape(-1, 128, 128)).reshape(N, K + 1, -1)
    Xtr = Z[:NTR, :-1].reshape(-1, 1024); Ytr = Z[:NTR, 1:].reshape(-1, 1024)

    net = nn.Sequential(nn.Linear(1024, 2048), nn.GELU(), nn.Linear(2048, 1024))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(3000):
        j = torch.randint(0, len(Xtr), (256,))
        loss = ((Xtr[j] + net(Xtr[j]) - Ytr[j]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    rl, pe, tf = {h: [] for h in HORIZONS}, {h: [] for h in HORIZONS}, {h: [] for h in HORIZONS}
    with torch.no_grad():
        for ti in range(NTR, N):
            true = trajs[ti]; zc = Z[ti, 0:1].clone(); preds = {}
            for s in range(1, K + 1):
                zc = zc + net(zc); preds[s] = zc.clone()
            f0 = (true[0] == 1)
            for h in HORIZONS:
                th = (true[h] == 1)
                rl[h].append(f1(torch.sigmoid(dec(preds[h])).squeeze().numpy() > 0.5, th))
                pe[h].append(f1(f0, th))
                tf[h].append(f1(torch.sigmoid(dec(Z[ti, h:h + 1])).squeeze().numpy() > 0.5, th))
    return ({h: float(np.mean(rl[h])) for h in HORIZONS},
            {h: float(np.mean(pe[h])) for h in HORIZONS},
            {h: float(np.mean(tf[h])) for h in HORIZONS})


# ----------------------------------------------------------------------------
# 2. N2
# ----------------------------------------------------------------------------
def pop_stats(fr):
    pops = fr.reshape(len(fr), -1).sum(1).astype(np.float32)
    feats = [pops.mean(), pops.std(), pops[0], pops[-1], pops.max(),
             pops[-1] / (pops[0] + 1e-6)]
    for t in [0, len(fr) // 2, len(fr) - 1]:
        feats.append(cc_label(fr[t])[1])
    return np.array(feats, np.float32)


def run_n2(enc, pool, labels, lifes_all, seed):
    rng = np.random.default_rng(seed)
    CENTER = engine.SEED_OFFSET_CENTER
    XL, XB, y = [], [], []
    for ci, cls in enumerate(CLASSES):
        ip = np.flatnonzero((labels == cls) & (lifes_all >= 8))
        pick = rng.choice(ip, size=N_PER_CLASS, replace=False)
        trajs = engine.simulate(np.asarray(pool.seeds[pick]), k=engine.K_DEFAULT,
                                offsets=np.full((N_PER_CLASS, 2), CENTER, np.int64))
        for i in range(N_PER_CLASS):
            L = int(lifes_all[pick[i]])
            aligned = trajs[i, np.linspace(0, L, T).astype(int)]
            XL.append(encode_all(enc, aligned).numpy().mean(0))
            XB.append(pop_stats(aligned)); y.append(ci)
    XL, XB, y = np.stack(XL), np.stack(XB), np.array(y)

    out = {}
    for name, X in [("latent", XL), ("stats", XB)]:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                              random_state=seed, stratify=y)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        lin = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        mlp = MLPClassifier((256,), max_iter=800, random_state=seed).fit(Xtr, ytr)
        out[name + "_lin"] = balanced_accuracy_score(yte, lin.predict(Xte))
        out[name + "_mlp"] = balanced_accuracy_score(yte, mlp.predict(Xte))
    return out


# ----------------------------------------------------------------------------
# 3. motif (glider)
# ----------------------------------------------------------------------------
def plant(grid, motif, tile_rc, off_rc):
    R = tile_rc[0] * TILE + off_rc[0]; C = tile_rc[1] * TILE + off_rc[1]
    grid[R:R + motif.shape[0], C:C + motif.shape[1]] |= motif


def tmpl_score(tile8, motif, mode):
    h, w = motif.shape
    pad = np.zeros((TILE + h - 1, TILE + w - 1), np.uint8); pad[:TILE, :TILE] = tile8
    on = int(motif.sum()); best = -1.0
    for r in range(TILE):
        for c in range(TILE):
            win = pad[r:r + h, c:c + w]
            inter = int((win & motif).sum())
            if mode == "iou":
                u = int((win | motif).sum()); s = (inter / u) if u else 0.0
            else:
                s = (inter - int((win & (1 - motif)).sum())) / on
            best = max(best, s)
    return best


def run_motif(enc, pool, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(pool.quartile_pools[2], size=40, replace=False)
    trajs = engine.simulate(np.asarray(pool.seeds[idx]), k=64,
                            offsets=engine.sample_center_biased_offsets(40, rng))
    real = trajs[:, ::8].reshape(-1, 128, 128)
    rc = trunk_codes(enc, real).reshape(-1, 128)
    rt = real.reshape(-1, NTILE, TILE, NTILE, TILE).transpose(0, 1, 3, 2, 4).reshape(-1, TILE, TILE)
    ne = np.flatnonzero(rt.reshape(len(rt), -1).sum(1) > 0)
    sel = rng.choice(ne, size=min(1500, len(ne)), replace=False)
    neg_codes, neg_tiles = rc[sel], rt[sel]

    offs_list = [(r, c) for r in range(TILE - 2) for c in range(TILE - 2)]
    g = np.zeros((len(offs_list), 128, 128), np.uint8)
    for k, o in enumerate(offs_list):
        plant(g[k], GLIDER, (8, 8), o)
    bank = torch.nn.functional.normalize(trunk_codes(enc, g)[:, 8 * NTILE + 8], dim=-1)

    res = {}
    for nf in [0, 1, 2, 3]:
        n_pos = 200
        gp = np.zeros((n_pos, 128, 128), np.uint8)
        pt = np.zeros((n_pos, TILE, TILE), np.uint8)
        for k in range(n_pos):
            m = GLIDER.copy()
            if nf:
                fl = m.reshape(-1).copy()
                fl[rng.choice(m.size, size=nf, replace=False)] ^= 1
                m = fl.reshape(m.shape)
            plant(gp[k], m, (8, 8), (int(rng.integers(0, TILE - 2)), int(rng.integers(0, TILE - 2))))
            pt[k] = gp[k][64:72, 64:72]
        pc = trunk_codes(enc, gp)[:, 8 * NTILE + 8]
        sb_p = (torch.nn.functional.normalize(pc, dim=-1) @ bank.T).max(1).values.numpy()
        sb_n = (torch.nn.functional.normalize(neg_codes, dim=-1) @ bank.T).max(1).values.numpy()
        best_t = max(
            auc(np.array([tmpl_score(t, GLIDER, "penalized") for t in pt]),
                np.array([tmpl_score(t, GLIDER, "penalized") for t in neg_tiles])),
            auc(np.array([tmpl_score(t, GLIDER, "iou") for t in pt]),
                np.array([tmpl_score(t, GLIDER, "iou") for t in neg_tiles])))
        res[nf] = (auc(sb_p, sb_n), best_t)
    return res


def main():
    torch.set_num_threads(8)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"Checkpoint step={ckpt['step']} F1={ckpt['metrics']['alive_f1']:.4f}")
    print(f"Seeds: {list(range(N_SEEDS))}  (mean +/- sample std, ddof=1)\n")
    enc, dec = Encoder(), Decoder(kernel_size=1)
    enc.load_state_dict(ckpt["encoder"]); enc.eval()
    dec.load_state_dict(ckpt["decoder"]); dec.eval()
    pool = data.TrainingSeedPool()

    # ---- N1 ----
    print("=" * 74)
    print("1. N1 ROLLOUT (learned / persistence / AE ceiling), F1 by horizon")
    print("=" * 74)
    RL, PE, TF = [], [], []
    for s in range(N_SEEDS):
        r, p, t = run_n1(enc, dec, pool, s); RL.append(r); PE.append(p); TF.append(t)
        print(f"  seed {s} done")
    print(f"\n  {'horizon':>8} {'learned':>16} {'persistence':>16} {'AE ceiling':>16}")
    for h in HORIZONS:
        print(f"  {h:>8} {ms([r[h] for r in RL]):>16} {ms([p[h] for p in PE]):>16} "
              f"{ms([t[h] for t in TF]):>16}")
    print("\n  -> AE ceiling flat across horizons while learned rollout collapses is")
    print("     the load-bearing claim; check the std bars do not overlap at h>=20.")

    # ---- N2 ----
    print("\n" + "=" * 74)
    print(f"2. N2 BEHAVIOUR CLASSIFICATION ({N_PER_CLASS}/class, chance 0.25)")
    print("=" * 74)
    R2 = [run_n2(enc, pool, np.load("data/labels.npy", allow_pickle=True).astype(str),
                 np.load("data/lifespans.npy"), s) for s in range(N_SEEDS)]
    for k in ["latent_lin", "latent_mlp", "stats_lin", "stats_mlp"]:
        print(f"  {k:>12}  balanced_acc = {ms([r[k] for r in R2])}")
    gaps = [r["stats_mlp"] - r["latent_mlp"] for r in R2]
    print(f"\n  stats_mlp - latent_mlp gap = {ms(gaps)}  "
          f"(>0 every seed: {all(g > 0 for g in gaps)})")

    # ---- motif ----
    print("\n" + "=" * 74)
    print("3. MOTIF DETECTION AUC — glider, latent bank vs best template")
    print("=" * 74)
    RM = [run_motif(enc, pool, s) for s in range(N_SEEDS)]
    print(f"  {'n_flip':>7} {'latent bank':>16} {'best template':>16} {'bank - tmpl':>16}")
    for nf in [0, 1, 2, 3]:
        b = [r[nf][0] for r in RM]; t = [r[nf][1] for r in RM]
        d = [x - y for x, y in zip(b, t)]
        print(f"  {nf:>7} {ms(b):>16} {ms(t):>16} {ms(d):>16}")
    print("\n  -> the constructive claim needs bank-tmpl > 0 with non-overlapping")
    print("     error bars at n_flip>=2.")

    np.save("logs/multi_seed_raw.npy",
            {"n1_learned": RL, "n1_persist": PE, "n1_ceiling": TF,
             "n2": R2, "motif": RM, "horizons": HORIZONS}, allow_pickle=True)
    print("\nRaw per-seed values -> logs/multi_seed_raw.npy (for figure error bars)")


if __name__ == "__main__":
    main()
