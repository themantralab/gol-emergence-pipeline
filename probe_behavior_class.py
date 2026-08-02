"""
Does the FROZEN latent separate known behaviour classes?

`labels.npy` holds per-seed hand classes (dying / still_life / oscillator /
glider) that were NEVER trained on. This is the validation set for the proposed
contrastive behaviour-embedding model. If a small classifier on frozen,
mean-pooled z already separates these classes well, the contrastive head is a
cheap, high-viability win (it only needs to ADD symmetry invariance). If not,
the encoder needs fine-tuning (much bigger effort).

Baseline: the same classifier on cheap population-statistics features — to check
whether the latent adds signal over trivial descriptors at all.

All trajectories placed at the CANONICAL CENTER (fixed position) so behaviour,
not position, is what's being separated. Reads best.pt only.
"""
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import label as cc_label
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import data, engine
from model import Encoder

CKPT = Path("checkpoints/best.pt")
CLASSES = ["dying", "still_life", "oscillator", "glider"]
N_PER_CLASS = 350
T = 8
CENTER = engine.SEED_OFFSET_CENTER
RNG = np.random.default_rng(0)


def encode_all(enc, frames, bs=64):
    zs = []
    with torch.no_grad():
        for i in range(0, len(frames), bs):
            x = torch.from_numpy(frames[i:i+bs].astype(np.float32)).unsqueeze(1)
            zs.append(enc(x))
    return torch.cat(zs).numpy()


def pop_stats(traj_frames):
    """Cheap trivial descriptor per trajectory: population curve stats + component
    counts at a few sampled times."""
    pops = traj_frames.reshape(len(traj_frames), -1).sum(1).astype(np.float32)
    feats = [pops.mean(), pops.std(), pops[0], pops[-1], pops.max(),
             pops[-1] / (pops[0] + 1e-6)]
    for t in [0, len(traj_frames)//2, len(traj_frames)-1]:
        n_cc = cc_label(traj_frames[t])[1]
        feats.append(n_cc)
    return np.array(feats, np.float32)


def main():
    torch.set_num_threads(8)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"Checkpoint step={ckpt['step']} F1={ckpt['metrics']['alive_f1']:.4f}")
    enc = Encoder(); enc.load_state_dict(ckpt["encoder"]); enc.eval()

    labels = np.load("data/labels.npy", allow_pickle=True).astype(str)
    lifes_all = np.load("data/lifespans.npy")
    pool = data.TrainingSeedPool()

    # balanced sample per class, lifespan >= 8 so there are frames to encode
    X_lat, X_base, y = [], [], []
    for ci, cls in enumerate(CLASSES):
        idx_pool = np.flatnonzero((labels == cls) & (lifes_all >= 8))
        pick = RNG.choice(idx_pool, size=N_PER_CLASS, replace=False)
        seeds = np.asarray(pool.seeds[pick])
        offs = np.full((N_PER_CLASS, 2), CENTER, np.int64)
        trajs = engine.simulate(seeds, k=engine.K_DEFAULT, offsets=offs)
        for i in range(N_PER_CLASS):
            L = int(lifes_all[pick[i]])
            ts = np.linspace(0, L, T).astype(int)
            aligned = trajs[i, ts]                          # (T,128,128)
            z = encode_all(enc, aligned).mean(0)            # mean-pool -> traj embedding
            X_lat.append(z); X_base.append(pop_stats(aligned)); y.append(ci)
        print(f"  encoded {cls:12s} ({N_PER_CLASS})")

    X_lat = np.stack(X_lat); X_base = np.stack(X_base); y = np.array(y)
    print(f"\nTotal {len(y)} trajectories, balanced {N_PER_CLASS}/class (chance = 25%)\n")

    def evaluate(X, name):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr = (Xtr-mu)/sd; Xte = (Xte-mu)/sd
        print(f"=== {name} (dim {X.shape[1]}) ===")
        for clf_name, clf in [("linear (logreg)", LogisticRegression(max_iter=2000, C=1.0)),
                              ("MLP (256)", MLPClassifier((256,), max_iter=800, random_state=0))]:
            clf.fit(Xtr, ytr)
            acc = clf.score(Xte, yte)
            bacc = balanced_accuracy_score(yte, clf.predict(Xte))
            print(f"  {clf_name:16s} accuracy={acc:.3f}  balanced_acc={bacc:.3f}")
        # confusion for the MLP
        cm = confusion_matrix(yte, clf.predict(Xte))
        print(f"  confusion (rows=true {CLASSES}):")
        for r, row in enumerate(cm):
            print(f"    {CLASSES[r]:12s} {row}")
        print()

    evaluate(X_lat,  "FROZEN LATENT (mean-pooled z)")
    evaluate(X_base, "BASELINE (population stats)")
    print("Verdict: latent >> baseline & high abs accuracy => contrastive head is a")
    print("cheap win on frozen z. latent ~ baseline or low => need encoder fine-tune.")


if __name__ == "__main__":
    main()
