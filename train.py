"""
Vanilla autoencoder training for the GoL world model.

Single architecture: decoder is kernel_size=1 throughout (halo-free by
construction — no spatial mixing across pixels). Bootstrapping out of the
"predict all dead" plateau is handled by a pos_weight curriculum on the BCE
reconstruction loss:

  pos_weight(step):
    start = POS_WEIGHT_START   (large; missing an alive cell is very costly)
    end   = ALIVE_POS_WEIGHT   (steady-state)
    schedule = cosine decay over POS_WEIGHT_DECAY_STEPS

  Diagnostic on a previous 1×1 run (step 500) showed the failure mode:
    - All latents collapsed to one direction (pairwise cos sim = 1.000)
    - Decoder learned a frame-independent spatial prior (max prob 0.41 < 0.5)
    - Encoder received no gradient signal because decoder ignored z

  Cure: high pos_weight forces the decoder to predict alive somewhere; the
  only way to predict alive in the right place per frame is to USE z; that
  demand back-propagates as strong encoder gradient and breaks the collapse.

All four losses active throughout (L1+L2+L3+L4).
"""

from pathlib import Path
import os
import time
import random
import threading
import queue
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import data
import engine
import losses
from model import Encoder, Decoder, LATENT_DIM


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SEED               = 0
BATCH_SIZE         = 32
FRAMES_PER_TRAJ    = 3       # frames sampled per trajectory per step (was 1) — more spread
LATE_BIAS_POW      = 0.5     # t = (U**pow)*lifespan; pow<1 skews toward late ("old") generations
LR                 = 3e-4
LR_MIN             = 3e-5
N_THREADS          = 8
TRAIN_STEPS        = 100_000
LOG_EVERY          = 50
VAL_EVERY          = 500
CKPT_EVERY         = 10_000
HOLDOUT_PER_Q      = 25
N_VAL_FRAMES_PER_Q = 80

ALIVE_F1_GATE      = 0.95
DEAD_F1_GATE       = 0.99
GATE_CONSECUTIVE   = 3

CKPT_DIR           = Path(__file__).resolve().parent / "checkpoints"
LOG_PATH           = Path(__file__).resolve().parent / "logs" / "train.log"


# -----------------------------------------------------------------------------

class Tee:
    """Write to stdout and a file at the same time."""
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("a", buffering=1)
    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        self.f.write(msg + "\n")


def to_tensor_frames(grids: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(grids.astype(np.float32)).unsqueeze(1)


def make_holdout(pool: data.TrainingSeedPool, rng: np.random.Generator) -> dict:
    """Pull HOLDOUT_PER_Q seeds from each quartile, remove from training pools,
    pre-simulate their trajectories so validation runs from RAM."""
    out = {}
    for q in range(pool.n_quartiles):
        chosen = rng.choice(pool.quartile_pools[q], size=HOLDOUT_PER_Q, replace=False)
        pool.quartile_pools[q] = np.setdiff1d(pool.quartile_pools[q], chosen, assume_unique=True)
        seeds = np.asarray(pool.seeds[chosen])
        # Per-trajectory center-biased offset, fixed for this holdout (sampled
        # once with the seeded rng) so validation measures varied placements
        # deterministically across the run.
        offsets = engine.sample_center_biased_offsets(len(chosen), rng)
        traj  = engine.simulate(seeds, k=engine.K_DEFAULT, offsets=offsets)
        out[q] = {"idx": chosen, "lifespans": pool.lifespans[chosen], "traj": traj}
    return out


# -----------------------------------------------------------------------------
# Single-frame batch construction
# -----------------------------------------------------------------------------

def fresh_training_batch(pool: data.TrainingSeedPool, rng: np.random.Generator) -> np.ndarray:
    """Sample BATCH_SIZE trajectories (stratified by lifespan quartile),
    simulate them fresh, and pick FRAMES_PER_TRAJ frames per trajectory from its
    live region [0, lifespan], time-biased toward late ("old") generations where
    evolved structures are most diverse. Returns
    (BATCH_SIZE * FRAMES_PER_TRAJ, 128, 128) uint8."""
    idx       = pool.sample_batch_indices(BATCH_SIZE, rng)
    seeds     = np.asarray(pool.seeds[idx])
    offsets   = engine.sample_center_biased_offsets(BATCH_SIZE, rng)
    traj      = engine.simulate(seeds, k=engine.K_DEFAULT, offsets=offsets)
    lifespans = pool.lifespans[idx].astype(np.int64)

    upper = lifespans + 1
    ar    = np.arange(BATCH_SIZE)
    # U**LATE_BIAS_POW with pow<1 pushes sampled timesteps toward the top of
    # [0, lifespan] (older generations), while still covering early frames.
    out = []
    for _ in range(FRAMES_PER_TRAJ):
        u     = rng.random(BATCH_SIZE) ** LATE_BIAS_POW
        t_idx = np.minimum((u * upper).astype(np.int64), lifespans)
        out.append(traj[ar, t_idx])
    return np.concatenate(out, axis=0)


class _PrefetchError:
    def __init__(self, exc): self.exc = exc


class BatchPrefetcher:
    """Background thread that simulates the next batch while the main thread
    is doing forward + backward. numpy ufuncs in engine.simulate release the
    GIL so the simulation overlaps with torch compute."""
    def __init__(self, pool: data.TrainingSeedPool, rng: np.random.Generator,
                 queue_size: int = 2):
        self.pool = pool
        self.rng  = rng
        self.queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._producer, daemon=True)
        self.thread.start()

    def _producer(self):
        while not self._stop.is_set():
            try:
                batch = fresh_training_batch(self.pool, self.rng)
            except Exception as exc:
                self.queue.put(_PrefetchError(exc))
                return
            while not self._stop.is_set():
                try:
                    self.queue.put(batch, timeout=0.5); break
                except queue.Full:
                    continue

    def next_batch(self):
        item = self.queue.get()
        if isinstance(item, _PrefetchError):
            raise item.exc
        return item

    def stop(self):
        self._stop.set()


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate_per_quartile(encoder: Encoder, decoder: Decoder,
                          holdout: dict, rng: np.random.Generator) -> dict:
    encoder.eval(); decoder.eval()
    per_q = {}
    all_logits, all_targets = [], []
    for q, h in holdout.items():
        n = len(h["lifespans"])
        upper = h["lifespans"].astype(np.int64) + 1
        tr_idx = rng.integers(0, n, size=N_VAL_FRAMES_PER_Q)
        t_idx  = (rng.random(N_VAL_FRAMES_PER_Q) * upper[tr_idx]).astype(np.int64)
        frames = h["traj"][tr_idx, t_idx]
        x = to_tensor_frames(frames)
        with torch.no_grad():
            logits = decoder(encoder(x))
        m = losses.reconstruction_metrics(logits, x)
        per_q[q] = m
        all_logits.append(logits); all_targets.append(x)
    overall = losses.reconstruction_metrics(
        torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0))
    encoder.train(); decoder.train()
    return {"per_q": per_q, "overall": overall}


# -----------------------------------------------------------------------------
# Full-state checkpointing (resumable training)
# -----------------------------------------------------------------------------

LATEST_PATH = CKPT_DIR / "latest.pt"


def save_full_state(path: Path, encoder, decoder, opt, sched, step,
                    best_alive_f1, gate_streak, gate_reached, np_rng) -> None:
    """Save everything needed to resume training bit-for-bit on the model side
    (weights + optimizer moments + LR schedule + bookkeeping + RNG states)."""
    torch.save({
        "encoder":       encoder.state_dict(),
        "decoder":       decoder.state_dict(),
        "optimizer":     opt.state_dict(),
        "scheduler":     sched.state_dict(),
        "step":          step,
        "best_alive_f1": best_alive_f1,
        "gate_streak":   gate_streak,
        "gate_reached":  gate_reached,
        "torch_rng":     torch.get_rng_state(),
        "np_rng_state":  np_rng.bit_generator.state,
    }, path)


# -----------------------------------------------------------------------------

def main(resume: bool = False, target_steps: int = TRAIN_STEPS,
         lr_restart: float | None = None) -> None:
    # --- reproducibility: seed every RNG source ---------------------------
    # All stochastic inputs are seeded and the RNG streams are isolated so the
    # run is reproducible: model init (torch), validation sampling (np_rng), and
    # the background batch producer (producer_rng, a SEPARATE stream so the
    # prefetch thread never races the main thread's RNG). The producer emits
    # batches in strict FIFO order, so the data sequence is deterministic
    # regardless of thread timing.
    #
    # NOT bitwise-identical across runs: torch.set_num_threads(N_THREADS) makes
    # intra-op FP reduction order vary, so two runs match in distribution but not
    # to the last bit. Full bitwise determinism needs single-thread or
    # torch.use_deterministic_algorithms(True) (~8× slower) — deliberately
    # rejected: not worth the compute for this work, and it conflicts with the
    # 8-core utilisation requirement.
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(N_THREADS)
    np_rng = np.random.default_rng(SEED)

    log = Tee(LOG_PATH)
    log(f"=== Vanilla AE training — restart 2026-06-30 (Jaccard→IoU L2 fix + geometry-weight decay) ===")
    log(f"threads={N_THREADS}  seed={SEED}  (reproducible-in-distribution; not bitwise — 8-thread FP order)")

    # --- data ---
    pool = data.TrainingSeedPool()
    log(pool.summary())
    holdout = make_holdout(pool, np_rng)
    log(f"\nHoldout: {HOLDOUT_PER_Q}/quartile × 4 = {HOLDOUT_PER_Q * 4} seeds (trajectories pre-simulated)")

    # --- model: 1×1 decoder throughout (halo-free) ---
    encoder = Encoder()
    decoder = Decoder(kernel_size=1)
    opt = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)
    sched = CosineAnnealingLR(opt, T_max=TRAIN_STEPS, eta_min=LR_MIN)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    log(f"\nModel: encoder ({n_enc:,}) + decoder ({n_dec:,}) = {n_enc + n_dec:,} parameters")
    log(f"Decoder: kernel_size=1 throughout (halo-free; no spatial mixing across pixels)")
    log(f"Latent: ℝ^{LATENT_DIM} (no sphere, no L2-normalise, soft-bounded to ‖z‖≈{losses.TARGET_NORM})")
    log(f"Sampling: {FRAMES_PER_TRAJ} frames/trajectory × {BATCH_SIZE} trajectories = "
        f"{FRAMES_PER_TRAJ * BATCH_SIZE} frames/step, late-biased (U**{LATE_BIAS_POW}) toward old generations")
    log(f"Seed placement: center-biased random offset "
        f"(mean={engine.SEED_OFFSET_CENTER}, std={engine.SEED_OFFSET_STD}, "
        f"clip=[{engine.SEED_OFFSET_MIN},{engine.SEED_OFFSET_MAX}]) — translation diversity")
    log(f"Losses:   L1 (BCE, curriculum pos_weight)  +  "
        f"L2 (cos_sim↔IoU, clamp≥0, w={losses.W_SMOOTH})  +  "
        f"L3 (soft ‖z‖²≈{losses.TARGET_NORM}², w={losses.W_NORM})  +  "
        f"L4 (angular uniformity, w={losses.W_UNIF})")
    log(f"pos_weight curriculum: cosine {losses.POS_WEIGHT_START} → "
        f"{losses.ALIVE_POS_WEIGHT} over {losses.POS_WEIGHT_DECAY_STEPS} steps, then held")
    log(f"L3 weight warmup: 0 → {losses.W_NORM} over {losses.NORM_WARMUP_STEPS} steps")
    log(f"geometry-weight decay (L2,L4): ×1.0 held to step {losses.GEOM_HOLD_STEPS}, "
        f"cosine → ×{losses.GEOM_FLOOR} over next {losses.GEOM_DECAY_STEPS} steps")
    log(f"LR: cosine {LR} → {LR_MIN} over {TRAIN_STEPS} steps")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # --- resume bookkeeping (holdout above is always rebuilt identically from
    #     the SEED-seeded np_rng BEFORE any state restore, so validation stays
    #     comparable; we then fast-forward np_rng to the checkpoint state) ---
    start_step    = 1
    best_alive_f1 = -1.0
    gate_streak   = 0
    gate_reached  = False
    if resume:
        if not LATEST_PATH.exists():
            raise FileNotFoundError(f"--resume given but {LATEST_PATH} not found")
        ck = torch.load(LATEST_PATH, map_location="cpu", weights_only=False)
        encoder.load_state_dict(ck["encoder"]); decoder.load_state_dict(ck["decoder"])
        opt.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["torch_rng"])
        np_rng.bit_generator.state = ck["np_rng_state"]
        start_step    = ck["step"] + 1
        best_alive_f1 = ck["best_alive_f1"]
        gate_streak   = ck["gate_streak"]
        gate_reached  = ck["gate_reached"]
        if lr_restart is not None:
            # Warm-restart: the finished run's cosine LR bottomed at LR_MIN, so a
            # plain resume barely moves. Bump the LR and run a fresh cosine over
            # the extension window so the extra steps actually do something.
            for g in opt.param_groups:
                g["lr"] = lr_restart
            remaining = max(target_steps - start_step + 1, 1)
            sched = CosineAnnealingLR(opt, T_max=remaining, eta_min=LR_MIN)
            log(f"\n[resume] LR warm-restart: cosine {lr_restart:g} → {LR_MIN:g} over {remaining} steps")
        else:
            sched.load_state_dict(ck["scheduler"])
        log(f"[resume] restored {LATEST_PATH.name}: continuing at step {start_step} "
            f"(best_alive_f1={best_alive_f1:.4f}, gate_reached={gate_reached})")

    losses_window = []
    t_start = time.perf_counter()

    log(f"\nStarting training: target step {target_steps}  (gate is non-terminal — training continues past it)")
    log(f"Gate: alive_F1 >= {ALIVE_F1_GATE} AND dead_F1 >= {DEAD_F1_GATE} for {GATE_CONSECUTIVE} consecutive val checks\n")

    # Producer RNG: isolated stream seeded deterministically from (SEED, start_step)
    # so the data sequence is reproducible for a given run/resume point.
    producer_rng = np.random.default_rng([SEED + 1, start_step])
    prefetcher   = BatchPrefetcher(pool, producer_rng, queue_size=2)

    for step in range(start_step, target_steps + 1):
        frames_np = prefetcher.next_batch()
        x = to_tensor_frames(frames_np)

        z = encoder(x)
        logits = decoder(z)
        pw = losses.pos_weight_at_step(step)
        w3 = losses.norm_weight_at_step(step, end=losses.W_NORM)
        gscale = losses.geometry_weight_scale(step)        # late-training decay
        w2 = losses.W_SMOOTH * gscale
        w4 = losses.W_UNIF   * gscale
        total, components = losses.combined_loss(logits, x, z, x,
                                                 w2=w2, w3=w3, w4=w4,
                                                 pos_weight=pw)

        opt.zero_grad()
        total.backward()
        opt.step()
        sched.step()

        losses_window.append((total.item(), components["l1"], components["l2"],
                              components["l3"], components["l4"],
                              z.detach().norm(dim=-1).mean().item()))

        if step % LOG_EVERY == 0:
            elapsed = time.perf_counter() - t_start
            arr = np.array(losses_window)
            mean_total, mean_l1, mean_l2, mean_l3, mean_l4, mean_znorm = arr.mean(axis=0)
            losses_window.clear()
            lr_now = opt.param_groups[0]['lr']
            wl1 = mean_l1 * losses.W_RECON
            wl2 = mean_l2 * w2                # time-varying (geometry decay)
            wl3 = mean_l3 * w3                # time-varying L3 weight (warmup)
            wl4 = mean_l4 * w4                # time-varying (geometry decay)
            log(f"  step={step:>6}  total={mean_total:+.4f}  "
                f"raw(L1={mean_l1:.4f} L2={mean_l2:.4f} L3={mean_l3:.4f} L4={mean_l4:+.4f})  "
                f"wtd(L1={wl1:.4f} L2={wl2:.4f} L3={wl3:.4f} L4={wl4:+.4f})  "
                f"pw={pw:.2f}  gscale={gscale:.3f}  ‖z‖={mean_znorm:.3f}  lr={lr_now:.2e}  ({elapsed / step:.2f}s/step)")

        if step % VAL_EVERY == 0:
            m = validate_per_quartile(encoder, decoder, holdout, np_rng)
            ov = m["overall"]
            log(f"  [val] step={step:>6}  overall  alive_F1={ov['alive_f1']:.4f}  dead_F1={ov['dead_f1']:.4f}  "
                f"alive_P={ov['alive_precision']:.4f}  alive_R={ov['alive_recall']:.4f}")
            for q in sorted(m["per_q"].keys()):
                qm = m["per_q"][q]
                log(f"        Q{q}: alive_F1={qm['alive_f1']:.4f}  alive_P={qm['alive_precision']:.4f}  alive_R={qm['alive_recall']:.4f}")

            if ov["alive_f1"] > best_alive_f1:
                best_alive_f1 = ov["alive_f1"]
                torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
                            "step": step, "metrics": ov, "per_q": m["per_q"]},
                           CKPT_DIR / "best.pt")
                log(f"  [ckpt] new best overall alive_F1 -> best.pt")

            # Gate is NON-TERMINAL: it marks a milestone (save gate_reached.pt
            # once + log) but training continues to target_steps so we can keep
            # refining past it. Resume with --resume to extend further.
            if ov["alive_f1"] >= ALIVE_F1_GATE and ov["dead_f1"] >= DEAD_F1_GATE:
                gate_streak += 1
                log(f"  [gate] met {gate_streak}/{GATE_CONSECUTIVE}")
                if gate_streak >= GATE_CONSECUTIVE and not gate_reached:
                    gate_reached = True
                    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
                                "step": step, "metrics": ov, "per_q": m["per_q"]},
                               CKPT_DIR / "gate_reached.pt")
                    log(f"\n  [gate] CLEARED at step {step} (alive_F1={ov['alive_f1']:.4f}, "
                        f"dead_F1={ov['dead_f1']:.4f}) -> gate_reached.pt. Training continues.\n")
            else:
                gate_streak = 0

        if step % CKPT_EVERY == 0:
            # Lightweight model-only milestone (browsable history) ...
            torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "step": step},
                       CKPT_DIR / f"step{step:06d}.pt")
            # ... plus full resumable state (overwrites — single rolling file).
            save_full_state(LATEST_PATH, encoder, decoder, opt, sched, step,
                            best_alive_f1, gate_streak, gate_reached, np_rng)
            log(f"  [ckpt] periodic snapshot -> step{step:06d}.pt  (+ latest.pt resumable state)")

    # Final resumable checkpoint so a completed run can still be extended later.
    save_full_state(LATEST_PATH, encoder, decoder, opt, sched, step,
                    best_alive_f1, gate_streak, gate_reached, np_rng)
    prefetcher.stop()
    log(f"\n=== done at step {step}. best overall alive_F1={best_alive_f1:.4f}. "
        f"gate_reached={gate_reached}. total elapsed={time.perf_counter() - t_start:.1f}s ===")
    log(f"    (resume/extend with:  python3 train.py --resume --steps <N>)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="GoL autoencoder training")
    p.add_argument("--resume", action="store_true",
                   help="continue from checkpoints/latest.pt")
    p.add_argument("--steps", type=int, default=TRAIN_STEPS,
                   help=f"target total step count (default {TRAIN_STEPS}; "
                        f"set higher than a completed run to extend it)")
    p.add_argument("--lr", type=float, default=None,
                   help="warm-restart peak LR for --resume (fresh cosine → LR_MIN "
                        "over the extension). Without it the loaded schedule "
                        "continues (near LR_MIN if the run had finished).")
    args = p.parse_args()
    main(resume=args.resume, target_steps=args.steps, lr_restart=args.lr)
