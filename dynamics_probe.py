"""
Decisive readiness tests (per advisor): sharpness can't tell good latents from
garbage (noise decodes sharp too), so we use the metrics that CAN.

  1. CYCLE-CONSISTENCY: latent -> decode -> binarize -> re-encode -> z'.
     On-manifold latents round-trip stably (z'~=z); confident-garbage drifts.
     This is the same quantity that governs multi-step rollout drift.
     Tested on: real, slerp-interpolated, perturbed, prior-sampled latents.

  2. LATENT-DYNAMICS ROLLOUT (the definitive test): train a small z_t->z_{t+1}
     predictor, then closed-loop roll it out on HELD-OUT trajectories and
     compare decoded frames to the true simulated frames over K steps. Separates
     dynamics error from the AE ceiling (teacher-forced) at each horizon.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

import data, engine
from model import Encoder, Decoder

CKPT = Path("checkpoints/best.pt")
RNG = np.random.default_rng(0); torch.manual_seed(0)


def f1(pred, true):
    tp=int((pred&true).sum()); fp=int((pred&~true).sum()); fn=int((~pred&true).sum())
    d=2*tp+fp+fn; return (2*tp/d) if d else 0.0

def slerp(a, b, t):
    an=a/a.norm(dim=-1,keepdim=True); bn=b/b.norm(dim=-1,keepdim=True)
    dot=(an*bn).sum(-1,keepdim=True).clamp(-1,1); om=torch.acos(dot); so=torch.sin(om)
    w=(torch.sin((1-t)*om)/so)*a + (torch.sin(t*om)/so)*b
    return torch.where(so<1e-6, (1-t)*a+t*b, w)

def encode_all(enc, frames, bs=64):
    zs=[]
    with torch.no_grad():
        for i in range(0,len(frames),bs):
            x=torch.from_numpy(frames[i:i+bs].astype(np.float32)).unsqueeze(1)
            zs.append(enc(x))
    return torch.cat(zs)


def main():
    torch.set_num_threads(8)
    ckpt=torch.load(CKPT,map_location="cpu",weights_only=False)
    print(f"Checkpoint step={ckpt['step']} F1={ckpt['metrics']['alive_f1']:.4f}\n")
    enc,dec=Encoder(),Decoder(kernel_size=1)
    enc.load_state_dict(ckpt["encoder"]); enc.eval()
    dec.load_state_dict(ckpt["decoder"]); dec.eval()
    pool=data.TrainingSeedPool()

    # gather real frames + latents
    frames=[]
    for q in range(4):
        idx=RNG.choice(pool.quartile_pools[q],size=40,replace=False)
        seeds=np.asarray(pool.seeds[idx]); offs=engine.sample_center_biased_offsets(40,RNG)
        trajs=engine.simulate(seeds,k=engine.K_DEFAULT,offsets=offs); lifes=pool.lifespans[idx]
        for i in range(40):
            frames.append(trajs[i,int(RNG.integers(0,int(lifes[i])+1))])
    frames=np.stack(frames); N=len(frames)
    z=encode_all(enc,frames)

    def cycle(zin):
        with torch.no_grad():
            p=torch.sigmoid(dec(zin)).squeeze(1).numpy()
        b=(p>0.5)
        z2=encode_all(enc,b.astype(np.uint8))
        drift=((z2-zin).norm(dim=-1)/zin.norm(dim=-1).clamp_min(1e-6)).numpy()
        zc=torch.nn.functional.cosine_similarity(z2,zin,dim=-1).numpy()
        n_alive=b.reshape(len(b),-1).sum(1)
        return drift, zc, n_alive

    print("="*66); print("1. CYCLE-CONSISTENCY  (decode->binarize->re-encode drift)"); print("="*66)
    print(f"  {'latent source':>16} {'drift ||z2-z||/||z||':>20} {'cos(z,z2)':>10} {'n_alive':>8}")
    d,c,na=cycle(z);           print(f"  {'real':>16} {d.mean():>20.3f} {c.mean():>10.3f} {na.mean():>8.1f}")
    ia,ib=RNG.integers(0,N,60),RNG.integers(0,N,60)
    zint=slerp(z[ia],z[ib],0.5)
    d,c,na=cycle(zint);        print(f"  {'slerp-mid':>16} {d.mean():>20.3f} {c.mean():>10.3f} {na.mean():>8.1f}")
    noise=torch.randn_like(z); noise=noise/noise.norm(dim=-1,keepdim=True)*z.norm(dim=-1,keepdim=True)
    d,c,na=cycle(z+0.2*noise); print(f"  {'perturb eps.2':>16} {d.mean():>20.3f} {c.mean():>10.3f} {na.mean():>8.1f}")
    samp=torch.randn(N,z.shape[1]); samp=samp/samp.norm(dim=-1,keepdim=True)*float(z.norm(dim=-1).mean())
    d,c,na=cycle(samp);        print(f"  {'prior-sample':>16} {d.mean():>20.3f} {c.mean():>10.3f} {na.mean():>8.1f}")
    print("  -> low drift / high cos = latent decodes to a frame the encoder")
    print("     maps back to the SAME place = genuinely on-manifold (real frame).")

    # ================= 2. dynamics rollout =================
    print("\n"+"="*66); print("2. LATENT-DYNAMICS ROLLOUT  (train z_t->z_{t+1}, closed-loop)"); print("="*66)
    K=60
    idx=RNG.choice(pool.quartile_pools[3],size=100,replace=False)
    seeds=np.asarray(pool.seeds[idx]); offs=engine.sample_center_biased_offsets(100,RNG)
    trajs=engine.simulate(seeds,k=K,offsets=offs)          # (100,K+1,128,128)
    Z=encode_all(enc,trajs.reshape(-1,128,128)).reshape(100,K+1,-1)  # (100,K+1,1024)
    tr,te=Z[:80],Z[80:]
    Xtr=tr[:,:-1].reshape(-1,1024); Ytr=tr[:,1:].reshape(-1,1024)

    net=nn.Sequential(nn.Linear(1024,2048),nn.GELU(),nn.Linear(2048,1024))
    opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    for it in range(3000):
        j=torch.randint(0,len(Xtr),(256,))
        pred=Xtr[j]+net(Xtr[j])                    # residual: predict delta
        loss=((pred-Ytr[j])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"  predictor trained (final MSE={loss.item():.4f})\n")

    # closed-loop rollout on held-out trajectories, compare decoded to TRUE frame
    net.eval()
    horizons=[1,2,5,10,20,40,60]
    tf_f1={h:[] for h in horizons}; rl_f1={h:[] for h in horizons}
    with torch.no_grad():
        for ti in range(80,100):
            true=trajs[ti]                          # (K+1,128,128)
            zc=Z[ti,0:1].clone()                    # start from encoded true frame 0
            preds={}
            for step in range(1,K+1):
                zc=zc+net(zc)
                preds[step]=zc.clone()
            for h in horizons:
                # rollout (dynamics) decoded frame at horizon h
                pr=(torch.sigmoid(dec(preds[h])).squeeze().numpy()>0.5)
                rl_f1[h].append(f1(pr, true[h]==1))
                # teacher-forced = AE ceiling: encode true frame h, decode
                tf=(torch.sigmoid(dec(Z[ti,h:h+1])).squeeze().numpy()>0.5)
                tf_f1[h].append(f1(tf, true[h]==1))
    print(f"  {'horizon':>8} {'rollout_F1':>11} {'AE_ceiling_F1':>14} {'gap(dynamics err)':>18}")
    for h in horizons:
        r,t=np.mean(rl_f1[h]),np.mean(tf_f1[h])
        print(f"  {h:>8} {r:>11.3f} {t:>14.3f} {t-r:>18.3f}")
    print("  -> rollout_F1 tracking AE_ceiling = dynamics learned; large gap that")
    print("     grows with horizon = error compounding / off-manifold drift.")


if __name__=="__main__":
    main()
