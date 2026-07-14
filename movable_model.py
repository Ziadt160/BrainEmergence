"""
movable_model.py -- the COUPLED movable-neuron sheet (the user's idea, for real)
===============================================================================
Neurons have LEARNABLE 2D positions. Two things are coupled:
  1. position -> connectivity: W_eff = W_base * exp(-dist(pos)^2 / 2 sigma^2),
     so who-connects-to-whom depends on where neurons currently sit.
  2. co-firing -> position: an attraction loss pulls co-active neurons together
     (fire together -> move closer). Positions are kept unit-variance so they
     can't collapse to a point.

The loop (position -> connectivity -> activity -> co-firing -> position) is what
makes any resulting map EMERGENT rather than imposed. We then check whether the
learned layout shows a smooth orientation map vs a random layout.

Honest caveat: the attraction loss optimizes "co-active close", so SOME structure
is expected; the test is whether orientation varies smoothly (a real map) and beats
the random-position baseline by a clear margin.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mnist_experiment import spike_func
from exp_alignment import grating, grating_batch, concept, DEVICE
from movable_test import local_scatter, response_matrix


class MovableSheet(nn.Module):
    def __init__(self, N=400, P=20, latent=8, t_steps=14, sigma=0.35, gen_gain=0.5):
        super().__init__()
        self.N, self.T, self.P, self.latent = N, t_steps, P, latent
        self.sigma, self.gen_gain = sigma, gen_gain
        self.pos = nn.Parameter(torch.rand(N, 2))                  # LEARNABLE positions
        self.W_base = nn.Parameter(torch.randn(N, N) / np.sqrt(N))
        self.tau_raw = nn.Parameter(torch.ones(N) * 2.0)
        self.thr_raw = nn.Parameter(torch.zeros(N))
        self.w_in = nn.Linear(P * P, N)
        self.enc = nn.Linear(N + 2, 2 * latent)                    # conditional encoder
        self.dec = nn.Linear(latent + 2, N)
        self.w_img = nn.Sequential(nn.Linear(N, 256), nn.ReLU(), nn.Linear(256, P * P))

    def npos(self):                                                # unit-variance (anti-collapse)
        p = self.pos
        return (p - p.mean(0)) / (p.std(0) + 1e-6)

    def W_eff(self):
        p = self.npos()
        d2 = ((p.unsqueeze(1) - p.unsqueeze(0)) ** 2).sum(2)
        W = self.W_base * torch.exp(-d2 / (2 * self.sigma ** 2))
        return W - torch.diag_embed(torch.diagonal(W))

    def _run(self, drive, gain):
        B = drive.shape[0]; W = self.W_eff()
        tau, thr = torch.sigmoid(self.tau_raw), F.softplus(self.thr_raw) + 0.5
        mem = torch.zeros(B, self.N, device=drive.device)
        spk = torch.zeros_like(mem); acc = torch.zeros_like(mem)
        for _ in range(self.T):
            mem = mem * tau + (drive + gain * (spk @ W.t())) - spk * thr
            spk = spike_func(mem - thr); acc = acc + spk
        return acc / self.T

    def up(self, x):
        return self._run(self.w_in(x), 1.0)

    def encode(self, A, c):
        return self.enc(torch.cat([A, c], 1)).chunk(2, 1)

    def generate(self, z, c):
        A = self._run(self.dec(torch.cat([z, c], 1)), self.gen_gain)
        return self.w_img(A), A

    def attract(self, A):
        """Pull co-firing neurons close: sum_ij corr(i,j) * dist(i,j)^2 (the user's rule)."""
        Z = (A - A.mean(0)) / (A.std(0) + 1e-6)
        C = (Z.t() @ Z) / A.shape[0]                               # (N,N) correlation
        p = self.npos()
        d2 = ((p.unsqueeze(1) - p.unsqueeze(0)) ** 2).sum(2)
        Cp = C.clamp(min=0)
        return (Cp * d2).sum() / (Cp.sum() + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--N", type=int, default=400)
    ap.add_argument("--P", type=int, default=20)
    ap.add_argument("--freq", type=float, default=3.0)
    ap.add_argument("--t-steps", type=int, default=14)
    ap.add_argument("--iters", type=int, default=3500)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lam-attract", type=float, default=1.0)
    args = ap.parse_args()
    if args.quick:
        args.iters, args.N = 300, 200
    print(f"Device {DEVICE}  N={args.N}  iters={args.iters}  lam_attract={args.lam_attract}")

    torch.manual_seed(0)
    model = MovableSheet(args.N, args.P, t_steps=args.t_steps).to(DEVICE)
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for it in range(args.iters):
        x, th = grating_batch(args.batch, args.P, args.freq, gen)
        c = concept(th)
        A = model.up(x)
        mu, logvar = model.encode(A, c)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        logits, _ = model.generate(z, c)
        recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.size(0)
        kl = torch.clamp((-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(0), min=0.2).sum()
        att = model.attract(A)
        loss = recon + kl + args.lam_attract * att
        opt.zero_grad(); loss.backward(); opt.step()
        if it % max(1, args.iters // 5) == 0:
            print(f"  iter {it}/{args.iters}  loss {loss.item():.1f}  attract {att.item():.3f}")

    # ---- did a map emerge in the LEARNED positions? ----
    R, thetas = response_matrix(model, K=24, reps=16, P=args.P, freq=args.freq, gen=gen)
    ang = 2 * thetas
    vx = (R * np.cos(ang)).sum(1); vy = (R * np.sin(ang)).sum(1)
    pref = 0.5 * np.arctan2(vy, vx)
    sel = np.sqrt(vx ** 2 + vy ** 2) / (R.sum(1) + 1e-6)
    keep = sel > np.median(sel)
    learned = model.npos().detach().cpu().numpy()[keep]
    rng = np.random.default_rng(0)
    rand_pos = rng.standard_normal((int(keep.sum()), 2))

    s_learned = local_scatter(learned, pref[keep])
    s_rand = local_scatter(rand_pos, pref[keep])
    print(f"\nLocal orientation scatter (low = organized map):")
    print(f"  learned positions (emergent): {s_learned:.3f}")
    print(f"  random positions  (baseline): {s_rand:.3f}")
    verdict = ("MAP EMERGED" if s_learned < s_rand - 0.1 else
               "weak/partial" if s_learned < s_rand - 0.03 else "NO emergent map")
    print(f"  -> {verdict}")

    plt.figure(figsize=(6, 5.5))
    sc = plt.scatter(learned[:, 0], learned[:, 1], c=pref[keep], cmap="twilight",
                     vmin=-np.pi/2, vmax=np.pi/2, s=30)
    plt.colorbar(sc, label="preferred orientation")
    plt.title(f"Emergent layout of movable neurons\nlocal scatter {s_learned:.2f} (random {s_rand:.2f})")
    plt.axis("equal"); plt.axis("off"); plt.tight_layout()
    plt.savefig("movable_model.png", dpi=120)
    print("Saved movable_model.png")


if __name__ == "__main__":
    main()
