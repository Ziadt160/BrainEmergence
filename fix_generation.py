"""
fix_generation.py  --  make the imagination OBEY before re-testing alignment
============================================================================
The alignment experiment was BLOCKED because generate(orientation=theta) did not
reliably draw a theta-grating (gen_valid ~ 0). Likely cause: the encoder leaks
orientation into the style latent z (it sees the image), so the decoder ignores
the orientation COMMAND; at generation time a random z then yields a random angle.

This script sweeps fixes and reports gen_valid (circular corr between requested and
produced orientation; ~1 = obeys, ~0 = ignores). No alignment / no zones -- just:
does the spiking recurrent generator respect the orientation it is told?

Fixes swept:
  baseline      encoder sees only the image  (reproduces the failure)
  cond_enc      encoder ALSO sees the orientation -> z carries only phase  (main fix)
  cond+gain0.5  cond_enc + weaker recurrence on the down-pass (less wash-out)
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
from exp_alignment import grating, grating_batch, concept, circ_corr, mexican_hat, gen_validity, DEVICE


class CondBiSheet(nn.Module):
    def __init__(self, H, W, P, latent=8, t_steps=14, cond_enc=True, gen_gain=1.0):
        super().__init__()
        self.H, self.W, self.N, self.T, self.P, self.latent = H, W, H * W, t_steps, P, latent
        self.cond_enc, self.gen_gain = cond_enc, gen_gain
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        pos = torch.stack([yy.reshape(-1), xx.reshape(-1)], 1).float()
        self.W_rec = nn.Parameter(mexican_hat(pos))
        self.tau_raw = nn.Parameter(torch.ones(self.N) * 2.0)
        self.thr_raw = nn.Parameter(torch.zeros(self.N))
        self.w_in = nn.Linear(P * P, self.N)
        self.enc = nn.Linear(self.N + (2 if cond_enc else 0), 2 * latent)
        self.dec = nn.Linear(latent + 2, self.N)
        self.w_img = nn.Sequential(nn.Linear(self.N, 256), nn.ReLU(), nn.Linear(256, P * P))

    def _run(self, drive, gain):
        B = drive.shape[0]
        tau, thr = torch.sigmoid(self.tau_raw), F.softplus(self.thr_raw) + 0.5
        mem = torch.zeros(B, self.N, device=drive.device)
        spk = torch.zeros(B, self.N, device=drive.device)
        acc = torch.zeros(B, self.N, device=drive.device)
        for _ in range(self.T):
            mem = mem * tau + (drive + gain * (spk @ self.W_rec.t())) - spk * thr
            spk = spike_func(mem - thr)
            acc = acc + spk
        return acc / self.T

    def up(self, x):
        return self._run(self.w_in(x), 1.0)

    def encode(self, A, c):
        h = torch.cat([A, c], 1) if self.cond_enc else A
        return self.enc(h).chunk(2, 1)

    def generate(self, z, c):
        A = self._run(self.dec(torch.cat([z, c], 1)), self.gen_gain)
        return self.w_img(A), A

    def forward(self, x, c):
        A = self.up(x)
        mu, logvar = self.encode(A, c)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.generate(z, c)[0], mu, logvar


def train_eval(cfg, seed, args):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = CondBiSheet(args.grid, args.grid, args.P, args.latent, args.t_steps,
                        cond_enc=cfg["cond_enc"], gen_gain=cfg["gen_gain"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for it in range(args.iters):
        x, th = grating_batch(args.batch, args.P, args.freq, gen)
        c = concept(th)
        logits, mu, logvar = model(x, c)
        recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.size(0)
        kl = torch.clamp((-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(0),
                         min=args.free_bits).sum()
        loss = recon + args.beta * kl
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if it % max(1, args.iters // 3) == 0:
            print(f"    [{cfg['name']} s{seed}] iter {it}/{args.iters} loss {loss.item():.1f}")
    gv = gen_validity(model, args.K, args.reps, args.P, args.freq)
    print(f"  [{cfg['name']} s{seed}] gen_valid = {gv:+.3f}")
    return gv, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--grid", type=int, default=28)
    ap.add_argument("--P", type=int, default=24)
    ap.add_argument("--freq", type=float, default=3.0)
    ap.add_argument("--latent", type=int, default=8)
    ap.add_argument("--t-steps", type=int, default=14)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--free-bits", type=float, default=0.2)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--reps", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    if args.quick:
        args.grid, args.P, args.iters, args.seeds, args.reps, args.t_steps = 14, 14, 150, 1, 12, 10
    print(f"Device: {DEVICE}  grid={args.grid}  iters={args.iters}  seeds={args.seeds}")

    configs = [
        dict(name="baseline", cond_enc=False, gen_gain=1.0),
        dict(name="cond_enc", cond_enc=True, gen_gain=1.0),
        dict(name="cond+gain0.5", cond_enc=True, gen_gain=0.5),
    ]
    results, best = {}, (None, -2, None)
    for cfg in configs:
        gvs = []
        for s in range(args.seeds):
            gv, model = train_eval(cfg, s, args)
            gvs.append(gv)
            if gv > best[1]:
                best = (cfg["name"], gv, model)
        results[cfg["name"]] = gvs

    print("\n=== gen_valid by fix (1=imagination obeys orientation, 0=ignores) ===")
    for name, gvs in results.items():
        print(f"  {name:>14}: mean {np.mean(gvs):+.3f}   per-seed {[round(g,2) for g in gvs]}")
    win_name, win_gv, win_model = best
    verdict = ("FIXED" if win_gv > 0.5 else "IMPROVED but not yet >0.5" if win_gv > 0.2 else "STILL BROKEN")
    print(f"\nBest: {win_name} (gen_valid={win_gv:+.3f}) -> {verdict}")
    if win_gv > 0.5:
        print("Generation now obeys orientation -> the alignment test is unblocked; "
              "port this fix into exp_alignment.py and re-run E1/E2a/E2b.")

    # show what the winner imagines at known orientations
    with torch.no_grad():
        ths = torch.linspace(0, np.pi, 8 + 1)[:-1]
        fig, axes = plt.subplots(1, 8, figsize=(14, 2.2))
        for j, th in enumerate(ths):
            z = torch.randn(1, args.latent, device=DEVICE)
            c = concept(torch.full((1,), float(th), device=DEVICE))
            img = torch.sigmoid(win_model.generate(z, c)[0]).cpu().reshape(args.P, args.P)
            axes[j].imshow(img, cmap="gray"); axes[j].axis("off")
            axes[j].set_title(f"{int(np.degrees(th))}°", fontsize=9)
        fig.suptitle(f"What '{win_name}' imagines when told each orientation (gen_valid={win_gv:+.2f})")
        plt.tight_layout(); plt.savefig("fix_generation_samples.png", dpi=120)
    print("Saved fix_generation_samples.png")


if __name__ == "__main__":
    main()
