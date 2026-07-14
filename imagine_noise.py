"""
imagine_noise.py -- does imagination help under NOISE (not just occlusion)?
==========================================================================
Generalizes the robustness story. Occlusion has a clean "visible vs missing"
structure; noise corrupts every pixel, so there is no mask to lean on -- the brain
must DENOISE by analysis-by-synthesis: reconstruct a clean-looking digit, partially
trust it, re-perceive, repeat.

Train recognize+imagine brain on clean MNIST, then sweep Gaussian-noise level. At
each level compare:
  noisy accuracy  (feedforward)
  imagined accuracy (iterative denoise: x <- beta*noisy + (1-beta)*reconstruction)
If imagined > noisy across levels, imagination cleans up degraded input in general,
not only when it can copy visible pixels.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from imagine_helps import RecallBrain, DEVICE


def train_model(seed, args, train_loader):
    """Denoising training: corrupt the input with random Gaussian noise, but classify
    the true label and reconstruct the CLEAN image -> the reconstruction learns to denoise."""
    torch.manual_seed(seed)
    model = RecallBrain(args.grid, args.grid, args.t_steps, movable=False).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    ce = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            std = torch.rand(1).item() * args.train_noise          # random noise per batch
            x_in = (x + std * torch.randn_like(x)).clamp(0, 1)
            A = model.encode(x_in)
            logits, recon = model.cls(A), model.dec(A)
            loss = ce(logits, y) + args.recon_w * F.binary_cross_entropy_with_logits(
                recon, x, reduction="sum") / x.size(0)             # reconstruct CLEAN x
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    print(f"  seed {seed} trained (final loss {loss.item():.1f})")
    return model


@torch.no_grad()
def evaluate_noise(model, loader, std, rounds, beta=0.5):
    model.eval()
    n = noisy_c = imag_c = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE); n += y.numel()
        x_noisy = (x + std * torch.randn_like(x)).clamp(0, 1)
        noisy_c += (model(x_noisy)[0].argmax(1) == y).sum().item()
        x_t = x_noisy.clone()
        for _ in range(rounds):                                  # iterative denoise
            _, recon = model(x_t)
            x_t = beta * x_noisy + (1 - beta) * torch.sigmoid(recon)
        imag_c += (model(x_t)[0].argmax(1) == y).sum().item()
    return noisy_c / n, imag_c / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--t-steps", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--recon-w", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--train-noise", type=float, default=0.5)   # max noise std during training
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    stds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    if args.quick:
        args.n_train, args.n_test, args.epochs, args.t_steps, args.seeds = 3000, 1000, 2, 12, 1
        stds = [0.0, 0.25, 0.5]
    print(f"Device {DEVICE}  Gaussian-noise sweep {stds}  seeds {args.seeds}")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    tr_full = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    te_full = datasets.MNIST("./data", train=False, transform=tfm)
    g = torch.Generator().manual_seed(0)
    tr = Subset(tr_full, torch.randperm(len(tr_full), generator=g)[:args.n_train].tolist())
    te = Subset(te_full, torch.randperm(len(te_full), generator=g)[:args.n_test].tolist())
    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(te, batch_size=256)

    noisy = np.zeros((args.seeds, len(stds)))
    imag = np.zeros((args.seeds, len(stds)))
    for s in range(args.seeds):
        torch.manual_seed(100 + s)                               # fix noise draw per seed
        model = train_model(s, args, train_loader)
        for si, std in enumerate(stds):
            noisy[s, si], imag[s, si] = evaluate_noise(model, test_loader, std, args.rounds, args.beta)

    nm, ns = noisy.mean(0), noisy.std(0)
    im, isd = imag.mean(0), imag.std(0)
    gain = imag - noisy
    gm, gs = gain.mean(0), gain.std(0)

    print(f"\n{'noise std':>9} {'noisy':>14} {'imagined':>14} {'gain':>14}")
    for si, std in enumerate(stds):
        print(f"{std:>9} {nm[si]:.3f}+/-{ns[si]:.3f} {im[si]:.3f}+/-{isd[si]:.3f} "
              f"{gm[si]:+.3f}+/-{gs[si]:.3f}")
    best = int(np.argmax(gm))
    print(f"\nImagination helps most at noise std {stds[best]} (gain {gm[best]:+.3f}).")
    helps = "HELPS under noise" if gm[best] > 0.02 else "no clear noise benefit (denoising too weak)"
    print(f"Verdict: {helps}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].errorbar(stds, nm, yerr=ns, fmt="s-", color="gray", label="noisy (feedforward)")
    ax[0].errorbar(stds, im, yerr=isd, fmt="o-", color="#378ADD", label="+ imagination (denoise)")
    ax[0].set_xlabel("Gaussian noise std"); ax[0].set_ylabel("digit accuracy")
    ax[0].set_ylim(0, 1); ax[0].set_title("Recognition vs noise"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].errorbar(stds, gm, yerr=gs, fmt="o-", color="#D85A30")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("Gaussian noise std"); ax[1].set_ylabel("imagination gain (imagined - noisy)")
    ax[1].set_title("Where imagination helps (noise)"); ax[1].grid(alpha=0.3)
    fig.suptitle("Does imagination denoise? (generalizing robustness beyond occlusion)")
    plt.tight_layout(); plt.savefig("imagine_noise.png", dpi=120)
    print("Saved imagine_noise.png")


if __name__ == "__main__":
    main()
