"""
imagine_robustness.py -- WHEN does imagination help perception?
==============================================================
Thread #1 follow-up. Train one recognize+imagine brain on clean MNIST, then sweep
occlusion severity (0 -> 20 px). At each severity, measure:
  occluded accuracy   (feedforward, no imagination)
  imagined  accuracy  (after iterative top-down completion)
and the GAIN between them, averaged over seeds.

Hypothesis: imagination is worthless on clean input, helps most at moderate
corruption (lots to fill, enough context to fill it), and may fade when almost
everything is gone (nothing left to reconstruct from). The shape of that curve is
the robustness story.
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

from imagine_helps import RecallBrain, evaluate_completion, DEVICE


def train_model(seed, args, train_loader):
    torch.manual_seed(seed)
    model = RecallBrain(args.grid, args.grid, args.t_steps, movable=False).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    ce = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            A = model.encode(x)
            logits, recon = model.cls(A), model.dec(A)
            loss = ce(logits, y) + args.recon_w * F.binary_cross_entropy_with_logits(
                recon, x, reduction="sum") / x.size(0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    print(f"  seed {seed} trained (final loss {loss.item():.1f})")
    return model


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
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dataset", default="mnist")          # 'mnist' or 'fashion'
    args = ap.parse_args()
    bands = [0, 4, 8, 12, 16, 20]
    if args.quick:
        args.n_train, args.n_test, args.epochs, args.t_steps, args.seeds = 3000, 1000, 2, 12, 1
        bands = [0, 8, 16]
    print(f"Device {DEVICE}  grid {args.grid}  occlusion sweep {bands}px  seeds {args.seeds}")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    DS = datasets.FashionMNIST if args.dataset == "fashion" else datasets.MNIST
    tr_full = DS("./data", train=True, download=True, transform=tfm)
    te_full = DS("./data", train=False, transform=tfm)
    g = torch.Generator().manual_seed(0)
    tr = Subset(tr_full, torch.randperm(len(tr_full), generator=g)[:args.n_train].tolist())
    te = Subset(te_full, torch.randperm(len(te_full), generator=g)[:args.n_test].tolist())
    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(te, batch_size=256)

    occ = np.zeros((args.seeds, len(bands)))
    imag = np.zeros((args.seeds, len(bands)))
    clean = []
    for s in range(args.seeds):
        model = train_model(s, args, train_loader)
        for bi, band in enumerate(bands):
            accs, _, cl = evaluate_completion(model, test_loader, args.rounds, band)
            occ[s, bi], imag[s, bi] = accs[0], accs[-1]
        clean.append(cl)

    occ_m, occ_s = occ.mean(0), occ.std(0)
    imag_m, imag_s = imag.mean(0), imag.std(0)
    gain = imag - occ
    gain_m, gain_s = gain.mean(0), gain.std(0)
    clean_m = float(np.mean(clean))

    print(f"\nClean ceiling: {clean_m:.3f}")
    print(f"{'occl px':>8} {'occluded':>14} {'imagined':>14} {'gain':>14}")
    for bi, band in enumerate(bands):
        print(f"{band:>8} {occ_m[bi]:.3f}+/-{occ_s[bi]:.3f} {imag_m[bi]:.3f}+/-{imag_s[bi]:.3f} "
              f"{gain_m[bi]:+.3f}+/-{gain_s[bi]:.3f}")
    best = int(np.argmax(gain_m))
    print(f"\nImagination helps most at {bands[best]}px occlusion (gain {gain_m[best]:+.3f}).")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].axhline(clean_m, color="green", ls="--", alpha=0.6, label="clean ceiling")
    ax[0].errorbar(bands, occ_m, yerr=occ_s, fmt="s-", color="gray", label="occluded (feedforward)")
    ax[0].errorbar(bands, imag_m, yerr=imag_s, fmt="o-", color="#378ADD", label="+ imagination")
    ax[0].set_xlabel("occlusion size (px)"); ax[0].set_ylabel("digit accuracy")
    ax[0].set_ylim(0, 1); ax[0].set_title("Recognition vs corruption"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].errorbar(bands, gain_m, yerr=gain_s, fmt="o-", color="#D85A30")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("occlusion size (px)"); ax[1].set_ylabel("imagination gain (imagined - occluded)")
    ax[1].set_title("Where imagination helps"); ax[1].grid(alpha=0.3)
    fig.suptitle(f"When does imagination help perception? [{args.dataset}]")
    out = f"imagine_robustness_{args.dataset}.png"
    plt.tight_layout(); plt.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
