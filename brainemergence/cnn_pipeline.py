"""
cnn_pipeline.py -- attack the worst link: swap the spiking encoder for a CNN.
============================================================================
The ablation said the worst link is the spiking classifier/substrate (~0.18 below a CNN), with
the generator second. This script builds the CNN version of the SAME recognize+imagine pipeline
(conv encoder + classifier head + deconv decoder, so it can still do completion) and runs it
head-to-head against the spiking RecallBrain on held-out missing-data.

We answer two things with numbers:
  1. How much accuracy does a CNN base RECOVER over the spiking base (raw / mean-fill / gen-fill / clean)?
  2. Does completion STILL help once the classifier is strong (gen-fill vs raw, gen-fill vs mean)?
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
from paper_exp1 import CORR, apply_corruption, train_broadaug


class ConvRI(nn.Module):
    """CNN recognize+imagine for 1x28x28: conv encoder -> (cls head, deconv decoder)."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),    # 14
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),   # 7
        )
        self.cls = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),   # 14
            nn.ConvTranspose2d(32, 1, 4, 2, 1),                                   # 28
        )

    def forward(self, x):
        f = self.enc(x.view(-1, 1, 28, 28))
        return self.cls(f), self.dec(f).view(-1, 784)


def train_convri(seed, heldout, args, loader):
    torch.manual_seed(seed + 7)
    model = ConvRI().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    aug = [k for k in CORR if k != heldout]
    for ep in range(args.epochs_cnn):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            k = aug[torch.randint(len(aug), (1,)).item()]
            x_in = apply_corruption(x, k)[0]
            logits, recon = model(x_in)
            loss = ce(logits, y) + args.recon_w * F.binary_cross_entropy_with_logits(recon, x)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def complete_image(model, x_c, mask, n, beta=0.5):
    x_t = x_c.clone()
    for _ in range(n):
        _, recon = model(x_t)
        x_t = (mask * x_c + (1 - mask) * torch.sigmoid(recon)) if mask is not None \
            else (beta * x_c + (1 - beta) * torch.sigmoid(recon))
    return x_t


@torch.no_grad()
def eval_pipeline(model, loader, heldout, n):
    model.eval()
    fills = ["raw", "mean", "genfill", "clean"]; acc = {f: 0 for f in fills}; tot = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE); tot += y.numel()
        x_c, mask = apply_corruption(x, heldout)
        m = torch.ones_like(x_c) if mask is None else mask
        obs_mean = (x_c * m).sum(1, keepdim=True) / (m.sum(1, keepdim=True) + 1e-6)
        x_mean = m * x_c + (1 - m) * obs_mean
        x_gen = complete_image(model, x_c, mask, n)
        for f, xf in {"raw": x_c, "mean": x_mean, "genfill": x_gen, "clean": x}.items():
            acc[f] += (model(xf)[0].argmax(1) == y).sum().item()
    return {f: acc[f] / tot for f in fills}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--t-steps", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=6)         # spiking
    ap.add_argument("--epochs-cnn", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--recon-w", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--heldout", default="band")
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.n_test, args.epochs, args.epochs_cnn, args.t_steps, args.seeds = 3000, 1000, 2, 2, 12, 1
    print(f"Device {DEVICE}  CNN-vs-spiking recognize+imagine ({args.dataset}, held-out={args.heldout})")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    DS = datasets.FashionMNIST if args.dataset == "fashion" else datasets.MNIST
    trf = DS("./data", train=True, download=True, transform=tfm)
    tef = DS("./data", train=False, transform=tfm)
    g = torch.Generator().manual_seed(0)
    tr = Subset(trf, torch.randperm(len(trf), generator=g)[:args.n_train].tolist())
    te = Subset(tef, torch.randperm(len(tef), generator=g)[:args.n_test].tolist())
    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(te, batch_size=256)

    R = {"spiking": [], "cnn": []}
    for s in range(args.seeds):
        sp = train_broadaug(s, args.heldout, args, train_loader)
        cn = train_convri(s, args.heldout, args, train_loader)
        R["spiking"].append(eval_pipeline(sp, test_loader, args.heldout, args.rounds))
        R["cnn"].append(eval_pipeline(cn, test_loader, args.heldout, args.rounds))
        print(f"  seed {s} done")

    def m(model, f): return np.mean([r[f] for r in R[model]]), np.std([r[f] for r in R[model]])
    fills = ["raw", "mean", "genfill", "clean"]
    print(f"\n=== recognize+imagine: spiking vs CNN base ({args.dataset}, held-out={args.heldout}, {args.seeds} seeds) ===")
    print(f"{'fill':>10} {'spiking':>16} {'CNN':>16} {'recovery (CNN-spk)':>20}")
    for f in fills:
        sp, cn = m("spiking", f), m("cnn", f)
        print(f"{f:>10} {sp[0]:.3f}+/-{sp[1]:.3f}   {cn[0]:.3f}+/-{cn[1]:.3f}   {cn[0]-sp[0]:>+20.3f}")

    sp_compl = m("spiking", "genfill")[0] - m("spiking", "raw")[0]
    cn_compl = m("cnn", "genfill")[0] - m("cnn", "raw")[0]
    cn_compl_vs_mean = m("cnn", "genfill")[0] - m("cnn", "mean")[0]
    recov_ff = m("cnn", "raw")[0] - m("spiking", "raw")[0]
    print("\n=== READS ===")
    print(f"  CNN recovery on raw corrupted (base-classifier fix) : {recov_ff:+.3f}")
    print(f"  completion gain, spiking (genfill - raw)            : {sp_compl:+.3f}")
    print(f"  completion gain, CNN     (genfill - raw)            : {cn_compl:+.3f}")
    print(f"  CNN: generative fill vs trivial mean-fill           : {cn_compl_vs_mean:+.3f}")
    print(f"  best spiking pipeline (genfill) {m('spiking','genfill')[0]:.3f}  vs  "
          f"best CNN pipeline (genfill) {m('cnn','genfill')[0]:.3f}")
    print("  => " + ("CNN base recovers a large gap; completion still helps on top."
                     if recov_ff > 0.05 else "CNN base ~ spiking here."))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(fills)); w = 0.38
    ax.bar(x - w/2, [m("spiking", f)[0] for f in fills], w, yerr=[m("spiking", f)[1] for f in fills],
           capsize=3, color="#378ADD", label="spiking RecallBrain")
    ax.bar(x + w/2, [m("cnn", f)[0] for f in fills], w, yerr=[m("cnn", f)[1] for f in fills],
           capsize=3, color="#1e8449", label="CNN recognize+imagine")
    ax.set_xticks(x); ax.set_xticklabels(["raw", "mean-fill", "gen-fill", "clean"])
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy")
    ax.set_title(f"Attack the worst link: CNN vs spiking base ({args.dataset}, {args.heldout})")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"cnn_pipeline_{args.dataset}_{args.heldout}.png", dpi=120)
    print(f"Saved cnn_pipeline_{args.dataset}_{args.heldout}.png")


if __name__ == "__main__":
    main()
