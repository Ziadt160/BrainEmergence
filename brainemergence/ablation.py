"""
ablation.py -- which part of the recognize+imagine pipeline is the worst link?
==============================================================================
Decompose accuracy on a held-out missing-data corruption into the contributions of the two
parts you can actually attack: the GENERATOR (the fill) and the CLASSIFIER/substrate.

We cross two classifiers x four fills:
  classifiers : ours (spiking RecallBrain, broad-aug)  |  strong (a plain CNN, broad-aug)
  fills       : raw corrupted | mean-fill (trivial) | our generative fill | clean (= ORACLE fill)

The gaps tell you where the loss is:
  GENERATOR weakness    = clean_acc - ourfill_acc   (how much a PERFECT fill would add over ours)
  CLASSIFIER weakness   = strong_acc - ours_acc      (how much a better classifier would add)
  completion value      = ourfill_acc - raw_acc      (does our completion help at all?)

Whichever gap is biggest is the part to attack. (Oracle fill = the true clean image, since filling
the hole with the true pixels reconstructs the clean input -- it is the ceiling of completion.)
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


class SimpleCNN(nn.Module):
    """Strong, plain discriminative reference (clean MNIST ~0.99)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x.view(-1, 1, 28, 28))


def train_cnn(seed, heldout, args, loader):
    torch.manual_seed(seed + 4242)
    model = SimpleCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    aug = [k for k in CORR if k != heldout]
    for ep in range(args.epochs_cnn):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            k = aug[torch.randint(len(aug), (1,)).item()]
            x_in = apply_corruption(x, k)[0]
            loss = ce(model(x_in), y)
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
def eval_grid(ours, cnn, loader, heldout, n):
    ours.eval(); cnn.eval()
    fills = ["raw", "mean", "genfill", "clean"]
    acc = {c: {f: 0 for f in fills} for c in ["ours", "strong"]}
    tot = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE); tot += y.numel()
        x_c, mask = apply_corruption(x, heldout)
        m = torch.ones_like(x_c) if mask is None else mask
        obs_mean = (x_c * m).sum(1, keepdim=True) / (m.sum(1, keepdim=True) + 1e-6)
        x_mean = m * x_c + (1 - m) * obs_mean
        x_gen = complete_image(ours, x_c, mask, n)
        inputs = {"raw": x_c, "mean": x_mean, "genfill": x_gen, "clean": x}
        for f, xf in inputs.items():
            acc["ours"][f] += (ours(xf)[0].argmax(1) == y).sum().item()
            acc["strong"][f] += (cnn(xf).argmax(1) == y).sum().item()
    for c in acc:
        for f in acc[c]:
            acc[c][f] /= tot
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--t-steps", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--epochs-cnn", type=int, default=5)
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
    print(f"Device {DEVICE}  ablation: which part is worst? ({args.dataset}, held-out={args.heldout})")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    DS = datasets.FashionMNIST if args.dataset == "fashion" else datasets.MNIST
    trf = DS("./data", train=True, download=True, transform=tfm)
    tef = DS("./data", train=False, transform=tfm)
    g = torch.Generator().manual_seed(0)
    tr = Subset(trf, torch.randperm(len(trf), generator=g)[:args.n_train].tolist())
    te = Subset(tef, torch.randperm(len(tef), generator=g)[:args.n_test].tolist())
    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(te, batch_size=256)

    accs = []
    for s in range(args.seeds):
        ours = train_broadaug(s, args.heldout, args, train_loader)
        cnn = train_cnn(s, args.heldout, args, train_loader)
        accs.append(eval_grid(ours, cnn, test_loader, args.heldout, args.rounds))
        print(f"  seed {s} done")

    def m(c, f): return np.mean([a[c][f] for a in accs]), np.std([a[c][f] for a in accs])
    fills = ["raw", "mean", "genfill", "clean"]
    print(f"\n=== accuracy grid ({args.dataset}, held-out={args.heldout}, {args.seeds} seeds) ===")
    print(f"{'fill':>10} {'ours (spiking)':>18} {'strong (CNN)':>18}")
    for f in fills:
        print(f"{f:>10} {m('ours',f)[0]:.3f}+/-{m('ours',f)[1]:.3f}   {m('strong',f)[0]:.3f}+/-{m('strong',f)[1]:.3f}")

    gen_gap = m("ours", "clean")[0] - m("ours", "genfill")[0]      # perfect fill - our fill (our clf)
    clf_gap_clean = m("strong", "clean")[0] - m("ours", "clean")[0]  # better clf at the ceiling
    clf_gap_corr = m("strong", "genfill")[0] - m("ours", "genfill")[0]  # better clf on our fill
    completion_value = m("ours", "genfill")[0] - m("ours", "raw")[0]  # does completion help
    genfill_vs_mean = m("ours", "genfill")[0] - m("ours", "mean")[0]  # generative vs trivial fill

    print("\n=== CONTRIBUTION DECOMPOSITION (bigger gap = worse / more to gain) ===")
    print(f"  GENERATOR weakness   (clean - our genfill, our clf)   : {gen_gap:+.3f}")
    print(f"  CLASSIFIER weakness  (strong - ours, on clean)         : {clf_gap_clean:+.3f}")
    print(f"  CLASSIFIER weakness  (strong - ours, on our fill)      : {clf_gap_corr:+.3f}")
    print(f"  completion value     (our genfill - raw corrupted)     : {completion_value:+.3f}")
    print(f"  generative vs trivial(our genfill - mean fill)         : {genfill_vs_mean:+.3f}")
    ranking = sorted([("GENERATOR (the fill)", gen_gap),
                      ("CLASSIFIER/substrate", max(clf_gap_clean, clf_gap_corr))],
                     key=lambda t: -t[1])
    print(f"\n  >>> WORST LINK TO ATTACK: {ranking[0][0]}  (recoverable gap {ranking[0][1]:+.3f})")
    print(f"      runner-up: {ranking[1][0]} ({ranking[1][1]:+.3f})")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(fills)); w = 0.38
    ax.bar(x - w/2, [m("ours", f)[0] for f in fills], w, yerr=[m("ours", f)[1] for f in fills],
           capsize=3, color="#378ADD", label="ours (spiking)")
    ax.bar(x + w/2, [m("strong", f)[0] for f in fills], w, yerr=[m("strong", f)[1] for f in fills],
           capsize=3, color="#D85A30", label="strong (CNN)")
    ax.set_xticks(x); ax.set_xticklabels(["raw", "mean-fill", "our gen-fill", "clean (oracle)"])
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy")
    ax.set_title(f"Which part is the worst link? ({args.dataset}, held-out {args.heldout})")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"ablation_{args.dataset}_{args.heldout}.png", dpi=120)
    print(f"Saved ablation_{args.dataset}_{args.heldout}.png")


if __name__ == "__main__":
    main()
