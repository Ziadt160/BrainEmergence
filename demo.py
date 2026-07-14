"""
demo.py -- a showable demonstration: the brain repairs occluded digits by imagining.
====================================================================================
Trains the recognize+imagine spiking brain on clean MNIST, then on occluded TEST
digits shows cases where the feedforward read is WRONG but, after the brain imagines
the missing band and re-perceives, the read becomes RIGHT.

Output: demo_imagination.png  (rows = examples; columns = original / occluded(pred) /
imagined(pred)). Green title = correct, red = wrong.
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

from imagine_helps import RecallBrain, occlude, DEVICE


def train(model, loader, epochs, recon_w):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            A = model.encode(x)
            logits, recon = model.cls(A), model.dec(A)
            loss = ce(logits, y) + recon_w * F.binary_cross_entropy_with_logits(
                recon, x, reduction="sum") / x.size(0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        print(f"  epoch {ep} loss {loss.item():.1f}")


@torch.no_grad()
def complete(model, x, band, rounds):
    x_occ, mask = occlude(x, band)
    x_t = x_occ.clone()
    pred0 = None
    for r in range(rounds + 1):
        logits, recon = model(x_t)
        if r == 0:
            pred0 = logits.argmax(1)
        x_t = mask * x_occ + (1 - mask) * torch.sigmoid(recon)
    predF = model(x_t)[0].argmax(1)
    return x_occ, x_t, pred0, predF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--band", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--n-show", type=int, default=6)
    args = ap.parse_args()
    torch.manual_seed(0)
    print(f"Device {DEVICE}  occlusion {args.band}px")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    tr = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    te = datasets.MNIST("./data", train=False, transform=tfm)
    g = torch.Generator().manual_seed(0)
    tr = Subset(tr, torch.randperm(len(tr), generator=g)[:12000].tolist())
    train_loader = DataLoader(tr, batch_size=128, shuffle=True)

    model = RecallBrain(24, 24, 16, movable=False).to(DEVICE)
    train(model, train_loader, args.epochs, recon_w=1.0)

    # gather a test batch and find "fixes": wrong when occluded, right after imagining
    xb = torch.stack([te[i][0] for i in range(400)]).to(DEVICE)
    yb = torch.tensor([te[i][1] for i in range(400)], device=DEVICE)
    x_occ, x_full, p0, pF = complete(model, xb, args.band, args.rounds)
    fixes = ((p0 != yb) & (pF == yb)).nonzero(as_tuple=True)[0]
    print(f"\nFound {len(fixes)} 'fix' cases in 400 test images "
          f"(occluded read wrong -> imagined read correct).")
    sel = fixes[: args.n_show].cpu().numpy()

    orig = xb.view(-1, 28, 28).cpu().numpy()
    occ = x_occ.view(-1, 28, 28).cpu().numpy()
    full = x_full.view(-1, 28, 28).cpu().numpy()
    p0, pF, yb = p0.cpu().numpy(), pF.cpu().numpy(), yb.cpu().numpy()

    rows = len(sel)
    fig, ax = plt.subplots(rows, 3, figsize=(5.2, 1.7 * rows))
    cols = ["original", "occluded input", "after imagination"]
    for r, i in enumerate(sel):
        for c, im in enumerate([orig[i], occ[i], full[i]]):
            a = ax[r, c]
            a.imshow(im, cmap="gray"); a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(cols[c], fontsize=10)
        ax[r, 1].set_ylabel(f"read: {p0[i]}", color="#c0392b", fontsize=11, rotation=0,
                            labelpad=22, va="center")
        ax[r, 2].set_ylabel(f"read: {pF[i]} ✓", color="#1e8449", fontsize=11, rotation=0,
                            labelpad=22, va="center")
    fig.suptitle(f"The brain imagines the missing band and re-reads the digit\n"
                 f"({args.band}px occlusion; red = wrong read, green = corrected)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("demo_imagination.png", dpi=120)
    print("Saved demo_imagination.png")


if __name__ == "__main__":
    main()
