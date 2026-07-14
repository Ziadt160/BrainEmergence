"""
cifar_imagine.py -- does the redundancy LAW hold on natural images?
==================================================================
Our prediction (from MNIST vs Fashion): top-down completion helps MORE when the input
has more spatial redundancy. Natural images (CIFAR-10) are far more redundant than
digits, so the effect should be LARGER here.

Architecture note: we test the PRINCIPLE, not the spiking brain (which our experiments
showed was decoration). A convolutional recognize+imagine net is the right tool for
natural images. Same protocol: train on clean CIFAR, occlude a band at test, let the net
imagine the missing region and re-perceive, sweep occlusion size, measure the gain.
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),    # 16
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),   # 8
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(), # 4
        )
        self.cls = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, 10))
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),  # 8
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),   # 16
            nn.ConvTranspose2d(32, 3, 4, 2, 1),                                   # 32
        )

    def forward(self, x):
        f = self.enc(x)
        return self.cls(f), self.dec(f)        # logits, recon-logits


def occlude(x, band):
    """Zero a horizontal band of `band` rows; return occluded image + observed mask."""
    B, _, H, W = x.shape
    r0 = H // 2 - band // 2
    img = x.clone(); mask = torch.ones(B, 1, H, W, device=x.device)
    img[:, :, r0:r0 + band, :] = 0.0
    mask[:, :, r0:r0 + band, :] = 0.0
    return img, mask


@torch.no_grad()
def eval_completion(model, loader, band, rounds):
    model.eval()
    n = occ_c = imag_c = clean_c = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE); n += y.numel()
        clean_c += (model(x)[0].argmax(1) == y).sum().item()
        x_occ, mask = occlude(x, band)
        occ_c += (model(x_occ)[0].argmax(1) == y).sum().item()
        x_t = x_occ.clone()
        for _ in range(rounds):
            _, recon = model(x_t)
            x_t = mask * x_occ + (1 - mask) * torch.sigmoid(recon)
        imag_c += (model(x_t)[0].argmax(1) == y).sum().item()
    return occ_c / n, imag_c / n, clean_c / n


def train_model(seed, args, loader):
    torch.manual_seed(seed)
    model = ConvBrain().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, recon = model(x)
            loss = ce(logits, y) + args.recon_w * F.binary_cross_entropy_with_logits(recon, x)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  seed {seed} epoch {ep} loss {loss.item():.3f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=25000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--recon-w", type=float, default=10.0)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    bands = [0, 4, 8, 12, 16]
    if args.quick:
        args.n_train, args.n_test, args.epochs, args.seeds = 4000, 1000, 3, 1
        bands = [0, 8, 16]
    print(f"Device {DEVICE}  CIFAR-10 occlusion sweep {bands} (of 32)  seeds {args.seeds}")

    tfm = transforms.ToTensor()
    tr_full = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
    te_full = datasets.CIFAR10("./data", train=False, transform=tfm)
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
            o, i, c = eval_completion(model, test_loader, band, args.rounds)
            occ[s, bi], imag[s, bi] = o, i
        clean.append(c)

    om, os = occ.mean(0), occ.std(0)
    im, isd = imag.mean(0), imag.std(0)
    gain = imag - occ
    gm, gs = gain.mean(0), gain.std(0)
    print(f"\nClean ceiling: {np.mean(clean):.3f}")
    print(f"{'occl rows':>9} {'occluded':>14} {'imagined':>14} {'gain':>14}")
    for bi, band in enumerate(bands):
        print(f"{band:>9} {om[bi]:.3f}+/-{os[bi]:.3f} {im[bi]:.3f}+/-{isd[bi]:.3f} "
              f"{gm[bi]:+.3f}+/-{gs[bi]:.3f}")
    print(f"\nPeak gain {gm.max():+.3f} at {bands[int(np.argmax(gm))]} rows. "
          f"(Prediction: should beat MNIST's +0.108 if the redundancy law holds.)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].errorbar(bands, om, yerr=os, fmt="s-", color="gray", label="occluded (feedforward)")
    ax[0].errorbar(bands, im, yerr=isd, fmt="o-", color="#378ADD", label="+ imagination")
    ax[0].set_xlabel("occlusion rows (of 32)"); ax[0].set_ylabel("CIFAR-10 accuracy")
    ax[0].set_ylim(0, 1); ax[0].set_title("Recognition vs occlusion"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].errorbar(bands, gm, yerr=gs, fmt="o-", color="#D85A30"); ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("occlusion rows (of 32)"); ax[1].set_ylabel("imagination gain")
    ax[1].set_title("Where imagination helps (CIFAR)"); ax[1].grid(alpha=0.3)
    fig.suptitle("Does the redundancy law hold on natural images? (CIFAR-10)")
    plt.tight_layout(); plt.savefig("cifar_imagine.png", dpi=120)
    print("Saved cifar_imagine.png")


if __name__ == "__main__":
    main()
