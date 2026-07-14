"""
imagine_patterns.py -- does imagination help for ANY shape of missing data?
==========================================================================
The robustness result used one occlusion shape (a center band). This checks whether
"imagination self-repairs missing data" is robust to the SHAPE of the hole -- center
band, top band, corner quadrant, random patch, scattered pixels -- each removing a
comparable ~25-28% of the image. If imagination helps across all shapes, the principle
("fills structured missing data") is solid, not a quirk of one mask.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from imagine_helps import RecallBrain, DEVICE
from imagine_robustness import train_model


def m_center(x, band=8):
    B = x.shape[0]; m = torch.ones(B, 28, 28, device=x.device)
    m[:, 14 - band // 2:14 - band // 2 + band, :] = 0
    return m.view(B, -1)

def m_top(x, band=8):
    B = x.shape[0]; m = torch.ones(B, 28, 28, device=x.device); m[:, :band, :] = 0
    return m.view(B, -1)

def m_quadrant(x, size=14):
    B = x.shape[0]; m = torch.ones(B, 28, 28, device=x.device); m[:, :size, :size] = 0
    return m.view(B, -1)

def m_patch(x, size=14):
    B = x.shape[0]; m = torch.ones(B, 28, 28, device=x.device)
    r = torch.randint(0, 28 - size + 1, (B,)); c = torch.randint(0, 28 - size + 1, (B,))
    for i in range(B):
        m[i, r[i]:r[i] + size, c[i]:c[i] + size] = 0
    return m.view(B, -1)

def m_scattered(x, frac=0.28):
    return (torch.rand(x.shape[0], 784, device=x.device) > frac).float()


PATTERNS = {"center band": m_center, "top band": m_top, "corner": m_quadrant,
            "random patch": m_patch, "scattered": m_scattered}


@torch.no_grad()
def eval_pattern(model, loader, mask_fn, rounds):
    model.eval()
    n = occ_c = imag_c = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE); n += y.numel()
        mask = mask_fn(x)
        x_occ = x * mask
        occ_c += (model(x_occ)[0].argmax(1) == y).sum().item()
        x_t = x_occ.clone()
        for _ in range(rounds):
            _, recon = model(x_t)
            x_t = mask * x_occ + (1 - mask) * torch.sigmoid(recon)
        imag_c += (model(x_t)[0].argmax(1) == y).sum().item()
    return occ_c / n, imag_c / n


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
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.n_test, args.epochs, args.t_steps, args.seeds = 3000, 1000, 2, 12, 1
    torch.manual_seed(0)
    print(f"Device {DEVICE}  occlusion-shape stress test  seeds {args.seeds}")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    tr_full = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    te_full = datasets.MNIST("./data", train=False, transform=tfm)
    g = torch.Generator().manual_seed(0)
    tr = Subset(tr_full, torch.randperm(len(tr_full), generator=g)[:args.n_train].tolist())
    te = Subset(te_full, torch.randperm(len(te_full), generator=g)[:args.n_test].tolist())
    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(te, batch_size=256)

    names = list(PATTERNS)
    occ = np.zeros((args.seeds, len(names)))
    imag = np.zeros((args.seeds, len(names)))
    for s in range(args.seeds):
        model = train_model(s, args, train_loader)
        for pi, nm in enumerate(names):
            occ[s, pi], imag[s, pi] = eval_pattern(model, test_loader, PATTERNS[nm], args.rounds)

    gain = imag - occ
    gm, gs = gain.mean(0), gain.std(0)
    om, im = occ.mean(0), imag.mean(0)
    print(f"\n{'pattern':>14} {'occluded':>12} {'imagined':>12} {'gain':>14}")
    for pi, nm in enumerate(names):
        print(f"{nm:>14} {om[pi]:.3f}       {im[pi]:.3f}       {gm[pi]:+.3f}+/-{gs[pi]:.3f}")
    allpos = np.all(gm - gs > 0)
    print(f"\nImagination helps for ALL occlusion shapes (robust): {bool(allpos)}")

    plt.figure(figsize=(7, 4.4))
    x = np.arange(len(names))
    plt.bar(x, gm, yerr=gs, capsize=5, color="#378ADD")
    plt.axhline(0, color="k", lw=0.8)
    plt.xticks(x, names, rotation=15)
    plt.ylabel("imagination gain (imagined - occluded)")
    plt.title("Does imagination help for any SHAPE of missing data? (MNIST, ~28% removed)")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig("imagine_patterns.png", dpi=120)
    print("Saved imagine_patterns.png")


if __name__ == "__main__":
    main()
