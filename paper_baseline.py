"""
paper_baseline.py -- the make-or-break baseline for a paper.
============================================================
Does imagination-completion add anything BEYOND simply training the classifier with
occlusion augmentation? A 2x2:

                         test: feedforward      test: + imagination
  clean-trained                A                       B
  occlusion-augmented          C  (the real baseline)  D

Key comparisons:
  B - A : imagination helps a clean-trained model (the original result).
  C - A : occlusion augmentation alone helps the feedforward classifier.
  D - C : DOES IMAGINATION ADD ON TOP OF AUGMENTATION?  <-- the make-or-break.
If D <= C, "just augment your data" wins and there is no paper. If D > C, imagination
contributes something augmentation does not.
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


def occlude_random(x):
    """Training-time augmentation: occlude a random horizontal band on ~70% of images."""
    B = x.shape[0]
    img = x.view(B, 28, 28).clone()
    mask = torch.ones_like(img)
    for i in range(B):
        if torch.rand(1).item() < 0.7:
            band = torch.randint(4, 13, (1,)).item()
            r0 = torch.randint(0, 28 - band + 1, (1,)).item()
            img[i, r0:r0 + band, :] = 0.0
            mask[i, r0:r0 + band, :] = 0.0
    return img.view(B, -1)


def train_model(seed, aug, args, loader):
    torch.manual_seed(seed)
    model = RecallBrain(args.grid, args.grid, args.t_steps, movable=False).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    ce = nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_in = occlude_random(x) if aug else x          # augmentation only changes INPUT
            A = model.encode(x_in)
            logits, recon = model.cls(A), model.dec(A)
            loss = ce(logits, y) + args.recon_w * F.binary_cross_entropy_with_logits(
                recon, x, reduction="sum") / x.size(0)       # reconstruct CLEAN x
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def eval_occ(model, loader, band, rounds):
    model.eval()
    n = ff = im = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE); n += y.numel()
        x_occ, mask = occlude(x, band)
        ff += (model(x_occ)[0].argmax(1) == y).sum().item()  # feedforward on occluded
        x_t = x_occ.clone()
        for _ in range(rounds):
            _, recon = model(x_t)
            x_t = mask * x_occ + (1 - mask) * torch.sigmoid(recon)
        im += (model(x_t)[0].argmax(1) == y).sum().item()    # + imagination
    return ff / n, im / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--t-steps", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--band", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--recon-w", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.n_test, args.epochs, args.t_steps, args.seeds = 3000, 1000, 2, 12, 1
    print(f"Device {DEVICE}  band {args.band}px  the make-or-break 2x2")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    tr_full = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    te_full = datasets.MNIST("./data", train=False, transform=tfm)
    g = torch.Generator().manual_seed(0)
    tr = Subset(tr_full, torch.randperm(len(tr_full), generator=g)[:args.n_train].tolist())
    te = Subset(te_full, torch.randperm(len(te_full), generator=g)[:args.n_test].tolist())
    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(te, batch_size=256)

    cells = {("clean", "ff"): [], ("clean", "im"): [], ("aug", "ff"): [], ("aug", "im"): []}
    for s in range(args.seeds):
        for aug in (False, True):
            model = train_model(s, aug, args, train_loader)
            ff, im = eval_occ(model, test_loader, args.band, args.rounds)
            tag = "aug" if aug else "clean"
            cells[(tag, "ff")].append(ff); cells[(tag, "im")].append(im)
            print(f"  seed {s} {tag}: feedforward {ff:.3f}  +imagination {im:.3f}")

    def ms(k): v = np.array(cells[k]); return v.mean(), v.std()
    A, B = ms(("clean", "ff")), ms(("clean", "im"))
    C, D = ms(("aug", "ff")), ms(("aug", "im"))
    print("\n=== 2x2 (occluded-test accuracy, mean +/- std) ===")
    print(f"  clean-trained:        feedforward {A[0]:.3f}+/-{A[1]:.3f}   +imagination {B[0]:.3f}+/-{B[1]:.3f}")
    print(f"  occlusion-augmented:  feedforward {C[0]:.3f}+/-{C[1]:.3f}   +imagination {D[0]:.3f}+/-{D[1]:.3f}")
    print(f"\n  imagination on clean-trained (B-A): {B[0]-A[0]:+.3f}")
    print(f"  augmentation alone     (C-A): {C[0]-A[0]:+.3f}")
    print(f"  >>> imagination ON TOP of augmentation (D-C): {D[0]-C[0]:+.3f} +/- {np.hypot(D[1],C[1]):.3f}")
    if D[0] - C[0] > 0.01:
        print("  VERDICT: imagination adds value BEYOND augmentation -> a paper has legs.")
    else:
        print("  VERDICT: augmentation alone matches imagination -> the result does not survive the baseline.")

    labels = ["clean\nfeedforward", "clean\n+imagination", "aug\nfeedforward", "aug\n+imagination"]
    means = [A[0], B[0], C[0], D[0]]; errs = [A[1], B[1], C[1], D[1]]
    plt.figure(figsize=(6.5, 4.3))
    plt.bar(range(4), means, yerr=errs, capsize=5,
            color=["#999", "#378ADD", "#bbb", "#1e8449"])
    plt.xticks(range(4), labels); plt.ylabel(f"occluded-test accuracy ({args.band}px)")
    plt.title("Does imagination beat 'just augment your training'?")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig("paper_baseline.png", dpi=120)
    print("Saved paper_baseline.png")


if __name__ == "__main__":
    main()
