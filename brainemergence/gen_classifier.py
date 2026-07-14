"""
gen_classifier.py -- Level-3: the GENERATIVE (Bayes) classifier vs the discriminative one.
=========================================================================================
Instead of learning pixels -> label directly (discriminative, p(y|x)), classify by asking
"which class can best EXPLAIN this image?" -- argmax_y p(x|y) (generative / analysis-by-
synthesis). We reuse BiBrain (a class-conditional VAE): for each candidate class y we condition
the decoder on y, reconstruct, and score how well that reconstruction matches the image. The
class whose generative model fits best wins.

The decisive design choice for robustness: when pixels are MISSING (occlusion/dropout), we score
each class only on the OBSERVED pixels (mask-aware reconstruction error) and ignore the hole.
A feedforward discriminative net cannot do this -- it must consume the corrupted pixels.

What we expect (and want to characterize, not beat SOTA):
  - CLEAN: discriminative > generative (the well-known accuracy tradeoff).
  - UNFORESEEN missing-data (no augmentation): generative degrades more GRACEFULLY, and the
    advantage should follow the same reconstructability rule (structure x redundancy).
  - NOISE: no missing pixels to ignore -> generative advantage should vanish (negative control).
Bonus: the per-class score gap is a built-in confidence / OOD signal -> abstention (risk-coverage).

This is a PROTOTYPE to test whether the Level-3 crossover exists at all. Mechanism is published
(Diffusion Classifier ICLR2023, RDC ICML2024); the wedge is the predictive rule + the uncertainty
signal, consistent with the current paper's framing.
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

from brain_generate import BiBrain, DEVICE
from gating import c_band, c_patch, c_scattered, c_noise

CORR = {"clean": None, "band": c_band, "patch": c_patch, "scattered": c_scattered, "noise": c_noise}
MISSING = {"band", "patch", "scattered"}


class Disc(nn.Module):
    """Strong, simple discriminative baseline (clean-trained). Deliberately a capable net so the
    generative classifier has to earn any robustness win against a real opponent."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


def corrupt(x, kind):
    if CORR[kind] is None:
        return x, None
    return CORR[kind](x)


def train_vae(seed, args, loader):
    torch.manual_seed(seed)
    model = BiBrain(t_steps=args.t_steps, latent=args.latent).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for ep in range(args.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            yoh = F.one_hot(y, 10).float()
            logits, mu, logvar = model(x, yoh)
            recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.size(0)
            kl_dim = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(0)
            kl = torch.clamp(kl_dim, min=args.free_bits).sum()
            loss = recon + args.beta * kl
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    return model


def train_disc(seed, args, loader):
    torch.manual_seed(seed + 999)
    model = Disc().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    for ep in range(args.epochs_disc):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = ce(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def gen_scores(model, x, mask, K):
    """Per-class generative score = -reconstruction error on OBSERVED pixels, averaged over K
    latent samples from q(z|x). Returns (B, 10); higher = the class explains the evidence better.
    KL is class-independent so it drops out of argmax_y -> reconstruction term alone discriminates."""
    B = x.shape[0]
    mu, logvar = model.encode(x)
    scores = torch.zeros(B, 10, device=x.device)
    for _ in range(K):
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if K > 1 else mu
        for y in range(10):
            yoh = F.one_hot(torch.full((B,), y, device=x.device), 10).float()
            logits = model.decode(z, yoh)
            bce = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
            bce = (bce * mask).sum(1) if mask is not None else bce.sum(1)
            scores[:, y] += -bce
    return scores / K


@torch.no_grad()
def evaluate(vae, disc, loader, kind, K):
    vae.eval(); disc.eval()
    tot = gen_c = disc_c = 0
    margins, gen_correct = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE); tot += y.numel()
        x_c, mask = corrupt(x, kind)
        # discriminative: must consume the corrupted input
        disc_c += (disc(x_c).argmax(1) == y).sum().item()
        # generative: score each class on observed evidence
        s = gen_scores(vae, x_c, mask, K)
        pred = s.argmax(1)
        gen_c += (pred == y).sum().item()
        # confidence = gap between best and 2nd-best class score (for abstention/OOD)
        top2 = s.topk(2, dim=1).values
        margins.append((top2[:, 0] - top2[:, 1]).cpu())
        gen_correct.append((pred == y).cpu())
    margins = torch.cat(margins); gen_correct = torch.cat(gen_correct)
    return gen_c / tot, disc_c / tot, margins, gen_correct


def risk_coverage(margins, correct, points=10):
    """Accuracy of the generative classifier when it only answers on its most-confident fraction
    (coverage). A useful 'know when you don't know' curve -- accuracy should rise as coverage drops."""
    order = torch.argsort(margins, descending=True)
    c = correct[order].float()
    out = []
    for frac in np.linspace(0.1, 1.0, points):
        k = max(1, int(frac * len(c)))
        out.append((frac, c[:k].mean().item()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--t-steps", type=int, default=12)
    ap.add_argument("--latent", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=10)         # VAE epochs
    ap.add_argument("--epochs-disc", type=int, default=6)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--free-bits", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--K", type=int, default=4)              # latent samples for the gen score
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dataset", default="mnist")
    args = ap.parse_args()
    if args.quick:
        (args.n_train, args.n_test, args.epochs, args.epochs_disc,
         args.t_steps, args.seeds, args.K) = 3000, 800, 3, 3, 10, 1, 2
    print(f"Device {DEVICE}  generative-vs-discriminative classifier ({args.dataset}, K={args.K})")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    DS = datasets.FashionMNIST if args.dataset == "fashion" else datasets.MNIST
    trf = DS("./data", train=True, download=True, transform=tfm)
    tef = DS("./data", train=False, transform=tfm)
    g = torch.Generator().manual_seed(0)
    tr = Subset(trf, torch.randperm(len(trf), generator=g)[:args.n_train].tolist())
    te = Subset(tef, torch.randperm(len(tef), generator=g)[:args.n_test].tolist())
    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(te, batch_size=256)

    kinds = list(CORR)
    res = {k: {"gen": [], "disc": []} for k in kinds}
    rc_clean, rc_band = None, None
    for s in range(args.seeds):
        vae = train_vae(s, args, train_loader)
        disc = train_disc(s, args, train_loader)
        for k in kinds:
            gen_a, disc_a, margins, correct = evaluate(vae, disc, test_loader, k, args.K)
            res[k]["gen"].append(gen_a); res[k]["disc"].append(disc_a)
            if s == 0 and k == "clean": rc_clean = risk_coverage(margins, correct)
            if s == 0 and k == "band":  rc_band = risk_coverage(margins, correct)
        print(f"  seed {s} done")

    def ms(k, m): v = np.array(res[k][m]); return v.mean(), v.std()
    print("\n=== Accuracy: discriminative vs generative (clean-trained, tested on each corruption) ===")
    print(f"{'corruption':>10} {'disc':>14} {'generative':>16} {'gen - disc':>14}")
    for k in kinds:
        d, gn = ms(k, "disc"), ms(k, "gen")
        tag = "  (neg ctrl)" if k == "noise" else ("" if k == "clean" else
              ("  <-- GEN MORE ROBUST" if gn[0] - gn[1] > d[0] + 0.005 else ""))
        print(f"{k:>10} {d[0]:.3f}+/-{d[1]:.3f} {gn[0]:.3f}+/-{gn[1]:.3f} "
              f"{gn[0]-d[0]:+.3f}{tag}")
    print("\nReconstructability rule (Level-3 form): the generative classifier should overtake the "
          "discriminative one on reconstructable held-out MISSING-DATA, not on noise.")

    # ---- figure: robustness bars + risk-coverage ----
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    x = np.arange(len(kinds)); w = 0.38
    ax[0].bar(x - w/2, [ms(k, "disc")[0] for k in kinds], w, yerr=[ms(k, "disc")[1] for k in kinds],
              capsize=3, color="#777", label="discriminative")
    ax[0].bar(x + w/2, [ms(k, "gen")[0] for k in kinds], w, yerr=[ms(k, "gen")[1] for k in kinds],
              capsize=3, color="#1e8449", label="generative (Bayes)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(kinds); ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("accuracy"); ax[0].set_title(f"Generative vs discriminative ({args.dataset})")
    ax[0].legend(fontsize=8); ax[0].grid(axis="y", alpha=0.3)
    for rc, lab, col in [(rc_clean, "clean", "#378ADD"), (rc_band, "band-occl", "#D85A30")]:
        if rc: ax[1].plot([c for c, _ in rc], [a for _, a in rc], "o-", color=col, label=lab)
    ax[1].set_xlabel("coverage (fraction answered, most-confident first)")
    ax[1].set_ylabel("accuracy on answered"); ax[1].set_ylim(0, 1.02)
    ax[1].set_title("Built-in 'know when you don't know'\n(generative score margin -> abstention)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"gen_classifier_{args.dataset}.png", dpi=120)
    print(f"Saved gen_classifier_{args.dataset}.png")


if __name__ == "__main__":
    main()
