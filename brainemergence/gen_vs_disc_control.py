"""
gen_vs_disc_control.py -- the make-or-break for the Level-3 generative-classifier result.
=========================================================================================
The generative (Bayes) classifier crushed the discriminative net on held-out missing-data
(+0.41 on band) -- but it got the missingness MASK (it scores only observed pixels). The
reviewer's objection: "that win is just having the mask, not generative modeling." This
script isolates the two by giving the discriminative baseline the mask too, in increasing
strength, while the generative model stays CLEAN-trained (zero-shot, no corruption training):

  gen (clean, mask-aware)        -- the generative classifier (reference; clean-trained).
  disc-plain (clean, no mask)    -- naive baseline (sees zeros in the hole).
  disc-mask (clean, mask input)  -- gets the mask as input, but never trained on masks
                                    -> tests zero-shot mask exploitation.
  disc-mask+aug (mask input,     -- gets the mask AND is TRAINED with random masking
    masking-augmented training)     augmentation -> LEARNS to ignore holes. The strongest
                                    discriminative opponent; it even gets MORE training
                                    information than the clean generative model.

VERDICT: if the clean generative classifier still beats disc-mask+aug on missing-data, the
advantage is the generative MODELING (structural), not mask access. If disc-mask+aug closes
the gap, then "mask + augmentation" suffices and the Level-3 win is not special (the honest
negative, mirroring how imagination lost to augmentation in the original project).
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
from gen_classifier import CORR, MISSING, corrupt, train_vae, gen_scores

TRAIN_CORR = ["band", "patch", "scattered"]   # missing-data family for masking augmentation


class DiscPlain(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x, m=None):
        return self.net(x)


class DiscMask(nn.Module):
    """Discriminative net that also receives the missingness mask as input."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1568, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x, m):
        return self.net(torch.cat([x, m], dim=1))


def rand_mask_corrupt(x, corr_list=TRAIN_CORR):
    k = corr_list[torch.randint(len(corr_list), (1,)).item()]
    xc, m = CORR[k](x)
    return xc, (m if m is not None else torch.ones_like(x))


def train_disc(use_mask, augment, seed, args, loader, corr_list=TRAIN_CORR):
    torch.manual_seed(seed + (321 if use_mask else 0) + (7 if augment else 0))
    model = (DiscMask() if use_mask else DiscPlain()).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    for ep in range(args.epochs_disc):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if augment:
                xc, m = rand_mask_corrupt(x, corr_list)
            else:
                xc, m = x, torch.ones_like(x)
            loss = ce(model(xc, m), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def run_loo(args, train_loader, test_loader):
    """Leave-one-out (paper-consistent): for each held-out missing-data corruption, the strongest
    disc (mask + augmentation) is trained on the OTHER missing-data corruptions only, then both it
    and the clean-trained generative classifier are tested on the held-out shape. The generative
    model handles any mask zero-shot; the augmented disc only generalizes to shapes it saw."""
    print(f"\n=== LEAVE-ONE-OUT (held-out mask unseen by the disc's augmentation), {args.seeds} seeds ===")
    print(f"{'held-out':>10} {'disc_mask_aug (LOO)':>20} {'GEN (clean)':>14}  verdict")
    rows = {}
    vaes = [train_vae(s, args, train_loader) for s in range(args.seeds)]   # gen is held-out-independent
    survive = False
    for heldout in TRAIN_CORR:
        others = [c for c in TRAIN_CORR if c != heldout]
        d_acc, g_acc = [], []
        for s in range(args.seeds):
            disc = train_disc(True, True, s, args, train_loader, corr_list=others)
            d_acc.append(eval_disc(disc, test_loader, heldout))
            g_acc.append(eval_gen(vaes[s], test_loader, heldout, args.K))
        d, dsd = np.mean(d_acc), np.std(d_acc)
        gmn, gsd = np.mean(g_acc), np.std(g_acc)
        win = gmn - gsd > d + 0.005
        survive = survive or win
        rows[heldout] = (d, dsd, gmn, gsd)
        print(f"{heldout:>10} {d:.3f}+/-{dsd:.3f}      {gmn:.3f}+/-{gsd:.3f}  "
              f"{'GEN WINS (zero-shot beats aug)' if win else 'disc-aug still wins/ties'}")
    print("\n--- LEAVE-ONE-OUT VERDICT ---")
    print("GEN SURVIVES: on an UNFORESEEN mask shape, the clean generative classifier beats the "
          "augmented disc -> zero-shot mask-handling is a real, paper-consistent edge." if survive else
          "GEN DOES NOT SURVIVE: even on held-out mask shapes, the mask-augmented disc generalizes "
          "(it learned the general 'ignore masked pixels' operation) -> generative offers no edge.")
    return rows


@torch.no_grad()
def eval_disc(model, loader, kind):
    model.eval(); tot = c = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE); tot += y.numel()
        xc, m = corrupt(x, kind)
        if m is None:
            m = torch.ones_like(xc)
        c += (model(xc, m).argmax(1) == y).sum().item()
    return c / tot


@torch.no_grad()
def eval_gen(vae, loader, kind, K):
    vae.eval(); tot = c = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE); tot += y.numel()
        xc, m = corrupt(x, kind)
        c += (gen_scores(vae, xc, m, K).argmax(1) == y).sum().item()
    return c / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)        # VAE
    ap.add_argument("--epochs-disc", type=int, default=8)
    ap.add_argument("--t-steps", type=int, default=12)
    ap.add_argument("--latent", type=int, default=20)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--free-bits", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--loo", action="store_true", help="leave-one-out: disc augments on all-but-held-out")
    args = ap.parse_args()
    if args.quick:
        (args.n_train, args.n_test, args.epochs, args.epochs_disc,
         args.seeds, args.K) = 3000, 800, 4, 4, 1, 2
    print(f"Device {DEVICE}  Level-3 fairness control ({args.dataset}, K={args.K})")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    DS = datasets.FashionMNIST if args.dataset == "fashion" else datasets.MNIST
    trf = DS("./data", train=True, download=True, transform=tfm)
    tef = DS("./data", train=False, transform=tfm)
    g = torch.Generator().manual_seed(0)
    tr = Subset(trf, torch.randperm(len(trf), generator=g)[:args.n_train].tolist())
    te = Subset(tef, torch.randperm(len(tef), generator=g)[:args.n_test].tolist())
    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(te, batch_size=256)

    if args.loo:
        run_loo(args, train_loader, test_loader)
        return

    kinds = list(CORR)                                   # clean, band, patch, scattered, noise
    methods = ["gen", "disc_plain", "disc_mask", "disc_mask_aug"]
    res = {m: {k: [] for k in kinds} for m in methods}
    for s in range(args.seeds):
        vae = train_vae(s, args, train_loader)
        d_plain = train_disc(False, False, s, args, train_loader)
        d_mask = train_disc(True, False, s, args, train_loader)
        d_mask_aug = train_disc(True, True, s, args, train_loader)
        for k in kinds:
            res["gen"][k].append(eval_gen(vae, test_loader, k, args.K))
            res["disc_plain"][k].append(eval_disc(d_plain, test_loader, k))
            res["disc_mask"][k].append(eval_disc(d_mask, test_loader, k))
            res["disc_mask_aug"][k].append(eval_disc(d_mask_aug, test_loader, k))
        print(f"  seed {s} done")

    def ms(m, k): v = np.array(res[m][k]); return v.mean(), v.std()
    print(f"\n=== {args.dataset}: accuracy by method x corruption (mean+/-std, {args.seeds} seeds) ===")
    print(f"{'corruption':>10} {'disc_plain':>14} {'disc_mask':>14} {'disc_mask_aug':>16} {'GEN(clean)':>14}")
    survives = True
    for k in kinds:
        dp, dm, dma, gn = ms("disc_plain", k), ms("disc_mask", k), ms("disc_mask_aug", k), ms("gen", k)
        best_disc = max(dp[0], dm[0], dma[0])
        if k in MISSING:
            tag = "  <-- GEN STILL WINS" if gn[0] > best_disc + 0.005 else "  (disc closes it)"
            if gn[0] <= best_disc + 0.005:
                survives = False
        else:
            tag = "  (neg ctrl)" if k == "noise" else ""
        print(f"{k:>10} {dp[0]:.3f}+/-{dp[1]:.3f} {dm[0]:.3f}+/-{dm[1]:.3f} {dma[0]:.3f}+/-{dma[1]:.3f} "
              f"{gn[0]:.3f}+/-{gn[1]:.3f}{tag}")
    print("\n--- LEVEL-3 FAIRNESS VERDICT ---")
    if survives:
        print("STRUCTURAL: the clean generative classifier beats even the mask-aware, masking-AUGMENTED "
              "discriminative net on held-out missing-data. The advantage is generative MODELING, not mask access.")
    else:
        print("NOT SPECIAL: a mask-aware + augmentation-trained discriminative net closes the gap on "
              "missing-data. The Level-3 win was mask access + augmentation, not generative modeling.")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(kinds)); w = 0.2
    for i, (m, col, lab) in enumerate([("disc_plain", "#bbb", "disc (no mask)"),
                                       ("disc_mask", "#888", "disc +mask"),
                                       ("disc_mask_aug", "#D85A30", "disc +mask +aug (strongest)"),
                                       ("gen", "#1e8449", "generative (clean)")]):
        ax.bar(x + (i - 1.5) * w, [ms(m, k)[0] for k in kinds], w,
               yerr=[ms(m, k)[1] for k in kinds], capsize=3, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(kinds); ax.set_ylim(0, 1); ax.set_ylabel("accuracy")
    ax.set_title(f"Level-3 fairness control ({args.dataset}): is the generative win structural?")
    ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"gen_vs_disc_control_{args.dataset}.png", dpi=120)
    print(f"Saved gen_vs_disc_control_{args.dataset}.png")


if __name__ == "__main__":
    main()
