"""
mask_control.py -- the missing fairness control for the MAIN paper.
===================================================================
Reviewer objection: imagination uses the missingness MASK (it pastes observed pixels and inpaints
the hole); the paper's ff/TTA/MEMO baselines do NOT. So is the band/patch win generative completion,
or just mask access? We add the two mask-aware baselines the paper lacks and re-run leave-one-out:

  imag        : broad-aug model + generative completion (uses mask)        -- the headline.
  mask_fill   : broad-aug model + trivial observed-MEAN fill (uses mask)   -- isolates whether the
                win is the GENERATIVE fill or just having the mask + any fill.
  maskaug_ff  : a model that TAKES the mask as input AND is trained with masking augmentation on the
                non-held-out corruptions, plain feed-forward (uses mask + trained-for-masks) -- the
                strongest "just augment, with the mask" discriminative baseline (mirrors Level-3).

The headline SURVIVES iff, on a held-out contiguous missing-data corruption, imagination still beats
BOTH mask_fill and maskaug_ff with separated error bars. If a mask-aware baseline closes it, the
band/patch win was mask access + augmentation, not generative completion.
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

from mnist_experiment import spike_func
from imagine_helps import RecallBrain, DEVICE
from paper_exp1 import CORR, MISSING, apply_corruption, train_broadaug, imagine


class RecallBrainMask(RecallBrain):
    """RecallBrain whose input layer also receives the missingness mask (concat)."""
    def __init__(self, H, W, t_steps):
        super().__init__(H, W, t_steps, movable=False)
        self.w_in = nn.Linear(784 * 2, self.N)

    def encode(self, x, mask):
        B = x.shape[0]; Wr = self.recurrent_W()
        tau, thr = torch.sigmoid(self.tau_raw), F.softplus(self.thr_raw) + 0.5
        mem = torch.zeros(B, self.N, device=x.device)
        spk = torch.zeros_like(mem); acc = torch.zeros_like(mem)
        drive = self.w_in(torch.cat([x, mask], dim=1))
        for _ in range(self.T):
            mem = mem * tau + (drive + spk @ Wr.t()) - spk * thr
            spk = spike_func(mem - thr); acc = acc + spk
        return acc / self.T

    def forward(self, x, mask):
        A = self.encode(x, mask)
        return self.cls(A), self.dec(A)


def mask_of(x_corr, mask):
    return torch.ones_like(x_corr) if mask is None else mask


def train_maskaug(seed, heldout, args, loader):
    """Train the mask-aware model with the SAME leave-one-out augmentation, feeding it the mask."""
    torch.manual_seed(seed + 555)
    model = RecallBrainMask(args.grid, args.grid, args.t_steps).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    ce = nn.CrossEntropyLoss()
    aug = [k for k in CORR if k != heldout]
    for ep in range(args.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            k = aug[torch.randint(len(aug), (1,)).item()]
            x_in, m = apply_corruption(x, k)
            logits = model(x_in, mask_of(x_in, m))[0]
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def eval_all(broad, maskaug, loader, heldout, n):
    broad.eval(); maskaug.eval()
    tot = imag = fill = maug = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE); tot += y.numel()
        x_c, mask = apply_corruption(x, heldout)
        m = mask_of(x_c, mask)
        # imagination (generative completion, broad-aug model)
        imag += (imagine(broad, x_c, mask, n) == y).sum().item()
        # mask-aware trivial mean-fill, broad-aug model
        obs_mean = (x_c * m).sum(1, keepdim=True) / (m.sum(1, keepdim=True) + 1e-6)
        x_fill = m * x_c + (1 - m) * obs_mean
        fill += (broad(x_fill)[0].argmax(1) == y).sum().item()
        # mask-aware augmented feed-forward
        maug += (maskaug(x_c, m)[0].argmax(1) == y).sum().item()
    return imag / tot, fill / tot, maug / tot


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
    ap.add_argument("--dataset", default="mnist")
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.n_test, args.epochs, args.t_steps, args.seeds = 3000, 1000, 2, 12, 1
    print(f"Device {DEVICE}  MASK-AWARE fairness control ({args.dataset})")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    DS = datasets.FashionMNIST if args.dataset == "fashion" else datasets.MNIST
    trf = DS("./data", train=True, download=True, transform=tfm)
    tef = DS("./data", train=False, transform=tfm)
    g = torch.Generator().manual_seed(0)
    tr = Subset(trf, torch.randperm(len(trf), generator=g)[:args.n_train].tolist())
    te = Subset(tef, torch.randperm(len(tef), generator=g)[:args.n_test].tolist())
    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(te, batch_size=256)

    res = {k: {"imag": [], "fill": [], "maug": []} for k in CORR}
    for s in range(args.seeds):
        for heldout in CORR:
            broad = train_broadaug(s, heldout, args, train_loader)     # no-mask model (imag + fill)
            maskaug = train_maskaug(s, heldout, args, train_loader)    # mask-aware augmented model
            im, fl, mg = eval_all(broad, maskaug, test_loader, heldout, args.rounds)
            res[heldout]["imag"].append(im); res[heldout]["fill"].append(fl); res[heldout]["maug"].append(mg)
        print(f"  seed {s} done")

    def ms(k, m): v = np.array(res[k][m]); return v.mean(), v.std()
    print(f"\n=== {args.dataset}: held-out | imagination | mask-fill | mask-aug-FF  (mean+/-std) ===")
    survives = True
    for k in CORR:
        im, fl, mg = ms(k, "imag"), ms(k, "fill"), ms(k, "maug")
        if k in MISSING:
            beats = im[0] - im[1] > max(fl[0], mg[0]) + 0.005
            tag = "  <-- IMAG STILL WINS" if beats else "  (mask-aware baseline closes it)"
            # only the contiguous win-cells decide the headline
            if k in ("band", "patch") and not beats:
                survives = False
        else:
            tag = "  (neg ctrl)"
        print(f"  {k:>10}: imag {im[0]:.3f}+/-{im[1]:.3f}   fill {fl[0]:.3f}+/-{fl[1]:.3f}   "
              f"maskaug {mg[0]:.3f}+/-{mg[1]:.3f}{tag}")
    print("\n--- MASK-AWARE FAIRNESS VERDICT ---")
    if survives:
        print("HEADLINE SURVIVES: on held-out band/patch, generative completion still beats BOTH the "
              "trivial mask-fill AND the mask-aware augmentation-trained model. The win is generative "
              "completion, not mask access.")
    else:
        print("HEADLINE WEAKENS: a mask-aware baseline (fill or augmented-FF) matches/beats imagination "
              "on a contiguous win-cell. The band/patch win was partly mask access + augmentation.")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ks = list(CORR); x = np.arange(len(ks)); w = 0.26
    for i, (mn, col, lab) in enumerate([("fill", "#bbb", "mask-fill (trivial)"),
                                        ("maug", "#8E44AD", "mask-aug FF (strong)"),
                                        ("imag", "#1e8449", "imagination (generative)")]):
        ax.bar(x + (i - 1) * w, [ms(k, mn)[0] for k in ks], w,
               yerr=[ms(k, mn)[1] for k in ks], capsize=3, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(ks); ax.set_ylim(0, 1); ax.set_ylabel("held-out accuracy")
    ax.set_title(f"Mask-aware fairness control ({args.dataset}): is the win generative, or just the mask?")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"mask_control_{args.dataset}.png", dpi=120)
    print(f"Saved mask_control_{args.dataset}.png")


if __name__ == "__main__":
    main()
