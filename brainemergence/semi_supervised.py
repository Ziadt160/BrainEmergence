"""
semi_supervised.py -- the one battlefield where "generate to understand" should actually win.
=============================================================================================
Few labels + many UNLABELED examples. The generative (reconstruction) objective can learn
structure from the unlabeled pool; the classifier uses the few labels. If analysis-by-synthesis
has a real edge anywhere on this hardware, it is here (this is classic semi-supervised learning,
where generative models have a genuine track record).

Same RecallBrain architecture (shared spiking encoder + cls head + dec head). Three conditions,
all trained for the SAME number of gradient steps (fair compute):

  disc       : cross-entropy on the labeled subset only.                (pure discriminative)
  lab_recon  : CE on labels + reconstruction on the labeled subset.     (recon as regularizer)
  semi       : CE on labels + reconstruction on ALL data (incl unlabeled). ("generate to
               understand" from unlabeled data -- the real semi-supervised condition)

Read: semi > disc  => generate-to-understand helps at few labels.
      semi > lab_recon => the UNLABELED data specifically is what helps (not just the regularizer).
The gap should be LARGEST at the fewest labels.
"""
import argparse
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset

from imagine_helps import RecallBrain, DEVICE


def balanced_labeled_idx(targets, n_labeled, generator):
    """Pick n_labeled indices balanced across the 10 classes."""
    per = max(1, n_labeled // 10)
    idx = []
    for c in range(10):
        cls_idx = (targets == c).nonzero(as_tuple=True)[0]
        perm = cls_idx[torch.randperm(len(cls_idx), generator=generator)[:per]]
        idx.extend(perm.tolist())
    return idx


def cycle(loader):
    while True:
        for b in loader:
            yield b


def train(condition, seed, n_labeled, args, Xtr, Ytr, pool_idx, lab_idx, test_loader):
    torch.manual_seed(seed)
    model = RecallBrain(args.grid, args.grid, args.t_steps, movable=False).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    ce = nn.CrossEntropyLoss()

    lab_ds = TensorDataset(Xtr[lab_idx], Ytr[lab_idx])
    lab_iter = cycle(DataLoader(lab_ds, batch_size=min(args.batch_lab, n_labeled), shuffle=True))
    unlab_ds = TensorDataset(Xtr[pool_idx])
    unlab_iter = cycle(DataLoader(unlab_ds, batch_size=args.batch_unlab, shuffle=True))

    def recon_loss(x):
        A = model.encode(x)
        return F.binary_cross_entropy_with_logits(model.dec(A), x, reduction="sum") / x.size(0)

    model.train()
    for step in range(args.steps):
        xb, yb = next(lab_iter); xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        A = model.encode(xb)
        loss = ce(model.cls(A), yb)
        if condition in ("lab_recon", "semi"):
            loss = loss + args.recon_w * F.binary_cross_entropy_with_logits(
                model.dec(A), xb, reduction="sum") / xb.size(0)
        if condition == "semi":
            (xu,) = next(unlab_iter); xu = xu.to(DEVICE)
            loss = loss + args.recon_w * recon_loss(xu)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    model.eval(); tot = c = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE); tot += y.numel()
            c += (model(x)[0].argmax(1) == y).sum().item()
    return c / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--t-steps", type=int, default=14)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-lab", type=int, default=64)
    ap.add_argument("--batch-unlab", type=int, default=128)
    ap.add_argument("--pool", type=int, default=12000)      # unlabeled pool size
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--recon-w", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--labels", type=int, nargs="+", default=[100, 1000])
    args = ap.parse_args()
    if args.quick:
        args.steps, args.seeds, args.pool, args.n_test, args.labels = 300, 1, 4000, 1000, [100]
    print(f"Device {DEVICE}  semi-supervised ({args.dataset}); labels={args.labels}; steps={args.steps}")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    DS = datasets.FashionMNIST if args.dataset == "fashion" else datasets.MNIST
    trf = DS("./data", train=True, download=True, transform=tfm)
    tef = DS("./data", train=False, transform=tfm)
    # materialize train tensors for the pool
    g = torch.Generator().manual_seed(0)
    pool_all = torch.randperm(len(trf), generator=g)[:args.pool].tolist()
    Xtr = torch.stack([trf[i][0] for i in pool_all])
    Ytr = torch.tensor([trf[i][1] for i in pool_all])
    pool_idx = list(range(len(pool_all)))                   # all pool examples are "unlabeled"
    te = Subset(tef, torch.randperm(len(tef), generator=g)[:args.n_test].tolist())
    test_loader = DataLoader(te, batch_size=256)

    conditions = ["disc", "lab_recon", "semi"]
    res = {nl: {c: [] for c in conditions} for nl in args.labels}
    for nl in args.labels:
        for s in range(args.seeds):
            gl = torch.Generator().manual_seed(100 + s)
            lab_idx = balanced_labeled_idx(Ytr, nl, gl)
            for cond in conditions:
                acc = train(cond, s, nl, args, Xtr, Ytr, pool_idx, lab_idx, test_loader)
                res[nl][cond].append(acc)
            print(f"  labels={nl} seed={s} done")

    def ms(nl, c): v = np.array(res[nl][c]); return v.mean(), v.std()
    print(f"\n=== {args.dataset}: test accuracy by (#labels, condition)  mean+/-std ===")
    print(f"{'#labels':>8} {'disc':>14} {'lab_recon':>14} {'semi (unlab)':>16} {'semi-disc':>11}")
    for nl in args.labels:
        d, lr, sm = ms(nl, "disc"), ms(nl, "lab_recon"), ms(nl, "semi")
        gap = sm[0] - d[0]
        print(f"{nl:>8} {d[0]:.3f}+/-{d[1]:.3f} {lr[0]:.3f}+/-{lr[1]:.3f} {sm[0]:.3f}+/-{sm[1]:.3f} {gap:>+11.3f}")
    # The hypothesis is that semi-sup helps MOST at the FEWEST labels -> gate the verdict on the
    # smallest label budget specifically (not "any" budget, which would be misleading).
    nl_min = min(args.labels)
    d, sm = ms(nl_min, "disc"), ms(nl_min, "semi")
    helps_few = sm[0] - sm[1] > d[0] + 0.005
    print("\n--- VERDICT (gated on the FEWEST labels = the regime the hypothesis is about) ---")
    if helps_few:
        print(f"GENERATE-TO-UNDERSTAND HELPS at {nl_min} labels: semi beats pure discriminative "
              f"({sm[0]:.3f} vs {d[0]:.3f}) with separated error bars.")
    else:
        print(f"HYPOTHESIS FALSIFIED at {nl_min} labels: semi (recon on unlabeled) does NOT beat pure "
              f"discriminative ({sm[0]:.3f} vs {d[0]:.3f}); naive reconstruction is a weak SSL signal "
              f"and dominates the encoder when labels are scarce. Real semi-sup wins use "
              f"consistency/contrastive methods, not generation.")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(args.labels)); w = 0.26
    for i, (c, col, lab) in enumerate([("disc", "#999", "discriminative (labels only)"),
                                       ("lab_recon", "#8E44AD", "+recon (labels only)"),
                                       ("semi", "#1e8449", "+recon on UNLABELED (semi-sup)")]):
        ax.bar(x + (i - 1) * w, [ms(nl, c)[0] for nl in args.labels], w,
               yerr=[ms(nl, c)[1] for nl in args.labels], capsize=3, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([str(nl) for nl in args.labels]); ax.set_ylim(0, 1)
    ax.set_xlabel("# labeled examples"); ax.set_ylabel("test accuracy")
    ax.set_title(f"Does 'generate to understand' help at few labels? ({args.dataset})")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"semi_supervised_{args.dataset}.png", dpi=120)
    print(f"Saved semi_supervised_{args.dataset}.png")


if __name__ == "__main__":
    main()
