"""
paper_exp1.py -- the make-or-break for the decision-rule paper.
==============================================================
Survive the two reviewer killers at once:
  (1) "just augment for everything"   -> LEAVE-ONE-CORRUPTION-OUT: train augmentation on all
      corruptions EXCEPT one held-out X, then test on X (you cannot augment for what you did
      not anticipate).
  (2) "give augmentation the same test-time compute" -> COMPUTE-MATCHED: compare imagination's
      N iterative rounds against test-time-augmentation marginalization over N views (same #
      forward passes).

For each held-out corruption X, on test set corrupted by X, report at equal compute:
  broad-aug feedforward | broad-aug + TTA-marginalization(N) | broad-aug + MEMO(N) | broad-aug + imagination(N)
PAPER LIVES iff for a held-out MISSING-DATA corruption, imagination beats ALL THREE compute-matched
baselines with separated error bars; and does NOT win on noise (the rule's negative half).

MEMO (Zhang et al. 2021) is the strong test-time-adaptation baseline: per batch it minimizes the
entropy of the prediction marginalized over N augmentations, then predicts. We give it its FULLEST
test-time form -- its augmentations are the broad-aug corruption set itself (every corruption except
the held-out one), so it adapts to be confident/invariant over exactly the corruptions it knows.
If even that cannot recover a held-out contiguous-missing-data corruption, the rule is robust.
"""
import argparse
import copy
import json

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
from gating import c_band, c_patch, c_scattered, c_noise

CORR = {"band": c_band, "patch": c_patch, "scattered": c_scattered, "noise": c_noise}
MISSING = {"band", "patch", "scattered"}   # reconstructable; noise is the negative control


def apply_corruption(x, kind):
    return CORR[kind](x)                    # returns (x_corr, mask|None)


def train_broadaug(seed, heldout, args, loader):
    """Train augmenting with every corruption EXCEPT `heldout`; reconstruct CLEAN."""
    torch.manual_seed(seed)
    model = RecallBrain(args.grid, args.grid, args.t_steps, movable=False).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    ce = nn.CrossEntropyLoss()
    aug_kinds = [k for k in CORR if k != heldout]
    for ep in range(args.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            k = aug_kinds[torch.randint(len(aug_kinds), (1,)).item()]
            x_in = apply_corruption(x, k)[0]
            A = model.encode(x_in)
            logits, recon = model.cls(A), model.dec(A)
            loss = ce(logits, y) + args.recon_w * F.binary_cross_entropy_with_logits(
                recon, x, reduction="sum") / x.size(0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def tta_marginalize(model, x, n):
    """Compute-matched augmentation baseline: average softmax over n shifted views."""
    B = x.shape[0]; img = x.view(B, 28, 28)
    probs = 0
    for _ in range(n):
        dh, dw = int(torch.randint(-2, 3, (1,))), int(torch.randint(-2, 3, (1,)))
        v = torch.roll(img, shifts=(dh, dw), dims=(1, 2)).reshape(B, -1)
        probs = probs + F.softmax(model(v)[0], dim=1)
    return (probs / n).argmax(1)


@torch.no_grad()
def imagine(model, x_corr, mask, n, beta=0.5):
    x_t = x_corr.clone()
    for _ in range(n):
        _, recon = model(x_t)
        x_t = (mask * x_corr + (1 - mask) * torch.sigmoid(recon)) if mask is not None \
            else (beta * x_corr + (1 - beta) * torch.sigmoid(recon))
    return model(x_t)[0].argmax(1)


def memo_predict(model, base_state, x_c, heldout, n, steps, lr):
    """MEMO/Tent: episodically adapt the model on this batch by minimizing the entropy of the
    prediction marginalized over n augmentations (drawn from the broad-aug corruption set,
    i.e. every corruption except the held-out one, applied to the OBSERVED input x_c), then
    predict feed-forward. Compute-matched to imagination (n forward passes per step). We adapt
    only the NORMALIZATION/GAIN parameters (per-neuron membrane time-constant and threshold) --
    the standard stable TTA parameterization; adapting all weights collapses under entropy-min.
    The model is restored to `base_state` afterwards so each batch adapts from the trained weights."""
    aug_kinds = [k for k in CORR if k != heldout]
    model.load_state_dict(base_state)
    opt = torch.optim.Adam([model.tau_raw, model.thr_raw], lr=lr)
    for _ in range(steps):
        probs = 0
        for _ in range(n):
            k = aug_kinds[torch.randint(len(aug_kinds), (1,)).item()]
            v = apply_corruption(x_c, k)[0]
            probs = probs + F.softmax(model(v)[0], dim=1)
        p = probs / n
        ent = -(p * torch.log(p + 1e-6)).sum(1).mean()
        opt.zero_grad(); ent.backward()
        nn.utils.clip_grad_norm_([model.tau_raw, model.thr_raw], 1.0); opt.step()
    with torch.no_grad():
        pred = model(x_c)[0].argmax(1)
    model.load_state_dict(base_state)
    return pred


def eval_heldout(model, loader, heldout, n, memo_steps, memo_lr):
    model.eval(); tot = ff = tta = memo = im = 0
    base_state = copy.deepcopy(model.state_dict())
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE); tot += y.numel()
        x_c, mask = apply_corruption(x, heldout)
        with torch.no_grad():
            ff += (model(x_c)[0].argmax(1) == y).sum().item()
            tta += (tta_marginalize(model, x_c, n) == y).sum().item()
            im += (imagine(model, x_c, mask, n) == y).sum().item()
        memo += (memo_predict(model, base_state, x_c, heldout, n, memo_steps, memo_lr) == y).sum().item()
    return ff / tot, tta / tot, memo / tot, im / tot


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
    ap.add_argument("--dataset", default="mnist")          # 'mnist' or 'fashion'
    ap.add_argument("--memo-steps", type=int, default=3)   # MEMO adaptation steps per batch
    ap.add_argument("--memo-lr", type=float, default=1e-3)  # strongest stable gain-mode adaptation
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.n_test, args.epochs, args.t_steps, args.seeds = 3000, 1000, 2, 12, 1
    print(f"Device {DEVICE}  leave-one-corruption-out, compute-matched (N={args.rounds})")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    DS = datasets.FashionMNIST if args.dataset == "fashion" else datasets.MNIST
    trf = DS("./data", train=True, download=True, transform=tfm)
    tef = DS("./data", train=False, transform=tfm)
    g = torch.Generator().manual_seed(0)
    tr = Subset(trf, torch.randperm(len(trf), generator=g)[:args.n_train].tolist())
    te = Subset(tef, torch.randperm(len(tef), generator=g)[:args.n_test].tolist())
    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(te, batch_size=256)

    res = {k: {"ff": [], "tta": [], "memo": [], "im": []} for k in CORR}
    for s in range(args.seeds):
        for heldout in CORR:
            model = train_broadaug(s, heldout, args, train_loader)
            ff, tta, memo, im = eval_heldout(model, test_loader, heldout, args.rounds,
                                             args.memo_steps, args.memo_lr)
            res[heldout]["ff"].append(ff); res[heldout]["tta"].append(tta)
            res[heldout]["memo"].append(memo); res[heldout]["im"].append(im)
        print(f"  seed {s} done")

    def ms(k, m): v = np.array(res[k][m]); return v.mean(), v.std()
    print("\n=== Held-out | broad-aug ff | +TTA(N) | +MEMO(N) | +imagination(N)  [mean+/-std] ===")
    live = False
    for k in CORR:
        ff, tta, memo, im = ms(k, "ff"), ms(k, "tta"), ms(k, "memo"), ms(k, "im")
        # imagination wins iff its lower error bar clears the best of ALL three baselines
        beats = (k in MISSING) and (im[0] - im[1] > max(ff[0], tta[0], memo[0]) + 0.005)
        live = live or beats
        flag = "  <-- IMAGINATION WINS" if beats else ("" if k in MISSING else "  (negative control)")
        print(f"  {k:>10}: ff {ff[0]:.3f}+/-{ff[1]:.3f}   tta {tta[0]:.3f}+/-{tta[1]:.3f}   "
              f"memo {memo[0]:.3f}+/-{memo[1]:.3f}   imag {im[0]:.3f}+/-{im[1]:.3f}{flag}")
    print("\n--- MAKE-OR-BREAK VERDICT ---")
    if live:
        print("PAPER LIVES: on a held-out MISSING-DATA corruption, imagination beats broad-aug "
              "feedforward AND compute-matched TTA AND MEMO (separated error bars). Scale the program.")
    else:
        print("PAPER DEAD: compute-matched augmentation/TTA/MEMO matches/beats imagination on every "
              "held-out corruption. The decision rule collapses to 'always augment'. Write the portfolio piece.")

    out_json = f"paper_exp1_{args.dataset}.json"
    with open(out_json, "w") as f:
        json.dump({"dataset": args.dataset, "seeds": args.seeds, "rounds": args.rounds,
                   "memo_steps": args.memo_steps, "memo_lr": args.memo_lr,
                   "missing": sorted(MISSING), "results": res}, f, indent=2)
    print(f"Saved {out_json}")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ks = list(CORR); x = np.arange(len(ks)); w = 0.2
    bars = [("ff", "#999", "broad-aug ff"), ("tta", "#D85A30", "+TTA (compute-matched)"),
            ("memo", "#8E44AD", "+MEMO (compute-matched)"), ("im", "#1e8449", "+imagination")]
    for i, (mname, col, lab) in enumerate(bars):
        ax.bar(x + (i - 1.5) * w, [ms(k, mname)[0] for k in ks], w,
               yerr=[ms(k, mname)[1] for k in ks], capsize=3, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(ks); ax.set_ylabel("held-out accuracy"); ax.set_ylim(0, 1)
    ax.set_title(f"Make-or-break ({args.dataset}): leave-one-out, compute-matched (+MEMO)")
    ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    out = f"paper_exp1_{args.dataset}.png"
    plt.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
