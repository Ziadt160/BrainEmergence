"""
paper_exp1_cifar.py -- the CIFAR-10 make-or-break (natural images).
==================================================================
Same leave-one-corruption-out + compute-matched protocol as paper_exp1.py, but with the
convolutional recognize+imagine net (ConvBrain) on real color images. This is the test that
decides whether the decision rule is general or MNIST-family-only.

Rule under test: imagination beats broad-augmentation feedforward AND compute-matched
TTA-marginalization AND MEMO (test-time entropy minimization) on held-out reconstructable
missing-data; does NOT win on noise (negative control). MEMO's test-time augmentations are
the broad-aug corruption set itself (its fullest test-time form).
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

from cifar_imagine import ConvBrain, DEVICE


def c_band(x, band=8):
    B, _, H, W = x.shape; r0 = H // 2 - band // 2
    m = torch.ones(B, 1, H, W, device=x.device); m[:, :, r0:r0 + band, :] = 0
    return x * m, m

def c_patch(x, size=14):
    B, _, H, W = x.shape; m = torch.ones(B, 1, H, W, device=x.device)
    r = torch.randint(0, H - size + 1, (B,)); c = torch.randint(0, W - size + 1, (B,))
    for i in range(B):
        m[i, :, r[i]:r[i] + size, c[i]:c[i] + size] = 0
    return x * m, m

def c_scattered(x, frac=0.30):
    B, _, H, W = x.shape
    m = (torch.rand(B, 1, H, W, device=x.device) > frac).float()
    return x * m, m

def c_noise(x, std=0.2):
    return (x + std * torch.randn_like(x)).clamp(0, 1), None

CORR = {"band": c_band, "patch": c_patch, "scattered": c_scattered, "noise": c_noise}
MISSING = {"band", "patch", "scattered"}


def train_broadaug(seed, heldout, args, loader):
    torch.manual_seed(seed)
    model = ConvBrain().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    aug = [k for k in CORR if k != heldout]
    for ep in range(args.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            k = aug[torch.randint(len(aug), (1,)).item()]
            x_in = CORR[k](x)[0]
            logits, recon = model(x_in)
            loss = ce(logits, y) + args.recon_w * F.binary_cross_entropy_with_logits(recon, x)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def tta(model, x, n):
    probs = 0
    for _ in range(n):
        dh, dw = int(torch.randint(-3, 4, (1,))), int(torch.randint(-3, 4, (1,)))
        v = torch.roll(x, shifts=(dh, dw), dims=(2, 3))
        probs = probs + F.softmax(model(v)[0], dim=1)
    return (probs / n).argmax(1)


@torch.no_grad()
def imagine(model, x_c, mask, n, beta=0.5):
    x_t = x_c.clone()
    for _ in range(n):
        _, recon = model(x_t)
        x_t = (mask * x_c + (1 - mask) * torch.sigmoid(recon)) if mask is not None \
            else (beta * x_c + (1 - beta) * torch.sigmoid(recon))
    return model(x_t)[0].argmax(1)


def memo_predict(model, base_state, x_c, heldout, n, steps, lr):
    """MEMO/Tent: episodically adapt on this batch by minimizing the entropy of the prediction
    marginalized over n augmentations (broad-aug corruption set applied to the observed x_c),
    then predict. Compute-matched to imagination. Restored to base_state after each batch.
    We adapt only the BatchNorm AFFINE parameters (the standard stable TTA parameterization);
    adapting all conv/linear weights collapses to a trivial single-class solution under
    entropy-min. Model stays in eval() so BN normalizes with the trained running statistics."""
    aug = [k for k in CORR if k != heldout]
    model.load_state_dict(base_state)
    bn_params = [p for m in model.modules() if isinstance(m, nn.BatchNorm2d)
                 for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(bn_params, lr=lr)
    for _ in range(steps):
        probs = 0
        for _ in range(n):
            k = aug[torch.randint(len(aug), (1,)).item()]
            v = CORR[k](x_c)[0]
            probs = probs + F.softmax(model(v)[0], dim=1)
        p = probs / n
        ent = -(p * torch.log(p + 1e-6)).sum(1).mean()
        opt.zero_grad(); ent.backward()
        nn.utils.clip_grad_norm_(bn_params, 1.0); opt.step()
    with torch.no_grad():
        pred = model(x_c)[0].argmax(1)
    model.load_state_dict(base_state)
    return pred


def eval_heldout(model, loader, heldout, n, memo_steps, memo_lr):
    model.eval(); tot = ff = t = memo = im = 0
    base_state = copy.deepcopy(model.state_dict())
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE); tot += y.numel()
        x_c, mask = CORR[heldout](x)
        with torch.no_grad():
            ff += (model(x_c)[0].argmax(1) == y).sum().item()
            t += (tta(model, x_c, n) == y).sum().item()
            im += (imagine(model, x_c, mask, n) == y).sum().item()
        memo += (memo_predict(model, base_state, x_c, heldout, n, memo_steps, memo_lr) == y).sum().item()
    return ff / tot, t / tot, memo / tot, im / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--recon-w", type=float, default=5.0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--memo-steps", type=int, default=3)
    ap.add_argument("--memo-lr", type=float, default=5e-4)  # strongest stable BN-affine adaptation
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.n_test, args.epochs, args.seeds = 3000, 1000, 3, 1
    print(f"Device {DEVICE}  CIFAR-10 leave-one-out, compute-matched (N={args.rounds})")

    tfm = transforms.ToTensor()
    trf = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
    tef = datasets.CIFAR10("./data", train=False, transform=tfm)
    g = torch.Generator().manual_seed(0)
    tr = Subset(trf, torch.randperm(len(trf), generator=g)[:args.n_train].tolist())
    te = Subset(tef, torch.randperm(len(tef), generator=g)[:args.n_test].tolist())
    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(te, batch_size=256)

    res = {k: {"ff": [], "tta": [], "memo": [], "im": []} for k in CORR}
    for s in range(args.seeds):
        for heldout in CORR:
            model = train_broadaug(s, heldout, args, train_loader)
            ff, t, memo, im = eval_heldout(model, test_loader, heldout, args.rounds,
                                           args.memo_steps, args.memo_lr)
            res[heldout]["ff"].append(ff); res[heldout]["tta"].append(t)
            res[heldout]["memo"].append(memo); res[heldout]["im"].append(im)
        print(f"  seed {s} done")

    def ms(k, m): v = np.array(res[k][m]); return v.mean(), v.std()
    print("\n=== Held-out | broad-aug ff | +TTA(N) | +MEMO(N) | +imagination(N)  [mean+/-std] (CIFAR-10) ===")
    live = False
    for k in CORR:
        ff, t, memo, im = ms(k, "ff"), ms(k, "tta"), ms(k, "memo"), ms(k, "im")
        beats = (k in MISSING) and (im[0] - im[1] > max(ff[0], t[0], memo[0]) + 0.005)
        live = live or beats
        flag = "  <-- IMAG WINS" if beats else ("" if k in MISSING else "  (neg control)")
        print(f"  {k:>10}: ff {ff[0]:.3f}+/-{ff[1]:.3f}  tta {t[0]:.3f}+/-{t[1]:.3f}  "
              f"memo {memo[0]:.3f}+/-{memo[1]:.3f}  imag {im[0]:.3f}+/-{im[1]:.3f}{flag}")
    print("\n--- VERDICT (CIFAR) ---")
    print("RULE GENERALIZES to natural images." if live else
          "RULE FAILS on natural images -> stays MNIST-family-only.")

    with open("paper_exp1_cifar.json", "w") as f:
        json.dump({"dataset": "cifar10", "seeds": args.seeds, "rounds": args.rounds,
                   "memo_steps": args.memo_steps, "memo_lr": args.memo_lr,
                   "missing": sorted(MISSING), "results": res}, f, indent=2)
    print("Saved paper_exp1_cifar.json")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ks = list(CORR); x = np.arange(len(ks)); w = 0.2
    for i, (mn, col, lab) in enumerate([("ff", "#999", "broad-aug ff"),
                                        ("tta", "#D85A30", "+TTA (compute-matched)"),
                                        ("memo", "#8E44AD", "+MEMO (compute-matched)"),
                                        ("im", "#1e8449", "+imagination")]):
        ax.bar(x + (i - 1.5) * w, [ms(k, mn)[0] for k in ks], w,
               yerr=[ms(k, mn)[1] for k in ks], capsize=3, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(ks); ax.set_ylabel("held-out CIFAR-10 accuracy"); ax.set_ylim(0, 1)
    ax.set_title("CIFAR-10 make-or-break: leave-one-out, compute-matched (+MEMO)")
    ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig("paper_exp1_cifar.png", dpi=120)
    print("Saved paper_exp1_cifar.png")


if __name__ == "__main__":
    main()
