"""
gating.py -- the leave-corruption-out decision-rule test (the last open slice).
==============================================================================
The paper died because occlusion AUGMENTATION beat imagination on occlusion. The
literature (DDA, RDC) says generative completion wins only on corruptions you did
NOT / could NOT augment for. So the one honest open question: train augmentation
ONLY for band-occlusion, then test on DIFFERENT corruptions and ask whether
imagination beats the augmentation-trained model where its augmentation does not
transfer.

For each model in {clean-trained, band-occlusion-augmented} and each test corruption
in {band-occlusion (in-aug), random-patch, scattered-dropout, Gaussian-noise}, report
feedforward vs +imagination accuracy. The decision rule is VIABLE only if there is a
corruption where (clean or aug) + imagination beats the augmentation-trained
feedforward at equal inference compute. Otherwise: always augment -> no paper.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from imagine_helps import RecallBrain, occlude, DEVICE
from paper_baseline import train_model


def c_band(x, band=8):
    return occlude(x, band)                                  # (x_occ, mask)

def c_patch(x, size=14):
    B = x.shape[0]; img = x.view(B, 28, 28).clone(); mask = torch.ones_like(img)
    r = torch.randint(0, 28 - size + 1, (B,)); c = torch.randint(0, 28 - size + 1, (B,))
    for i in range(B):
        img[i, r[i]:r[i] + size, c[i]:c[i] + size] = 0.0
        mask[i, r[i]:r[i] + size, c[i]:c[i] + size] = 0.0
    return img.view(B, -1), mask.view(B, -1)

def c_scattered(x, frac=0.30):
    m = (torch.rand(x.shape[0], 784, device=x.device) > frac).float()
    return x * m, m

def c_noise(x, std=0.4):
    return (x + std * torch.randn_like(x)).clamp(0, 1), None


CORRUPTIONS = {"band-occ (in-aug)": c_band, "random-patch": c_patch,
               "scattered-drop": c_scattered, "gaussian-noise": c_noise}


@torch.no_grad()
def eval_corr(model, loader, corr_fn, rounds, beta=0.5):
    model.eval(); n = ff = im = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE); n += y.numel()
        x_c, mask = corr_fn(x)
        ff += (model(x_c)[0].argmax(1) == y).sum().item()
        x_t = x_c.clone()
        for _ in range(rounds):
            _, recon = model(x_t)
            x_t = (mask * x_c + (1 - mask) * torch.sigmoid(recon)) if mask is not None \
                else (beta * x_c + (1 - beta) * torch.sigmoid(recon))
        im += (model(x_t)[0].argmax(1) == y).sum().item()
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
    print(f"Device {DEVICE}  leave-corruption-out gating test")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    trf = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    tef = datasets.MNIST("./data", train=False, transform=tfm)
    g = torch.Generator().manual_seed(0)
    tr = Subset(trf, torch.randperm(len(trf), generator=g)[:args.n_train].tolist())
    te = Subset(tef, torch.randperm(len(tef), generator=g)[:args.n_test].tolist())
    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(te, batch_size=256)

    corr = list(CORRUPTIONS)
    # results[model][corruption] = {ff:[...seeds], im:[...]}
    R = {m: {c: {"ff": [], "im": []} for c in corr} for m in ["clean", "aug"]}
    for s in range(args.seeds):
        for mtag, aug in [("clean", False), ("aug", True)]:
            model = train_model(s, aug, args, train_loader)
            for c in corr:
                ff, im = eval_corr(model, test_loader, CORRUPTIONS[c], args.rounds)
                R[mtag][c]["ff"].append(ff); R[mtag][c]["im"].append(im)
            print(f"  seed {s} {mtag} done")

    def m(model, c, k): return float(np.mean(R[model][c][k]))
    print("\n=== Accuracy by (model, corruption): feedforward / +imagination ===")
    hdr = "corruption".ljust(20) + "  " + "  ".join(f"{c:>22}" for c in ["clean", "aug"])
    print("                          clean-trained        aug-trained")
    print(f"{'corruption':>18}  {'ff':>7} {'+imag':>7}   {'ff':>7} {'+imag':>7}   {'WEDGE?':>8}")
    wedge_found = False
    for c in corr:
        cf, ci, af, ai = m("clean", c, "ff"), m("clean", c, "im"), m("aug", c, "ff"), m("aug", c, "im")
        best_imag = max(ci, ai)            # best imagination-using option
        best_aug = af                      # "just augment" feedforward
        wins = best_imag > best_aug + 0.01
        wedge_found = wedge_found or wins
        print(f"{c:>18}  {cf:>7.3f} {ci:>7.3f}   {af:>7.3f} {ai:>7.3f}   {'YES' if wins else 'no':>8}")
    print("\nWEDGE? = does any imagination option beat the augmentation-trained feedforward "
          "on that corruption (at equal compute)?")
    if wedge_found:
        print("VERDICT: a corruption exists where imagination beats 'just augment' -> the decision "
              "rule has ONE live cell; pursue the scoping/workshop framing on that corruption.")
    else:
        print("VERDICT: augmentation-trained feedforward wins (or ties) every corruption -> the "
              "decision rule collapses to 'always augment'. No paper. Closure confirmed.")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(corr)); w = 0.2
    ax.bar(x - 1.5 * w, [m("clean", c, "ff") for c in corr], w, label="clean ff", color="#bbb")
    ax.bar(x - 0.5 * w, [m("clean", c, "im") for c in corr], w, label="clean +imag", color="#378ADD")
    ax.bar(x + 0.5 * w, [m("aug", c, "ff") for c in corr], w, label="aug ff (baseline)", color="#777")
    ax.bar(x + 1.5 * w, [m("aug", c, "im") for c in corr], w, label="aug +imag", color="#1e8449")
    ax.set_xticks(x); ax.set_xticklabels(corr, rotation=12, fontsize=8)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Leave-corruption-out: does imagination ever beat 'just augment'?")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig("gating.png", dpi=120)
    print("Saved gating.png")


if __name__ == "__main__":
    main()
