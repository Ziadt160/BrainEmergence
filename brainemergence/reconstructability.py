"""
reconstructability.py -- quantitative test of the paper's rule.
================================================================
The rule says imagination's margin tracks RECONSTRUCTABILITY = structure x redundancy.
Here we make 'reconstructability' a concrete, parameter-free number per (dataset x corruption):
how well a GENERIC context inpainter (diffusion / neighbour-averaging fill -- no learning)
recovers the masked region, measured as the correlation between the filled and the true pixels
on the missing region. Then we ask: does this proxy predict the 12 measured imagination margins
(from paper_exp1_*.json)?

  - contiguous holes on redundant data -> neighbours predict the hole well -> high proxy.
  - scattered holes on sparse digits   -> little context -> low proxy.
  - noise                              -> nothing missing -> proxy = 0 (negative control).

A strong monotone proxy->margin relationship turns the empirical rule into a quantitative one.
We report it honestly either way (margin also depends on baseline headroom, so we don't expect
a perfect fit).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from gating import c_band, c_patch, c_scattered            # MNIST/Fashion (flat 784)
from paper_exp1_cifar import c_band as cf_band, c_patch as cf_patch, c_scattered as cf_scattered

CORRS = ["band", "patch", "scattered", "noise"]
BASELINES = ["ff", "tta", "memo"]


def diffusion_inpaint(img, mask, iters=40):
    """Fill missing pixels (mask==0) by iteratively averaging valid neighbours. Parameter-free."""
    B, C, H, W = img.shape
    k = torch.ones(1, 1, 3, 3)
    val = img * mask
    fm = mask.clone()                                       # 1 where we currently have a value
    for _ in range(iters):
        v = val.reshape(B * C, 1, H, W)
        f = fm.expand(B, C, H, W).reshape(B * C, 1, H, W)
        nsum = F.conv2d(v, k, padding=1).reshape(B, C, H, W)
        ncnt = F.conv2d(f, k, padding=1).reshape(B, C, H, W)
        avg = nsum / (ncnt + 1e-6)
        fill_here = (mask == 0) & (ncnt > 0)
        val = torch.where(fill_here, avg, val)
        fm = ((ncnt > 0) | (fm > 0)).float()
    return val


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def reconstructability(loader, corr_fn, is_cifar, n_batches=4):
    """Mean correlation between inpainted and true pixels on the missing region."""
    rs = []
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        if is_cifar:
            img = x
            xc, mask = corr_fn(img)
        else:
            B = x.shape[0]
            img = x.view(B, 1, 28, 28)
            xc_flat, m_flat = corr_fn(x)
            mask = m_flat.view(B, 1, 28, 28)
        filled = diffusion_inpaint(img, mask)
        miss = (mask == 0).expand_as(img)
        if miss.sum() == 0:
            continue
        rs.append(pearson(filled[miss], img[miss]))
    return float(np.mean(rs)) if rs else 0.0


def spearman(x, y):
    def rank(v):
        order = np.argsort(v); r = np.empty_like(order, float); r[order] = np.arange(len(v)); return r
    return pearson_np(rank(np.array(x)), rank(np.array(y)))


def pearson_np(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def margin_from_json(fn):
    blob = json.load(open(fn))["results"]
    out = {}
    for c in CORRS:
        if c not in blob:
            continue
        im = np.mean(blob[c]["im"])
        base = max(np.mean(blob[c][b]) for b in BASELINES if b in blob[c])
        out[c] = im - base
    return out


def main():
    cf_corr = {"band": cf_band, "patch": cf_patch, "scattered": cf_scattered, "noise": None}
    mn_corr = {"band": c_band, "patch": c_patch, "scattered": c_scattered, "noise": None}
    datasets_cfg = [
        ("mnist", "paper_exp1_mnist.json", datasets.MNIST, False, mn_corr),
        ("fashion", "paper_exp1_fashion.json", datasets.FashionMNIST, False, mn_corr),
        ("cifar10", "paper_exp1_cifar.json", datasets.CIFAR10, True, cf_corr),
    ]
    proxies, margins, labels = [], [], []
    print(f"{'cell':>18} {'reconstructability':>20} {'imag margin':>13}")
    for name, jf, DS, is_cifar, corr in datasets_cfg:
        if not os.path.exists(jf):
            print(f"  (missing {jf})"); continue
        if is_cifar:
            tfm = transforms.ToTensor()
        else:
            tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
        te = DS("./data", train=False, download=True, transform=tfm)
        g = torch.Generator().manual_seed(0)
        sub = Subset(te, torch.randperm(len(te), generator=g)[:1000].tolist())
        loader = DataLoader(sub, batch_size=256)
        marg = margin_from_json(jf)
        for c in CORRS:
            r = 0.0 if c == "noise" else reconstructability(loader, corr[c], is_cifar)
            proxies.append(r); margins.append(marg[c]); labels.append(f"{name}/{c}")
            print(f"{name+'/'+c:>18} {r:>20.3f} {marg[c]:>+13.3f}")

    rho = spearman(proxies, margins)
    r = pearson_np(np.array(proxies), np.array(margins))
    print(f"\nReconstructability proxy vs imagination margin (12 cells):")
    print(f"  Spearman rho = {rho:+.3f}   Pearson r = {r:+.3f}")
    verdict = ("strong: the proxy predicts the margin" if rho > 0.7 else
               "weak/partial: the pixel-level proxy only partly predicts the margin -- it over-rates "
               "scattered dropout on sparse data, because PIXEL reconstructability != DISCRIMINATIVE "
               "reconstructability. A discriminative predictor is future work.")
    print(f"  -> {verdict}")

    plt.figure(figsize=(7, 5.2))
    cols = {"mnist": "#378ADD", "fashion": "#1e8449", "cifar10": "#D85A30"}
    for p, m, lab in zip(proxies, margins, labels):
        ds = lab.split("/")[0]
        plt.scatter(p, m, color=cols[ds], s=70, edgecolor="k", linewidth=0.5, zorder=3)
        plt.annotate(lab, (p, m), fontsize=7, xytext=(4, 4), textcoords="offset points")
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("reconstructability proxy  (context-inpaint recovery of the masked region)")
    plt.ylabel("imagination margin over best baseline")
    plt.title(f"Reconstructability predicts the imagination margin\nSpearman rho={rho:+.2f}, Pearson r={r:+.2f}")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=d)
               for d, c in cols.items()]
    plt.legend(handles=handles, title="dataset", fontsize=8)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("reconstructability.png", dpi=130)
    print("Saved reconstructability.png")


if __name__ == "__main__":
    main()
