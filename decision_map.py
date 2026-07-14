"""
decision_map.py -- the paper's headline Figure 1.
=================================================
Consumes the JSON written by paper_exp1.py / paper_exp1_cifar.py and renders the unified
DECISION MAP: for each (dataset x corruption), which test-time strategy wins at equal compute,
organized along the reconstructability axis  reconstructability = structure x redundancy.

  - structure  (rows)    : how reconstructable the corruption's geometry is
                           band (contiguous) > patch > scattered > noise (none)
  - redundancy (columns) : how much the data lets you fill a gap from context
                           MNIST (sparse strokes) < Fashion (simple texture) < CIFAR-10 (natural)

Two panels:
  A. grid of imagination's MARGIN over the strongest compute-matched baseline
     (best of broad-aug ff / TTA / MEMO). Green = generative completion wins; a ring marks a
     win with separated error bars; red = augmentation wins (incl. the noise negative control).
  B. that same margin vs data redundancy, one line per corruption -- the unifying view: the
     scattered corruption FLIPS from a loss (sparse MNIST) to the biggest win (redundant CIFAR),
     while noise stays a loss everywhere. That flip is the whole rule in one picture.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# datasets ordered by data redundancy (low -> high)
DATASETS = [("paper_exp1_mnist.json", "MNIST"),
            ("paper_exp1_fashion.json", "Fashion"),
            ("paper_exp1_cifar.json", "CIFAR-10")]
# corruptions ordered by structure / contiguity (high -> low), noise last (no structure)
CORRS = ["band", "patch", "scattered", "noise"]
BASELINES = ["ff", "tta", "memo"]
WIN_EPS = 0.005          # margin a win must clear, matching the experiment scripts


def load():
    cols = []
    for fn, label in DATASETS:
        if not os.path.exists(fn):
            print(f"  (missing {fn} -- skipping {label})")
            continue
        with open(fn) as f:
            cols.append((label, json.load(f)))
    return cols


def stats(res, corr):
    """Return (imag_mean, imag_std, best_baseline_mean, best_baseline_name)."""
    im = np.array(res[corr]["im"]); im_m, im_s = im.mean(), im.std()
    bvals = {b: np.array(res[corr][b]).mean() for b in BASELINES}
    bname = max(bvals, key=bvals.get)
    return im_m, im_s, bvals[bname], bname


def main():
    cols = load()
    if not cols:
        print("No result JSON found. Run paper_exp1.py / paper_exp1_cifar.py first.")
        return
    labels = [c[0] for c in cols]
    nC, nD = len(CORRS), len(cols)

    margin = np.full((nC, nD), np.nan)      # imag - best baseline
    won = np.zeros((nC, nD), bool)          # separated-error-bar win
    imag_s = np.zeros((nC, nD))
    best_name = [[""] * nD for _ in range(nC)]
    for j, (_, blob) in enumerate(cols):
        res = blob["results"]
        for i, corr in enumerate(CORRS):
            if corr not in res:
                continue
            im_m, im_sd, b_m, b_nm = stats(res, corr)
            margin[i, j] = im_m - b_m
            imag_s[i, j] = im_sd
            best_name[i][j] = b_nm
            won[i, j] = (im_m - im_sd) > (b_m + WIN_EPS)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- Panel A: the decision-map grid ----
    vlim = np.nanmax(np.abs(margin)) if np.isfinite(margin).any() else 0.1
    im = axA.imshow(margin, cmap="RdYlGn", vmin=-vlim, vmax=vlim, aspect="auto")
    axA.set_xticks(range(nD)); axA.set_xticklabels(labels)
    axA.set_yticks(range(nC)); axA.set_yticklabels(CORRS)
    axA.set_xlabel("data redundancy  -->", fontsize=10)
    axA.set_ylabel("<--  corruption structure", fontsize=10)
    axA.set_title("Decision map: imagination margin over the\nstrongest compute-matched baseline",
                  fontsize=11)
    for i in range(nC):
        for j in range(nD):
            if not np.isfinite(margin[i, j]):
                continue
            axA.text(j, i - 0.13, f"{margin[i, j]:+.3f}", ha="center", va="center",
                     fontsize=10, fontweight="bold")
            axA.text(j, i + 0.20, f"vs {best_name[i][j]}", ha="center", va="center", fontsize=7.5)
            if won[i, j]:
                axA.add_patch(Circle((j, i), 0.42, fill=False, ec="#0b3d0b", lw=2.2))
    cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04)
    cb.set_label("imag - best baseline  (green = completion wins)", fontsize=8)
    axA.text(0.5, -0.17, "ring = win with separated error bars",
             transform=axA.transAxes, ha="center", fontsize=8, style="italic")

    # ---- Panel B: margin vs redundancy, one line per corruption ----
    xs = np.arange(nD)
    colmap = {"band": "#1e6f1e", "patch": "#3a86c8", "scattered": "#e07b00", "noise": "#999999"}
    for i, corr in enumerate(CORRS):
        ys = margin[i, :]
        axB.errorbar(xs, ys, yerr=imag_s[i, :], fmt="o-", lw=2, capsize=3,
                     color=colmap[corr], label=corr)
    axB.axhline(0, color="k", lw=0.9)
    axB.set_xticks(xs); axB.set_xticklabels(labels)
    axB.set_xlabel("data redundancy  -->", fontsize=10)
    axB.set_ylabel("imagination margin over best baseline", fontsize=10)
    axB.set_title("The rule in one view: reconstructability =\nstructure x redundancy",
                  fontsize=11)
    axB.legend(fontsize=9, title="corruption"); axB.grid(alpha=0.3)

    fig.suptitle("When does test-time generative completion beat compute-matched augmentation?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    import os; os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/decision_map.png", dpi=130)
    print("Saved figures/decision_map.png")

    # text summary
    print("\n=== Decision map (imag margin over strongest of ff/tta/memo) ===")
    print("corruption".ljust(12) + "".join(f"{l:>14}" for l in labels))
    for i, corr in enumerate(CORRS):
        row = corr.ljust(12)
        for j in range(nD):
            if np.isfinite(margin[i, j]):
                mark = "*" if won[i, j] else " "
                row += f"{margin[i, j]:+.3f}{mark}".rjust(14)
            else:
                row += "n/a".rjust(14)
        print(row)
    print("\n* = win with separated error bars (imag mean - std > best baseline + 0.005)")


if __name__ == "__main__":
    main()
