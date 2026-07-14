"""
results_table.py -- emit the paper's main results table (markdown) from the run JSONs,
and a per-cell verdict, so PAPER.md can be populated without hand-transcription.
"""
import json
import os

import numpy as np

DATASETS = [("paper_exp1_mnist.json", "MNIST"),
            ("paper_exp1_fashion.json", "Fashion"),
            ("paper_exp1_cifar.json", "CIFAR-10")]
CORRS = ["band", "patch", "scattered", "noise"]
BASELINES = ["ff", "tta", "memo"]
WIN_EPS = 0.005


def ms(v):
    v = np.array(v)
    return v.mean(), v.std()


def cell(v):
    m, s = ms(v)
    return f"{m:.3f}±{s:.3f}"


def main():
    lines = ["| dataset (seeds) | corruption | broad-aug ff | +TTA | +MEMO | +imagination | verdict |",
             "|---|---|---|---|---|---|---|"]
    summary = []
    for fn, label in DATASETS:
        if not os.path.exists(fn):
            continue
        blob = json.load(open(fn))
        res = blob["results"]
        seeds = blob.get("seeds", "?")
        missing = set(blob.get("missing", ["band", "patch", "scattered"]))
        for corr in CORRS:
            if corr not in res:
                continue
            ff_m = ms(res[corr]["ff"])[0]
            tta_m = ms(res[corr]["tta"])[0]
            memo_m = ms(res[corr]["memo"])[0] if "memo" in res[corr] else float("nan")
            im_m, im_s = ms(res[corr]["im"])
            base_best = np.nanmax([ff_m, tta_m, memo_m])
            won = (corr in missing) and (im_m - im_s > base_best + WIN_EPS)
            if corr not in missing:
                verdict = "**no win** (neg. control) [ok]" if im_m <= base_best else "spurious -- check"
            else:
                verdict = "**imag wins**" if won else "tie/loss"
            memo_str = cell(res[corr]["memo"]) if "memo" in res[corr] else "--"
            lines.append(f"| {label} ({seeds}) | {corr} | {cell(res[corr]['ff'])} | "
                         f"{cell(res[corr]['tta'])} | {memo_str} | **{cell(res[corr]['im'])}** | {verdict} |")
            summary.append((label, corr, won, corr in missing, im_m - base_best))

    wins = [(l, c, marg) for (l, c, w, miss, marg) in summary if w]
    negctrl_ok = all(im_marg <= 0 for (l, c, w, miss, im_marg) in summary if c == "noise")
    lines += ["", "### one-line rule outcome",
              f"- imagination wins ({len(wins)} cells): "
              + ", ".join(f"{l}/{c} (+{m:.3f})" for l, c, m in wins),
              f"- noise negative control holds everywhere: {negctrl_ok}"]

    with open("results_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Saved results_table.md")
    # ASCII-safe console echo
    print("\n".join(lines).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
