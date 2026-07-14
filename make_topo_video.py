"""
make_topo_video.py -- neurons self-organize: "fire together, move together", animated.
=======================================================================================
Neurons start at RANDOM 2D positions. During training, a co-firing attraction rule pulls
neurons that respond to similar things toward each other (the user's idea). We snapshot the
positions throughout training and watch a scattered cloud sort itself into smooth ORIENTATION
DOMAINS -- a self-organizing cortical-style map. (Honest note: this is self-organization under
an explicit attraction rule, not spontaneous emergence -- see the project's negative results.)

Output: demo_topo.gif
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from movable_model import MovableSheet
from exp_alignment import grating_batch, concept, DEVICE
from movable_test import response_matrix, local_scatter

BG = "#0c0f14"; FG = "#e8edf2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=400)
    ap.add_argument("--P", type=int, default=20)
    ap.add_argument("--freq", type=float, default=3.0)
    ap.add_argument("--t-steps", type=int, default=14)
    ap.add_argument("--iters", type=int, default=3200)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lam-attract", type=float, default=1.0)
    ap.add_argument("--snap", type=int, default=55)
    args = ap.parse_args()
    torch.manual_seed(0)
    print(f"Device {DEVICE}  building TOPOGRAPHY animation  N={args.N} iters={args.iters}")

    model = MovableSheet(args.N, args.P, t_steps=args.t_steps).to(DEVICE)
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    snaps = [model.npos().detach().cpu().numpy().copy()]
    snap_iters = [0]
    for it in range(args.iters):
        x, th = grating_batch(args.batch, args.P, args.freq, gen)
        c = concept(th)
        A = model.up(x)
        mu, logvar = model.encode(A, c)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        logits, _ = model.generate(z, c)
        recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.size(0)
        kl = torch.clamp((-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(0), min=0.2).sum()
        loss = recon + kl + args.lam_attract * model.attract(A)
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % args.snap == 0:
            snaps.append(model.npos().detach().cpu().numpy().copy()); snap_iters.append(it + 1)
    print(f"  captured {len(snaps)} snapshots")

    # preferred orientation per neuron (fixed colour); keep the neurons that are actually tuned
    R, thetas = response_matrix(model, K=24, reps=16, P=args.P, freq=args.freq, gen=gen)
    ang = 2 * thetas
    vx = (R * np.cos(ang)).sum(1); vy = (R * np.sin(ang)).sum(1)
    pref = 0.5 * np.arctan2(vy, vx)
    sel = np.sqrt(vx ** 2 + vy ** 2) / (R.sum(1) + 1e-6)
    keep = sel > np.quantile(sel, 0.45)                     # the orientation-selective neurons
    snaps_k = [s[keep] for s in snaps]
    pref_k = pref[keep]
    s_final = local_scatter(snaps[-1][keep], pref_k)
    print(f"  final local orientation scatter (selective neurons) {s_final:.3f}  "
          f"[{keep.sum()} of {args.N}]")

    plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG})
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.08)
    ax.set_facecolor(BG); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#2a3340")
    fig.text(0.5, 0.955, "Fire together, move together", ha="center", color=FG,
             fontsize=17, weight="bold")
    fig.text(0.5, 0.905, "orientation-selective neurons self-organise into domains",
             ha="center", color="#9fb0c0", fontsize=10.5)
    cap = fig.text(0.5, 0.02, "", ha="center", color="#7fd4e0", fontsize=11.5, family="monospace")
    lim = np.abs(np.concatenate(snaps_k)).max() * 1.08
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    rgba = plt.cm.twilight((pref_k + np.pi / 2) / np.pi); rgba[:, 3] = 0.92
    sc = ax.scatter(snaps_k[0][:, 0], snaps_k[0][:, 1], c=rgba, s=64, edgecolor="#0c0f14", linewidth=0.4)

    def update(i):
        sc.set_offsets(snaps_k[i])
        cap.set_text(f"iteration {snap_iters[i]:>4}   ·   local scatter "
                     f"{local_scatter(snaps_k[i], pref_k):.2f}   ·   colour = orientation")
        return [sc, cap]

    hold = [len(snaps) - 1] * 12
    order = list(range(len(snaps))) + hold
    import os; os.makedirs("media", exist_ok=True)
    anim = animation.FuncAnimation(fig, lambda k: update(order[k]), frames=len(order),
                                   interval=90, blit=False)
    anim.save("media/demo_topo.gif", writer=animation.PillowWriter(fps=11))
    print(f"Saved media/demo_topo.gif  ({len(order)} frames)")
    update(len(snaps) - 1); fig.savefig("media/demo_topo_poster.png", dpi=130)
    print("Saved media/demo_topo_poster.png")


if __name__ == "__main__":
    main()
