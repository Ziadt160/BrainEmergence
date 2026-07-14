"""
energy.py -- does the spiking brain actually win on energy? (the SNN-native scoreboard)
======================================================================================
We stop comparing on accuracy (the spiking net loses) and measure the axis where a spiking net
can legitimately win: ENERGY. Standard SNN accounting (45nm CMOS figures used across the SNN
literature): a dense multiply-accumulate costs E_MAC=4.6 pJ; a spike-driven accumulate costs
E_AC=0.9 pJ. ANNs pay MACs everywhere; SNNs pay MACs only for non-spiking layers and pay
event-driven SOPs (spikes x fan-out) for the spiking (recurrent) part.

We report, per image: accuracy, energy (nJ), and accuracy-per-energy -- and the firing rate, which
decides whether the spiking brain is actually SPARSE (efficient) or dense-and-always-on (not).
NOTE: this is an analytic estimate (no neuromorphic hardware), exactly as SNN papers report it.
"""
import argparse
import os

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
from ablation import SimpleCNN
from gen_classifier import Disc as MLP        # 784->256->128->10, the fair small-ANN baseline

E_MAC = 4.6e-12   # J per multiply-accumulate (45nm)
E_AC = 0.9e-12    # J per accumulate / spike-driven op (45nm)


def train_spiking(seed, args, loader):
    torch.manual_seed(seed)
    model = RecallBrain(args.grid, args.grid, args.t_steps, movable=False).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    ce = nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = ce(model.cls(model.encode(x)), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


def train_ann(make, seed, args, loader):
    torch.manual_seed(seed + 4242)
    model = make().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    for ep in range(args.epochs_cnn):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = ce(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def cnn_macs(model, x):
    macs = [0]; hooks = []
    def conv_hook(m, inp, out):
        oh, ow = out.shape[2], out.shape[3]
        macs[0] += oh * ow * m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
    def lin_hook(m, inp, out):
        macs[0] += m.in_features * m.out_features
    for m in model.modules():
        if isinstance(m, nn.Conv2d): hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear): hooks.append(m.register_forward_hook(lin_hook))
    with torch.no_grad(): model(x[:1])
    for h in hooks: h.remove()
    return macs[0]


@torch.no_grad()
def eval_cnn(model, loader):
    model.eval(); tot = c = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE); tot += y.numel()
        c += (model(x).argmax(1) == y).sum().item()
    return c / tot


@torch.no_grad()
def eval_spiking(model, loader):
    model.eval(); tot = c = 0; spk_sum = 0.0
    N, T = model.N, model.T
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE); tot += y.numel()
        A = model.encode(x)                      # firing RATE = acc/T, shape (B, N)
        c += (model.cls(A).argmax(1) == y).sum().item()
        spk_sum += (A.sum(1) * T).sum().item()   # total spikes (all neurons, all T), summed over batch
    avg_spikes = spk_sum / tot                   # avg total spikes per image
    sops_rec = avg_spikes * N                    # dense recurrent fan-out: each spike -> N post-neurons
    macs_static = 784 * N + N * 10               # w_in (input is continuous -> MAC) + readout
    energy = macs_static * E_MAC + sops_rec * E_AC
    firing_rate = avg_spikes / (N * T)           # spikes per neuron per timestep
    return c / tot, energy, avg_spikes, firing_rate, sops_rec, macs_static


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--t-steps", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--epochs-cnn", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.n_test, args.epochs, args.epochs_cnn, args.t_steps, args.seeds = 3000, 1000, 2, 2, 12, 1
    print(f"Device {DEVICE}  ENERGY: spiking brain vs CNN (E_MAC={E_MAC*1e12}pJ, E_AC={E_AC*1e12}pJ)")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    trf = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    tef = datasets.MNIST("./data", train=False, transform=tfm)
    g = torch.Generator().manual_seed(0)
    tr = Subset(trf, torch.randperm(len(trf), generator=g)[:args.n_train].tolist())
    te = Subset(tef, torch.randperm(len(tef), generator=g)[:args.n_test].tolist())
    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(te, batch_size=256)

    sp_acc, sp_en, frates = [], [], []
    cn_acc, cn_en, ml_acc, ml_en = [], [], [], []
    macs_cnn = macs_mlp = None
    for s in range(args.seeds):
        sp = train_spiking(s, args, train_loader)
        cn = train_ann(SimpleCNN, s, args, train_loader)
        ml = train_ann(MLP, s, args, train_loader)
        a, e, spikes, fr, sops, macs_static = eval_spiking(sp, test_loader)
        sp_acc.append(a); sp_en.append(e); frates.append(fr)
        xb = next(iter(test_loader))[0].to(DEVICE)
        ca = eval_cnn(cn, test_loader); macs_cnn = cnn_macs(cn, xb)
        cn_acc.append(ca); cn_en.append(macs_cnn * E_MAC)
        ma = eval_cnn(ml, test_loader); macs_mlp = cnn_macs(ml, xb)
        ml_acc.append(ma); ml_en.append(macs_mlp * E_MAC)
        print(f"  seed {s}: spk {a:.3f}/{e*1e9:.0f}nJ (fr {fr*100:.0f}%)  "
              f"mlp {ma:.3f}/{macs_mlp*E_MAC*1e9:.0f}nJ  cnn {ca:.3f}/{macs_cnn*E_MAC*1e9:.0f}nJ")

    rows = [("spiking brain", np.mean(sp_acc), np.mean(sp_en)),
            ("MLP (small ANN)", np.mean(ml_acc), np.mean(ml_en)),
            ("CNN", np.mean(cn_acc), np.mean(cn_en))]
    fr = np.mean(frates)
    print("\n=== ENERGY / ACCURACY (per image, MNIST, mean of seeds) ===")
    print(f"{'model':>16} {'accuracy':>10} {'energy(nJ)':>12} {'acc per uJ':>12}")
    for name, a, e in rows:
        print(f"{name:>16} {a:>10.3f} {e*1e9:>12.1f} {a/(e*1e6):>12.1f}")
    print(f"\n  spiking firing rate: {fr*100:.1f}% (spikes/neuron/timestep)  [<10% = sparse/efficient]")
    print(f"  MACs/image -- CNN: {macs_cnn:,}   MLP: {macs_mlp:,}")

    spa, spe = np.mean(sp_acc), np.mean(sp_en)
    mla, mle = np.mean(ml_acc), np.mean(ml_en)
    cna, cne = np.mean(cn_acc), np.mean(cn_en)
    print("\n--- VERDICT (fair: compare to the SMALL MLP, not just the big CNN) ---")
    print(f"  spiking vs CNN: {cne/spe:.1f}x less energy (spiking wins -- conv is MAC-heavy)")
    print(f"  spiking vs MLP: {mle/spe:.2f}x   (>1 = spiking cheaper than the small MLP)")
    if mle < spe and mla >= spa - 0.01:
        print(f"  => MLP DOMINATES: a plain MLP gives >= accuracy ({mla:.3f} vs {spa:.3f}) at LOWER energy "
              f"({mle*1e9:.0f} vs {spe*1e9:.0f} nJ). On this analytic model the spiking brain is NOT "
              f"Pareto-optimal; the recurrent loop is overhead. The spiking energy win is only vs conv nets.")
        print(f"  => Honest path: the spiking advantage needs (a) neuromorphic hardware (not this analytic "
              f"45nm model), AND (b) far sparser firing (now {fr*100:.0f}%, want <10%). Both are real levers.")
    else:
        print(f"  => Spiking is competitive even vs the small MLP -- a genuine energy result to build on.")

    # ---- accuracy-vs-energy scoreboard figure (up-and-left = better) ----
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    pts = [("spiking brain", spa, spe * 1e9, "#378ADD"),
           ("MLP (small ANN)", mla, mle * 1e9, "#1e8449"),
           ("CNN", cna, cne * 1e9, "#D85A30")]
    for name, a, e, col in pts:
        ax.scatter(e, a, s=170, color=col, edgecolor="#111", linewidth=1.2, zorder=3)
        ax.annotate(name, (e, a), xytext=(9, -4), textcoords="offset points", fontsize=10, color=col)
    ax.set_xscale("log")
    ax.set_xlabel("energy per image  (nJ, log scale)  —  cheaper →", fontsize=10)
    ax.set_ylabel("accuracy  —  better ↑", fontsize=10)
    ax.set_title("Accuracy vs energy (MNIST, analytic 45nm)\nthe MLP is Pareto-optimal; spiking only beats the conv net")
    ax.grid(alpha=0.3, which="both")
    ax.annotate("", xy=(0.06, 0.94), xytext=(0.24, 0.78), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#888"))
    ax.text(0.03, 0.965, "better", transform=ax.transAxes, color="#888", fontsize=9)
    plt.tight_layout(); plt.savefig("figures/energy_scoreboard.png", dpi=130)
    print("Saved figures/energy_scoreboard.png")


if __name__ == "__main__":
    main()
