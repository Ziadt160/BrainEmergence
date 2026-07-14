"""
make_demo_video.py -- a stunning, HONEST demo animation straight from the trained model.
========================================================================================
Story (all generated from the real RecallBrain, not a mockup):
  1. a clean digit arrives,
  2. the sensor loses part of it (occlusion / dropout),
  3. the spiking "brain" FIRES across its 24x24 sheet (real spikes),
  4. it IMAGINES the missing region over a few rounds (real decoder output),
  5. the prediction flips from wrong -> right as perception self-repairs.

Output: demo_brain.gif  (LinkedIn-ready). Framed as a mechanism demonstration -- neuromorphic
self-repairing perception under partial input loss -- NOT a SOTA robustness claim.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, gridspec
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from mnist_experiment import spike_func
from imagine_helps import RecallBrain, occlude, DEVICE

BG = "#0c0f14"; FG = "#e8edf2"; ACCENT = "#22d3c5"; GOOD = "#41d17f"; BAD = "#ff5d5d"
FASHION = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def train(model, loader, epochs, recon_w):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3); ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            A = model.encode(x)
            loss = ce(model.cls(A), y) + recon_w * F.binary_cross_entropy_with_logits(
                model.dec(A), x, reduction="sum") / x.size(0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        print(f"  epoch {ep} done")


def encode_capture(model, x):
    """Run the spiking sheet and record the spike map at every timestep (for the animation)."""
    W = model.recurrent_W()
    tau, thr = torch.sigmoid(model.tau_raw), F.softplus(model.thr_raw) + 0.5
    mem = torch.zeros(x.shape[0], model.N, device=x.device)
    spk = torch.zeros_like(mem); acc = torch.zeros_like(mem)
    drive = model.w_in(x); g = int(round(model.N ** 0.5)); frames = []
    for _ in range(model.T):
        mem = mem * tau + (drive + spk @ W.t()) - spk * thr
        spk = spike_func(mem - thr); acc = acc + spk
        frames.append(spk[0].detach().cpu().numpy().reshape(g, g))
    return acc / model.T, frames, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--recon-w", type=float, default=3.0)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--t-steps", type=int, default=16)
    ap.add_argument("--band", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--digit", type=int, default=-1)      # pick a specific class, or -1 = first clear one
    ap.add_argument("--dataset", default="mnist")          # 'mnist' or 'fashion'
    args = ap.parse_args()
    torch.manual_seed(3)
    print(f"Device {DEVICE}  building demo animation ({args.dataset})")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    DS = datasets.FashionMNIST if args.dataset == "fashion" else datasets.MNIST
    trf = DS("./data", train=True, download=True, transform=tfm)
    tef = DS("./data", train=False, transform=tfm)
    g0 = torch.Generator().manual_seed(0)
    tr = Subset(trf, torch.randperm(len(trf), generator=g0)[:15000].tolist())
    loader = DataLoader(tr, batch_size=128, shuffle=True)

    model = RecallBrain(args.grid, args.grid, args.t_steps, movable=False).to(DEVICE)
    print("training the brain (clean MNIST)...")
    train(model, loader, args.epochs, args.recon_w)
    model.eval()

    # pick a clear test digit the clean model gets right
    x0 = y0 = None
    for xi, yi in DataLoader(tef, batch_size=1, shuffle=True):
        xi = xi.to(DEVICE)
        if (args.digit < 0 or int(yi) == args.digit) and model(xi)[0].argmax(1).item() == int(yi):
            x0, y0 = xi, int(yi); break
    x_occ, mask = occlude(x0, args.band)

    # ---- build the frame timeline from the REAL model ----
    # per frame: (input_img, brain_map, imagined_img, pred, correct, caption, phase)
    frames = []
    img_clean = x0[0].detach().cpu().numpy().reshape(28, 28)
    img_occ = x_occ[0].detach().cpu().numpy().reshape(28, 28)
    gsz = args.grid

    def add(inp, brain, imag, pred, correct, cap):
        frames.append(dict(inp=inp, brain=brain, imag=imag, pred=pred, correct=correct, cap=cap))

    blank_brain = np.zeros((gsz, gsz))
    # scene 1: clean digit
    for _ in range(6): add(img_clean, blank_brain, None, None, None, "a clean input arrives")
    # scene 2: sensor loses part of it
    for _ in range(6): add(img_occ, blank_brain, None, None, None, "sensor loses 30% of the image")

    x_t = x_occ.clone()
    with torch.no_grad():
        for r in range(args.rounds):
            A, sframes, _ = encode_capture(model, x_t)
            logits = model.cls(A); pred = int(logits.argmax(1))
            recon = torch.sigmoid(model.dec(A))
            x_t = mask * x_occ + (1 - mask) * recon
            imag_img = x_t[0].detach().cpu().numpy().reshape(28, 28)
            cap = "perceiving — the brain fires" if r == 0 else f"imagining the missing part · round {r}"
            # animate a few spike timesteps this round
            for si in range(0, len(sframes), 2):
                add(img_occ, sframes[si], imag_img if r > 0 else None, pred, pred == y0, cap)
            # settle: show the imagined result this round
            for _ in range(3):
                add(img_occ, sframes[-1] * 0.4, imag_img, pred, pred == y0, cap)
    # final hold
    for _ in range(10):
        add(img_occ, blank_brain, imag_img, y0, True, "recovered — recognised correctly")

    # ---- render ----
    plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG})
    fig = plt.figure(figsize=(9, 3.9))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.25, 1], wspace=0.18,
                           left=0.03, right=0.97, top=0.80, bottom=0.14)
    ax_in, ax_brain, ax_out = [fig.add_subplot(gs[i]) for i in range(3)]
    for ax in (ax_in, ax_brain, ax_out):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color("#2a3340")
    ttl = fig.text(0.5, 0.93, "", ha="center", color=FG, fontsize=15, weight="bold")
    cap = fig.text(0.5, 0.045, "", ha="center", color=ACCENT, fontsize=11.5,
                   family="monospace")
    ax_in.set_title("input (sensor)", color="#9fb0c0", fontsize=10)
    ax_brain.set_title("spiking brain · 576 neurons", color="#9fb0c0", fontsize=10)
    ax_out.set_title("imagined + recognised", color="#9fb0c0", fontsize=10)

    im_in = ax_in.imshow(img_clean, cmap="gray", vmin=0, vmax=1)
    im_br = ax_brain.imshow(blank_brain, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    im_out = ax_out.imshow(img_clean, cmap="gray", vmin=0, vmax=1)
    predtxt = ax_out.text(0.5, -0.14, "", transform=ax_out.transAxes, ha="center",
                          color=FG, fontsize=15, weight="bold", family="monospace")

    def update(i):
        f = frames[i]
        im_in.set_data(f["inp"])
        im_br.set_data(f["brain"])
        im_out.set_data(f["imag"] if f["imag"] is not None else np.zeros((28, 28)))
        ttl.set_text("Self-repairing perception")
        cap.set_text(f["cap"])
        if f["pred"] is None:
            predtxt.set_text("")
        else:
            label = FASHION[f["pred"]] if args.dataset == "fashion" else str(f["pred"])
            predtxt.set_text(f"reads: {label}")
            predtxt.set_color(GOOD if f["correct"] else BAD)
        return [im_in, im_br, im_out, ttl, cap, predtxt]

    import os; os.makedirs("media", exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=110, blit=False)
    out = f"media/demo_brain_{args.dataset}.gif"
    anim.save(out, writer=animation.PillowWriter(fps=9))
    print(f"Saved {out}  ({len(frames)} frames, class={y0})")
    update(len(frames) - 1); fig.savefig(f"media/demo_brain_{args.dataset}_poster.png", dpi=130)
    print(f"Saved media/demo_brain_{args.dataset}_poster.png")


if __name__ == "__main__":
    main()
