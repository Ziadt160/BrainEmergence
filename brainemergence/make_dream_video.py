"""
make_dream_video.py -- the brain DREAMS (top-down generation), animated from BiBrain.
====================================================================================
No input at all. We clamp a concept (a digit class + a random latent "style") at the motor
zone and let activity flow DOWN through the shared recurrent sheet to the sensory zone (the
"eyes"), which paints an image. We watch the spiking sheet fire and the dreamed digit SHARPEN
as spikes accumulate, then move to the next concept. Pure top-down imagination.

Output: demo_dream.gif
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
from brain_generate import BiBrain, DEVICE

BG = "#0c0f14"; FG = "#e8edf2"; ACCENT = "#c58cff"; WARM = "#ffd479"


def train(model, loader, epochs, beta, free_bits):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE); yoh = F.one_hot(y, 10).float()
            logits, mu, logvar = model(x, yoh)
            recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.size(0)
            kl = torch.clamp((-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(0), min=free_bits).sum()
            loss = recon + beta * kl
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        print(f"  epoch {ep} done")


def dream_capture(model, z, yoh):
    """Run the DOWN-pass, recording the full spiking sheet and the image forming each timestep."""
    concept = model.dec_in(torch.cat([z, yoh], 1))
    drive = model._inject(concept, model.motor_idx)
    tau, thr = torch.sigmoid(model.tau_raw), F.softplus(model.thr_raw) + 0.5
    mem = torch.zeros(1, model.N, device=z.device)
    spk = torch.zeros_like(mem); acc_sen = torch.zeros(1, model.sensory_idx.numel(), device=z.device)
    W = model.W_rec; frames = []
    for t in range(model.T):
        cur = drive + spk @ W.t()
        mem = mem * tau + cur - spk * thr
        spk = spike_func(mem - thr)
        acc_sen = acc_sen + spk[:, model.sensory_idx]
        img = torch.sigmoid(model.w_img(acc_sen / (t + 1)))
        frames.append((spk[0].detach().cpu().numpy().reshape(28, 28),
                       img[0].detach().cpu().numpy().reshape(28, 28)))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=9)
    ap.add_argument("--latent", type=int, default=20)
    ap.add_argument("--t-steps", type=int, default=12)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--free-bits", type=float, default=0.25)
    ap.add_argument("--styles-of", type=int, default=-1)   # >=0: sweep the latent for ONE fixed class
    args = ap.parse_args()
    torch.manual_seed(1)
    print(f"Device {DEVICE}  building DREAM animation")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    trf = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    g0 = torch.Generator().manual_seed(0)
    tr = Subset(trf, torch.randperm(len(trf), generator=g0)[:15000].tolist())
    loader = DataLoader(tr, batch_size=128, shuffle=True)

    model = BiBrain(t_steps=args.t_steps, latent=args.latent).to(DEVICE)
    print("training the generative brain...")
    train(model, loader, args.epochs, args.beta, args.free_bits)
    model.eval()

    seq = []
    with torch.no_grad():
        if args.styles_of >= 0:
            # sweep the latent "style" for ONE fixed class -> same digit, many handwritings
            d = args.styles_of
            yoh = F.one_hot(torch.tensor([d], device=DEVICE), 10).float()
            keys = [torch.randn(1, args.latent, device=DEVICE) * 0.95 for _ in range(8)]
            keys.append(keys[0])                                 # loop back
            for k in range(len(keys) - 1):
                for t in np.linspace(0, 1, 7):
                    z = (1 - t) * keys[k] + t * keys[k + 1]
                    frames = dream_capture(model, z, yoh)
                    seq.append((d, frames[len(frames) // 2][0], frames[-1][1]))
        else:
            # dream each digit 0..9 with a random style
            for d in range(10):
                z = torch.randn(1, args.latent, device=DEVICE) * 0.8
                yoh = F.one_hot(torch.tensor([d], device=DEVICE), 10).float()
                frames = dream_capture(model, z, yoh)
                for si in range(0, len(frames), 2):
                    seq.append((d, frames[si][0], frames[si][1]))
                for _ in range(4):                               # hold the finished dream
                    seq.append((d, frames[-1][0] * 0.25, frames[-1][1]))

    plt.rcParams.update({"figure.facecolor": BG, "savefig.facecolor": BG})
    fig = plt.figure(figsize=(7.6, 4.1))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.15, 1], wspace=0.16,
                           left=0.04, right=0.96, top=0.80, bottom=0.13)
    ax_br, ax_im = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for ax in (ax_br, ax_im):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color("#2a3340")
    ax_br.set_title("spiking sheet · top-down flow", color="#9fb0c0", fontsize=10)
    ax_im.set_title("the dreamed image forms", color="#9fb0c0", fontsize=10)
    fig.text(0.5, 0.93, "The brain dreams", ha="center", color=FG, fontsize=15, weight="bold")
    cap = fig.text(0.5, 0.04, "", ha="center", color=ACCENT, fontsize=12, family="monospace")

    im_br = ax_br.imshow(np.zeros((28, 28)), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    im_im = ax_im.imshow(np.zeros((28, 28)), cmap="bone", vmin=0, vmax=1)

    styles = args.styles_of >= 0

    def update(i):
        d, br, img = seq[i]
        im_br.set_data(br); im_im.set_data(img)
        cap.set_text(f"one concept, many styles  ·  a {d}" if styles else f"imagining a  {d}")
        return [im_br, im_im, cap]

    import os; os.makedirs("media", exist_ok=True)
    name = f"demo_dream_styles{args.styles_of}" if styles else "demo_dream"
    anim = animation.FuncAnimation(fig, update, frames=len(seq), interval=110, blit=False)
    anim.save(f"media/{name}.gif", writer=animation.PillowWriter(fps=10))
    print(f"Saved media/{name}.gif  ({len(seq)} frames)")
    update(len(seq) // 3); fig.savefig(f"media/{name}_poster.png", dpi=130)
    print(f"Saved media/{name}_poster.png")


if __name__ == "__main__":
    main()
