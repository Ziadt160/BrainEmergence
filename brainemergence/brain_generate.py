"""
brain_generate.py
=================
A BIDIRECTIONAL spiking brain (option 3): one shared recurrent sheet that runs
both ways.

  * UP-pass  (recognise / infer): clamp the image at the eyes (sensory zone),
    let activity flow up, read the motor zone -> a latent "style" code.
  * DOWN-pass (imagine / generate): clamp a concept (digit + latent) at the
    motor zone, let activity flow back DOWN through the SAME wiring, read the
    sensory zone (the eyes) -> a painted image.

Trained as a conditional VAE: up-pass infers a latent distribution from the
image, down-pass reconstructs the image from (latent + digit). Because the down
pass is conditioned on the digit, sampling a random latent for a chosen digit
generates a NEW, similar example of that digit.

The recurrent weights W_rec are SHARED by both directions -- that is what makes
it one brain running forwards and backwards, not two separate networks.

Outputs:
  generate_samples.png      grid of generated digits (rows = class, cols = samples)
  generate_recon.png        originals vs. reconstructions (sanity check)
  generate_loss.png         training loss
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from mnist_experiment import spike_func
from brain_predict import zone_masks, mexican_hat

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiBrain(nn.Module):
    def __init__(self, H=28, W=28, sensory_pct=0.25, motor_pct=0.25, t_steps=12, latent=20):
        super().__init__()
        self.H, self.W, self.N, self.T, self.latent = H, W, H * W, t_steps, latent
        coords, sen, asc, mot = zone_masks(H, W, sensory_pct, motor_pct, "cpu")
        self.register_buffer("sensory_idx", sen.nonzero(as_tuple=True)[0])
        self.register_buffer("motor_idx", mot.nonzero(as_tuple=True)[0])
        n_sen, n_mot = int(sen.sum()), int(mot.sum())

        self.W_rec = nn.Parameter(mexican_hat(coords))     # SHARED both directions
        self.tau_raw = nn.Parameter(torch.ones(self.N) * 2.0)
        self.thr_raw = nn.Parameter(torch.zeros(self.N))

        # up-pass: image -> eyes ;  motor activity -> latent (mu, logvar)
        self.w_in = nn.Linear(784, n_sen)
        self.enc_head = nn.Linear(n_mot, 2 * latent)
        # down-pass: (latent + digit) -> motor concept ;  eyes activity -> image
        self.dec_in = nn.Linear(latent + 10, n_mot)
        self.w_img = nn.Sequential(nn.Linear(n_sen, 256), nn.ReLU(), nn.Linear(256, 784))

    def _run(self, drive, read_idx):
        """Run the shared recurrent sheet with a constant injected drive; return
        the time-averaged spike activity of the read-out zone."""
        B = drive.shape[0]
        tau = torch.sigmoid(self.tau_raw)
        thr = F.softplus(self.thr_raw) + 0.5
        mem = torch.zeros(B, self.N, device=drive.device)
        spk = torch.zeros(B, self.N, device=drive.device)
        acc = torch.zeros(B, read_idx.numel(), device=drive.device)
        for _ in range(self.T):
            cur = drive + spk @ self.W_rec.t()
            mem = mem * tau + cur - spk * thr
            spk = spike_func(mem - thr)
            acc = acc + spk[:, read_idx]
        return acc / self.T

    def _inject(self, signal, idx):
        """Scatter a per-zone signal (B, n_zone) into a full (B, N) drive vector."""
        B = signal.shape[0]
        drive = torch.zeros(B, self.N, device=signal.device)
        return drive.index_copy(1, idx, signal)

    def encode(self, x):                                   # UP: eyes -> concept
        drive = self._inject(self.w_in(x), self.sensory_idx)
        motor_act = self._run(drive, self.motor_idx)
        mu, logvar = self.enc_head(motor_act).chunk(2, dim=1)
        return mu, logvar

    def decode(self, z, y_onehot):                         # DOWN: concept -> eyes
        concept = self.dec_in(torch.cat([z, y_onehot], dim=1))
        drive = self._inject(concept, self.motor_idx)
        eyes_act = self._run(drive, self.sensory_idx)
        return self.w_img(eyes_act)                         # logits over 784 pixels

    def forward(self, x, y_onehot):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z, y_onehot), mu, logvar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t-steps", type=int, default=12)
    ap.add_argument("--latent", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--beta", type=float, default=1.0)     # KL weight
    ap.add_argument("--free-bits", type=float, default=0.25)  # min nats/latent-dim (anti-collapse)
    ap.add_argument("--n-train", type=int, default=12000)
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.epochs, args.t_steps = 3000, 3, 10

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"Device: {DEVICE}  T={args.t_steps}  latent={args.latent}  beta={args.beta}")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    full = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    g = torch.Generator().manual_seed(args.seed)
    tr = Subset(full, torch.randperm(len(full), generator=g)[:args.n_train].tolist())
    loader = DataLoader(tr, batch_size=args.batch, shuffle=True)

    model = BiBrain(t_steps=args.t_steps, latent=args.latent).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}  "
          f"(eyes={model.sensory_idx.numel()}, motor={model.motor_idx.numel()})")

    loss_hist = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for i, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            yoh = F.one_hot(yb, 10).float()
            opt.zero_grad()
            logits, mu, logvar = model(xb, yoh)
            recon = F.binary_cross_entropy_with_logits(logits, xb, reduction="sum") / xb.size(0)
            # free-bits KL: each latent dim must carry >= free_bits nats, so the
            # latent can't collapse to the prior and is forced to encode style.
            kl_dim = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(0)
            kl = torch.clamp(kl_dim, min=args.free_bits).sum()
            loss = recon + args.beta * kl
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += loss.item()
            if i % 20 == 0:
                print(f"  epoch {epoch} batch {i}/{len(loader)}  loss {loss.item():.1f} "
                      f"(recon {recon.item():.1f}, kl {kl.item():.2f})")
        loss_hist.append(running / len(loader))
        print(f"=== epoch {epoch} | avg loss {loss_hist[-1]:.1f} ===")

    # ---- generate: clamp each digit at the output, sample styles, read the eyes ----
    model.eval()
    n_samples = 8
    fig, axes = plt.subplots(10, n_samples, figsize=(n_samples * 1.1, 11))
    with torch.no_grad():
        for d in range(10):
            yoh = F.one_hot(torch.full((n_samples,), d), 10).float().to(DEVICE)
            z = torch.randn(n_samples, args.latent, device=DEVICE)
            imgs = torch.sigmoid(model.decode(z, yoh)).cpu().view(-1, 28, 28)
            for s in range(n_samples):
                ax = axes[d, s]
                ax.imshow(imgs[s], cmap="gray"); ax.axis("off")
                if s == 0:
                    ax.set_ylabel(str(d), rotation=0, labelpad=12, fontsize=12)
    fig.suptitle("Digits imagined by the brain (output clamped per row, eyes read out)")
    plt.tight_layout(); plt.savefig("generate_samples.png", dpi=120)

    # ---- reconstruction sanity check ----
    xb, yb = next(iter(DataLoader(tr, batch_size=8, shuffle=True)))
    with torch.no_grad():
        rec = torch.sigmoid(model(xb.to(DEVICE), F.one_hot(yb, 10).float().to(DEVICE))[0]).cpu()
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.2))
    for j in range(8):
        axes[0, j].imshow(xb[j].view(28, 28), cmap="gray"); axes[0, j].axis("off")
        axes[1, j].imshow(rec[j].view(28, 28), cmap="gray"); axes[1, j].axis("off")
    axes[0, 0].set_ylabel("real", rotation=0, labelpad=20)
    axes[1, 0].set_ylabel("recon", rotation=0, labelpad=20)
    fig.suptitle("Reconstruction (up-pass infers, down-pass repaints)")
    plt.tight_layout(); plt.savefig("generate_recon.png", dpi=120)

    plt.figure(figsize=(6, 4))
    plt.plot(loss_hist, "o-"); plt.xlabel("epoch"); plt.ylabel("loss (recon + KL)")
    plt.title("Training loss"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("generate_loss.png", dpi=120)

    print("\nSaved generate_samples.png, generate_recon.png, generate_loss.png")


if __name__ == "__main__":
    main()
