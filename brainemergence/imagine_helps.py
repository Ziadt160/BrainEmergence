"""
imagine_helps.py -- does IMAGINATION clean up PERCEPTION, and does the
MOVABLE/ATTRACTING-neuron substrate do it better than a fixed grid?
=====================================================================
A recognize+imagine brain (spiking recurrent sheet that CLASSIFIES a digit and
RECONSTRUCTS the image). Trained on clean MNIST. Tested on occluded digits with
predictive-coding pattern completion: keep visible pixels, let the brain imagine
the missing band, re-feed, classify; repeat.

Two substrates compared head-to-head:
  fixed   : fixed Mexican-hat recurrent weights, neurons don't move.
  movable : LEARNABLE neuron positions; connectivity = W_base * exp(-dist^2/2sigma^2)
            so position shapes wiring, PLUS a co-firing attraction loss (the user's
            "fire together -> move closer" rule).

Controls so a win is trustworthy:
  - raw occluded (round 0)          : do nothing.
  - dumb mean-fill                  : fill the hole with the visible-pixel mean.
  Imagination only "genuinely helps" if it beats BOTH.
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
from brain_predict import mexican_hat, zone_masks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RecallBrain(nn.Module):
    def __init__(self, H=24, W=24, t_steps=16, movable=False, sigma=0.35):
        super().__init__()
        self.N, self.T, self.movable, self.sigma = H * W, t_steps, movable, sigma
        coords, *_ = zone_masks(H, W, 0.2, 0.2, "cpu")
        if movable:
            self.pos = nn.Parameter(torch.rand(self.N, 2))                  # movable positions
            self.W_base = nn.Parameter(torch.randn(self.N, self.N) / np.sqrt(self.N))
        else:
            self.W_rec = nn.Parameter(mexican_hat(coords))                  # fixed Mexican-hat
        self.tau_raw = nn.Parameter(torch.ones(self.N) * 2.0)
        self.thr_raw = nn.Parameter(torch.zeros(self.N))
        self.w_in = nn.Linear(784, self.N)
        self.cls = nn.Linear(self.N, 10)
        self.dec = nn.Sequential(nn.Linear(self.N, 512), nn.ReLU(),
                                 nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 784))

    def npos(self):
        p = self.pos
        return (p - p.mean(0)) / (p.std(0) + 1e-6)

    def recurrent_W(self):
        if not self.movable:
            return self.W_rec
        p = self.npos()
        d2 = ((p.unsqueeze(1) - p.unsqueeze(0)) ** 2).sum(2)
        W = self.W_base * torch.exp(-d2 / (2 * self.sigma ** 2))
        return W - torch.diag_embed(torch.diagonal(W))

    def encode(self, x):
        B = x.shape[0]; W = self.recurrent_W()
        tau, thr = torch.sigmoid(self.tau_raw), F.softplus(self.thr_raw) + 0.5
        mem = torch.zeros(B, self.N, device=x.device)
        spk = torch.zeros_like(mem); acc = torch.zeros_like(mem)
        drive = self.w_in(x)
        for _ in range(self.T):
            mem = mem * tau + (drive + spk @ W.t()) - spk * thr
            spk = spike_func(mem - thr); acc = acc + spk
        return acc / self.T

    def forward(self, x):
        A = self.encode(x)
        return self.cls(A), self.dec(A)

    def attract(self, A):
        """Co-firing attraction: pull correlated neurons close (only used if movable)."""
        Z = (A - A.mean(0)) / (A.std(0) + 1e-6)
        C = (Z.t() @ Z) / A.shape[0]
        p = self.npos()
        d2 = ((p.unsqueeze(1) - p.unsqueeze(0)) ** 2).sum(2)
        Cp = C.clamp(min=0)
        return (Cp * d2).sum() / (Cp.sum() + 1e-6)


def occlude(x, band=8):
    B = x.shape[0]
    img = x.view(B, 28, 28).clone()
    r0 = 28 // 2 - band // 2
    mask = torch.ones_like(img)
    img[:, r0:r0 + band, :] = 0.0
    mask[:, r0:r0 + band, :] = 0.0
    return img.view(B, -1), mask.view(B, -1)


@torch.no_grad()
def evaluate_completion(model, loader, rounds, band):
    model.eval()
    n = 0
    correct = np.zeros(rounds + 1)
    mean_correct = clean_correct = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        n += y.numel()
        clean_correct += (model(x)[0].argmax(1) == y).sum().item()
        x_occ, mask = occlude(x, band)
        obs_mean = (x_occ * mask).sum(1, keepdim=True) / (mask.sum(1, keepdim=True) + 1e-6)
        x_mean = mask * x_occ + (1 - mask) * obs_mean
        mean_correct += (model(x_mean)[0].argmax(1) == y).sum().item()
        x_t = x_occ.clone()
        for r in range(rounds + 1):
            logits, recon = model(x_t)
            correct[r] += (logits.argmax(1) == y).sum().item()
            x_t = mask * x_occ + (1 - mask) * torch.sigmoid(recon)
    return correct / n, mean_correct / n, clean_correct / n


def run_substrate(movable, seed, args, train_loader, test_loader):
    name = "movable" if movable else "fixed"
    torch.manual_seed(seed)
    model = RecallBrain(args.grid, args.grid, args.t_steps, movable=movable).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    ce = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            A = model.encode(x)
            logits, recon = model.cls(A), model.dec(A)
            loss = ce(logits, y) + args.recon_w * F.binary_cross_entropy_with_logits(
                recon, x, reduction="sum") / x.size(0)
            if movable:
                loss = loss + args.lam_attract * model.attract(A)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if epoch == args.epochs - 1:
            print(f"  [{name} s{seed}] final loss {loss.item():.2f}")
    accs, mean_fill, clean = evaluate_completion(model, test_loader, args.rounds, args.band)
    return dict(name=name, seed=seed, clean=clean, occ=accs[0], mean=mean_fill,
                imag=accs[-1], accs=list(accs))


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
    ap.add_argument("--lam-attract", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.n_test, args.epochs, args.t_steps, args.seeds = 3000, 1000, 2, 12, 2
    print(f"Device {DEVICE}  grid {args.grid}  occlusion {args.band}px  (fixed vs movable)")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    tr_full = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    te_full = datasets.MNIST("./data", train=False, transform=tfm)
    g = torch.Generator().manual_seed(0)
    tr = Subset(tr_full, torch.randperm(len(tr_full), generator=g)[:args.n_train].tolist())
    te = Subset(te_full, torch.randperm(len(te_full), generator=g)[:args.n_test].tolist())
    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(te, batch_size=256)

    rows = []
    for movable in [False, True]:
        for s in range(args.seeds):
            rows.append(run_substrate(movable, s, args, train_loader, test_loader))

    print(f"\n=== Substrate comparison ({args.seeds} seeds, mean +/- std) ===")
    print(f"{'substrate':>9} {'clean':>14} {'occluded':>14} {'imagined':>14} "
          f"{'gain(im-occ)':>16} {'im - dumbfill':>16}")
    summ = {}
    for name in ["fixed", "movable"]:
        sub = [r for r in rows if r["name"] == name]
        ms = lambda f: (np.mean([f(r) for r in sub]), np.std([f(r) for r in sub]))
        cl, oc, im = ms(lambda r: r["clean"]), ms(lambda r: r["occ"]), ms(lambda r: r["imag"])
        gn, cm = ms(lambda r: r["imag"] - r["occ"]), ms(lambda r: r["imag"] - r["mean"])
        summ[name] = dict(gain=gn, sub=sub)
        print(f"{name:>9} {cl[0]:.3f}+/-{cl[1]:.3f} {oc[0]:.3f}+/-{oc[1]:.3f} {im[0]:.3f}+/-{im[1]:.3f} "
              f"{gn[0]:+.3f}+/-{gn[1]:.3f} {cm[0]:+.3f}+/-{cm[1]:.3f}")
    print("\nRobust if (gain mean - std) > 0 -- the effect survives seed noise:")
    for name in ["fixed", "movable"]:
        g = summ[name]["gain"]
        print(f"  {name:>9}: imagination gain {g[0]:+.3f} +/- {g[1]:.3f} -> "
              f"{'ROBUST positive' if g[0] - g[1] > 0 else 'within noise'}")

    plt.figure(figsize=(6.8, 4.4))
    for name, col in zip(["fixed", "movable"], ["#378ADD", "#D85A30"]):
        A = np.array([r["accs"] for r in summ[name]["sub"]])      # (seeds, rounds+1)
        m, sd = A.mean(0), A.std(0)
        xs = range(A.shape[1])
        plt.plot(xs, m, "o-", color=col, label=name)
        plt.fill_between(xs, m - sd, m + sd, color=col, alpha=0.2)
    plt.xlabel("imagination rounds"); plt.ylabel("digit accuracy"); plt.ylim(0, 1)
    plt.title(f"Imagination-aided recognition ({args.seeds} seeds, mean +/- std)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("imagine_helps.png", dpi=120)
    print("\nSaved imagine_helps.png")


if __name__ == "__main__":
    main()
