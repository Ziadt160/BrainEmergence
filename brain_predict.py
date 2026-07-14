"""
brain_predict.py
================
Merges the two halves of the project into one trainable model:

  * the SPATIAL, zoned brain from snn_emergence.py  (sensory at the bottom,
    a recurrent association sheet, motor at the top)  -> gives interpretable
    per-digit "brain scans" you can look at;
  * the TRAINABLE-SPIKE machinery from mnist_experiment.py (surrogate
    gradients) -> lets us put a LOSS on the brain and teach it.

What the loss does (your three goals, in one objective):
  1. "reach the output": we read predictions from the MOTOR zone only, so the
     classification loss is *forced* to route each digit's signal all the way
     up to the output (a dead motor zone can't lower the loss).
  2. "unique scan per digit": a supervised-contrastive term pushes motor scans
     of different digits APART and same-digit scans TOGETHER.
  3. "use it for prediction": a linear readout on the motor zone classifies.

Outputs:
  predict_accuracy.png        test accuracy per epoch
  predict_scans_per_digit.png average brain scan per digit (now distinct)
  predict_output_similarity.png  how distinct the 10 motor outputs are
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

from mnist_experiment import spike_func  # surrogate-gradient spike

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------- the spatial brain ----------------------------
def zone_masks(H, W, sensory_pct, motor_pct, device):
    y = torch.arange(H, device=device).repeat_interleave(W)
    x = torch.arange(W, device=device).repeat(H)
    coords = torch.stack([y, x], 1).float()
    motor = y < (H * motor_pct)                       # top rows
    sensory = y >= (H * (1 - sensory_pct))            # bottom rows
    assoc = ~(motor | sensory)
    return coords, sensory, assoc, motor


def mexican_hat(coords, sigma_exc=2.0, sigma_inh=4.0, w_exc=2.5, w_inh=0.8):
    d2 = ((coords.unsqueeze(1) - coords.unsqueeze(0)) ** 2).sum(2)
    W = w_exc * torch.exp(-d2 / (2 * sigma_exc**2)) - w_inh * torch.exp(-d2 / (2 * sigma_inh**2))
    W.fill_diagonal_(0.0)
    # normalise each row's total |weight| to ~1 so initial dynamics are stable
    W = W / (W.abs().sum(1, keepdim=True) + 1e-6)
    return W


class SpatialBrain(nn.Module):
    def __init__(self, H=30, W=30, sensory_pct=0.2, motor_pct=0.2, t_steps=16):
        super().__init__()
        self.H, self.W, self.N, self.T = H, W, H * W, t_steps
        coords, sen, asc, mot = zone_masks(H, W, sensory_pct, motor_pct, "cpu")
        self.register_buffer("sensory", sen.float())
        self.register_buffer("motor_idx", mot.nonzero(as_tuple=True)[0])
        self.register_buffer("assoc", asc.float())
        n_motor = int(mot.sum())

        # input -> sensory zone only (masked); recurrent sheet; motor -> readout
        self.w_in = nn.Linear(784, self.N)
        self.W_rec = nn.Parameter(mexican_hat(coords))   # trainable, brain-like init
        self.w_out = nn.Linear(n_motor, 10)

        # trainable LIF params, constrained for stability (tau in (0,1), thr>0)
        self.tau_raw = nn.Parameter(torch.ones(self.N) * 2.0)     # sigmoid -> ~0.88
        self.thr_raw = nn.Parameter(torch.zeros(self.N))          # softplus+1 -> 1

    def forward(self, x):
        B = x.shape[0]
        tau = torch.sigmoid(self.tau_raw)
        thr = F.softplus(self.thr_raw) + 0.5
        mem = torch.zeros(B, self.N, device=x.device)
        spk = torch.zeros(B, self.N, device=x.device)
        scan = torch.zeros(B, self.N, device=x.device)
        logits = torch.zeros(B, 10, device=x.device)

        drive = self.w_in(x) * self.sensory               # input enters at the eyes
        for _ in range(self.T):
            current = drive + spk @ self.W_rec.t()        # input + recurrent sheet
            mem = mem * tau + current - spk * thr          # leaky integrate (soft reset)
            spk = spike_func(mem - thr)                    # fire (surrogate gradient)
            scan = scan + spk
            logits = logits + self.w_out(spk[:, self.motor_idx])
        scan = scan / self.T
        return logits / self.T, scan[:, self.motor_idx], scan   # logits, motor scan, full scan


# --------------------------- unique-scan loss -----------------------------
def supcon_loss(feats, labels, temp=0.1):
    """Supervised contrastive: pull same-label scans together, push others apart."""
    z = F.normalize(feats, dim=1)
    sim = z @ z.t() / temp
    B = feats.size(0)
    not_self = ~torch.eye(B, dtype=torch.bool, device=feats.device)
    pos = (labels.view(-1, 1) == labels.view(1, -1)) & not_self
    sim = sim.masked_fill(~not_self, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_count = pos.sum(1)
    valid = pos_count > 0
    if not valid.any():
        return feats.new_tensor(0.0)
    loss = -(log_prob.masked_fill(~pos, 0.0).sum(1)[valid] / pos_count[valid])
    return loss.mean()


# ------------------------------- training ---------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", type=int, default=30)
    ap.add_argument("--t-steps", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lambda-con", type=float, default=0.3)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-test", type=int, default=2000)
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.n_test, args.epochs, args.t_steps = 3000, 1000, 2, 12

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Device: {DEVICE}  grid={args.grid}x{args.grid}  T={args.t_steps}")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    train_full = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test_full = datasets.MNIST("./data", train=False, transform=tfm)
    g = torch.Generator().manual_seed(args.seed)
    tr = Subset(train_full, torch.randperm(len(train_full), generator=g)[:args.n_train].tolist())
    te = Subset(test_full, torch.randperm(len(test_full), generator=g)[:args.n_test].tolist())
    train_loader = DataLoader(tr, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(te, batch_size=256)

    model = SpatialBrain(args.grid, args.grid, t_steps=args.t_steps).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}  "
          f"(motor neurons read for output: {len(model.motor_idx)})")

    acc_hist = []
    for epoch in range(args.epochs):
        model.train()
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits, motor_scan, _ = model(xb)
            loss = ce(logits, yb) + args.lambda_con * supcon_loss(motor_scan, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if i % 20 == 0:
                print(f"  epoch {epoch} batch {i}/{len(train_loader)}  loss {loss.item():.3f}")

        # eval
        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits, _, _ = model(xb)
                correct += (logits.argmax(1) == yb).sum().item()
        acc = correct / len(te)
        acc_hist.append(acc)
        print(f"=== epoch {epoch} | test accuracy (predicting from MOTOR zone): {acc:.4f} ===")

    # ---- collect per-digit scans + motor outputs on the test set ----
    model.eval()
    scans = torch.zeros(10, model.N); counts = torch.zeros(10)
    motor_means = torch.zeros(10, len(model.motor_idx))
    with torch.no_grad():
        for xb, yb in test_loader:
            _, mscan, fscan = model(xb.to(DEVICE))
            for d in range(10):
                m = yb == d
                if m.any():
                    scans[d] += fscan[m].sum(0).cpu()
                    motor_means[d] += mscan[m].sum(0).cpu()
                    counts[d] += m.sum().item()
    scans /= counts.clamp(min=1).unsqueeze(1)
    motor_means /= counts.clamp(min=1).unsqueeze(1)

    # Figure 1: accuracy curve
    plt.figure(figsize=(6, 4))
    plt.plot(range(args.epochs), acc_hist, "o-")
    plt.axhline(0.1, color="gray", ls=":", label="chance")
    plt.xlabel("epoch"); plt.ylabel("test accuracy"); plt.ylim(0, 1)
    plt.title("Prediction accuracy (read from the motor/output zone)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("predict_accuracy.png", dpi=120)

    # Figure 2: per-digit brain scans (should now be distinct)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for d in range(10):
        ax = axes[d // 5, d % 5]
        im = ax.imshow(scans[d].reshape(model.H, model.W), cmap="plasma", interpolation="nearest")
        ax.set_title(f"Digit {d}"); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Average Brain Scan per Digit (taught with prediction + unique-scan loss)")
    plt.tight_layout(); plt.savefig("predict_scans_per_digit.png", dpi=120)

    # Figure 3: how distinct are the 10 motor outputs? (cosine similarity matrix)
    z = F.normalize(motor_means, dim=1)
    sim = (z @ z.t()).numpy()
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(sim, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="cosine similarity")
    plt.xticks(range(10)); plt.yticks(range(10))
    plt.xlabel("digit"); plt.ylabel("digit")
    plt.title("Motor-output similarity\n(diagonal=1; low off-diagonal = unique per digit)")
    plt.tight_layout(); plt.savefig("predict_output_similarity.png", dpi=120)

    off = sim[~np.eye(10, dtype=bool)]
    print("\nSaved predict_accuracy.png, predict_scans_per_digit.png, predict_output_similarity.png")
    print(f"Final test accuracy: {acc_hist[-1]:.4f}")
    print(f"Motor-output distinctness: mean off-diagonal similarity = {off.mean():.3f} "
          f"(lower = more unique per digit)")


if __name__ == "__main__":
    main()
