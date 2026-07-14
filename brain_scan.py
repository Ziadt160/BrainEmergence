"""
brain_scan.py
=============
Turns the self-organizing NeuroGrid into a "brain" that produces an activity
"scan" for each MNIST digit, then measures EMERGENCE quantitatively.

Pipeline
--------
1. Encode a digit -> drive the SENSORY zone with a fixed random projection
   (rate-coded Poisson spikes). The digit actually enters the sheet now.
2. Let the recurrent sheet run for T steps and record per-neuron spike counts
   -> that count map IS the brain "scan" for the image.
3. EMERGENCE METRIC: read the MOTOR zone (the "end" of the sheet) and fit a
   simple LINEAR decoder (logistic regression) to predict the digit from it.
   Information has to propagate sensory -> association -> motor and be shaped
   by the recurrent wiring to land there, so motor-zone decodability above
   chance = structure has emerged. We track this accuracy as STDP exposes the
   sheet to more digits -> a rising curve is emergence, as a number.

Crucially the grid is trained UNSUPERVISED (STDP only, no labels). Labels are
used only by the linear probe that *measures* the representation, never to shape
the sheet -- that is what makes a rising curve evidence of self-organization.

Outputs:
  - emergence_curve.png      decoding accuracy vs. # digits the sheet has seen
  - brain_scan_per_digit.png average scan per digit class (digit-driven)
"""

import argparse

import matplotlib
matplotlib.use("Agg")  # save figures without a display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from snn_emergence import NeuroGrid, GRID_H, GRID_W, DEVICE


# ----------------------------- Encoding -----------------------------------
def sensory_geometry(grid):
    """The sensory zone is a contiguous block of full rows at the bottom of the
    grid; return its neuron indices and (rows, cols) shape."""
    sensory_idx = grid.sensory_mask.nonzero(as_tuple=True)[0]
    h_s = int(grid.sensory_mask.sum().item() // grid.W)
    return sensory_idx, h_s, grid.W


def encode_image(x, sensory_idx, h_s, w_s, n_neurons, max_rate):
    """Topographic encoding: map the 28x28 image onto the sensory band so its
    spatial layout (and thus digit identity) is preserved, then read the pixels
    as per-neuron Poisson firing rates. A dense *random* projection instead
    central-limit-smears every digit into the same vector (verified ~0.997
    cross-class cosine similarity) and makes decoding impossible."""
    small = F.interpolate(x.view(1, 1, 28, 28), size=(h_s, w_s),
                          mode="bilinear", align_corners=False).reshape(-1)
    small = small / (small.max() + 1e-6)
    rate = torch.zeros(n_neurons, device=x.device)
    rate[sensory_idx] = small * max_rate
    return rate


def run_image(grid, rate, t_steps, warmup, input_pulse, learn):
    """Present one stimulus for t_steps; return accumulated spike counts (N,)."""
    grid.reset_state()
    counts = torch.zeros(grid.N, device=grid.device)
    for t in range(t_steps):
        spikes_in = (torch.rand(grid.N, device=grid.device) < rate).float() * input_pulse
        grid.dynamics_step(external_input=spikes_in, learn=learn, return_maps=False)
        if t >= warmup:
            counts += grid.spikes
    return counts


# ----------------------------- Read-out -----------------------------------
def collect_scans(grid, imgs, encode, args, learn):
    feats = torch.zeros(imgs.shape[0], grid.N, device=grid.device)
    for i in range(imgs.shape[0]):
        rate = encode(imgs[i])
        feats[i] = run_image(grid, rate, args.t_steps, args.warmup, args.input_pulse, learn)
    return feats


def decode(feats, labels, subset_idx, tr_idx, te_idx, epochs):
    """Fit a linear (logistic-regression) probe on a feature subset; return test acc."""
    X = feats[:, subset_idx]
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = labels[tr_idx], labels[te_idx]

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6      # standardise on train split only
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    clf = nn.Linear(X.shape[1], 10).to(feats.device)
    opt = optim.Adam(clf.parameters(), lr=0.05, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(clf(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return (clf(Xte).argmax(1) == yte).float().mean().item()


def evaluate(grid, imgs, labels, encode, zones, tr_idx, te_idx, args):
    """Freeze weights, scan the eval set, decode the digit from each zone."""
    feats = collect_scans(grid, imgs, encode, args, learn=False)
    accs = {name: decode(feats, labels, idx, tr_idx, te_idx, args.decoder_epochs)
            for name, idx in zones.items()}
    return accs, feats


# ----------------------------- Experiment ---------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny run to smoke-test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--chunks", type=int, default=5)
    ap.add_argument("--t-steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--input-pulse", type=float, default=5.0)
    ap.add_argument("--max-rate", type=float, default=0.6)
    ap.add_argument("--decoder-epochs", type=int, default=300)
    # connectivity / homeostasis knobs (defaults = original module constants)
    ap.add_argument("--sigma-exc", type=float, default=2.0)
    ap.add_argument("--sigma-inh", type=float, default=4.0)
    ap.add_argument("--conn-radius", type=float, default=20.0)
    ap.add_argument("--w-exc", type=float, default=2.5)
    ap.add_argument("--w-inh", type=float, default=0.4)
    ap.add_argument("--gain", type=float, default=3.0)
    ap.add_argument("--scale-every", type=int, default=1)
    args = ap.parse_args()

    if args.quick:
        args.n_eval, args.n_train, args.chunks = 120, 300, 3
        args.t_steps, args.warmup, args.decoder_epochs = 12, 4, 150

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Data ---
    print(f"Device: {DEVICE}")
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)

    order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed))
    eval_ids = order[: args.n_eval]
    train_ids = order[args.n_eval : args.n_eval + args.n_train]

    eval_imgs = torch.stack([ds[i][0] for i in eval_ids]).to(DEVICE)
    eval_labels = torch.tensor([ds[i][1] for i in eval_ids], device=DEVICE)
    train_imgs = torch.stack([ds[i][0] for i in train_ids]).to(DEVICE)

    # fixed train/test split of the eval set (same images probed at every checkpoint)
    sp = torch.randperm(args.n_eval, generator=torch.Generator().manual_seed(args.seed + 1))
    cut = int(0.7 * args.n_eval)
    tr_idx, te_idx = sp[:cut].to(DEVICE), sp[cut:].to(DEVICE)

    # --- Brain ---
    grid = NeuroGrid(sigma_exc=args.sigma_exc, sigma_inh=args.sigma_inh,
                     conn_radius=args.conn_radius, w_exc=args.w_exc,
                     w_inh=args.w_inh, gain=args.gain, scale_every=args.scale_every)
    sensory_idx, h_s, w_s = sensory_geometry(grid)
    encode = lambda x: encode_image(x, sensory_idx, h_s, w_s, grid.N, args.max_rate)

    # Decode each anatomical zone separately. 'sensory' = input fidelity (should
    # be high & flat); 'association' = where emergent representations would form;
    # 'motor' = the far "end" of the sheet; 'whole' = everything.
    zones = {
        "sensory": sensory_idx,
        "association": grid.association_mask.nonzero(as_tuple=True)[0],
        "motor": grid.motor_mask.nonzero(as_tuple=True)[0],
        "whole": torch.arange(grid.N, device=grid.device),
    }
    zone_names = list(zones.keys())

    # --- Emergence over training ---
    per_chunk = args.n_train // args.chunks
    history = []  # (images_seen, accs_dict)

    def show(seen, accs):
        cells = "  ".join(f"{n}={accs[n]:.3f}" for n in zone_names)
        print(f"  seen={seen:<6} {cells}")

    print("\nCheckpoint 0 (untrained sheet)...  (chance=0.100)")
    accs, feats = evaluate(grid, eval_imgs, eval_labels, encode, zones, tr_idx, te_idx, args)
    history.append((0, accs)); show(0, accs)

    for c in range(args.chunks):
        chunk = train_imgs[c * per_chunk:(c + 1) * per_chunk]
        for i in range(chunk.shape[0]):
            run_image(grid, encode(chunk[i]), args.t_steps, args.warmup,
                      args.input_pulse, learn=True)
        seen = (c + 1) * per_chunk
        accs, feats = evaluate(grid, eval_imgs, eval_labels, encode, zones, tr_idx, te_idx, args)
        history.append((seen, accs)); show(seen, accs)

    # --- Figure 1: per-zone emergence curves ---
    seen = [h[0] for h in history]
    styles = {"sensory": "^:", "association": "o-", "motor": "s-", "whole": "d--"}
    plt.figure(figsize=(7, 5))
    for n in zone_names:
        plt.plot(seen, [h[1][n] for h in history], styles[n], label=n)
    plt.axhline(0.1, color="gray", ls=":", label="chance (10%)")
    plt.xlabel("Digits the sheet has seen (unsupervised STDP)")
    plt.ylabel("Linear-decoding accuracy")
    plt.title("Emergence: digit decodability per brain zone vs. self-organization")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("emergence_curve.png", dpi=120)
    print("\nSaved emergence_curve.png")

    # --- Figure 2: average brain scan per digit (digit-driven, final sheet) ---
    feats_np = feats.cpu().numpy()
    labels_np = eval_labels.cpu().numpy()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for d in range(10):
        ax = axes[d // 5, d % 5]
        mask = labels_np == d
        avg = feats_np[mask].mean(0).reshape(GRID_H, GRID_W) if mask.any() else np.zeros((GRID_H, GRID_W))
        im = ax.imshow(avg, cmap="plasma", interpolation="nearest")
        ax.set_title(f"Digit {d}"); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Average Brain Scan per Digit Class (final sheet)")
    plt.tight_layout()
    plt.savefig("brain_scan_per_digit.png", dpi=120)
    print("Saved brain_scan_per_digit.png")

    # --- Summary ---
    print("\n=== Emergence summary (linear-decoding accuracy, chance=0.100) ===")
    header = f"{'seen':>8} " + " ".join(f"{n:>12}" for n in zone_names)
    print(header)
    for s, accs in history:
        print(f"{s:>8} " + " ".join(f"{accs[n]:>12.3f}" for n in zone_names))
    print("\nChange over training (first -> last):")
    for n in zone_names:
        d = history[-1][1][n] - history[0][1][n]
        print(f"  {n:>12}: {history[0][1][n]:.3f} -> {history[-1][1][n]:.3f}  ({d:+.3f})")


if __name__ == "__main__":
    main()
