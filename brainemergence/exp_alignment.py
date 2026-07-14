"""
exp_alignment.py  --  the de-risking experiment from PROPOSAL.md (E1 / E2a / E2b)
=================================================================================
Question: in a SHARED-WEIGHT, spatially-embedded SPIKING bidirectional network,
does the top-down GENERATIVE pathway form an orientation map that ALIGNS with the
bottom-up PERCEPTUAL map -- and is a spatial WIRING-COST prior what causes it?

Design (all three share ONE recurrent weight matrix used both ways; only the
regularizer differs -- this is the shared-weight row of the 2x2 in the proposal):
  E1   spatial wiring-cost prior   (penalize |W_ij| * distance(i,j))
  E2a  no cost                     (lambda = 0)              -> weight-sharing floor
  E2b  matched L1 sparsity         (penalize |W_ij|)         -> the KEY control

If  align(E1) >> align(E2b)  across seeds (permutation-tested) -> the spatial prior
causes alignment -> the paper is live.  If  E1 ~= E2b  -> it's just sparsity -> stop.

Stimulus: oriented gratings (classic V1 orientation-map probe). Generation is
conditioned on orientation via c = [cos 2theta, sin 2theta]; a VAE latent absorbs phase.

This is a runnable SKELETON meant for a GPU. `--quick` runs a tiny CPU smoke test that
only validates the pipeline end-to-end (numbers will be noise at that scale).
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mnist_experiment import spike_func  # surrogate-gradient spike

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------- stimuli ----------------------------------
def grating(P, theta, phase, freq):
    """Single oriented sinusoidal grating, flattened, in [0,1]."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, P), torch.linspace(-1, 1, P), indexing="ij")
    proj = xx * torch.cos(theta) + yy * torch.sin(theta)
    g = 0.5 * (1 + torch.sin(2 * np.pi * freq * proj + phase))
    return g.reshape(-1)


def grating_batch(B, P, freq, gen):
    thetas = torch.rand(B, generator=gen) * np.pi          # orientation in [0, pi)
    phases = torch.rand(B, generator=gen) * 2 * np.pi
    x = torch.stack([grating(P, thetas[i], phases[i], freq) for i in range(B)])
    return x.to(DEVICE), thetas.to(DEVICE)


def concept(thetas):
    """Orientation -> 2D circular code (pi-periodic)."""
    return torch.stack([torch.cos(2 * thetas), torch.sin(2 * thetas)], dim=1)


# ------------------------------- model ------------------------------------
def mexican_hat(pos, sigma_e=2.0, sigma_i=4.0, we=2.5, wi=0.8):
    d2 = (pos.unsqueeze(1) - pos.unsqueeze(0)).pow(2).sum(2)
    W = we * torch.exp(-d2 / (2 * sigma_e**2)) - wi * torch.exp(-d2 / (2 * sigma_i**2))
    W.fill_diagonal_(0.0)
    return W / (W.abs().sum(1, keepdim=True) + 1e-6)


class BiSheet(nn.Module):
    """One recurrent spiking sheet on a 2D grid, run bottom-up (perceive) and
    top-down (generate) through the SAME recurrent weights.

    cond_enc + gen_gain<1 are the generation-validity fix (see fix_generation.py):
    a conditional encoder frees the latent from carrying orientation, and a weaker
    recurrent gain on the down-pass stops the dynamics from washing out the command,
    so generate(theta) reliably draws a theta-grating in perception's convention."""
    def __init__(self, H, W, P, latent=8, t_steps=12, cond_enc=True, gen_gain=0.5, init="mexican"):
        super().__init__()
        self.H, self.W, self.N, self.T, self.P = H, W, H * W, t_steps, P
        self.cond_enc, self.gen_gain = cond_enc, gen_gain
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        pos = torch.stack([yy.reshape(-1), xx.reshape(-1)], 1).float()
        dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).pow(2).sum(2).sqrt()
        self.register_buffer("dist", dist)
        self.register_buffer("mean_dist", dist.sum() / (self.N * self.N - self.N))
        if init == "random":
            # Random dense init -> the wiring-cost penalty is the ONLY possible source
            # of spatial structure (removes the Mexican-hat init confound). Same
            # row-abs-sum scale as the mexican-hat so dynamics stay comparable.
            W0 = torch.randn(self.N, self.N)
            W0.fill_diagonal_(0.0)
            self.W_rec = nn.Parameter(W0 / (W0.abs().sum(1, keepdim=True) + 1e-6))
        else:
            self.W_rec = nn.Parameter(mexican_hat(pos))    # SHARED both directions
        self.tau_raw = nn.Parameter(torch.ones(self.N) * 2.0)
        self.thr_raw = nn.Parameter(torch.zeros(self.N))
        self.w_in = nn.Linear(P * P, self.N)               # image  -> sheet (up)
        self.enc = nn.Linear(self.N + (2 if cond_enc else 0), 2 * latent)  # sheet[,c] -> (mu,logvar)
        self.dec = nn.Linear(latent + 2, self.N)           # (z, orient) -> sheet (down)
        self.w_img = nn.Sequential(nn.Linear(self.N, 256), nn.ReLU(), nn.Linear(256, P * P))
        self.latent = latent

    def _run(self, drive, gain=1.0):
        B = drive.shape[0]
        tau, thr = torch.sigmoid(self.tau_raw), F.softplus(self.thr_raw) + 0.5
        mem = torch.zeros(B, self.N, device=drive.device)
        spk = torch.zeros(B, self.N, device=drive.device)
        acc = torch.zeros(B, self.N, device=drive.device)
        for _ in range(self.T):
            mem = mem * tau + (drive + gain * (spk @ self.W_rec.t())) - spk * thr
            spk = spike_func(mem - thr)
            acc = acc + spk
        return acc / self.T                                # (B, N) sheet activity

    def up(self, x):                                       # UP: image -> sheet activity
        return self._run(self.w_in(x), 1.0)

    def encode(self, A, c):                                # sheet activity -> latent (training)
        h = torch.cat([A, c], 1) if self.cond_enc else A
        return self.enc(h).chunk(2, 1)

    def generate(self, z, c):                              # DOWN
        A = self._run(self.dec(torch.cat([z, c], 1)), self.gen_gain)
        return self.w_img(A), A

    def reg(self, kind):
        # spatial is normalized by mean distance so it applies the SAME total pressure
        # as L1 at equal lambda; the only difference is that spatial weights the penalty
        # by connection length (long links cost more) -- that is the manipulation under test.
        if kind == "spatial":
            return (self.W_rec.abs() * self.dist).sum() / (self.N * self.mean_dist)
        if kind == "l1":
            return self.W_rec.abs().sum() / self.N
        return self.W_rec.new_tensor(0.0)


# --------------------- orientation tuning & alignment ---------------------
@torch.no_grad()
def tuning(model, direction, K, reps, P, freq, gen):
    """Per-neuron preferred orientation, measured in 'perceive' or 'generate'."""
    thetas = torch.linspace(0, np.pi, K + 1)[:-1]
    resp = torch.zeros(model.N, K, device=DEVICE)
    for k, th in enumerate(thetas):
        if direction == "perceive":
            phases = torch.rand(reps, generator=gen) * 2 * np.pi
            x = torch.stack([grating(P, th, ph, freq) for ph in phases]).to(DEVICE)
            A = model.up(x)
        else:  # generate: clamp orientation concept, sample styles
            z = torch.randn(reps, model.latent, generator=gen).to(DEVICE)
            c = concept(torch.full((reps,), float(th), device=DEVICE))
            _, A = model.generate(z, c)
        resp[:, k] = A.mean(0)
    ang = (2 * thetas).to(DEVICE)                          # pi-periodic -> double angle
    vx = (resp * torch.cos(ang)).sum(1)                    # real vector average (no complex
    vy = (resp * torch.sin(ang)).sum(1)                    # tensors -> avoids CUDA nvrtc JIT)
    pref = 0.5 * torch.atan2(vy, vx)                       # preferred orientation per neuron
    sel = torch.sqrt(vx * vx + vy * vy) / (resp.sum(1) + 1e-6)  # orientation selectivity
    return pref.cpu().numpy(), sel.cpu().numpy()


def circ_corr(a, b):
    """Circular correlation between two angle vectors (already doubled = orientation)."""
    a, b = 2 * a, 2 * b
    a0, b0 = a - _cmean(a), b - _cmean(b)
    num = np.sum(np.sin(a0) * np.sin(b0))
    den = np.sqrt(np.sum(np.sin(a0) ** 2) * np.sum(np.sin(b0) ** 2)) + 1e-9
    return num / den


def _cmean(a):
    return np.angle(np.mean(np.exp(1j * a)))


@torch.no_grad()
def gen_validity(model, K, reps, P, freq):
    """Does generate(orientation=theta) actually produce a theta-grating? Returns a
    circular correlation in [-1,1] between requested and produced orientation. If this
    is near 0 the generator ignores orientation and the TD tuning map is meaningless."""
    thetas = torch.linspace(0, np.pi, K + 1)[:-1]
    req, meas = [], []
    for th in thetas:
        z = torch.randn(reps, model.latent, device=DEVICE)
        c = concept(torch.full((reps,), float(th), device=DEVICE))
        imgs = torch.sigmoid(model.generate(z, c)[0]).reshape(-1, P, P)
        for im in imgs:
            gy, gx = torch.gradient(im)
            Jxx, Jyy, Jxy = (gx * gx).sum(), (gy * gy).sum(), (gx * gy).sum()
            ori = 0.5 * torch.atan2(2 * Jxy, Jxx - Jyy)   # dominant gradient orientation
            req.append(float(th)); meas.append(ori.item())
    return circ_corr(np.array(req), np.array(meas))


def alignment(pref_bu, pref_td, sel_bu, sel_td, n_perm=2000, seed=0):
    """Selectivity-weighted circular correlation + permutation p-value."""
    w = (sel_bu * sel_td)
    keep = w > np.percentile(w, 50)                        # focus on well-tuned neurons
    a, b = pref_bu[keep], pref_td[keep]
    obs = circ_corr(a, b)
    rng = np.random.default_rng(seed)
    null = np.array([circ_corr(a, rng.permutation(b)) for _ in range(n_perm)])
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, p


def smoothness(pref, sel, H, W):
    """Selectivity-WEIGHTED neighbour orientation difference. Each neighbour pair is
    weighted by the product of the two neurons' selectivities, so untuned neurons
    (which default to preferred-orientation 0) contribute ~nothing. This measures
    whether the WELL-TUNED population is spatially organized -- not just whether most
    neurons sit at 0. Low = smooth topographic map; ~0.785 = scattered/salt-and-pepper."""
    p, s = pref.reshape(H, W), sel.reshape(H, W)
    def wdiff(pa, pb, sa, sb):
        d = np.abs(np.angle(np.exp(1j * 2 * (pa - pb))) / 2)   # circular, mod pi
        w = sa * sb
        return (d * w).sum(), w.sum()
    nh, dh = wdiff(p[:, :-1], p[:, 1:], s[:, :-1], s[:, 1:])
    nv, dv = wdiff(p[:-1, :], p[1:, :], s[:-1, :], s[1:, :])
    return float((nh + nv) / (dh + dv + 1e-9))


# ------------------------------- training ---------------------------------
def train_one(cfg, seed, args):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = BiSheet(args.grid, args.grid, args.P, args.latent, args.t_steps,
                    cond_enc=args.cond_enc, gen_gain=args.gen_gain, init=args.init).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for it in range(args.iters):
        x, th = grating_batch(args.batch, args.P, args.freq, gen)
        c = concept(th)
        A = model.up(x)
        mu, logvar = model.encode(A, c)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        logits, _ = model.generate(z, c)
        recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.size(0)
        kl_dim = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(0)
        kl = torch.clamp(kl_dim, min=args.free_bits).sum()
        loss = recon + args.beta * kl + args.lreg * model.reg(cfg)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if it % max(1, args.iters // 4) == 0:
            print(f"    [{cfg} seed{seed}] iter {it}/{args.iters} loss {loss.item():.1f}")
    # measure both maps
    pbu, sbu = tuning(model, "perceive", args.K, args.reps, args.P, args.freq, gen)
    ptd, std = tuning(model, "generate", args.K, args.reps, args.P, args.freq, gen)
    obs, p = alignment(pbu, ptd, sbu, std, args.n_perm, seed)
    valid = gen_validity(model, args.K, args.reps, args.P, args.freq)
    wsp = model.W_rec.abs().mean().item()
    smooth_bu = smoothness(pbu, sbu, args.grid, args.grid)
    smooth_td = smoothness(ptd, std, args.grid, args.grid)
    return dict(cfg=cfg, seed=seed, align=obs, p=p, valid=valid, wsp=wsp,
                smooth_bu=smooth_bu, smooth_td=smooth_td, maps=(pbu, ptd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--P", type=int, default=20)            # stimulus patch size
    ap.add_argument("--freq", type=float, default=3.0)
    ap.add_argument("--latent", type=int, default=8)
    ap.add_argument("--t-steps", type=int, default=12)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--free-bits", type=float, default=0.2)
    ap.add_argument("--lreg", type=float, default=0.05)     # regularizer strength (match E1/E2b!)
    ap.add_argument("--K", type=int, default=12)            # orientations sampled for tuning
    ap.add_argument("--reps", type=int, default=16)         # phases/styles per orientation
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cond-enc", type=int, default=1)      # 1 = conditional encoder (gen-validity fix)
    ap.add_argument("--gen-gain", type=float, default=0.5)  # <1 reduces down-pass wash-out
    ap.add_argument("--init", default="mexican")            # 'mexican' (built-in topo) or 'random' (clean causal test)
    args = ap.parse_args()
    args.cond_enc = bool(args.cond_enc)
    if args.quick:
        args.grid, args.P, args.iters, args.seeds, args.reps, args.t_steps, args.n_perm = \
            14, 14, 120, 2, 8, 8, 500
    print(f"Device: {DEVICE}  grid={args.grid}x{args.grid}  P={args.P}  seeds={args.seeds}")

    conditions = {"E1_spatial": "spatial", "E2a_none": "none", "E2b_l1": "l1"}
    results = {name: [] for name in conditions}
    example_maps = {}
    results_file = f"exp_alignment_results_{args.init}.jsonl"
    open(results_file, "w").close()   # fresh log; crash-safe per-seed append
    for name, kind in conditions.items():
        for s in range(args.seeds):
            r = train_one(kind, s, args)
            results[name].append(r)
            if s == 0:
                example_maps[name] = r["maps"]
            with open(results_file, "a") as f:   # persist before next seed
                rec = {"cond": name, "seed": int(r["seed"]),
                       **{k: float(r[k]) for k in    # cast numpy floats -> JSON-safe
                          ("align", "p", "valid", "smooth_bu", "smooth_td", "wsp")}}
                f.write(json.dumps(rec) + "\n")
            print(f"  [{name} seed{s}] alignment={r['align']:+.3f}  p={r['p']:.3f}  "
                  f"gen_valid={r['valid']:+.2f}  smooth(BU/TD)={r['smooth_bu']:.3f}/{r['smooth_td']:.3f}  "
                  f"mean|W|={r['wsp']:.4f}")

    # ---- summary table + decision ----
    print("\n=== Per-condition summary (alignment + TOPOGRAPHY; chance align ~ 0) ===")
    print(f"{'condition':>12}  {'align':>7}  {'gen_valid':>9}  {'smoothBU':>8}  "
          f"{'smoothTD':>8}  {'mean|W|':>8}")
    means, smTD = {}, {}
    for name in conditions:
        a = np.array([r["align"] for r in results[name]])
        vv = np.mean([r["valid"] for r in results[name]])
        sbu = np.mean([r["smooth_bu"] for r in results[name]])
        std_ = np.mean([r["smooth_td"] for r in results[name]])
        ww = np.mean([r["wsp"] for r in results[name]])
        means[name] = a.mean(); smTD[name] = std_
        print(f"{name:>12}  {a.mean():+7.3f}  {vv:>+9.2f}  {sbu:>8.3f}  {std_:>8.3f}  {ww:>8.4f}")
    print("smoothness: LOW = smooth cortex-like map; ~0.785 = salt-and-pepper (random).")

    print("\n--- DECISION RULE (refined hypothesis: wiring cost -> TOPOGRAPHY) ---")
    e1_valid = np.mean([r["valid"] for r in results["E1_spatial"]])
    if e1_valid < 0.3:
        print(f"BLOCKED: gen_valid={e1_valid:+.2f} < 0.3 -- generation still invalid, maps meaningless.")
    else:
        sm_e1, sm_e2b, sm_e2a = smTD["E1_spatial"], smTD["E2b_l1"], smTD["E2a_none"]
        if sm_e1 < sm_e2b - 0.03 and sm_e1 < sm_e2a - 0.03:
            print(f"SIGNAL: E1 generation map is SMOOTHER ({sm_e1:.3f}) than L1 ({sm_e2b:.3f}) and "
                  f"none ({sm_e2a:.3f}) -> the spatial wiring-cost prior produces topographic "
                  "organization the controls lack. This is the contribution; scale up + stats.")
        else:
            print(f"NULL on topography: E1 smoothTD={sm_e1:.3f} not clearly below L1 ({sm_e2b:.3f}) / "
                  f"none ({sm_e2a:.3f}). Wiring cost does not produce smoother maps at this scale.")
    print("(Preliminary GTX-1650 run; a real verdict needs the proposal's full settings + >=5 seeds.)")

    # ---- figures ----
    twl = "twilight"
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    pbu, ptd = example_maps["E1_spatial"]
    for ax, m, t in [(axes[0], pbu, "perception map (BU)"), (axes[1], ptd, "generation map (TD)")]:
        im = ax.imshow(m.reshape(args.grid, args.grid), cmap=twl, vmin=-np.pi/2, vmax=np.pi/2)
        ax.set_title(t); ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.046, label="preferred orientation")
    fig.suptitle("E1: do the two orientation maps align?")
    plt.savefig("exp_alignment_maps.png", dpi=120, bbox_inches="tight")

    plt.figure(figsize=(6, 4))
    xs = list(conditions)
    ys = [np.mean([r["align"] for r in results[n]]) for n in xs]
    es = [np.std([r["align"] for r in results[n]]) for n in xs]
    plt.bar(xs, ys, yerr=es, capsize=5, color=["#2a9d8f", "#999", "#e76f51"])
    plt.axhline(0, color="k", lw=0.8); plt.ylabel("BU/TD alignment")
    plt.title("Map alignment by regularizer (E1 vs controls)")
    plt.tight_layout(); plt.savefig("exp_alignment_bars.png", dpi=120)
    print("\nSaved exp_alignment_maps.png, exp_alignment_bars.png")


if __name__ == "__main__":
    main()
