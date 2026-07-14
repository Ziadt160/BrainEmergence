"""
movable_test.py -- does the topographic map appear if neurons ARRANGE THEMSELVES?
================================================================================
The grid experiments asked "is function smooth on the grid we imposed?" -> no.
This asks the user's question instead: if neurons that FIRE TOGETHER move close
together (co-firing attracts), does a clean orientation map emerge?

Cheap post-hoc test (no movable-neuron model yet): train a normal net, then embed
its neurons by co-firing similarity (spectral / Laplacian eigenmaps = "fire together
-> placed close") and colour each by preferred orientation. Compare:
  LEFT  : the imposed grid          (what we measured -> scattered)
  RIGHT : self-arranged by co-firing (the user's idea -> organized?)

Honest caveat: oriented neurons' tuning curves live on a circle, so the embedding
will TEND to recover a ring. The real question is whether orientation varies
*smoothly* across it (a map) vs fragments.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from exp_alignment import BiSheet, grating, grating_batch, concept, DEVICE


def train(model, iters, P, freq, batch, seed):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for it in range(iters):
        x, th = grating_batch(batch, P, freq, gen)
        c = concept(th)
        A = model.up(x)
        mu, logvar = model.encode(A, c)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        logits, _ = model.generate(z, c)
        recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.size(0)
        kl = torch.clamp((-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(0), min=0.2).sum()
        loss = recon + kl
        opt.zero_grad(); loss.backward(); opt.step()
        if it % max(1, iters // 4) == 0:
            print(f"  iter {it}/{iters} loss {loss.item():.1f}")
    return gen


@torch.no_grad()
def response_matrix(model, K, reps, P, freq, gen):
    thetas = torch.linspace(0, np.pi, K + 1)[:-1]
    R = torch.zeros(model.N, K, device=DEVICE)
    for k, th in enumerate(thetas):
        phases = torch.rand(reps, generator=gen) * 2 * np.pi
        x = torch.stack([grating(P, th, ph, freq) for ph in phases]).to(DEVICE)
        R[:, k] = model.up(x).mean(0)
    return R.cpu().numpy(), thetas.numpy()


def local_scatter(pos, pref, k=6):
    """Mean orientation difference to the k nearest neighbours in `pos` (radians;
    low = locally organized). pos: (n,2), pref: (n,)."""
    D = np.sqrt(((pos[:, None] - pos[None, :]) ** 2).sum(2))
    np.fill_diagonal(D, np.inf)
    out = []
    for i in range(len(pos)):
        nn = np.argsort(D[i])[:k]
        d = np.abs(np.angle(np.exp(1j * 2 * (pref[i] - pref[nn]))) / 2)
        out.append(d.mean())
    return float(np.mean(out))


def main():
    torch.manual_seed(0)
    H = W = 22; P = 20; freq = 3.0; T = 14
    model = BiSheet(H, W, P, latent=8, t_steps=T, cond_enc=True, gen_gain=0.5, init="random").to(DEVICE)
    print(f"Device {DEVICE}  grid {H}x{W}  (random init)")
    gen = train(model, iters=2500, P=P, freq=freq, batch=48, seed=0)

    R, thetas = response_matrix(model, K=24, reps=16, P=P, freq=freq, gen=gen)
    ang = 2 * thetas
    vx = (R * np.cos(ang)).sum(1); vy = (R * np.sin(ang)).sum(1)
    pref = 0.5 * np.arctan2(vy, vx)
    sel = np.sqrt(vx ** 2 + vy ** 2) / (R.sum(1) + 1e-6)

    # focus on the well-tuned population (the ones that actually carry orientation)
    keep = sel > np.median(sel)
    Rk, prefk = R[keep], pref[keep]

    # co-firing similarity = correlation of tuning curves
    Rc = Rk - Rk.mean(1, keepdims=True)
    norms = np.sqrt((Rc ** 2).sum(1)) + 1e-9
    S = (Rc @ Rc.T) / np.outer(norms, norms)

    # spectral embedding (Laplacian eigenmaps): "fire together -> placed close"
    Aff = np.clip(S, 0, None); np.fill_diagonal(Aff, 0.0)
    dinv = 1.0 / np.sqrt(Aff.sum(1) + 1e-9)
    Lsym = np.eye(len(Aff)) - (dinv[:, None] * Aff * dinv[None, :])
    vals, vecs = np.linalg.eigh(Lsym)
    emb = vecs[:, 1:3]   # first two non-trivial eigenvectors -> 2D layout

    # imposed-grid positions of the kept neurons
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    grid_pos = np.stack([yy.reshape(-1), xx.reshape(-1)], 1).astype(float)[keep]

    s_grid = local_scatter(grid_pos, prefk)
    s_emb = local_scatter(emb, prefk)
    print(f"\nLocal orientation scatter (low = organized):")
    print(f"  imposed grid         : {s_grid:.3f}")
    print(f"  self-arranged (co-fire): {s_emb:.3f}")
    print(f"  -> {'ORGANIZED when self-arranged' if s_emb < s_grid - 0.05 else 'no improvement'}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    sc0 = ax[0].scatter(grid_pos[:, 1], grid_pos[:, 0], c=prefk, cmap="twilight",
                        vmin=-np.pi/2, vmax=np.pi/2, s=28)
    ax[0].set_title(f"Imposed grid (what we measured)\nlocal scatter {s_grid:.2f}")
    ax[0].invert_yaxis(); ax[0].set_aspect("equal"); ax[0].axis("off")
    ax[1].scatter(emb[:, 0], emb[:, 1], c=prefk, cmap="twilight",
                  vmin=-np.pi/2, vmax=np.pi/2, s=28)
    ax[1].set_title(f"Self-arranged by co-firing (your idea)\nlocal scatter {s_emb:.2f}")
    ax[1].set_aspect("equal"); ax[1].axis("off")
    fig.colorbar(sc0, ax=ax, fraction=0.04, label="preferred orientation")
    fig.suptitle("Same neurons, two layouts: does the map appear when they self-arrange?")
    plt.savefig("movable_test.png", dpi=120, bbox_inches="tight")
    print("\nSaved movable_test.png")


if __name__ == "__main__":
    main()
