# Research Proposal — Aligned Perception & Generation Topography in a Shared-Weight Spiking Network

> Status: novelty VERIFIED by deep literature review (2026-06). A defensible gap exists:
> no prior work combines (1) a shared recurrent weight matrix used bidirectionally,
> (2) spiking neurons, (3) analysis of the generative/top-down topographic map, and
> (4) a perception-vs-generation map-alignment test with a spatial wiring-cost prior
> as candidate mechanism. Re-check arXiv/bioRxiv at submission time (fast-moving area).

## 1. One-sentence thesis
A spatially-embedded recurrent **spiking** network trained to **both perceive and
generate through shared weights** develops a **topographic map in its generative
(top-down) pathway that aligns with its perceptual (bottom-up) map**, and a
**spatial wiring-cost prior is necessary** for that alignment.

## 2. Contribution claims (what a reviewer must be able to check)
- **C1 — Phenomenon:** top-down generative activity is topographically organized and
  spatially aligned with the bottom-up perceptual map (quantified, not eyeballed).
- **C2 — Cause (necessity, not sufficiency):** the spatial wiring-cost prior is
  NECESSARY for the alignment — removing it (and matched generic-sparsity controls)
  significantly degrades alignment. Do NOT claim sufficiency: the lit review refuted
  "spatial prior alone drives brain-like organization" (3-0); effects live in a
  regularization sweet spot.
- **C3 — Mechanism / confound control:** isolate alignment caused by the spatial prior
  from alignment that weight-sharing produces *mechanically*. Weight-sharing may force
  some alignment on its own, so the design needs a shared-weight + no-spatial-cost cell.
- **C4 — Grounding (optional, strengthens):** the emergent maps recapitulate known
  cortical structure (orientation pinwheels / retinotopy) and/or alignment predicts
  better generation quality.

## 2.5 Related work — the four camps, and the wedge (lit-review grounded)
Each closest neighbour misses exactly ONE required ingredient:
- **Topographic DNNs** — TDANN (Margalit, *Neuron* 2024), All-TNNs (*Nat Hum Behav*
  2025), TopoLM/TDSNN/TopoNets (2024-25): feedforward, discriminative; topography from a
  spatial-smoothness loss on the *bottom-up* pathway. → no generation, no top-down map.
- **Shared recognition+generation** — spiking Helmholtz machine (Sountsov & Miller,
  *Front. Comput. Neurosci.* 2015); predictive coding made generative (Sun & Orchard,
  *Neural Computation* 2020): bidirectional, some spiking, but **separate** recognition
  vs generative weights and **no topographic analysis**.
- **Generative topography** — Topographic VAE (Keller & Welling, NeurIPS 2021): topography
  in a generative model, but non-spiking, latent-grid (not a recurrent spatial wiring-cost
  prior), and **no perception-vs-generation alignment test**.
- **Spatially-embedded recurrent nets** — seRNN (Achterberg, *Nat Mach Intell* 2023):
  spatial wiring-cost yields functional clustering (absent under matched L1), but
  **discriminative and never run generatively**.
- **Biology (motivation)** — Senden et al. (*Brain Struct Funct* 2019, 7T fMRI):
  perception and mental imagery share retinotopic maps in V1-V3 — phenomenon is real in
  brains, untested in a shared-weight spiking model.

## 3. The gaps that block "paper" today → how each is closed
| Gap | Why it blocks publication | Action to close it |
|-----|---------------------------|--------------------|
| Novelty | known combo of known parts | lit review confirms the precise wedge (generative-side topography + map alignment in a shared-weight spiking net); position vs TDANN / spiking-VAE / predictive coding |
| Toy data | MNIST ≈ auto-reject | move to V1-style stimuli with *known* ground-truth topography (see §4) |
| No baselines | "isn't it just X?" | a control per claim (§6) — wiring-cost, weight-sharing, spatial-embedding, spiking, feedforward-only |
| No rigor | single seed, no stats | ≥5 seeds, CIs on every metric, significance tests across conditions |
| Cosmetic biology | zones/Mexican-hat imposed | make topography *emergent & measured*, not assumed; compare to cortex quantitatively |
| Won't scale | dense N×N on CPU | GPU + sparse/convolutional connectivity |

## 4. Stimuli (chosen so topography is measurable & cortically comparable)
- **Primary:** natural image patches (e.g. van Hateren / BSDS crops) + oriented
  gratings → enables orientation-tuning and retinotopy analysis with known ground truth.
- **Sanity rung:** Fashion-MNIST (drop-in, harder than MNIST) for pipeline checks only.
- **Optional category-topography arm:** small faces/objects/scenes set (IT-level), to
  connect to the TDANN line — only if compute allows.

## 5. Metrics (must be quantitative & falsifiable)
- **Topographic quality:** map smoothness = mean |Δ(preferred feature)| over neighbours;
  pinwheel density for orientation maps (compare to cortical ~π/hypercolumn²);
  retinotopy = corr(neuron position, RF-centre position).
- **Map alignment (core quantity):** per neuron, preferred feature on the UP pass
  (θ^BU) vs preferred feature on the DOWN/generation pass (θ^TD); **alignment =
  (circular) spatial cross-correlation / RSA between the θ^BU and θ^TD cortical-sheet
  maps, PERMUTATION-TESTED** (seRNN p_perm framework). This is the headline number;
  pre-register it before running.
- **Generation quality:** FID (or patch log-likelihood / reconstruction error) — proves
  the generator works, otherwise alignment is vacuous.
- **Wiring cost:** connection-length distribution Σ|W_ij|·dist(i,j).

## 6. Experiment matrix (each row kills one objection)
Core design is a 2×2: {shared vs separate weights} × {spatial-cost vs L1-only}, plus refs.
| ID | Condition | Tests claim | Expected |
|----|-----------|-------------|----------|
| E1 | full model: shared-weight, spatial wiring-cost, spiking | C1 | aligned BU/TD maps, high permutation-tested alignment |
| E2a | shared-weight, **no** spatial cost | C2 + weight-share confound | alignment degrades; residual = weight-sharing's mechanical floor |
| E2b | shared-weight, **matched L1** (generic sparsity, no spatial cost) | C2 (the key control) | alignment ≈ E2a, ≪ E1 → spatial cost, not sparsity, drives it |
| E3 | **separate** encoder/decoder weights + spatial cost | C3 | lower alignment than E1 → weight-sharing matters |
| E4 | shuffle neuron positions (no spatial embedding) | sanity control | no topography at all |
| E5 | non-spiking (rate) version | isolate spiking's role | report honestly: helps / neutral / efficiency-only |
| E6 | feedforward-only topographic net (TDANN-style) | reference | perception map only, no generation map → our delta |
| E7 (opt) | alignment vs generation quality (FID) / PC error | C4 | alignment predicts better samples |

## 7. Expected figures
1. BU map vs TD map side by side (full model) + alignment scatter.
2. Alignment score across conditions E1–E6 with error bars (the money figure).
3. Emergent orientation/retinotopy maps vs cortical reference.
4. Wiring-cost vs alignment trade-off curve.
5. Generated samples + FID table.

## 8. Rigor protocol
≥5 seeds per condition; report mean ± 95% CI; paired tests (e.g. permutation /
bootstrap) for E1-vs-E2/E3; pre-register the alignment metric; release code + configs.

## 9. Compute
Current CPU + dense matrices is a prototype only. Paper needs GPU and sparse/conv
connectivity to reach the grid sizes where topographic maps are legible.

## 10. Target venues (realistic, ascending)
- Cosyne abstract/poster (low-bar milestone, fast feedback).
- Workshop: NeurReps / SVRHM / NeuroAI @ NeurIPS-ICLR.
- Conference: CCN (Cognitive Computational Neuroscience).
- Journal (full version): Neural Computation / PLoS Comp Biol / eLife.

## 11. Risks & gating
- **Primary risk:** lit review finds the gap already filled → pivot the claim
  (narrower spiking-specific result, or shift to the necessity-of-wiring-cost angle).
- **Secondary:** spiking adds nothing over rate (E5) → reframe spiking as the
  efficiency/biological-plausibility axis, not the performance axis.
- **Scope creep:** the category-topography arm (§4) is optional; cut if compute-bound.

## 11b. Novelty maintenance
Re-run the literature check (arXiv/bioRxiv) for topographic-generative or shared-weight
bidirectional spiking variants immediately before submission — the topographic-DNN /
spiking-topography area is moving fast (TopoLM, TDSNN, TopoNets all 2024-25).

## 12. Minimal first milestone (de-risk before committing)
Run **E1 + E2a + E2b** on oriented gratings / natural-image patches at a legible grid
size on GPU, ≥5 seeds, permutation-tested alignment. Decision rule:
- E1 alignment ≫ chance AND E1 ≫ E2b (L1 control) → **spatial prior causes alignment →
  paper is live.**
- E1 ≈ E2b → alignment is just sparsity/weight-sharing → **kill or pivot** (per §11).
Cost: three training conditions, one stimulus set. Cheapest possible test of the core claim.

## 13. Closest prior works (cite these)
- Margalit et al., *Neuron* 2024 — TDANN (feedforward topographic).
- All-TNNs, *Nature Human Behaviour* 2025 (feedforward topographic, smoothness loss).
- Keller & Welling, NeurIPS 2021 — Topographic VAE (generative topography, non-spiking).
- Sountsov & Miller, *Front. Comput. Neurosci.* 2015 — spiking Helmholtz machine (separate weights).
- Achterberg et al., *Nature Machine Intelligence* 2023 — seRNN (spatial cost, discriminative).
- Sun & Orchard, *Neural Computation* 2020 — making PC networks generative.
- Senden et al., *Brain Struct. Funct.* 2019 — shared perception/imagery retinotopy (7T fMRI).
