# BrainEmergence — Session Handoff (resume here)

This document is the single source of truth to continue the project in a fresh session.
Read it top to bottom. Companion docs: `README.md` (public story), `PAPER_PLAN.md` (paper roadmap),
`PROPOSAL.md` (the older, abandoned topography proposal — historical).

---

## 0. How to run (environment)

- **GPU env (use this):** `~/miniconda3/envs/bacteria_gpu/python.exe` — torch 2.5.1, **CUDA True**,
  NVIDIA GTX 1650 (4 GB, shared with the user's separate "bacteria" work — watch VRAM).
- **CPU-only fallback:** `~/miniconda3/python.exe` (torch 2.10 cpu). Anaconda3 has NO torch.
- **GPU gotcha:** this CUDA build lacks an nvrtc JIT DLL, so **complex-number ops fail on GPU**
  (`torch.exp(1j*x)`). Use real `cos/sin/atan2` instead. (Already handled in the code.)
- Long runs: launch in background. Data (MNIST/Fashion/CIFAR) auto-downloads to `./data`.
- Deps: `requirements.txt` (torch, torchvision, numpy, matplotlib).

---

## 1. TL;DR — current state (2026-06-24, evening — READ THE CRITICAL UPDATE)

> **⚠ CRITICAL UPDATE (2026-06-24 PM): the headline did NOT survive the mask-aware fairness control.**
> A fresh-eyes review caught that imagination uses the missingness MASK (it pastes observed pixels +
> inpaints) but the ff/TTA/MEMO baselines do NOT. Adding the missing mask-aware baselines
> (`mask_control.py`) — a trivial observed-mean fill, and a mask-input model trained with masking
> augmentation — **largely defeats the headline**: across MNIST+Fashion, imagination beats the
> mask-aware baselines on only **1 of 8 missing-data cells (MNIST band)**; on Fashion *every* cell
> loses to a mask-aware baseline, and Fashion patch loses to a TRIVIAL mean-fill (separated bars).
> Since imagination itself REQUIRES the mask, there's no regime where it wins but a mask-aware
> baseline can't be used. ⇒ **The paper as framed ("generative completion beats compute-matched
> augmentation on reconstructable missing-data") is NOT submittable** — the wins were largely a
> mask-access artifact. Same lesson as Level-3. CIFAR (the dramatic scattered +0.32) is UNTESTED with
> this control (needs a conv mask model) and is likely also vulnerable (trivial fill on 70%-observed
> natural images is strong) — TEST BEFORE CLAIMING. What honestly survives = a NEGATIVE/NULL result
> (ICBINB-style): "for occlusion-augmented models, test-time generative completion ≈ a trivial
> mask-aware fill; analysis-by-synthesis is subsumed by knowing the mask." Logs: run_mask_control.log;
> figs mask_control_{mnist,fashion}.png.

The project started as a vague "self-organizing spiking brain" and, through ~20 controlled
experiments, became **two things**:
1. A **rigorous negative-results study** (topographic maps don't emerge; brain-like structure is
   decoration; "imagination helps recognition" loses to data augmentation — and, per the update
   above, even the leave-one-out/compute-matched rescue loses to a *mask-aware* baseline).
2. (SUPERSEDED) what looked like a live workshop paper — a predictive decision rule for when
   generative completion beats augmentation. Its baselines were not mask-aware; once they are, the
   claim collapses to a near-null (see CRITICAL UPDATE).

**The original paper claim (now REFUTED by the mask-aware control):**
> ~~Test-time generative completion beats compute-matched data augmentation iff the corruption is
> reconstructable missing-data outside the training augmentation~~ — DEAD: a trivial mask-fill matches
> it in 7/8 MNIST+Fashion cells. Noise negative control still holds.

Honest tier NOW: **a negative/null result (ICBINB)** at best, pending the CIFAR mask-control check.
The mechanism (analysis-by-synthesis) is published (DDA, RDC); with a fair mask-aware baseline there is
no positive contribution left except possibly MNIST-band (single cell, non-replicating). Drop ALL
spiking/topographic/"brain" framing regardless — decoration.

---

## 2. The paper evidence so far (the live result — FINAL, 5 seeds, +MEMO, 2026-06-24 PM)

Protocol (`paper_exp1.py`, `paper_exp1_cifar.py`): **leave-one-corruption-out** (train augmentation on
all corruptions EXCEPT the held-out one) **+ compute-matched** (imagination's N iterative rounds vs
N-view TTA-marginalization AND MEMO entropy-min TTA). "Imagination wins" = beats ALL THREE baselines
(ff, TTA, MEMO) with separated error bars. **All three datasets now at 5 seeds with the MEMO baseline.**
Results JSON: `paper_exp1_{mnist,fashion,cifar}.json`. Headline figure: `decision_map.png`. Draft: `PAPER.md`.

DECISION MAP — imagination's margin over the strongest of ff/TTA/MEMO (**bold** = separated error bars):

| corruption \ dataset | MNIST | Fashion | CIFAR-10 |
|---|---|---|---|
| **band** (contig)    | **+0.059** | **+0.036** | +0.016 |
| **patch** (contig)   | +0.002 | **+0.068** | **+0.046** |
| **scattered**        | −0.024 | −0.051 | **+0.322** |
| **noise** (control)  | −0.067 | −0.054 | −0.000 |

5 clean wins (band MNIST/Fashion, patch Fashion/CIFAR, scattered CIFAR). Raw held-out accuracies in
`results_table.md`.

**Key reads:** (a) imagination wins on held-out *reconstructable missing-data*. (b) **MEMO SURVIVED —
the make-or-break passed.** Both compute-matched baselines fail to fill holes: TTA actively HURTS
(−0.06…−0.19), MEMO ≈ ff in all 12 cells (verified genuinely adapting via param-movement diagnostic —
NOT a no-op; entropy-min sharpens but can't reconstruct). So *no discriminative test-time method fills
structured holes; only generative completion does.* (c) Negative control holds (noise: no win; cleanest
on MNIST/Fashion, neutral on CIFAR). (d) **The contiguous-vs-scattered boundary FLIPS with redundancy** —
scattered LOSES on sparse MNIST/Fashion but is the BIGGEST win on rich CIFAR (+0.322). A single-factor
account can't make that sign flip; the *product* (reconstructability = structure × redundancy) can.

**MEMO config (locked, defensible):** adapt ONLY normalization/gain params (BN affine for conv; per-neuron
tau_raw/thr_raw for spiking) — adapting all weights COLLAPSES under entropy-min. steps=3; lr=1e-3 (spiking)
/ 5e-4 (conv). Found via a strength sweep (see git/log); these are the strongest stable settings.

**Supporting characterization results (earlier, all in repo):**
- `imagine_robustness_*.png`: imagination self-repairs occlusion; peak gain MNIST +0.108 / Fashion +0.189 / CIFAR +0.098 (scales with redundancy).
- `imagine_noise.png`: imagination HURTS under noise (−0.03 to −0.04) — bottom-up handles noise.
- `imagine_patterns.png`: helps for ALL occlusion shapes on a clean-trained model.
- `paper_baseline.png`: on IN-distribution occlusion, augmentation beats imagination 3:1 (+0.231 vs +0.073), imagination adds nothing on top. (This is why the leave-one-out framing is essential.)

---

## 3. EXACT next steps to a submission (in order)

**DONE this session (2026-06-24 PM):** ✅ MEMO baseline added to both scripts + verified (the make-or-break);
✅ all 3 datasets at 5 seeds with MEMO; ✅ decision-map figure (`decision_map.png`) + `results_table.py`;
✅ `PAPER.md` draft (abstract→limits) with the final table. **The paper survived MEMO.** Remaining:

1. **Polish `PAPER.md`** — it's a complete draft; needs a real intro pass, figure embedding, and the
   reference list fleshed out (DDA/RDC/Mintun/Tent/MEMO already cited). Decide venue (ICBINB / robustness-TTA
   workshop / short empirical track).
2. **(optional) Tighten the CIFAR band cell** (+0.016, just misses separated bars) and the MNIST patch
   tie (+0.002) — more seeds or a slightly stronger completion would convert near-ties. The 5 wins already
   carry the rule; don't over-engineer.
3. **(optional) Formalize the rule** information-theoretically: test whether a conditional-entropy /
   mutual-information proxy of the corruption predicts the 12 measured margins (we already have the data).
   This converts the empirical rule into a small derivation-with-evidence. Check prior art (value-of-information,
   inpainting bounds) before claiming novelty.
4. **(new direction, user is keen) Level-3 generative classifier** — see §3.5 below.

**Make-or-break STATUS: PASSED.** MEMO did not close the gap (it provides no benefit on held-out
missing-data; verified live-not-broken). Residual risk is still scale (Mintun): only tested to CIFAR-10
on a 4 GB GPU.

## 3.5 Level-3 / generative-classifier direction (user's chosen next thread)
User is excited about the GENERATIVE (Bayes) classifier: classify by which class best *explains* x
(argmax_y p(x|y)), reusing `BiBrain` (the conditional VAE in `brain_generate.py`). Prototype built
(`gen_classifier.py`): scores each class by reconstruction of OBSERVED pixels (mask-aware) → can ignore
holes a feedforward net can't; includes an abstention/risk-coverage demo. **RUNG 0 DONE & it WORKS:** the
`--quick` smoke test was at chance (~0.18) but that was **UNDERTRAINING (3 epochs), NOT an encoder leak**
(my earlier "encoder class-leak" call was WRONG). At 10 epochs the generative classifier hits **0.875 clean
MNIST** at latent=20, with a large true-vs-wrong-y reconstruction gap (+121 → the decoder strongly USES the
class label). Constraining the latent HURT (0.875→0.819) — so keep latent=20. Lesson: train adequately
before diagnosing a failure mode. FULL COMPARISON (`gen_classifier.py`): vs a NAIVE disc, the generative classifier crushes missing-data
(band +0.41, patch +0.29), loses on clean (−0.07) and noise (−0.35). BUT that naive disc was unfair →
ran the fairness control (`gen_vs_disc_control.py`, the make-or-break, mirrors the paper's killer-baseline
discipline):

LEVEL-3 RESULT (MNIST, 3 seeds) — TWO REGIMES:
- **In-distribution** (disc gets mask + trained ON the test corruption): disc_mask_aug WINS (band 0.797 >
  gen 0.744, patch 0.781 > 0.757) AND wins clean (0.905 > 0.873). "Just augment for it" beats generative.
  Mask access ALONE (disc_mask, no aug) does NOT help (band 0.354) — the win needs augmentation.
- **Leave-one-out** (disc augments on the OTHER shapes, tested on a HELD-OUT/unforeseen shape): **GEN WINS**
  on contiguous — band gen 0.744 > disc 0.668 (+0.076, separated), patch 0.760 > 0.672 (+0.088, separated);
  scattered ties (gen 0.863 ≈ disc 0.870); loses clean & noise. The disc's hole-handling COLLAPSES on an
  unseen shape (band 0.797→0.668) while the clean generative classifier is shape-agnostic (0.744 always).

MNIST ⇒ Level-3 APPEARED to survive the unforeseen case (gen beats disc+aug LOO: band 0.744>0.668, patch
0.760>0.672). **BUT FASHION DID NOT REPLICATE** (`run_control_fashion.log`): Fashion LOO — band disc 0.757 >
gen 0.659, patch tie, scattered disc 0.793 > gen 0.701. So even on unforeseen shapes the augmented disc beats
the generative classifier on Fashion. WHY: (1) the VAE generative classifier is WEAKER on Fashion (clean 0.713
vs MNIST 0.873 — texture harder to model), (2) the augmented disc GENERALIZES BETTER on Fashion (band LOO 0.757
vs MNIST 0.668). Both push against generative.

⇒ **LEVEL-3 IS DEAD (with this generator): the MNIST win DID NOT REPLICATE on Fashion → not a robust result.**
A one-dataset positive that fails replication is not a finding. Matches the deep-research prediction: the
generative-classifier edge needs a STRONG generator (diffusion-scale), not a cheap VAE — with a weak generator
it wins only on the simplest data (MNIST) and loses as complexity rises (Fashion; CIFAR would be worse, and
CIFAR isn't even built). DO NOT pursue Level-3 further on this hardware/substrate (4GB GPU = no scale). Logs:
run_genclf_mnist.log, run_control_mnist.log, run_loo_mnist.log, run_control_fashion.log. Figs:
gen_vs_disc_control_{mnist,fashion}.png. Scripts gen_classifier.py / gen_vs_disc_control.py kept for the record.
THE PAPER (test-time completion) STANDS — robust across 3 datasets; Level-3 was the exploration beyond it and
it did not pan out.
Honest framing: mechanism is published (Diffusion Classifier ICLR2023, RDC ICML2024); wedge = the
reconstructability rule applied to generative-vs-discriminative SELECTION + the built-in uncertainty signal.
Other discussed threads (lower priority): generative REPLAY for continual learning (imagine past data to
avoid forgetting; reuses BiBrain; Deep Generative Replay Shin 2017 = prior art). Drop: stacking-for-language
(wrong battlefield — attention wins), "more powerful than backprop" learning (not reachable solo).

---

## 4. Script map (what to keep / what's historical)

**PAPER track (the live work):**
- `PAPER.md` — the workshop-paper DRAFT (abstract→limitations, final 5-seed+MEMO table). **The write-up.**
- `paper_exp1.py` — make-or-break, leave-one-out + compute-matched TTA+MEMO, MNIST/Fashion (`--dataset`). **CORE.** Writes `paper_exp1_<ds>.json`.
- `paper_exp1_cifar.py` — same protocol (incl. MEMO) with the conv net on CIFAR-10. **CORE.** Writes `paper_exp1_cifar.json`.
- `decision_map.py` — builds the headline Figure 1 (`decision_map.png`) from the 3 JSONs. **CORE figure.**
- `results_table.py` — emits the paper's main results table (`results_table.md`) from the 3 JSONs.
- `gen_classifier.py` — Level-3 generative (Bayes) classifier prototype (see §3.5). NEW direction; needs the latent fix.
- `gating.py` — earlier leave-corruption-out (one fixed held-out); superseded by paper_exp1 but useful.
- `paper_baseline.py` — the 2×2 that established augmentation beats imagination in-distribution (the autopsy).
- `imagine_helps.py` — `RecallBrain` (the spiking recognize+imagine model) + occlude + completion eval. **Imported by most paper scripts.**
- `imagine_robustness.py` / `imagine_noise.py` / `imagine_patterns.py` — the characterization sweeps (occlusion severity, noise, occlusion shape).
- `cifar_imagine.py` — `ConvBrain` (conv recognize+imagine) + CIFAR occlusion sweep. Imported by paper_exp1_cifar.
- `demo.py` — the visual demo (brain repairs occluded digits → `demo_imagination.png`).

**Earlier milestones (keep for the story, not the paper):**
- `brain_predict.py` (93% spiking classifier), `brain_generate.py` (bidirectional spiking VAE),
  `brain_scan.py` (per-zone emergence probes), `fix_generation.py` (generation-validity fix).

**Topography / emergence (the NEGATIVE results — historical):**
- `exp_alignment.py` (perception-vs-generation map alignment; null), `movable_model.py` (movable/co-firing
  neurons; map imposed not emergent), `movable_test.py` (dependency of movable_model), `snn_emergence.py`
  + `mnist_experiment.py` (the two original scripts).

---

## 5. The honest project arc (so you don't repeat dead ends)

1. Goal was "emergence" of brain-like structure. **Result: it does NOT emerge** — topographic maps must be
   imposed (init or explicit loss), shown 3 ways incl. a falsification control. The brain-like features
   (spiking, topographic zones, movable neurons) are **decoration**: a plain fixed grid matches them.
2. The one real positive: iterative top-down "imagination" self-repairs occluded input. BUT it **lost 3:1
   to data augmentation** on the corruption it was tested on (occlusion is augmentable). Paper died.
3. **The fix (current paper):** generative completion only beats augmentation on corruptions you did NOT /
   could NOT augment for. Leave-one-out + compute-matched demonstrates this. Survives across 3 datasets.

**A measurement bug we caught (important):** the first "smoothness" metric was dominated by untuned neurons
(everything looked "smooth ~0.06"); fixed to selectivity-weighted. Lesson: always check whether a metric is
measuring the population you think it is.

---

## 6. Deep-research findings (the lit landscape — read before claiming novelty)

A 13-agent web-research + adversarial-verification workflow concluded:
- **No flagship paper is reachable** from these assets. The core idea (analysis-by-synthesis at test time)
  is published: **DDA** (CVPR2023, arXiv:2207.03442), **Diffusion-TTA** (NeurIPS2023), **RDC** (ICML2024,
  arXiv:2305.15241), **CausalDiff** (NeurIPS2024, arXiv:2410.23091), **Generative Classifiers Avoid Shortcuts**
  (ICLR2025, arXiv:2512.25034). Where generative wins: **unforeseen/un-augmentable threats** (RDC 89% vs 3%).
- **The one unpublished slice = our paper:** a *predictive reconstructability decision rule* for
  compute-matched generative-completion-vs-augmentation selection. Workshop tier.
- **DO NOT pursue:** missing-modality medical synthesis (npj Prec Onc 2025: dropout beats synthesis);
  beating augmentation on standard ImageNet-C (DeepAugment+AugMix ~25 pts ahead); spiking/efficiency wedge
  (VAR generative classifier arXiv:2510.12060 doing it better); adversarial purification (discredited under
  adaptive attacks); SNN-robustness-as-advantage (arXiv:2512.22522 shows it's a measurement artifact).
- **Most interesting to FOLLOW (read, don't out-compete):** the generative-classifier / unforeseen-threat
  line — it's where the user's generative intuition is genuinely right, but the win comes from the modeling
  objective at scale, not a cheap/brain substrate.

---

## 7. Honest framing for the user (a strong self-taught researcher who values bluntness)

- This is **good research PROCESS, modest CONTENT**. The transferable win is the rigor: pre-registering
  predictions, running the killer baseline BEFORE writing, honest nulls, catching confounds.
- The live paper is **real but workshop-tier**, and could still narrow or die under the MEMO baseline.
- The bigger picture: the generative/mechanistic intuition is good but was repeatedly aimed at the wrong
  battlefield (augmentable corruption) on the wrong substrate (spiking/topographic). The right battlefield
  is **un-augmentable / unforeseen** corruption — which is exactly what the leave-one-out paper now targets.

---

## 8. First action in the new session
Run step 3.1 (CIFAR 5 seeds) and 3.2 (add MEMO baseline) — those two most determine whether the result
holds up. Then the decision-map figure (3.4). Everything needed is in `paper_exp1.py` / `paper_exp1_cifar.py`.
