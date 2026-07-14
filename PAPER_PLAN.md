# Paper plan — "When does imagining the input beat just augmenting?"
### A reconstructability decision rule for test-time generative completion vs data augmentation

> Honest tier: **workshop / modest empirical-study venue** (ICBINB, a robustness/TTA workshop,
> or a short empirical-study track). NOT a flagship. It is a *characterization* paper, not a
> "we beat SOTA" method paper. Built on the project's gating + reconstructability results.

## 1. The contribution (one sentence)
A predictive, **compute-matched** rule for *when* test-time generative completion helps over
data augmentation: it wins **iff** the corruption is (a) **reconstructable** (structured
missing-data — not noise, contiguous > scattered) **AND** (b) **outside the training
augmentation distribution** (un-anticipated). It ties/loses otherwise.

## 2. Why it's a contribution (the gap — confirmed open by the deep lit review)
- DDA (CVPR 2023) shows generative test-time projection helps; Mintun (NeurIPS 2021) shows
  augmentation generalizes only to *perceptually similar* corruptions. **Neither gives a
  predictive rule for which corruption favors which strategy at equal compute.**
- The deep review (verifier) found **no published predictive reconstructability decision rule
  for compute-matched generative-completion-vs-augmentation selection.** That is the wedge.
- Framing is *characterization*, never "our method is more robust" — that claim already lost
  3:1 to augmentation in this project and is owned by DDA at scale.

## 3. The MAKE-OR-BREAK experiment (do this FIRST — `paper_exp1.py`)
Everything hinges on surviving the reviewer's two killers: *"just augment for everything"* and
*"give augmentation the same test-time compute."* So the first experiment is **leave-one-
corruption-out + compute-matched**:
- Train a model augmented on **all corruptions except one held-out** X.
- On held-out X, compare at **equal inference compute (N forward passes)**:
  1. broad-aug **feedforward** (reference),
  2. broad-aug **+ test-time-augmentation marginalization** (N views — the compute-matched augmentation baseline),
  3. broad-aug **+ iterative imagination** (N rounds — our method).
- **The paper lives iff:** for at least one held-out *missing-data* corruption, imagination
  beats BOTH (1) and (2) with non-overlapping error bars (>=5 seeds). On noise / in-distribution
  corruptions it should NOT win (that's the rule's other half).
- **If TTA-marginalization matches imagination on the held-out missing-data corruptions, the
  paper is dead** (augmentation+cheap-TTA covers it) — stop and write the portfolio piece.

## 4. Full experiment program (only if make-or-break survives)
- Datasets: MNIST, Fashion-MNIST, CIFAR-10 (infra exists), + ideally one more.
- Corruption taxonomy along the two axes: {missing-data: band/patch/scattered/quadrant} x
  {in-aug, held-out} + {noise, blur} as the "should-not-help" negative controls.
- Baselines (all compute-matched): broad augmentation, TTA-marginalization, MEMO (entropy TTA),
  and a clean discriminative model. Position against DDA conceptually (cite, don't re-run).
- Deliverable headline figure: a **decision map** — for each (corruption, in/out-of-aug),
  which strategy wins — plus the predictive rule fit and held-out validation of the rule.
- Rigor: >=5 seeds, error bars, significance tests, pre-registered metric, released code.

## 5. Honest risks
- **Scale erosion:** Mintun shows a few augmentations can cover many corruptions; at scale,
  broad-aug + TTA may close the gap the gating test opened at toy scale. The make-or-break tests this.
- **Reviewer:** "augment for more / it's DDA with a heuristic." Answer only via the
  *un-augmentable* framing + compute-matching + the predictive rule (the novelty is the RULE).
- **Compute:** stays small (MNIST-family + CIFAR-10) — feasible on the 4 GB GPU. CIFAR needs the conv net (`cifar_imagine.py`).

## 6. What to DROP (dead weight, per the project's own findings)
Spiking neurons, topographic zones, movable neurons, "brain emergence" framing. None are
load-bearing; all are decoration. This is a plain ML robustness-characterization paper.

## 7. The path
1. Run `paper_exp1.py` (make-or-break, leave-one-out + compute-matched). ~30 min.
2. If it survives -> scale the program (§4), ~weeks of careful work.
3. If it dies -> write the portfolio/process piece; the honest answer was "no."
