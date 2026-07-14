# I taught a spiking "brain" to imagine — then spent months proving it wasn't good enough

*An honest post-mortem of a solo AI project, built on a 4 GB laptop GPU, that answered its
question instead of defending its thesis.*

---

I started with a romantic idea: build a neural network that works a little more like a brain.
Not a metaphor-brain — an actual **spiking** network, where neurons fire discrete pulses over
time, wired into a 2D sheet with zones, the way cortex is. And I wanted it to do something
brains are thought to do: **recognize things by imagining them.** Show it a corrupted image, and
instead of just guessing, it would *reconstruct* what it expected to see, look again, and decide.

Perception as a loop. Analysis by synthesis. It's an old, beautiful idea.

Over a few months and about thirty experiments, I built it, and it worked — in the sense that it
produced genuinely striking behavior. And then, one careful experiment at a time, I proved to
myself that it wasn't actually better than much simpler methods. This is the story of that,
because I think **the second half is the more useful half.**

## The part that looked like magic

Here's the thing I built, doing its trick. I take an image, delete a third of it — simulating a
dead sensor — and the spiking network fires across its sheet, imagines the missing strip, and
re-reads the object:

![self-repairing perception](media/demo_brain_fashion.gif)

Every frame of that is the real trained model. No stored copy of the answer; it reconstructs the
missing region from what it can still see, then recognizes it.

And because the same network runs *both directions* — it can recognize *and* generate with one
shared set of weights — it can also run with **no input at all.** Clamp a concept, let activity
flow the other way through the sheet, and watch a digit form out of the spikes:

![the brain dreams](media/demo_dream.gif)

Fix the *class* and sweep the latent "style," and it imagines the same digit in many
handwritings — it has learned to separate *what* a thing is from *how* it's drawn:

![one concept, many styles](media/demo_dream_styles3.gif)

It felt like magic.

**That's exactly when I got suspicious.**

## The one rule I gave myself

Here's the rule, and it's the most important thing in this whole post:

> **Before you believe your idea works, run the experiment that could prove it doesn't.**

Not after you write it up. Not "future work." First. The baseline a skeptical reviewer would
reach for on their first read — you run *that*, and you run it honestly, and you let it win if it
wins. Most of the value in this project came from obeying that rule even when I didn't want to.

So I asked the obvious skeptical question: *does imagining the missing pixels actually beat just
training the model on occluded images in the first place?*

## Death by a thousand baselines

**Round 1 — "just augment your data."** I trained the same classifier with occlusion
augmentation and compared. Augmentation improved occluded accuracy by **+0.23**; my imagination
loop, **+0.07** — and adding imagination on top of augmentation did *nothing*. A one-line
reviewer comment ("why not just augment?") beat months of work, 3 to 1. The project looked dead.

**Round 2 — the rescue.** But there was a real idea hiding in the wreckage. Augmentation can only
prepare you for corruptions you *anticipated*. What about corruptions you didn't? I set up a
**leave-one-corruption-out** test: train broad augmentation on every corruption *except* a
held-out one, then, at *equal test-time compute*, compare feedforward vs. test-time augmentation
vs. MEMO (a strong test-time-adaptation baseline) vs. my imagination loop — on the held-out
corruption. And it **survived**: on unforeseen, structured missing-data, imagination beat all
three, across MNIST, Fashion-MNIST, and CIFAR-10. There was even a gorgeous result — scattered
pixel-dropout was a *loss* on sparse digits but the *biggest win* on rich natural images. Same
corruption, opposite outcome, because redundancy makes the gaps fillable. I had a rule:
*reconstructability = structure × redundancy.* I thought I had a paper.

**Round 3 — the control I didn't want to run.** Then I noticed the confound. My imagination loop
gets the **mask** — it knows *which* pixels are missing. My baselines didn't. So I gave a plain
model the mask too, in the strongest form: trained with masking augmentation, mask as input. And
the result was brutal: a **zero-parameter trick — just filling the hole with the average of the
visible pixels — matched my generative model in 7 of 8 cases.** The headline was mostly "I had
the mask and the baselines didn't." Only one narrow cell survived (large contiguous holes, where
a blur-fill genuinely isn't enough).

**Round 4 — the more ambitious version.** Maybe the wrong move was *repairing the input*; maybe I
should classify *generatively* — pick the class whose imagined version best explains the image. On
MNIST it looked spectacular (+0.41 over a naive baseline!). On a *fair* baseline it shrank, and on
Fashion-MNIST it **didn't replicate at all.** A one-dataset win that dies on the second dataset
isn't a result; it's noise with good PR.

**Round 5 — its home turf.** "Generate to understand" is supposed to shine when labels are
scarce. So I tested it semi-supervised, 100 labels. It didn't just fail to help — at the fewest
labels it **actively hurt** (0.73 vs 0.76). Reconstruction is a weak learning signal; it drags the
features toward pixels and away from the class.

**Round 6 — the substrate itself.** Fine — is the brain-like part even pulling its weight? A
plain CNN beat my spiking network on clean accuracy by **18 points**. And on energy — the one
scoreboard where spiking is *supposed* to win — I measured it honestly (spikes → synaptic
operations → joules) and found a plain MLP **dominates** the spiking brain: same accuracy, a third
of the energy. Spiking only beat the *convolutional* net, which is a bar so low it doesn't count.

![accuracy vs energy — the MLP is Pareto-optimal](figures/energy_scoreboard.png)

**Round 7 — the original dream.** The whole project was named "BrainEmergence" for the hope that
brain-like *structure* would emerge on its own. It doesn't. Topographic maps only appear if you
*impose* them — I showed it three ways, including a falsification control. The map in the
self-organization demo below is real, but it needs an explicit "fire together, move together" rule;
it does **not** emerge from the dynamics:

![self-organizing map](media/demo_topo.gif)

Seven rounds. Seven honest losses. Every time I gave the boring baseline a fair shot, it won.

## The one thing that lived

Stripped of everything that didn't survive, here is the honest, narrow, *true* finding:

> Test-time generative completion beats a trivial fill **only** when the missing region has
> texture that averaging can't reproduce — and even then, the advantage *shrinks* as the
> underlying classifier gets stronger.

That's it. It's small. It's also correct, reproducible, and — checked against a 2025 literature
review — a slice nobody had characterized quite this way. A modest true thing is worth more than a
big false one.

## Where "brain-inspired" actually pays off (spoiler: not accuracy)

The most useful thing I learned is *where the whole family of ideas belongs.* I'd been grading
brain-inspired models on a CNN's report card — accuracy — and that's the wrong scoreboard. Reading
the current literature (topographic nets like TDANN and TopoLM; spiking transformers like
Spikformer; the 2025 predictive-coding surveys; the Sensorium neural-prediction benchmarks), the
pattern is consistent: **brain-inspired methods compete on *energy* (neuromorphic hardware), on
*explaining neural data*, and on *biological plausibility* — almost never on raw accuracy.** My
experiments "failing" on accuracy is exactly what that literature would predict. Spiking pays off
on spiking hardware. That's not a disappointment; it's a map.

## What I actually learned

- **Run the killer baseline first.** It's the difference between research and marketing. It cost
  me my thesis and saved me from publishing something wrong.
- **Negative results are results.** "Here's what doesn't work, and precisely why" is a
  contribution — it saves the next person months.
- **You can be right about the idea and wrong about the battlefield.** Generative perception is a
  real, powerful idea; I kept aiming it at problems a cheap discriminative model already solves.
- **Watch your confounds.** My best "win" was my method quietly having information the baselines
  didn't. The mask. Always ask what your method knows that its opponent doesn't.
- **Know when *not* to refactor.** Late on, I nearly "cleaned up" duplicated helper code — until I
  noticed the copies had *diverged*, and merging them would have silently changed results I'd
  already reported. Sometimes the disciplined move is to leave working code alone.

## Was it worth it?

I did not build a state-of-the-art anything. I built a spiking network that recognizes at ~93%,
dreams legible digits, and repairs its own degraded perception — and then I demonstrated, seven
different ways, that simpler methods do the actual job better. If you're keeping score on models,
that's a loss.

But I don't think that's the score that matters. What I actually have is a way of working: I
generate bold ideas *and* I kill them honestly, with the exact tests that would embarrass me if I
skipped them. In a field full of confident overclaiming, that's the rarer skill — and it's the one
I'd want a collaborator, or an employer, to see.

The ideas were fun. Killing them cleanly was the point.

---

*Code, all thirty-odd experiments, and the demos above are in the repository. Everything is real
model output; every claim has a figure; every negative result is reported as plainly as the
positive ones.*
