# 01 — Background: why a brain-like chip for slow, changing dangers

This note explains the *scientific motivation* in plain language first, then
connects each idea to the circuit that implements it. It is written so that a
non-specialist can follow the reasoning while every claim stays checkable.

---

## 1. The monitoring problem, in plain words

Put a vibration sensor on a bridge. Today's "smart" sensor is usually trained
once, in a lab, on recordings someone collected beforehand. Then it is frozen.

The world does not stay still:

- The **noise floor changes** with temperature, wind, traffic, and seasons.
- The **structure itself ages** — a cable relaxes, a bearing wears, concrete
  cracks slowly.
- The **sensor drifts** as its electronics age.
- The **load changes** — a factory runs faster, a reservoir fills, a new truck
  class crosses the span.

A frozen detector quietly degrades as these changes accumulate. Nobody re-tunes
it because there is no technician on every bridge. And when it *does* fire, it
can only say "alarm" or "no alarm" — it never says *how sure it is*.

Two failures therefore matter:

1. **It stops adapting** (concept drift / catastrophic forgetting).
2. **It is overconfident** — sure about a decision it has no right to be sure of.

Both are exactly what a brain does well, and both are what NeuroContinuum-1
builds into silicon.

---

## 2. What a brain does differently

Three properties of biological sensing are worth stealing:

**Sparingly (events, not frames).** The eye and the ear do not stream a full
picture to the brain 30 times a second. Neurons send *spikes* — brief,
all-or-nothing pulses — and only when something changes. Silence is the default.
A monitor that only wakes on *events* uses a tiny fraction of the energy of one
that recomputes a full window every cycle.

**Slowly and locally (learning at the synapse).** Learning in cortex is local:
a synapse changes based on the activity of the two neurons it connects plus a
slow chemical "how did that go?" signal. There is no global backpropagation
pass pushing error backwards through layers. This locality is what lets a brain
keep learning a lifetime without a teacher re-running it on stored data.

**Hesitantly (an honest "I'm not sure").** Biological systems behave
differently under uncertainty than certainty. A brain does not force a decision
when evidence is thin — it dithers, waits, or keeps multiple hypotheses alive.
That behavior *is* a confidence estimate.

---

## 3. Mapping biology onto circuits

| Biological idea | Circuit realization here | Where |
|---|---|---|
| Leaky membrane potential, threshold, fire | Fixed-point leaky integrate-and-fire neuron | `rtl/lif_neuron.v` |
| Sliding "modification threshold" (BCM) | Weight update judged against a slow trace of recent activity | `rtl/bcm_plasticity.v` |
| Stochastic vesicle release / channel noise | LFSR-driven membrane noise → Monte-Carlo trials | `rtl/stochastic_neuron.v` |
| "First neuron to respond decides" | Race-based winner latch + timestamp | `rtl/first_spike_decoder.v` |
| Synapses failing with age / radiation | Write-refusal detection + spare-column remap | `rtl/fault_monitor.v`, `synapse_remap.v` |

### 3.1 Novelty #1 — a threshold that moves (continual learning)

Most hardware Hebbian/STDP rules compare activity to a *fixed* threshold. That
means a strong input ratchets its synapse up forever, and a quiet period slowly
erases what was learned — forgetting.

BCM (Bienenstock–Cooper–Munro) fixes this by making the threshold itself a slow
function of recent activity: when the postsynaptic neuron has been very active,
the threshold rises, so only unusually strong responses keep potentiating. When
activity is quiet, it falls, so weak connections can re-grow. The rule therefore
*flips its own sign* depending on context. `test_novel_features.py` demonstrates
this directly: the same synapse potentiates when activity is above the threshold
and depresses when below, and the threshold measurably slides up under
sustained drive and back down when quiet.

Pair that with synaptic scaling (homeostasis) and the network tracks a drifting
input distribution without the whole system saturating or collapsing — measured
in the demo as accuracy recovering after a regime change.

### 3.2 Novelty #2 — noise as a feature (calibrated uncertainty)

A deterministic chip gives the same answer every time for the same input, so
running it twice tells you nothing new. Inject a controlled amount of noise into
the membrane and the *same* input now produces *different* spike patterns on
each run. Repeat a handful of times and count how often each output neuron wins:
that histogram is a posterior over classes.

This is on-chip Monte-Carlo sampling with no backpropagation and no cloud. From
it we read three quantities every window:

- the **decision** (arg-max of the vote histogram),
- the **confidence** (probability mass on the winner),
- the **entropy** (how spread the vote was).

The honesty is measured as **ECE** (Expected Calibration Error): average
`confidence − accuracy`. The demo achieves ECE 0.10–0.24 — meaning when the chip
says "70% sure" it is right about 70% of the time.

### 3.3 Novelty #3 — decide on the first spike

Rate coding spends tens of spikes to convey one number, then needs accumulators
to read it back. Temporal coding encodes strength in *when*: a stronger input
makes the neuron fire earlier. A race decoder then only needs a priority encoder
and a latch — the first neuron to fire wins, and its identity is the answer.
Sub-millisecond, one spike, no multiply.

### 3.4 Novelty #4 — a chip that repairs itself

Silicon develops stuck-at faults; analog/memristive synapses drift and pin to a
rail. A learning core is especially vulnerable because a dead synapse silently
corrupts what it learns.

Detection here is based on **write refusal**: a healthy synapse always stores
what its plasticity rule asked for. A stuck cell rejects the write, so
`what-was-requested ≠ what-it-holds`. Crucially this does **not** confuse "the
weight is at the rail" with "the weight is broken" — a synapse legitimately
resting at maximum shows request == hold and is never flagged. Once detected,
the logical synapse is re-pointed at a spare column, so the network sees the
same input width either way: the repair is transparent.

---

## 4. Why bridges, dams, and factories

The sensor is always an accelerometer/geo-sensor; only the meaning of the
classes changes. That makes one chip and one datapath serve three verticals,
demonstrated directly by `simulator/domains.py`:

- **Bridge** — traffic rumble, wind buffeting, crack acoustic emissions,
  expansion-joint impacts.
- **Dam** — hydrostatic baseline, seepage-channel trickle, flood loading,
  freeze cracking.
- **Factory** — running-speed harmonics (imbalance), periodic bearing impacts,
  sub-harmonic looseness.

Each generator also deliberately *drifts* over time (load rises, sensor noise
floor climbs), so the continual-learning claim is tested against a moving target
rather than a static test set.

---

## 5. Where else this applies

The same "slow, changing danger, far from a power outlet, where being confident
without cause is dangerous" shape fits: earthquake and volcano early warning,
pipeline and border monitoring, predictive maintenance on remote machinery,
grid/transformer health, and seismic networks on other planets. All want a
device that keeps learning on-site, states its uncertainty, and survives a few
dead transistors without failing.

---

**Next:** `02_architecture.md` for the concrete chip specification.
