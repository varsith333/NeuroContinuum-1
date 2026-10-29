# NeuroContinuum-1

**A continual-learning, uncertainty-aware, self-healing neuromorphic processor
for always-on structural & industrial health monitoring**
(bridges · dams · factories — and anywhere a silent, slow, changing danger hides).

```
  vibration  ──►  sensor ──► ADC ──► ╔══════════════════════════╗ ──► alarm / radio / dashboard
  (accel/geo)                    ║   NeuroContinuum-1  (chip)  ║       + confidence  "92% sure"
                                  ╚════════════════════════════╝
```

---

## The problem in one paragraph

Today's "smart" sensors are trained once in a lab, then frozen. Put one on a
bridge and it works until the world changes — the geology shifts, the seasons
change the noise floor, a bearing wears, a sensor ages — and then it quietly
fails. Worse, it can only ever say *"DANGER"* or *"SAFE"*, with no idea whether
it's right. For a safety-critical monitor that is not good enough.
**NeuroContinuum-1 keeps learning on the job without forgetting, is honest about
how sure it is, reacts in a blink, and repairs its own failing synapses.**

## Four novel contributions

Nothing below is in a shipped chip (Loihi 2, TrueNorth, SpiNNaker2, BrainScaleS-2):

| # | Novelty | What it buys | Python model | RTL |
|---|---------|--------------|--------------|-----|
| 1 | **Meta-plastic BCM + reward learning** — a *sliding* modification threshold | Continual adaptation, no catastrophic forgetting, no cloud retraining | `plasticity.py` | `bcm_plasticity.v` |
| 2 | **Stochastic neuron** — programmable membrane noise → on-chip Monte-Carlo | A **calibrated posterior** (confidence + entropy) on every decision | `stochastic.py` | `stochastic_neuron.v` |
| 3 | **First-spike temporal decoder** — race-based readout of *when*, not *how often* | Sub-millisecond, ultra-sparse inference (1 spike = 1 bit) | `temporal.py` | `first_spike_decoder.v` |
| 4 | **Self-healing synapse repair** — write-refusal detection + spare-column remap | Survives stuck/radiated cells; graceful degradation, never a crash | `selfhealing.py` | `fault_monitor.v` + `synapse_remap.v` |

## What it does (the demo)

A streaming **vibration classifier** over three domains with deliberately
*non-stationary* statistics (so continual learning is measured, not asserted):

- **bridge** — traffic / wind / crack-impact events
- **dam** — normal / seepage / flood / freeze
- **factory** — healthy / imbalance / bearing-fault / looseness

Each window the chip outputs **event class + confidence + entropy**, learns
continually from a single scalar reward signal (no backprop), and repairs
injected synapse faults.

**Measured results** (see `docs/04_benchmarking.md` to reproduce; the stream
changes regime **abruptly** at the halfway mark — class priors flip and the
noise floor steps up — and `pre`/`post` are accuracies before/after that change):

```
[LEARNING] bridge  -> pre 0.480  post 0.525  conf 0.497  ECE 0.117  calibrated ✓
[LEARNING] dam     -> pre 0.555  post 0.530  conf 0.625  ECE 0.240  calibrated ✓
[LEARNING] factory -> pre 0.425  post 0.400  conf 0.508  ECE 0.100  calibrated ✓
A/B after the change:  learning 0.37   vs   frozen 0.22   <- learning clearly wins
(by domain, post-change: bridge 0.53 vs 0.36, dam 0.53 vs 0.35, factory 0.40 vs 0.26)
```

Chance is 0.25. Energy vs conventional compute (event-driven, measured workload):

```
NeuroCore 0.06 µJ  vs  CPU 0.22 µJ (3.9×)  vs  GPU 0.13 µJ (2.3×)
```

## Repository layout

```
neuromorphic-chip/
├── README.md                 ← you are here
├── requirements.txt
├── run_all.ps1               ← Windows one-pass verification (no make needed)
├── Makefile                  ← top-level: demo, test, bench, rtl
├── simulator/                ← Python golden model (the "brain")
│   ├── neurons.py            LIF / Izhikevich populations
│   ├── plasticity.py         NOVEL#1 BCM meta-plasticity + reward-STDP
│   ├── stochastic.py         NOVEL#2 noisy neuron → Monte-Carlo uncertainty
│   ├── temporal.py           NOVEL#3 latency coding + first-spike decoder
│   ├── selfhealing.py        NOVEL#4 stuck-synapse repair (golden model)
│   ├── stream.py             drifting seismic hazard generator
│   ├── domains.py            bridge / dam / factory vibration generators
│   ├── encoding.py           rate / latency / poisson spike encodings
│   ├── network.py            generic lock-step spiking container
│   ├── continuum_net.py      the flagship end-to-end classifier
│   └── examples/             demos + self-checking tests
├── rtl/                      ← synthesizable Verilog (the "chip")
│   ├── lif_neuron.v  bcm_plasticity.v  stochastic_neuron.v
│   ├── first_spike_decoder.v  fault_monitor.v  synapse_remap.v
│   ├── neuro_core.v          structural top integrating all four novelties
│   ├── tb/                   self-checking testbenches
│   ├── run_tests.ps1         Windows test runner, Icarus Verilog (-Wave dumps wave.vcd)
│   ├── run_xsim.ps1          same tests on Vivado xsim (second independent simulator)
│   ├── create_project.tcl    builds rtl\NeuroContinuum\*.xpr for the Vivado GUI
│   └── Makefile              iverilog / Vivado-xsim flows
├── architecture/             ← cycle + energy analysis
│   ├── energy_model.py       per-op energy constants (Horowitz ISSCC'14)
│   ├── arch_sim.py           event/cycle simulator + CPU/GPU baselines
│   ├── workloads.py          extract hardware counters from real runs
│   ├── benchmark.py          the benchmark (writes results/)
│   └── results/              generated reports
└── docs/                     ← research & design notes
    ├── 01_background.md      why neuromorphic, the biology → circuit mapping
    ├── 02_architecture.md    chip spec (cores, precision, NoC, area, power)
    ├── 03_rtl_guide.md       module hierarchy, fixed point, how to simulate
    ├── 04_benchmarking.md    methodology + how to reproduce every number
    └── references.md         real papers & datasheets
```

## Quick start

```bash
pip install -r requirements.txt

# 1. verify the four novel blocks
python -m simulator.examples.test_novel_features

# 2. run the flagship continual-learning demo (bridge/dam/factory)
python -m simulator.examples.demo_continual --n 400

# 3. compare against a frozen network (why continual learning matters)
python -m simulator.examples.demo_continual --n 400 --no-learn

# 4. architecture + energy benchmark vs CPU/GPU
python -m architecture.benchmark

# 5. RTL simulation — Windows (no make needed; finds iverilog in C:\iverilog):
powershell -ExecutionPolicy Bypass -File rtl\run_tests.ps1
powershell -ExecutionPolicy Bypass -File rtl\run_tests.ps1 -Wave   # + wave.vcd

#    ...or with Vivado xsim (same PASS tokens; finds Vivado in C:\Xilinx / D:\Xilinx):
powershell -ExecutionPolicy Bypass -File rtl\run_xsim.ps1

#    ...and to open it in the Vivado GUI with waveforms (run once, then
#    double-click rtl\NeuroContinuum\NeuroContinuum.xpr):
& "D:\Xilinx\2025.2\Vivado\bin\vivado.bat" -mode batch -source rtl\create_project.tcl

#    ...or Unix with GNU make:  make -C rtl     (or `make -C rtl xsim` for Vivado)
```

Or run everything at once:

```bash
make          # = test + demo + bench
```

```powershell
# Windows, no make needed — same checks in one pass (stops at first failure):
powershell -ExecutionPolicy Bypass -File run_all.ps1
```

## Where does the reward come from in the field?

The demo feeds the learning rule a scalar reward from the known ground-truth
label — that is the honest way to *measure* the algorithm. On a real bridge or
factory floor there is no label, and the chip must source its reward some other
way. Three deployment options, all compatible with the existing hardware
interface (a single scalar into the `reward` port):

1. **Technician confirmation** — an inspector marks a report ("that was a truck,
   not a crack"); one bit of human feedback aligns the readout.
2. **Cross-sensor consensus** — neighbouring accelerometers or a second sensing
   modality (strain gauge, acoustic) vote; disagreement produces a negative
   reward, agreement a positive one.
3. **Unsupervised surprise** — the chip's own calibrated confidence acts as the
   teacher: high-confidence predictions that keep being surprising (persistent
   entropy) suppress themselves; this needs no labels at all and is the
   fallback for truly remote sites.

## Honest limitations

- Accuracy on the drifting benchmark is **~0.40–0.56** (chance 0.25; a linear
  probe on the raw features reaches ~0.9). The gap is the **latency-encoding +
  small-network** representation, not the readout — see `docs/04_benchmarking.md`.
  This is stated plainly rather than hidden.
- Energy figures are **design estimates**, not silicon measurements; every
  constant is re-tunable in `architecture/energy_model.py`.
- The RTL is a **small parameterised core** (4-in/4-hidden/4-out) meant to prove
  the blocks and their wiring, not a full 65 k-neuron tape-out layout.

## Where it's used
Mining is only the first example — the same chip fits **earthquake/volcano
early warning, predictive maintenance, pipelines, borders, and anywhere a slow,
changing danger hides far from a power outlet.**  See `docs/01_background.md`.
