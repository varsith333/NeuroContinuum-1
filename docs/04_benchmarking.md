# 04 — Benchmarking: methodology and how to reproduce every number

This note records *exactly* what was measured, with what parameters, and what
the numbers do and do not mean. Reproduce with commands at the end.

---

## 1. Datasets / streams

Three synthetic-but-physical vibration domains, from `simulator/domains.py`
(`DomainStream`), each with deliberate **concept drift** so continual learning
is measured rather than asserted:

| domain | classes | physical basis | drift applied |
|---|---|---|---|
| `bridge` | traffic / wind / crack / impact | damped resonances at ~8, ~25, ~140, ~18 Hz | load term `1 + 0.5·sin(2πk/500)`; noise floor rises with `k` |
| `dam` | normal / seepage / flood / freeze | steady hiss, ~60 Hz trickle, ~5 Hz flood, ~95 Hz freeze snap | same schedule |
| `factory` | healthy / imbalance / bearing / looseness | running-speed harmonics `f0=24·g` Hz, periodic impacts, sub-harmonics | RPM `f0` scales with load; noise rises |

On top of the slow drift, the demo passes `regime_at = n//2`, which triggers an
**abrupt regime change**: the class priors are reversed (rare ⇄ common — hard
concept drift) and the noise floor steps up by 0.6 (e.g. a sensor mount
loosening). This is a real, persistent environment change at a known point, not
a label on smooth sine drift.

Feature front-end: `featurise()` = 6 log band-powers + zero-crossing rate +
crest factor + RMS (9 features, all clipped to [0,1]). Sampling 512 Hz,
0.5 s window.

**Feature ceiling (linear probe, no spiking):** bridge 0.911, dam 0.983,
factory 0.961 test accuracy. This is the reference point for judging the SNN.

---

## 2. The network under test

`simulator/continuum_net.py :: NeuroContinuum`, locked configuration:

```
n_feat 9, n_hidden 48, n_classes 4, window_ms 40, mc_trials 7,
noise_sigma 0.10, input_gain 6.0, output_gain 1.0, syn_tau 5.0, rstdp_eta 0.05
```

Pipeline per window: features → **latency encode** (each input spikes exactly
once, `spike_t = round(lat)`) → exponential synaptic current (`syn_tau=5 ms`)
→ **stochastic hidden LIF** + **BCM** + **self-healing** → output racers with
**winner-take-all** → decision = arg-max of accumulated graded output drive;
**posterior** = vote histogram over `mc_trials=7` noisy trials.

Readout learning: **reward-gated competitive Hebbian** (three-factor: local
presynaptic activity × scalar reward routed to the correct-class row), no
backprop. Rationale in §5.

---

## 3. Result A — accuracy and calibration

Command:

```bash
python -m simulator.examples.demo_continual --n 400
```

Measured (chance = 0.25); `pre`/`post` split at the regime change (window 200):

| domain | acc pre | acc post | sliding overall | mean confidence | ECE | calibrated? |
|---|---:|---:|---:|---:|---:|---|
| bridge | 0.480 | 0.525 | 0.500 | 0.497 | 0.117 | ✓ (<0.25) |
| dam | 0.555 | 0.530 | 0.740 | 0.625 | 0.240 | ✓ |
| factory | 0.425 | 0.400 | 0.500 | 0.508 | 0.100 | ✓ |

`pre`/`post` are plain segment accuracies; `sliding overall` is the trailing
50-window accuracy the `DriftTracker` reports at the end of the run (it lags
the segment it is summarising, which is why dam reads 0.740).

**ECE** = mean `|confidence − accuracy|` over probability bins. ECE 0.10–0.24
means confidence tracks correctness — when the chip says "70 % sure" it is right
roughly 70 % of the time.

Note that `acc post ≥ acc pre − 0.055` on every domain *despite* the abrupt
regime change — the network holds through the environment change rather than
collapsing (§4 shows what happens without learning).

### 3.1 Why accuracy is ~0.5 and not ~0.9 (stated plainly)

The *features* support ~0.91–0.98 (§1). The gap is **not** the readout:

- A linear probe on the **hidden representation** reaches only **0.45–0.57**
  (`probe_ceiling.py`) — the latency encoding + small stochastic network throws
  away feature information before the readout ever sees it.
- With the readout fixed, accuracy lands near that hidden ceiling, confirming
  the hidden representation is the binding constraint.

Known contributors: latency encoding compresses features into a 40-step window;
leaky integration with `syn_tau=5 ms` is a lossy temporal filter; membrane noise
required for calibration adds variance.

Cheap fixes were tried and measured (`sweep_repr.py`):

| config | avg acc | avg ECE |
|---|---:|---:|
| hid24 win25 σ0.05 (baseline) | 0.556 | 0.305 |
| hid48 win40 σ0.05 | 0.578 | 0.307 |
| **hid64 win40 σ0.10** | 0.544 | **0.213** |
| hid64 win40 σ0.15 | 0.528 | **0.141** |

More hidden units barely help (0.556 → 0.578): the ceiling is the encoder, not
the layer width. σ trades accuracy against calibration (§3.2). The locked
config takes a middle point.

### 3.2 The accuracy/calibration trade (measured)

Noise sweep with the fixed readout (`sweep_noise2.py`, bridge / dam / factory):

| σ | acc | ECE | confidence |
|---:|---:|---:|---:|
| 0.00 | 0.63 / 0.62 / 0.48 | 0.37 / 0.38 / 0.52 | 1.00 (overconfident) |
| 0.02 | 0.68 / 0.62 / 0.48 | 0.29 / 0.39 / 0.49 | 0.96 |
| 0.05 | 0.62 / 0.55 / 0.50 | 0.18 / 0.38 / 0.36 | 0.78–0.82 |
| **0.10** | 0.43 / 0.48 / 0.45 | **0.11 / 0.14 / 0.11** | 0.54–0.60 |
| 0.25 | 0.43 / 0.45 / 0.32 | 0.07 / 0.09 / 0.17 | 0.49–0.54 |

**This is a real, honest tension:** with accuracy ~0.5, *well-calibrated*
confidence must also be ~0.5, which requires enough noise to soften an otherwise
overconfident posterior. σ=0.00–0.02 gives the best accuracy but a confidently
wrong chip (ECE ~0.5) — unusable for a safety monitor. σ=0.10 keeps accuracy
above 0.43 on every domain with ECE 0.11–0.14. **Chosen: σ = 0.10.**

---

## 4. Result B — continual learning beats frozen

```bash
python -m simulator.examples.demo_continual --n 400           # learning ON
python -m simulator.examples.demo_continual --n 400 --no-learn # frozen
```

The regime change is **real** (priors reversed + noise step at window 200).
Post-change accuracy, learning vs frozen, full run (`--n 400`):

| domain | learning (post) | frozen (post) |
|---|---:|---:|
| bridge | 0.525 | 0.360 |
| dam | 0.530 | 0.350 |
| factory | 0.400 | 0.255 |

The demo's built-in A/B (`factory`, 200 windows, change at 100) reports the
same direction:

```
learning : post 0.37   (pre 0.39)
frozen   : post 0.22   (pre 0.28)
```

The frozen network drifts toward/below chance because it never aligns its
readout while the stream changes under it; the learning network holds its
post-change accuracy within ~0.05 of its pre-change accuracy. That gap is the
continual-learning claim, measured rather than asserted.

---

## 5. Why the readout uses reward-gated Hebbian, not raw spike-timing R-STDP

Recorded because it was a genuine finding, not a default:

1. A spike-timing R-STDP eligibility trace is small (`a_plus ≈ 0.01`) and
   depends on output spikes actually occurring. With sparse output firing the
   trace rarely builds, so a 300–400 window run barely moves the weights
   (`sweep_rstdp.py`: η swept 0.05 → 20, best gap only +0.07).
2. Fixing **winner-take-all on the output** removed the "all four classes fire
   together ⇒ arg-max is a tie" failure (a real bug: every output neuron was
   spiking on every burst).
3. Reading the **graded output drive** instead of sparse spike tallies removed
   the resolution problem (≈5 races/window is a 5-count histogram).
4. The remaining blocker was `W_out` **row-norm bias** — random rows differ in
   magnitude, so arg-max just picked the largest row. Unit-normalizing rows made
   the readout a pure direction match.
5. The **reward-gated competitive update** finally converged, because it pulls
   the correct row directly onto the active population code.

Full spike-timing `RewardSTDP` remains in the codebase and is verified in
`test_novel_features.py` (grows under +reward, shrinks under −reward) — it is
kept for timing-critical variants, but the shipped readout uses the gated rule.

---

## 6. Result C — energy and latency vs CPU/GPU

```bash
python -m architecture.benchmark
```

Model: `architecture/energy_model.py`. CPU/GPU/DRAM constants are the 45 nm
values from M. Horowitz, *Computing's Energy Problem*, ISSCC 2014. The
neuromorphic per-event costs (synapse op 1.03 pJ, neuron update 0.06 pJ, route
2.00 pJ, …) are stated in that file and are deliberately conservative — no
ideal-memristor assumptions.

Measured workload (200 windows/domain, from `workloads.py`, real spike counts):

| domain | spikes | synapse ops | NeuroCore | CPU | GPU | ×CPU | ×GPU | latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bridge | 12 659 | 48 236 | 0.06 µJ | 0.22 µJ | 0.13 µJ | **3.9×** | **2.3×** | ~0 µs |
| dam | 12 841 | 48 964 | 0.06 µJ | 0.22 µJ | 0.13 µJ | 3.9× | 2.3× | ~0 µs |
| factory | 12 408 | 47 232 | 0.05 µJ | 0.21 µJ | 0.13 µJ | 3.9× | 2.3× | ~0 µs |

Chip configuration: 16 cores × 4096 neurons, 1 GHz, AER 4 events/clock,
`core_ops_per_cycle` 8. Reported as bottleneck: **NoC/AER** (routing, not
compute, is the limit — the honest scaling knob).

**Framing decision (important):** the headline is **dynamic (compute) energy**.
Static/standby power is reported *separately* as `standby_mw`, not folded into
the per-window number. An earlier version folded in `0.1 W × 8 s of listening =
0.8 J`, which made the chip look millions of times worse than a CPU that was
never charged its own idle power — apples to oranges that would have
discredited the result. Every benchmark comparison here is compute-only on all
three sides.

**Interpretation:** 3.9× vs CPU and 2.3× vs GPU on this workload is modest and
real. It comes purely from event-driven operation (energy only on actual spikes)
and avoiding DRAM streaming. It is *not* the 1000× often claimed for
neuromorphics — the demo network has only 624 synapses, so DRAM traffic is small.
The advantage grows with network size and activity sparsity, which is exactly the
always-on-sensor regime.

---

## 7. Result D — the four novel blocks in isolation

```bash
python -m simulator.examples.test_novel_features
```

Self-checking assertions (all must print `True`):

1. **BCM** — LTP above threshold, LTD below; threshold slides 0.20 → 0.93 under
   activity → 0.016 when quiet; 500 steps with homeostasis keep weights in
   [0.165, 0.337] at row norm ≈ target 1.5.
2. **Reward-STDP** — causal pre→post protocol: grows 0.500 → 1.000 under +reward,
   shrinks to 0.000 under −reward.
3. **Stochastic neuron** — deterministic always fires 3/100; stochastic varies
   (3/100 then 5/100); skewed trials give confidence 0.67 / entropy 0.71 vs
   coin-flip confidence 0.32 / entropy 0.96 (higher entropy when uncertain ✓).
4. **Self-healing** — zero false positives on 30 healthy steps; exactly one
   repair for one injected stuck fault; live width stays 10 (spares are *extra*
   columns, never stolen from inputs); `spares_left = n_spare − 1`.

---

## 8. RTL simulation

```bash
# Unix / with GNU make:
make -C rtl            # or: make -C rtl wave   (VCD) / make -C rtl xsim

# Windows (no make needed) — Icarus Verilog:
powershell -ExecutionPolicy Bypass -File rtl\run_tests.ps1
powershell -ExecutionPolicy Bypass -File rtl\run_tests.ps1 -Wave   # + wave.vcd

# Windows — Vivado xsim (second independent simulator, same PASS tokens):
powershell -ExecutionPolicy Bypass -File rtl\run_xsim.ps1

# Windows — build the Vivado GUI project (waveforms, synthesis readiness):
& "D:\Xilinx\2025.2\Vivado\bin\vivado.bat" -mode batch -source rtl\create_project.tcl
#   -> then double-click rtl\NeuroContinuum\NeuroContinuum.xpr
```

Self-checking; must print `TB_LIF_PASS` and `TB_CORE_PASS`. The Icarus
runner finds Verilog in `C:\iverilog\bin` or on PATH (`winget install
Icarus.Verilog`); the xsim runner finds Vivado under `C:\Xilinx` / `D:\Xilinx`.
Verified with **Icarus Verilog 12.0 and Vivado 2025.2 xsim** — running both
tools caught one real bug (a declaration ordering error Vivado rejects but
Icarus tolerates).

The core testbench proves the integrated loop end to end: it runs the core,
injects a **stuck synapse** through the `inject_stuck` DFT port, and asserts
that (a) the write-refusal detector fires, (b) the spare-column remap performs
a repair, (c) the decoder had already latched a winner, and (d) the core keeps
emitting spikes *after* the repair. It also checks there is no repair storm
(exactly the injected faults are detected).

---

## 8.1 FPGA synthesis & implementation (measured)

```powershell
& "D:\Xilinx\2025.2\Vivado\bin\vivado.bat" -mode batch -source rtl\report_util.tcl
```

`neuro_core` taken through full Vivado synthesis **and** place-and-route
(xc7a35tcpg236-1, Artix-7, −1 speed grade), measured with Vivado 2025.2:

| Metric | Value |
|---|---|
| Slice LUTs | 1263 / 20800 (**6.1 %**) |
| Slice registers (flip-flops) | 869 / 41600 (**2.1 %**) |
| DSP blocks (the y² multipliers) | 4 |
| Bonded I/O pins | 24 |
| Worst post-route path | 10.93 ns → **≈91 MHz** max clock |
| Hold timing | clean (+0.13 ns slack) |
| On-chip power @100 MHz | 0.10 W total (≈33 mW core dynamic) |

Honest notes — read before quoting these:

- These are **FPGA** numbers, not an ASIC tapeout. An FPGA LUT/FF is an order
  of magnitude bulkier than an ASIC standard cell; the µJ-level claims in §3
  come from `energy_model.py` (event-driven ASIC estimates), **not** from
  these reports. Use FPGA numbers for "how much logic", energy_model for
  "how much energy".
- Power uses Vivado's default switching-activity estimates (no SAIF file
  from the testbench) — order-of-magnitude only.
- ≈91 MHz is far above the kHz-scale timestep rate the spiking architecture
  needs; timing is not a constraint on this design.
- Reports land in `rtl\synth_run\` (`impl_util.rpt`, `impl_timing.rpt`,
  `impl_power.rpt`); the directory is git-ignored and the command rebuilds
  it from scratch.

---

## 9. One-pass reproduction

```bash
pip install -r requirements.txt
make            # = test + demo + bench   (then: make rtl)
```

Windows (no `make` installed) — one script runs the entire project in order
and stops at the first failure:

```powershell
powershell -ExecutionPolicy Bypass -File run_all.ps1        # smoke → tests →
                                                            # demo → frozen →
                                                            # benchmark → RTL
powershell -ExecutionPolicy Bypass -File run_all.ps1 -Wave  # + rtl\wave.vcd
```

(Equivalent to the individual commands below.)

```powershell
python -m simulator.examples.test_novel_features
python -m simulator.examples.demo_continual --n 400
python -m simulator.examples.demo_continual --n 400 --no-learn
python -m architecture.benchmark
powershell -ExecutionPolicy Bypass -File rtl\run_tests.ps1
```

Outputs land in `architecture/results/` (`benchmark_results.md`,
`benchmark_results.json`, `continual_learn_*.json`, `continual_frozen_*.json`)
and `rtl/work/` (+ `rtl/wave.vcd` with `-Wave`).

**Environment:** Python 3.14, numpy 2.5, matplotlib 3.11, no scikit-learn.
On PowerShell use `;` to chain commands, not `&&`.
