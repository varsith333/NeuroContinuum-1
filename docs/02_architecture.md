# 02 — Architecture specification: NeuroContinuum-1

This document turns the four novel mechanisms into a concrete chip description:
hierarchy, data precision, interconnect, memory, and a first area/power budget.
All numbers are *design targets for exploration*, not silicon measurements, and
every energy constant is re-tunable in `architecture/energy_model.py`.

---

## 1. Design goals

| Goal | Metric | Target |
|---|---|---|
| Always-on battery operation | dynamic energy per processed window | < 1 µJ |
| Continuous hazard sensing | standby / idle power | < 50 mW |
| Never wrong-and-confident | calibration error (ECE) | < 0.25 |
| Survives aging / radiation | repair coverage of stuck synapses | 100 % of injected faults, transparent to network |
| Fast alarm path | decision latency after a feature settles | < 1 ms |
| Learns in the field | retraining required | none (local reward only) |

---

## 2. Hierarchy

```
NeuroContinuum-1  (chip)
└── NeuroCore  x  N            (e.g. N = 16)
    ├── Input spike encoder    (latency codes from fixed-function DSP front-end)
    ├── Synapse crossbar  n_in x n_hid
    │     ├── BCM meta-plasticity   (novel #1)
    │     ├── fault monitor         (novel #4a)
    │     └── spare columns + remap (novel #4b)
    ├── Hidden neuron array  (stochastic LIF — novel #2)
    ├── Readout array        (racing outputs → first-spike decoder, novel #3)
    └── AER interface        (event out / reward line in)
└── Shared interconnect: AER bus  (address-event routing between cores)
```

The reference instantiation used in simulation and RTL is deliberately small
(**4 inputs, 48 hidden, 4 classes**) to keep runs fast while exercising every
block and its wiring; `N_*` are parameters everywhere.

---

## 3. Data path and precision

**Number format:** signed two's-complement throughout.

| Signal | Width | Notes |
|---|---|---|
| Membrane potential `v` | 16-bit (Q8.8) | headroom for overshoot before saturation |
| Synaptic weight `w` | 8-bit signed | symmetric range, saturating on write |
| Activity / threshold traces | 12-bit signed | `y`, `th` |
| Spike events | 1-bit (addressed by core/id) | address = neuron index |
| LFSR | 32-bit | maximal-length taps 32,22,2,1 |
| Reward (teaching line) | 8-bit signed | ±1 scalar, broadcast |

Leak is an arithmetic right shift (`v >>> LEAK_SHIFT`), so a neuron's time
constant is `2**LEAK_SHIFT` timesteps and costs no multiplier. The BCM weight
step is likewise `((y_now − th) + ROUND) >>> ETA_SHIFT` on a pre/post
coincidence — the learning rate is literally `1/2**ETA_SHIFT` and the `+ROUND`
makes it round-to-nearest instead of always truncating toward −∞.

---

## 4. The four blocks, as hardware

### 4.1 BCM meta-plasticity (`bcm_plasticity.v`) — continual learning

Per update, one shift-and-add each (traces advance every clock):

```
y  <- y − (y >> ACC_SHIFT) + (post ? ACC_INC : 0)          # activity trace
th <- th + ((y_sq − th) >> TH_SHIFT)                        # SLIDING threshold
                                                     y_sq = sat((y*y) >> YSQ_SHIFT)
if (pre && post):  w <- sat( w + ((y_now − th + ROUND) >> ETA_SHIFT) )
```

`th` is a slow low-pass follower of **activity squared** — the fixed-point
realization of BCM's `theta += eta_theta·(post² − theta)`. The sign of
`y_now − th` is what flips each synapse between LTP and LTD, so a synapse that
used to help gets turned around when the world stops rewarding it.

The weight write happens only on a pre/post **coincidence** (the Python rule's
`pre·post` factor), and because the physical weight cell lives outside this
module (`w_cur` in, `w_next`/`move` out), a stuck cell can refuse that write —
which is exactly what the self-healing block detects.

### 4.2 Stochastic neuron (`stochastic_neuron.v`) — uncertainty

A 32-bit maximal LFSR taps four bit-groups and subtracts pairs to approximate a
zero-mean Gaussian cheaply; the result is scaled by `NOISE_SHIFT` and added to
the membrane. `noise_on` (runtime) and `NOISE_EN` (elaboration) allow the same
module to be instantiated as a deterministic neuron where noise is unwanted.

### 4.3 First-spike decoder (`first_spike_decoder.v`) — fast sparse readout

A priority encoder (lowest set index wins a same-cycle tie) latches the first
arriving spike, freezes an armed flag, and captures the race timestamp. Output:
`winner`, `winner_valid` (one clock), `race_t` (decision latency in steps).

### 4.4 Self-healing (`fault_monitor.v` + `synapse_remap.v`) — resilience

- **Detect:** compare `w_want` (what the rule requested) with `w_actual` (what
  the cell holds). Persistent mismatch with `rule_wanted_change` high for
  `STUCK_CYCLES` confirmations → `fault` pulse. Debounce prevents transient
  glitches from triggering a repair.
- **Remap:** a spare-column allocator re-points the logical→physical map for the
  failed synapse. A spare mirrors the faulty synapse's input so the network
  sees a full-width matrix either way. No spare left → `degraded` (graceful).

Both halves are **wired in the structural top** (`neuro_core.v`): every hidden
row carries its own `fault_monitor`, requests queue in a `pending` mask, and a
shared arbiter services one remap per clock (`pending`/`servicing`). A row
whose request lands clears it (`repair_now`) so no false positives fire after a
repair, and the `inject_stuck` test/DFT input poisons a row's live cell so the
whole loop — refuse → detect → remap → resume — is proven in
`rtl/tb/tb_neuro_core.v` (exactly 1 detection, 1 repair, 0 false positives,
43 output spikes after the repair).

The Python golden model (`selfhealing.py`) runs the identical write-refusal
logic and reports `repairs`, `spares_left`, and `degraded_rows`.

---

## 5. Interconnect (AER bus)

Spike events are addressed and moved on a shared address-event bus. The
architecture simulator (`arch_sim.py`) treats it as the primary bottleneck:

- `routes_per_cycle` events/clock (default 4),
- `core_ops_per_cycle` synapse+neuron ops per core per clock (default 8),
- chip-level latency = `max(bus backlog, core compute / cores) / clk`.

The benchmark reports which side is the bottleneck for the measured workload —
this is the knob a larger/multi-plane NoC would scale, and the report says so
rather than implying infinite routing bandwidth.

---

## 6. Memory

- **Local SRAM per core** holds weights and traces (near-memory: a synapse read
  and accumulate costs ~1 pJ, per `energy_model.py`).
- **No per-window DRAM streaming.** The CPU/GPU baselines *do* pay DRAM
  (`e_dram_read_32b = 640 pJ`), which is the structural source of the
  neuromorphic advantage on sparse event traffic.
- Weight updates are in-place; there is no separate "download a checkpoint"
  phase, which is what lets learning run continuously on the sensor itself.

---

## 7. Budget (exploration-grade)

For the modeled 16-core, 4096-neuron/core, 1 GHz configuration:

- **Dynamic energy / window:** ~0.06 µJ measured (vs 0.22 µJ CPU, 0.13 µJ GPU
  → 3.9× / 2.3× advantage; see `docs/04_benchmarking.md`).
- **Standby power:** reported separately as `standby_mw` (default 100 mW,
  tunable) — this is the "listening" cost, deliberately *not* folded into the
  per-window compute energy, because charging it against a microsecond of work
  over an 8-second listening window would hide the event-driven advantage.
- **Area drivers:** the synapse array (weight × count), the per-synapse stall
  counter (4 bits), and spare columns (`spare_frac ≈ 15 %` of columns, capped).

---

## 8. Training / control model

There is no backpropagation and no stored training set on the sensor.

1. **Hidden layer** learns *unsupervised* with BCM + synaptic scaling — feature
   extraction that tracks drift.
2. **Readout** learns with a **reward-gated competitive Hebbian** update: one
   scalar reward from a teacher/critic, routed to the class rows. Per synapse
   the update only uses that synapse's own hidden input plus the gated reward,
   so it stays hardware-local (a global reward line + row select), not a
   gradient flowing backwards.
3. Full **spike-timing R-STDP** (`RewardSTDP`) remains in the design as a
   timing-critical alternative and is exercised in
   `test_novel_features.py`; the readout uses the more effective gated variant,
   and `04_benchmarking.md` records why.

---

**Next:** `03_rtl_guide.md` for module details and how to simulate.
