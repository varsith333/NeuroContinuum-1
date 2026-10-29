# 03 — RTL guide: module hierarchy, fixed point, and how to simulate

The RTL is **synthesizable Verilog-2001** (`-g2012` is used only for the
testbench conveniences). Every module is a structural implementation of a Python
model, and each carries a header naming that reference file so the two can be
diffed side by side.

---

## 1. Module map

| File | Role | Python reference |
|---|---|---|
| `lif_neuron.v` | base leaky integrate-and-fire neuron | `simulator/neurons.py :: LIFLayer` |
| `bcm_plasticity.v` | **NOVEL #1** BCM meta-plasticity with sliding threshold | `simulator/plasticity.py :: BCMPlasticity` |
| `stochastic_neuron.v` | **NOVEL #2** LFSR membrane noise | `simulator/stochastic.py :: StochasticLIFLayer` |
| `first_spike_decoder.v` | **NOVEL #3** race / priority-encode winner | `simulator/temporal.py :: FirstSpikeDecoder` |
| `fault_monitor.v` | **NOVEL #4a** write-refusal stuck detection | `simulator/selfhealing.py` (`_screen`) |
| `synapse_remap.v` | **NOVEL #4b** spare-column allocator | `simulator/selfhealing.py` (`_repair`) |
| `neuro_core.v` | structural top wiring the four blocks together | `simulator/continuum_net.py` |
| `tb/tb_lif_neuron.v` | self-checking testbench for the neuron | — |
| `tb/tb_neuro_core.v` | self-checking testbench for the core (incl. fault injection → repair) | — |
| `run_tests.ps1` | Windows test runner, Icarus Verilog (no `make` needed); `-Wave` dumps `wave.vcd` | — |
| `run_xsim.ps1` | Windows test runner, Vivado xsim — second independent simulator, same PASS tokens | — |
| `create_project.tcl` | builds the Vivado GUI project (`NeuroContinuum.xpr`) and proves it in batch | — |
| `report_util.tcl` | Vivado synthesis + place-and-route of `neuro_core` (area / timing / power reports) | — |

**Interfaces are intentionally uniform:** every behavioral module takes
`clk`, sync-`rst`, and `en` (advance one timestep when high). Spike signals are
single-clock pulses, and the core's input interface holds each input spike for
the whole timestep (registered), so a 1-clock post-spike pulse still overlaps
its input window and the BCM coincidence (`pre && post`) is well defined. This
makes the whole system lock-step, which is exactly why the Python model and the
RTL can be co-simulated and why the design is easy to formal-verify later.

---

## 2. Fixed-point conventions

- **Two's-complement signed** everywhere; widths set by `DATA_W` (membrane,
  default 16) and `W_W` (weight, default 8).
- **Leak = arithmetic shift** (`v >>> LEAK_SHIFT`) ⇒ time constant is
  `2**LEAK_SHIFT` steps, zero multipliers.
- **Learning rate = arithmetic shift** (`… >>> ETA_SHIFT`) ⇒ `eta = 1/2**ETA_SHIFT`.
- **Saturation, not wrap:** weight adds clamp at `W_MAX`/`W_MIN`, and membrane
  clamps when it would exceed the signed width — wrap-around would be
  catastrophic in a safety monitor.
- **Refractory:** a counter holds the membrane at `RESET_VAL` for `REFRACTORY`
  steps and suppresses spikes.

Example: `lif_neuron.v`

```
v_next = v + i_ext - (v >>> LEAK_SHIFT)
fire   = (v_next >= THRESHOLD) && (ref_cnt == 0)
```

---

## 3. What each novel block proves in simulation

### 3.1 `bcm_plasticity.v`
Two signed traces advance every clock: the activity `y` (leaky trace of post
spikes) and the threshold `th`, which slides toward `y²` — computed as
`(y*y) >>> YSQ_SHIFT`, saturated, then a slow low-pass follower (a multiplier-
free hardware form of Python's `theta += eta_theta·(post² − theta)`). The weight
step is gated by the **pre/post coincidence** (matching Python's `pre*post`
factor), uses round-to-nearest (`(diff + ROUND) >>> ETA_SHIFT`, which removes
the truncation bias that would otherwise always round negative steps down), and
saturates at `W_MAX`/`W_MIN`.

**The weight cell lives OUTSIDE this module:** `w_cur` is an input, `w_next`/
`move` are outputs. That is what lets a stuck physical cell refuse the write and
be detected by `fault_monitor` (novel #4 depends on this split). The behavioral
equivalent (LTP/LTD sign flip + threshold sliding up then down) is asserted in
`simulator/examples/test_novel_features.py`.

### 3.2 `stochastic_neuron.v`
An LFSR (`x³²+x²²+x²+x+1`, seed parameterized) feeds a XOR/subtract tap-folding
approximation to a zero-mean Gaussian, scaled by `NOISE_SHIFT` and gated by
`noise_on`. Same input ⇒ different spike pattern across runs — the property the
Monte-Carlo posterior depends on. `NOISE_EN=0` yields the deterministic neuron.

### 3.3 `first_spike_decoder.v`
Priority-encodes the lowest set spike bit (deterministic tie-break), latches it,
freezes the armed flag, captures `race_t`. `clr` re-arms after each reward.
Output: `winner`, `winner_valid`, `race_t`.

### 3.4 `fault_monitor.v` + `synapse_remap.v` + the wiring in `neuro_core.v`
`fault_monitor` samples on each **write attempt** (a BCM coincidence, since the
rule only ever moves a weight then) and counts consecutive confirmations of
`w_want ≠ w_actual` *where the rule wanted a change*; after `STUCK_CYCLES` it
emits a **one-shot** `fault` pulse (firing clears the counter, so one fault =
one repair request — no repair storms). `synapse_remap` takes
`fault`/`fault_id`/`fault_src`/`cur_col` and answers one clock later on
`done_valid`/`done_ok`/`done_id`/`map_out`: it picks a spare that is free or
already mirrors the same input, and never re-picks the row's current (failed)
column. `neuro_core` holds per-row `pending` flags and services **one repair
per clock** through the shared allocator; a row with no spare left latches
`degraded` (graceful degradation) and stops requesting. The `inject_stuck`
input port is a test/DFT pin that poisons a row's live cell so the whole
loop — refuse, detect, remap, resume — can be proven end to end in
`tb_neuro_core.v`.

---

## 4. Running the simulations

**Icarus Verilog, Unix (GNU make):**

```bash
make -C rtl            # runs tb_lif_neuron + tb_neuro_core, greps for PASS
make -C rtl sim_lif    # just the neuron
make -C rtl sim_core   # just the core
make -C rtl wave       # core sim with a VCD for GTKWave
```

**Icarus Verilog, Windows (no `make` required):**

```powershell
powershell -ExecutionPolicy Bypass -File rtl\run_tests.ps1          # both TBs
powershell -ExecutionPolicy Bypass -File rtl\run_tests.ps1 -Wave    # + wave.vcd
```

The runner locates `iverilog`/`vvp` on `PATH` or in `C:\iverilog\bin` (where
`winget install Icarus.Verilog` puts it) and exits non-zero if either pass
token is missing.

**Vivado xsim (second independent simulator, Windows):**

```powershell
powershell -ExecutionPolicy Bypass -File rtl\run_xsim.ps1   # both TBs, same PASS tokens
```

```bash
make -C rtl xsim       # same flow on Unix, if Vivado is on PATH
```

`run_xsim.ps1` finds Vivado under `C:\Xilinx` / `D:\Xilinx` (any version) or
on `PATH`, compiles with `xvlog -sv`, and gates on the same `TB_*_PASS`
tokens. Testing on this second tool caught a real portability bug —
`sel_valid` was used before its declaration, which Icarus tolerates but
Vivado rejects (`VRFC 10-3380`); declarations now precede every use.

**Open it in the Vivado GUI (project + waveforms):**

```powershell
& "D:\Xilinx\2025.2\Vivado\bin\vivado.bat" -mode batch -source rtl\create_project.tcl
```

(use `vivado -mode batch -source rtl\create_project.tcl` if Vivado's `bin`
is on `PATH`; adjust the install path/version to yours). The script creates
`rtl\NeuroContinuum\NeuroContinuum.xpr` with:

- design sources = every `rtl\*.v` (synthesis top: `neuro_core`),
- simulation sources = both testbenches (simulation top: `tb_neuro_core`),

then runs the behavioral simulation once in batch mode — the run that
prints `TB_CORE_PASS` — so the project opens already proven. Afterwards
**double-click `rtl\NeuroContinuum\NeuroContinuum.xpr`**; in the Flow
Navigator choose **Run Simulation → Run Behavioral Simulation** for
interactive waveforms. (The project directory is generated output and is
git-ignored; re-run the script to rebuild it.)

**Size & speed on real hardware (synthesis + place-and-route):**

```powershell
& "D:\Xilinx\2025.2\Vivado\bin\vivado.bat" -mode batch -source rtl\report_util.tcl
```

Runs Vivado synthesis and implementation of `neuro_core` for the Artix-7
`xc7a35tcpg236-1` in a throwaway project (`rtl\synth_run\`, git-ignored) and
writes utilization / timing / power reports. Measured numbers are in
`docs/04_benchmarking.md` §8.1. These are *FPGA* numbers — an honest area
estimate for the design, not an ASIC tapeout.

Both testbenches are **self-checking**: they `$fatal` on failure and print
`TB_LIF_PASS` / `TB_CORE_PASS` on success, and the Makefile, the Icarus
runner and the xsim runner all `grep` for those tokens, so CI can gate on
them.

> Note: if `iverilog` is not installed, install Icarus Verilog
> (`winget install Icarus.Verilog`), or use `rtl\run_xsim.ps1` with Vivado.
> The Python side needs no simulator.

---

## 5. Reading a waveform

`make -C rtl wave` (Unix) or `rtl\run_tests.ps1 -Wave` (Windows) dumps
`rtl/wave.vcd`. With Vivado, open the GUI project (`create_project.tcl`,
§4) and choose **Run Simulation → Run Behavioral Simulation** for an
interactive waveform window instead. Useful probes:

- `v` on any neuron — watch the leaky sawtooth approach threshold,
- `y` and `th` in `bcm_plasticity` — watch the threshold *chase* the
  activity² (the visual form of novel #1: `th` bumps up after every spike
  burst and decays back down between them),
- `w_cur` / `w_req` in a `g_row` of `neuro_core` — before a fault the two
  track each other; after `inject_stuck` they diverge (that gap *is* the
  write refusal), and `fault_row` pulses when the monitor confirms it,
- `live_col` / `stuck` / `map_out` — watch the row re-point from its failed
  column to a spare when the repair lands,
- `lfsr` and `v` in `stochastic_neuron` — same drive, jittered crossings,
- `winner_valid` / `race_t` — one pulse per decision, the latency readout.

---

## 6. Relationship to the Python golden model

The Python simulator is the **executable specification**; the RTL is the
structural realization. Where a hardware approximation is unavoidable it is
called out in the module header and justified. The current list:

- the BCM threshold uses a fixed-point `y²` follower (one 12×12 multiply,
  arithmetic-shifted) instead of Python's float `eta_theta` rate — the
  follower gain (`TH_SHIFT`) and scaling (`YSQ_SHIFT`) are tuned so the
  threshold rises and falls on the same timescale as the Python model,
- BCM writes are gated by a same-clock `pre && post` coincidence; with the
  core's registered input interface (input spikes held for the whole
  timestep) this fires exactly once per input/hidden pair, matching Python's
  `pre*post` factor,
- write-refusal is screened only at those coincidence instants (the rule
  never wants a change anywhere else, so screening elsewhere could only
  produce false resets), so "3 consecutive confirmations" means 3
  consecutive coincidences rather than 3 consecutive clocks,
- the LFSR tap-folding replaces a true Gaussian (cheap, zero-mean, bounded),
- the compact core carries one synapse per row and spare columns mirror the
  row's input (no input mux); a full crossbar would add the mux the
  Python `live_view()` already models,
- `synapse_remap` arbitrates one repair per clock across rows.

If you change a model in Python, update the corresponding header comment and
re-run both `python -m simulator.examples.test_novel_features` and
`rtl\run_tests.ps1` (or `make -C rtl`) to keep them honest.

---

**Next:** `04_benchmarking.md` for methodology and reproduction steps.
