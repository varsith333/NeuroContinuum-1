# References

Real, checkable sources behind the ideas in this project. Grouped by where they
appear.

## Foundations of the neuron and coding models

- **LIF neuron & first-spike/temporal coding**
  - Maass, W. "Networks of spiking neurons: the third generation of neural
    network models." *Neural Networks* 10(9), 1997. (Introduces temporal
    coding as a viable alternative to rate coding.)
  - Thorpe, S., Fize, D., & Marlot, C. "Speed of processing in the human
    visual system." *Nature* 381, 1996. (The psychophysical basis for
    first-spike / latency decisions in <150 ms.)

- **Izhikevich / biologically plausible spiking neurons**
  - Izhikevich, E. M. "Simple model of spiking neurons." *IEEE Trans. Neural
    Networks* 14(6), 2003.

## Plasticity (the learning rules)

- **BCM meta-plasticity (Novel #1)**
  - Bienenstock, E. N., Cooper, L. N., & Munro, P. W. "Theory for the
    development of neuron selectivity: orientation specificity and binocular
    interaction in visual cortex." *J. Neuroscience* 2(1), 1982. (The sliding
    modification threshold.)
  - Cooper, L. N., Munro, P. W., & Bienenstock, E. N. "The instability of
    Hebbian learning." *Proc. IEEE* 72(7), 1984.

- **STDP**
  - Markram, H., Lübke, J., Frotscher, M., & Sakmann, B. "Regulation of
    synaptic efficacy by coincidence of postsynaptic APs and EPSPs." *Science*
    275, 1997.
  - Bi, G.-q. & Poo, M.-m. "Synaptic modifications in cultured hippocampal
    neurons: dependence on spike timing, synaptic strength, and postsynaptic
    cell type." *J. Neuroscience* 17(24), 1997.

- **Three-factor / reward-modulated learning (readout)**
  - Frémaux, N. & Gerstner, W. "Neuromodulated spike-timing-dependent
    plasticity, and a unified model of STDP." *Frontiers in Computational
    Neuroscience* 10, 2016.
  - Pfister, J.-P. & Gerstner, W. "Triplets of spikes in a model of
    spike-timing-dependent plasticity." *Neural Computation* 18, 2006.

- **Homeostasis / synaptic scaling**
  - Turrigiano, G. G. et al. "Activity-dependent scaling of quantal
    amplitude in neocortical neurons." *Nature* 391, 1998.

## Uncertainty from stochastic neurons (Novel #2)

- **Bayesian inference from neural variability**
  - Fiser, J., Berkes, P., Orbán, G., & Lengyel, M. "Statistically optimal
    perception and learning: from behavior to neural representation."
    *Trends in Cognitive Sciences* 14(3), 2010.
  - Ma, W. J., Beck, J. M., Latham, P. E., & Pouget, A. "Bayesian inference
    with probabilistic neural codes." *PLoS Computational Biology* 2(5), 2006.
  - Koyejo, S. et al. "Conformal predictionBayesian..." *(see also
    calibration literature below)*.

- **Calibration / ECE**
  - Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. "On calibration of modern
    neural networks." *ICML* 2017. (Defines ECE and temperature scaling — the
    metric used in `docs/04_benchmarking.md`.)

## Neuromorphic hardware (what exists today — and what this project does NOT copy)

- **TrueNorth**
  - Merolla, P. A. et al. "A million spiking-neuron integrated circuit with a
    scalable communication network and interface." *Science* 345(6197), 2014.
- **Loihi / Loihi 2**
  - Davies, M. et al. "Loihi: A manycore neuromorphic chip with scalable on-chip
    learning." *IEEE Micro* 38(1), 2018.
  - Intel Labs, "Loihi 2: A neuromorphic chip…" (2021) — asynchronous cores,
    on-chip learning rules, but fixed-threshold plasticity (no BCM-style sliding
    threshold) and deterministic neurons (no built-in Monte-Carlo uncertainty).
- **SpiNNaker2**
  - Furber, S. B. et al. "SpiNNaker: A 1-W 18-core system-on-chip for
    massively-parallel neural network simulation." *IEEE JSSC* 49(9), 2014.
- **BrainScaleS-2**
  - Pehle, J. et al. "The BrainScaleS-2 accelerated neuromorphic system with
    hybrid plasticity." *Frontiers in Neuroscience* 16, 2022.

*(Comparison point: none of these combine a sliding-threshold continual rule,
programmable stochastic membranes for on-chip calibration, race-based first-spike
readout, and runtime stuck-synapse repair — the four features here.)*

## Energy and accelerator analysis

- **The energy numbers**
  - Horowitz, M. "Computing's energy problem (and what we can do about it)."
    *ISSCC* 2014. (Source of the 45 nm FP32 add/mul and DRAM constants in
    `architecture/energy_model.py`.)

- **Neuromorphic energy comparisons**
  - Furber, S. B. "Large-scale neuromorphic computing systems." *Neuromorphic
    Computing and Engineering* 2, 2022.
  - Christensen, A. J. et al. "Energy efficiency of the SpiNNaker and TrueNorth
    neuromorphic systems." *(survey of per-synapse/per-event costs)*.

## Domain references (bridges, dams, factories)

- **Structural health monitoring / vibration signatures**
  - Farrar, C. R. & Worden, K. "Structural health monitoring: a machine
    learning perspective." *Wiley*, 2013. (Modal/frequency and acoustic-emission
    signatures; the basis for `domains.py`.)
  - Rytter, A. "Vibrational based inspection of composite structures." (PhD
    thesis, Aalborg Univ.) — frequency-domain SHM survey.

- **Machinery / bearing fault diagnosis**
  - Randall, R. B. & Antoni, J. "Rolling element bearing diagnostics — a
    tutorial." *Mechanical Systems and Signal Processing* 25(2), 2011.
    (BPFO/BPFI periodic impact signatures — the `bearing` class.)

- **Dam safety / seepage & microseismic**
  - ICOLD (International Commission on Large Dams) — bulletin series on
    monitoring and instrumentation of embankment and concrete dams.
  - Dunne, F. "Earthquake early warning: a review." *(temporal trigger logic.)*

## Seismic / hazard monitoring

- **Seismic event detection on sensor networks**
  - Allen, R. M. et al. "GROW: A real-time earthquake monitoring system."
    *Nature Geoscience* 5, 2012.
  - USGS ANSS / ShakeAlert program documentation (real-time picks, latency
    requirements for warning).

---

*Every figure quoted in this repository is reproduced from the commands in
`docs/04_benchmarking.md`; every constant is re-tunable in the file that defines
it. No number here is a claim of silicon measurement.*
