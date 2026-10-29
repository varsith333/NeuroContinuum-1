"""Energy model for NeuroContinuum-1 vs conventional compute.

All numbers are per-operation dynamic energy in picojoules.  The CPU/GPU/DRAM
figures are the widely-cited 45 nm values from M. Horowitz, "Computing's Energy
Problem (and what we can do about it)", ISSCC 2014.  The neuromorphic figures
are per-event/per-synapse costs for an event-driven, near-memory 8-bit core and
are stated explicitly so they can be re-tuned; they are deliberately
conservative (they do NOT assume an ideal memristor crossbar).

Why an energy model belongs in this project: the whole reason to build a
brain-like chip for an always-on battery sensor is ENERGY.  A rate-coded,
always-on accelerator burns power re-processing the same window; an event-driven
neuromorphic core only spends energy when a spike actually occurs.  This module
makes that trade quantitative instead of hand-wavy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnergyModel:
    # ---- NeuroContinuum-1 event-driven core (8-bit, near-memory SRAM) ------
    e_synapse_op: float = 1.03   # 8b weight read + 8b accumulate (pJ)
    e_neuron_update: float = 0.06  # leak + integrate + threshold compare (pJ)
    e_spike_detect: float = 0.10  # spike generation & event enqueue (pJ)
    e_spike_route: float = 2.00  # AER event: router traversal + wire hop (pJ)
    e_stdp_trace: float = 0.20  # trace decay + one plasticity update (pJ)
    e_repair: float = 50.0      # fault detect+remap (rare, amortised) (pJ)
    e_sram_read_32b: float = 5.0   # local SRAM word read (pJ)
    e_dram_read_32b: float = 640.0  # off-chip DRAM word read (pJ)

    # ---- conventional baseline (Horowitz ISSCC'14, 45 nm) ------------------
    e_fp32_add: float = 0.90
    e_fp32_mul: float = 3.10
    e_gpu_op: float = 0.80      # amortised per-op on a utilized GPU (est.)
    e_gpu_dram: float = 40.0    # HBM/GDDR byte-ish access (est., pJ/word)

    @property
    def e_cpu_mac(self) -> float:
        """One float multiply-accumulate = mul + add."""
        return self.e_fp32_mul + self.e_fp32_add


# Default model instance used across the architecture simulator.
DEFAULT_MODEL = EnergyModel()
