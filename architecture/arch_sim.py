"""Cycle / event architecture simulator for NeuroContinuum-1.

Turns a *measured* spike workload (produced by running the simulator) into a
latency + energy estimate for the RTL architecture, and contrasts it with a
conventional CPU and GPU doing the "same" math the von-Neumann way.

The model is event-driven and deliberately simple but honest:

  * The chip only does work when a spike occurs (that is the entire point).
    Synapse ops  = sum over spikes of their fan-out.
    Neuron updates = synapse ops (each arriving event integrates one neuron).
  * A shared AER (address-event) bus carries spike events between cores.  It is
    the bottleneck: it can move ``routes_per_cycle`` events per clock.  Backlog
    on this bus is what makes a large sparse network slow — exactly the trade a
    real neuromorphic NoC must size for.
  * Conventional baselines stream weights from DRAM on every pass and do a
    multiply-accumulate per synapse even when nothing is spiking.

Everything is parameterised so the numbers can be re-tuned; see
``energy_model.py`` for the per-op constants and their provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .energy_model import EnergyModel, DEFAULT_MODEL


@dataclass
class WorkloadStats:
    """Spike-workload summary extracted from a run of the simulator."""
    name: str = "workload"
    n_neurons: int = 0
    n_synapses: int = 0
    duration_ms: float = 0.0
    total_spikes: int = 0
    synapse_ops: int = 0          # weight reads + accumulates (spikes x fan-out)
    routing_events: int = 0       # spike events that must cross the NoC
    stdp_updates: int = 0         # plasticity updates applied
    repairs: int = 0              # self-healing repairs performed


@dataclass
class NeuroCoreArch:
    """Hardware configuration of the chip (one instance)."""
    n_cores: int = 16
    neurons_per_core: int = 4096
    routes_per_cycle: int = 4     # AER bus event throughput (events / clk)
    core_ops_per_cycle: int = 8   # synapse+neuron ops a core retires per clk
    clk_hz: float = 1.0e9         # 1 GHz event-driven clock
    p_static_w: float = 0.10      # chip static/leakage power (W)
    model: EnergyModel = field(default_factory=lambda: DEFAULT_MODEL)

    # ------------------------------------------------------------------ energy
    def energy(self, s: WorkloadStats) -> dict:
        m = self.model
        e_syn   = s.synapse_ops * m.e_synapse_op
        e_neu   = s.synapse_ops * m.e_neuron_update          # 1 update / event
        e_det   = s.total_spikes * m.e_spike_detect
        e_route = s.routing_events * m.e_spike_route
        e_stdp  = s.stdp_updates * m.e_stdp_trace
        e_rep   = s.repairs * m.e_repair
        dyn_pj  = e_syn + e_neu + e_det + e_route + e_stdp + e_rep
        # Headline metric = DYNAMIC (compute) energy, which is the fair thing to
        # compare against CPU/GPU (they are also charged compute-only).  Static /
        # standby power is a SEPARATE always-on spec, not folded into the run
        # energy — otherwise the 8 s of *listening* would dominate the μs of
        # *computing* and hide the real event-driven advantage.
        return {
            "synapse_pj": e_syn, "neuron_pj": e_neu, "detect_pj": e_det,
            "route_pj": e_route, "stdp_pj": e_stdp, "repair_pj": e_rep,
            "total_uj": dyn_pj * 1e-6,          # dynamic compute energy (headline)
            "standby_mw": self.p_static_w * 1e3,  # always-on idle power (spec)
        }

    # ----------------------------------------------------------------- latency
    def latency(self, s: WorkloadStats) -> dict:
        # shared AER bus is the routing bottleneck
        route_cycles = s.routing_events / max(self.routes_per_cycle, 1)
        # core compute is spread across cores
        core_cycles = s.synapse_ops / max(self.core_ops_per_cycle * self.n_cores, 1)
        cycles = max(route_cycles, core_cycles)
        t_s = cycles / self.clk_hz
        return {
            "route_cycles": route_cycles, "core_cycles": core_cycles,
            "bottleneck": "NoC/AER" if route_cycles >= core_cycles else "core compute",
            "cycles": cycles, "latency_ms": t_s * 1e3,
            "realtime_factor": (s.duration_ms / (t_s * 1e3)) if t_s > 0 else float("inf"),
        }

    def run(self, s: WorkloadStats) -> dict:
        e = self.energy(s); l = self.latency(s)
        return {"workload": s.name, "energy": e, "latency": l,
                "config": {"cores": self.n_cores,
                           "neurons_per_core": self.neurons_per_core}}


# --------------------------------------------------------------------- baselines
def cpu_baseline(s: WorkloadStats, m: EnergyModel = DEFAULT_MODEL,
                 dram_reuse: float = 64.0) -> dict:
    """Conventional CPU: a MAC per synapse + stream weights from DRAM.

    ``dram_reuse`` = how many synapse-ops each DRAM weight fetch amortises over
    (a good tiled kernel reuses each fetched weight; 64 is optimistic).
    """
    e_mac = s.synapse_ops * m.e_cpu_mac
    dram_words = s.n_synapses
    e_dram = (dram_words * 4 / dram_reuse) * m.e_dram_read_32b  # 4 B/word
    total_pj = e_mac + e_dram
    return {"compute_uj": e_mac * 1e-6, "dram_uj": e_dram * 1e-6,
            "total_uj": total_pj * 1e-6}


def gpu_baseline(s: WorkloadStats, m: EnergyModel = DEFAULT_MODEL,
                 utilization: float = 0.3, dram_reuse: float = 256.0) -> dict:
    """GPU: cheaper per-op when utilized, but utilization is often poor for
    sparse/event workloads, and it still pays memory traffic."""
    e_op = s.synapse_ops * m.e_gpu_op / max(utilization, 1e-6)
    dram_words = s.n_synapses
    e_dram = (dram_words * 4 / dram_reuse) * m.e_gpu_dram
    total_pj = e_op + e_dram
    return {"compute_uj": e_op * 1e-6, "dram_uj": e_dram * 1e-6,
            "total_uj": total_pj * 1e-6}
