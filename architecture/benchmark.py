"""Benchmark: NeuroContinuum-1 vs a conventional CPU and GPU on the same task.

Runs the real network over the three target domains (bridge / dam / factory),
extracts the hardware event counters, then feeds them to:
  * the NeuroContinuum architecture simulator  (event-driven, near-memory),
  * a float32 CPU baseline  (MAC per synapse + DRAM streaming),
  * a GPU baseline          (cheaper per-op but poor utilization on sparse data),

and writes a markdown + JSON report to architecture/results/.

Run from repo root:
    python -m architecture.benchmark
"""

from __future__ import annotations

import json
from pathlib import Path

from .arch_sim import NeuroCoreArch, cpu_baseline, gpu_baseline
from .workloads import extract_workload

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def fmt_uj(x: float) -> str:
    if x >= 1000:
        return f"{x/1000:.2f} mJ"
    return f"{x:.2f} uJ"


def run_one(domain: str, arch: NeuroCoreArch) -> dict:
    s = extract_workload(domain, windows=200, seed=0, learn=True)
    nc = arch.run(s)
    cpu = cpu_baseline(s)
    gpu = gpu_baseline(s)
    return {
        "domain": domain,
        "workload": {"spikes": s.total_spikes, "synapse_ops": s.synapse_ops,
                     "routing": s.routing_events, "neurons": s.n_neurons,
                     "synapses": s.n_synapses, "duration_ms": s.duration_ms},
        "neurocore": nc,
        "cpu": cpu, "gpu": gpu,
        "speedup_vs_cpu_latency": None,  # latency baseline is throughput-bound; see report
        "energy_ratio_cpu": cpu["total_uj"] / max(nc["energy"]["total_uj"], 1e-9),
        "energy_ratio_gpu": gpu["total_uj"] / max(nc["energy"]["total_uj"], 1e-9),
    }


def main() -> None:
    arch = NeuroCoreArch(n_cores=16, neurons_per_core=4096,
                         routes_per_cycle=4, core_ops_per_cycle=8)
    domains = ["bridge", "dam", "factory"]
    rows = [run_one(d, arch) for d in domains]

    # ---- console -----------------------------------------------------------
    print("=" * 78)
    print("NeuroContinuum-1  |  architecture & energy benchmark")
    print(f"  chip: {arch.n_cores} cores x {arch.neurons_per_core} neurons, "
          f"{arch.clk_hz/1e9:.0f} GHz, AER {arch.routes_per_cycle} ev/clk")
    print("=" * 78)
    hdr = f"{'domain':9s} {'spikes':>8s} {'syn_ops':>9s} {'NC(uJ)':>9s} " \
          f"{'CPU(uJ)':>10s} {'GPU(uJ)':>10s} {'xCPU':>6s} {'xGPU':>6s} {'lat(ms)':>8s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        w, nc = r["workload"], r["neurocore"]
        print(f"{r['domain']:9s} {w['spikes']:8d} {w['synapse_ops']:9d} "
              f"{nc['energy']['total_uj']:9.2f} {r['cpu']['total_uj']:10.2f} "
              f"{r['gpu']['total_uj']:10.2f} {r['energy_ratio_cpu']:6.1f} "
              f"{r['energy_ratio_gpu']:6.1f} {nc['latency']['latency_ms']:8.3f}")
    print("-" * len(hdr))
    print("xCPU / xGPU = energy advantage of NeuroContinuum (higher = better).")
    print("lat(ms)    = wall-clock to process the stream; realtime_factor below 1 "
          "means faster than real time.")

    # ---- markdown report ---------------------------------------------------
    lines = ["# NeuroContinuum-1 — Architecture & Energy Benchmark", ""]
    lines.append(f"Chip: **{arch.n_cores} cores x {arch.neurons_per_core} neurons**, "
                 f"**{arch.clk_hz/1e9:.0f} GHz** event-driven, shared AER bus "
                 f"**{arch.routes_per_cycle} events/clock**, "
                 f"static {arch.p_static_w*1000:.0f} mW.")
    lines.append("")
    lines.append("Energy model: Horowitz ISSCC'14 (45 nm) for CPU/GPU/DRAM; "
                 "per-event neuromorphic costs in `architecture/energy_model.py`.")
    lines.append("")
    lines.append("| domain | spikes | synapse ops | NeuroCore | CPU | GPU | xCPU | xGPU | latency (ms) | realtime x |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        w, nc = r["workload"], r["neurocore"]
        lines.append(
            f"| {r['domain']} | {w['spikes']} | {w['synapse_ops']} | "
            f"{nc['energy']['total_uj']:.2f} uJ | {r['cpu']['total_uj']:.2f} uJ | "
            f"{r['gpu']['total_uj']:.2f} uJ | {r['energy_ratio_cpu']:.1f}x | "
            f"{r['energy_ratio_gpu']:.1f}x | {nc['latency']['latency_ms']:.3f} | "
            f"{nc['latency']['realtime_factor']:.0f}x |")
    lines.append("")
    bottleneck = rows[0]["neurocore"]["latency"]["bottleneck"]
    lines.append(f"**Latency bottleneck:** {bottleneck}. "
                 "For sparse event traffic the shared AER bus is the limiting "
                 "resource — this is the key knob a larger NoC would scale.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- NeuroCore only spends energy on *actual spikes* (event-driven); "
                 "the CPU/GPU pay a multiply-accumulate per synapse every pass and "
                 "stream weights from DRAM.")
    lines.append("- The advantage grows as activity sparsity increases — exactly the "
                 "regime of an always-on hazard sensor.")
    lines.append("- These are *estimates* for design exploration, not silicon "
                 "measurements; every constant is re-tunable in `energy_model.py`.")
    (RESULTS / "benchmark_results.md").write_text("\n".join(lines))
    (RESULTS / "benchmark_results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nsaved: {RESULTS/'benchmark_results.md'}")
    print(f"saved: {RESULTS/'benchmark_results.json'}")


if __name__ == "__main__":
    main()
