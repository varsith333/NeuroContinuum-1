"""A small, general spiking network container.

``continuum_net.NeuroContinuum`` is the *applied* flagship (built for streaming
hazard data).  ``SNNetwork`` here is the *generic* container underneath it:
you assemble layers, connect them with weights, optionally attach a plasticity
rule, and step the whole thing forward.  It exists so that:

  * the demos in ``simulator/examples/`` can build custom networks concisely,
  * the RTL ``neuro_core.v`` has a clean, documented behavioural reference, and
  * anyone extending the project can try a new layer type without touching the
    flagship code.

It deliberately mirrors the hardware execution model: **one timestep = one
``step()`` call, all layers update, then all plasticity rules update.**  That
"lock-step" schedule is exactly what makes it mappable onto clocked RTL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .neurons import LIFLayer
from .plasticity import BCMPlasticity, RewardSTDP
from .synapses import STDPRule, Connection


@dataclass
class SpikeRecord:
    """Recorded spikes for a run: ``data[name]`` is a (T, n) boolean array."""

    data: dict = field(default_factory=dict)

    def rate(self, name: str, duration_ms: float) -> np.ndarray:
        sp = self.data[name]
        return sp.sum(axis=0) * (1000.0 / max(duration_ms, 1e-9))


class SNNetwork:
    """Lock-step spiking network: layers + weighted (optionally plastic) links."""

    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.layers: dict[str, object] = {}
        self.order: list[str] = []
        self.connections: list[Connection] = []
        self._bcm: dict[int, BCMPlasticity] = {}
        self._rstdp: dict[int, RewardSTDP] = {}
        self.reward = 0.0

    # ------------------------------------------------------------ construction
    def add_layer(self, name: str, layer) -> "SNNetwork":
        if name in self.layers:
            raise KeyError(f"duplicate layer name {name!r}")
        layer.name = name
        self.layers[name] = layer
        self.order.append(name)
        return self

    def add_lif(self, name: str, n: int, **kw) -> "SNNetwork":
        return self.add_layer(name, LIFLayer(n, dt=self.dt, name=name, **kw))

    def connect(self, src: str, dst: str, w, rule: STDPRule | None = None,
                plastic: str | None = None, normalize: bool = False) -> Connection:
        """Link two layers.  ``w`` is a scalar (random uniform init) or a matrix.

        ``plastic`` = ``'bcm'`` or ``'rstdp'`` selects one of the novel rules
        instead of classic STDP; see ``simulator/plasticity.py``.
        """
        c = Connection(self.layers[src], self.layers[dst], w, rule=rule,
                       normalize=normalize, label=f"{src}->{dst}")
        idx = len(self.connections)
        self.connections.append(c)
        if plastic == "bcm":
            self._bcm[idx] = BCMPlasticity(c.W.shape[1], c.W.shape[0])
        elif plastic == "rstdp":
            self._rstdp[idx] = RewardSTDP(c.W.shape[1], c.W.shape[0])
        elif plastic not in (None, "stdp"):
            raise ValueError(f"unknown plastic rule {plastic!r}")
        return c

    # ------------------------------------------------------------------ run
    def step(self, external: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """Advance one timestep.  ``external[name]`` = injected current vector."""
        external = external or {}
        spikes: dict[str, np.ndarray] = {}
        for name in self.order:
            layer = self.layers[name]
            i_ext = np.asarray(external.get(name, 0.0), dtype=float)
            i_ext = i_ext + np.asarray(getattr(layer, "bias", 0.0), dtype=float)
            # sum feed-forward drive from every connection targeting this layer
            for ci, c in enumerate(self.connections):
                if c.dst is layer:
                    if ci in self._bcm or ci in self._rstdp:
                        i_ext = i_ext + c.live_drive() if hasattr(c, "live_drive") \
                            else i_ext + c.drive()
                    else:
                        i_ext = i_ext + c.drive()
            # winner-take-all: only the strongest spiking neuron survives
            sp = layer.step(i_ext)
            if getattr(layer, "wta", False) and sp.sum() > 1:
                keep = int(np.argmax(layer.v))
                sp = np.zeros_like(sp)
                sp[keep] = True
                layer.spikes = sp
            spikes[name] = sp

        # plasticity pass, after all layers have settled for this timestep
        for ci, c in enumerate(self.connections):
            if ci in self._bcm:
                self._bcm[ci].update(c.W, c.src.spikes, c.dst.spikes)
            elif ci in self._rstdp:
                self._rstdp[ci].update(c.W, c.src.spikes, c.dst.spikes, self.reward)
            else:
                c.learn()
        return spikes

    def run(self, T: int, external: dict[str, np.ndarray] | None = None,
            record: bool = True) -> SpikeRecord:
        """Run ``T`` timesteps; returns a :class:`SpikeRecord` of all spikes."""
        ext_seq = external or {}
        out = SpikeRecord(data={n: np.zeros((T, self.layers[n].n), dtype=bool)
                                for n in self.order}) if record else SpikeRecord()
        for t in range(T):
            per_layer = {}
            for n, arr in ext_seq.items():
                per_layer[n] = arr[t] if np.ndim(arr) > 1 else arr
            sp = self.step(per_layer)
            if record:
                for n, s in sp.items():
                    out.data[n][t] = s
        return out

    def reset(self) -> None:
        for layer in self.layers.values():
            layer.reset_state()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (f"SNNetwork(layers={self.order}, "
                f"connections={len(self.connections)}, plastic="
                f"{sorted(set(list(self._bcm) + list(self._rstdp)))})")
