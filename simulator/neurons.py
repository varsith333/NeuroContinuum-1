"""Spike neuron models for the NeuroCore simulator.

All models are *vectorised*: one ``LIFLayer`` object holds an entire population
of ``n`` identical neurons as NumPy arrays, so a 65 k-neuron network costs the
same Python object overhead as a single neuron.  Time is discretised with a
fixed step ``dt`` (milliseconds), matching the fixed-timestep update used by
the RTL neuron (``rtl/lif_neuron.v``).
"""

from __future__ import annotations

import numpy as np


class LIFLayer:
    """Leaky integrate-and-fire (LIF) neuron population.

    Membrane dynamics (forward Euler, ``dt`` in milliseconds)::

        dv/dt = ( -(v - v_rest) + R * I ) / tau

    A spike is emitted when ``v >= v_threshold``.  The membrane is then
    clamped to ``v_reset`` for ``t_ref`` ms (absolute refractory period).

    The same equations are implemented (in fixed point) by ``rtl/lif_neuron.v``:
    ``tau`` maps to ``2**LEAK_SHIFT`` integration steps and ``R`` to the
    scale of the input current.
    """

    def __init__(
        self,
        n: int,
        tau: float = 20.0,
        v_rest: float = 0.0,
        v_reset: float = 0.0,
        v_thresh: float = 1.0,
        r: float = 1.0,
        t_ref: float = 2.0,
        dt: float = 1.0,
        bias: float = 0.0,
        name: str = "layer",
    ):
        self.n = int(n)
        self.tau = float(tau)
        self.v_rest = float(v_rest)
        self.v_reset = float(v_reset)
        self.v_thresh = float(v_thresh)
        self.r = float(r)
        self.dt = float(dt)
        self.t_ref = float(t_ref)
        self.bias = float(bias)
        self.name = name
        self.ref_steps = int(round(t_ref / dt)) if t_ref > 0 else 0
        self.wta = False  # if True, only the max-v spiking neuron survives
        self.reset_state()

    def reset_state(self) -> None:
        """Return the population to its resting state (clears all history)."""
        self.v = np.full(self.n, self.v_rest, dtype=float)
        self.ref = np.zeros(self.n, dtype=int)
        self.spikes = np.zeros(self.n, dtype=bool)

    def step(self, i_ext) -> np.ndarray:
        """Advance one timestep; returns a boolean array of emitted spikes."""
        i_ext = np.broadcast_to(np.asarray(i_ext, dtype=float), (self.n,))
        active = self.ref == 0
        dv = (-(self.v - self.v_rest) + self.r * i_ext) * (self.dt / self.tau)
        self.v[active] += dv[active]
        fired = active & (self.v >= self.v_thresh)
        self.v[fired] = self.v_reset
        self.ref[self.ref > 0] -= 1
        if self.ref_steps:
            self.ref[fired] = self.ref_steps
        self.spikes = fired
        return fired

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"LIFLayer(name={self.name!r}, n={self.n}, tau={self.tau})"


class IzhikevichLayer:
    """Izhikevich mixed-mode neuron population (bursting / regular spiking).

        dv/dt = 0.04 v^2 + 5 v + 140 - u + I
        du/dt = a (b v - u)
        v >= 30 mV  ->  spike, v <- c, u <- u + d

    With standard parameters this reproduces regular spiking (a=0.02, b=0.2,
    c=-65, d=8) and can produce bursting / chattering with other settings.
    Included to show that the simulator framework is model-agnostic.
    """

    def __init__(self, n: int, a=0.02, b=0.2, c=-65.0, d=8.0, v_init=-65.0,
                 dt: float = 1.0, name: str = "izh"):
        self.n = int(n)
        self.a, self.b, self.c, self.d = float(a), float(b), float(c), float(d)
        self.dt = float(dt)
        self.name = name
        self.v = np.full(self.n, v_init, dtype=float)
        self.u = self.b * self.v
        self.spikes = np.zeros(self.n, dtype=bool)

    def reset_state(self) -> None:
        self.v[:] = -65.0
        self.u[:] = self.b * self.v
        self.spikes[:] = False

    def step(self, i_ext) -> np.ndarray:
        i_ext = np.broadcast_to(np.asarray(i_ext, dtype=float), (self.n,))
        h = self.dt  # Izhikevich uses dt=1 ms conventionally
        dv = (0.04 * self.v ** 2 + 5.0 * self.v + 140.0 - self.u + i_ext) * h
        du = (self.a * (self.b * self.v - self.u)) * h
        self.v += dv
        self.u += du
        fired = self.v >= 30.0
        self.v[fired] = self.c
        self.u[fired] += self.d
        self.spikes = fired
        return fired

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"IzhikevichLayer(name={self.name!r}, n={self.n})"
