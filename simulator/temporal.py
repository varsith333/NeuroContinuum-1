"""Temporal (latency) coding and the first-spike decoder.

Rate coding — the default on essentially every shipped neuromorphic chip —
wastes energy: a neuron must fire tens of spikes to communicate one number.
Temporal coding instead encodes intensity in *when* a neuron fires:

    latency = t_max - gain * intensity        (stronger input -> earlier spike)

A single spike per neuron then carries the full message.  Combined with a
race-based readout this yields sub-millisecond, ultra-sparse inference —
the exact mechanism behind biological sensory systems (e.g. auditory brainstem
time-coding) and the reason insect brains are so fast on so little power.

``FirstSpikeDecoder`` implements the race readout: the first output neuron to
spike *wins* and its identity is the decision.  Because neurons that lost the
race are simply cancelled, the decoder needs no accumulator and no multiply —
just a latch and a timer, which is what ``rtl/first_spike_decoder.v`` builds.
"""

from __future__ import annotations

import numpy as np


def latency_encode(values: np.ndarray, t_max: float = 40.0, gain: float = 36.0,
                   t_min: float = 2.0) -> np.ndarray:
    """Map real values in [0,1] to per-neuron first-spike latencies (ms).

    ``values`` 0..1 -> latencies t_max .. t_min.  Neurons whose value is <= 0
    are silent (latency = +inf) so the encoding is naturally sparse.
    """
    v = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    lat = t_max - gain * v
    lat[v <= 0.0] = np.inf
    return np.maximum(lat, t_min)


def latency_to_spike_train(latencies: np.ndarray, duration: float,
                           dt: float = 1.0) -> np.ndarray:
    """Materialise (T, N) binary spike trains from per-neuron first-spike times."""
    n = len(latencies)
    T = int(round(duration / dt))
    train = np.zeros((T, n), dtype=bool)
    idx = np.round(latencies / dt).astype(int)
    for j in range(n):
        if np.isfinite(latencies[j]) and 0 <= idx[j] < T:
            train[idx[j], j] = True
    return train


class FirstSpikeDecoder:
    """Race-based winner-take-first readout.

    Feed it one boolean spike vector per timestep.  The first neuron to spike
    latches as the winner and every subsequent spike is ignored until
    :meth:`reset`.  ``race_time`` reports how long the decision took.

    ``allow_multi`` (default False) supports a *soft* variant used by the
    uncertainty estimator: instead of a single winner it counts first spikes
    per class across the population, returning a histogram that plays the role
    of a posterior without any stochastic trials.
    """

    def __init__(self, n_classes: int, allow_multi: bool = False):
        self.n_classes = int(n_classes)
        self.allow_multi = allow_multi
        self.reset()

    def reset(self, t: float = 0.0) -> None:
        self.winner = None
        self.race_time = None
        self.t = t
        self.first_spike = np.full(self.n_classes, np.inf)
        self.counts = np.zeros(self.n_classes)

    def step(self, spikes: np.ndarray, t: float) -> int | None:
        self.t = t
        sp = np.asarray(spikes, dtype=bool)
        if self.allow_multi:
            for j in np.flatnonzero(sp):
                c = j % self.n_classes
                if not np.isfinite(self.first_spike[c]):
                    self.first_spike[c] = t
                    self.counts[c] += 1
            return None
        if self.winner is None and sp.any():
            self.winner = int(np.flatnonzero(sp)[0])
            self.race_time = t
        return self.winner

    def posterior(self) -> np.ndarray:
        """Soft-decoder posterior: mass concentrated on earliest first spikes."""
        if not self.allow_multi:
            p = np.full(self.n_classes, 1e-3)
            if self.winner is not None:
                p[self.winner] += 1.0
            return p / p.sum()
        # weight each class by how much earlier it fired than the population
        f = self.first_spike.copy()
        f[~np.isfinite(f)] = np.inf
        span = np.max(f[np.isfinite(f)]) if np.isfinite(f).any() else 1.0
        w = (span + 1.0) - f
        w[~np.isfinite(f)] = 0.0
        w = np.clip(w, 0.0, None)
        if w.sum() == 0:
            w = np.ones(self.n_classes)
        return w / w.sum()
