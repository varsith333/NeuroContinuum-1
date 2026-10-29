"""Spike *encodings* — turning real numbers into spikes (and back).

Every neuromorphic system needs a translator between the continuous world and
the all-or-nothing spike world.  Three encodings are provided because different
applications genuinely want different ones:

* **Rate coding** — number = how *often* a neuron fires.  Simple, robust, but
  spends many spikes per bit (what most shipped chips use).
* **Latency coding** — number = *when* the neuron fires (first-spike time).
  One spike carries the whole message: sparse and fast (``temporal.py`` races
  these).  NeuroContinuum's default.
* **Poisson / stochastic** — draws spikes as a random process with the given
  rate, matching the noisy, asynchronous behaviour of real cortex and giving the
  stochastic neuron something realistic to chew on.

Also included is the inverse readout (spikes -> rate) for analysing results.
"""

from __future__ import annotations

import numpy as np


def rate_to_current(rate_hz: float, tau: float = 10.0, v_rest: float = 0.0,
                    v_thresh: float = 1.0, r: float = 1.0) -> float:
    """Steady current needed to make an LIF neuron fire at ``rate_hz``.

    Inverts the f-I curve so a desired firing rate can be requested directly.
    """
    if rate_hz <= 0:
        return 0.0
    isi_ms = 1000.0 / rate_hz
    # steady-state membrane must reach threshold within isi - t_ref; solve for I
    frac = np.clip(1.0 - np.exp(-max(isi_ms, 1.0) / tau), 1e-6, 1.0)
    return (v_thresh - v_rest) / (r * frac)


def poisson_spikes(rate_hz: float, duration_ms: float, dt: float = 1.0,
                   seed: int | None = None) -> np.ndarray:
    """Draw one binary spike train with mean rate ``rate_hz`` (independent trials)."""
    rng = np.random.default_rng(seed)
    T = int(round(duration_ms / dt))
    p = np.clip(rate_hz * dt / 1000.0, 0.0, 1.0)
    return rng.random(T) < p


def poisson_matrix(rates_hz, duration_ms: float, dt: float = 1.0,
                   seed: int | None = None) -> np.ndarray:
    """Draw a (T, N) matrix of Poisson trains for a population of ``rates_hz``."""
    rng = np.random.default_rng(seed)
    T = int(round(duration_ms / dt))
    rates = np.atleast_1d(np.asarray(rates_hz, dtype=float))
    p = np.clip(rates * dt / 1000.0, 0.0, 1.0)[None, :]
    return rng.random((T, len(rates))) < p


def pattern_input(pattern: np.ndarray, duration_ms: float, dt: float = 1.0,
                  active_current: float = 2.0) -> np.ndarray:
    """Build a (T, N) *current* matrix that stimulates only ``pattern`` neurons.

    ``pattern`` is a 0/1 (or real-valued) mask selecting which input neurons are
    driven during the whole window.  Used by the clustering / classification demos
    to present "A vs B" stimuli cleanly.
    """
    T = int(round(duration_ms / dt))
    drive = np.asarray(pattern, dtype=float) * active_current
    return np.tile(drive, (T, 1))


def spikes_to_rate(spikes: np.ndarray, duration_ms: float) -> np.ndarray:
    """Inverse readout: spike train (T, N) -> mean firing rate (N,) in Hz."""
    T = spikes.shape[0]
    if T == 0:
        return np.zeros(spikes.shape[1])
    return spikes.sum(axis=0) * (1000.0 / duration_ms)


def spikes_to_bins(spikes: np.ndarray, duration_ms: float,
                   n_bins: int = 10) -> np.ndarray:
    """Smooth a spike train into ``n_bins`` rate bins (for plotting/analysis)."""
    T, N = spikes.shape
    edges = np.linspace(0, T, n_bins + 1).astype(int)
    out = np.zeros((n_bins, N))
    for i in range(n_bins):
        seg = spikes[edges[i]:edges[i + 1]]
        if seg.shape[0]:
            out[i] = seg.mean(axis=0) * 1000.0 / (duration_ms / n_bins)
    return out
