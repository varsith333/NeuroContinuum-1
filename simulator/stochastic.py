"""Stochastic spiking neurons — on-chip Monte-Carlo sampling for uncertainty.

Every commercial neuromorphic chip to date (Loihi 2, TrueNorth, SpiNNaker2,
BrainScaleS-2) is *deterministic*: same input, same output, every time.  That is
fine for classification, but for a safety-critical sensor (a mine-hazard
monitor, a seismic alarm) a bare class label is not enough — you also want to
know *how sure* the chip is.

NeuroContinuum-1 injects a controlled amount of noise directly into the
membrane potential::

    v <- v + (-(v - v_rest) + R*I)/tau + sigma * N(0,1)

Because the membrane noise is Gaussian and its variance ``sigma^2`` is
programmable, the population of output neurons performs an approximate
Gibbs/MCMC sample from a Boltzmann distribution over neuron identities.  Run the
*same* noisy network for K stochastic trials on the *same* input and count how
often each output neuron wins: that histogram is a Monte-Carlo estimate of the
posterior P(class | input).  From it we read out both a decision **and** a
calibrated uncertainty (entropy / margin), entirely on-chip and without a single
line of backpropagation.

The noise source in hardware is a 32-bit maximal-length LFSR plus a
Box-Muller-style transform; ``rtl/stochastic_neuron.v`` mirrors this model
exactly (uniform LFSR taps -> gaussian via sum-of-uniforms approximation).
"""

from __future__ import annotations

import numpy as np

from .neurons import LIFLayer


class StochasticLIFLayer(LIFLayer):
    """LIF population with programmable Gaussian membrane noise.

    ``sigma`` is the std-dev of the noise added to the membrane each step, in
    the same units as ``v``.  ``sigma=0`` recovers the deterministic LIF.
    ``noise_mode`` selects how the hardware draws noise:

    * ``'gauss'`` — NumPy Gaussian (models the ideal Box-Muller hardware path)
    * ``'lfsr'``  — 32-bit LFSR uniform + sum-of-8 uniforms CLT approximation,
                    matching ``rtl/stochastic_neuron.v`` bit-for-bit in spirit.
    """

    def __init__(self, n: int, sigma: float = 0.05, noise_mode: str = "gauss",
                 seed: int = 0, **kw):
        super().__init__(n, **kw)
        self.sigma = float(sigma)
        self.noise_mode = noise_mode
        self._rng = np.random.default_rng(seed)
        # hardware-style LFSR state (one 32-bit word per neuron)
        self._lfsr = np.full(n, 0xACE1 + 1 + seed, dtype=np.uint32)

    def _draw_noise(self) -> np.ndarray:
        if self.sigma == 0.0:
            return np.zeros(self.n)
        if self.noise_mode == "gauss":
            return self.sigma * self._rng.standard_normal(self.n)
        # --- hardware-mirroring path: 32-bit LFSR, sum of 8 uniforms ---
        return self.sigma * self._lfsr_uniform_sum(k=8)

    def _lfsr_uniform_sum(self, k: int = 8) -> np.ndarray:
        """Advance each neuron's 32-bit maximal LFSR k times, sum the uniform
        draws in [0,1), and CLT-normalise to approx N(0,1)."""
        s = np.zeros(self.n)
        for _ in range(k):
            # maximal-length 32-tap LFSR: x^32 + x^22 + x^2 + x^1 + 1
            bit = ((self._lfsr >> 31) ^ (self._lfsr >> 21) ^
                   (self._lfsr >> 1) ^ self._lfsr) & np.uint32(1)
            self._lfsr = (self._lfsr << np.uint32(1)) | bit
            s += (self._lfsr.astype(np.float64) / 2**32)
        u = s / k                       # mean 0.5
        return (u - 0.5) * np.sqrt(12.0)  # zero-mean, unit-variance approx

    def step(self, i_ext) -> np.ndarray:
        spikes = super().step(i_ext)
        if self.sigma > 0:
            self.v += self._draw_noise()
        return spikes


def posterior_from_trials(winners_by_trial: np.ndarray, n_classes: int,
                          smooth: float = 1e-3) -> np.ndarray:
    """Turn per-trial winner identities into a smoothed posterior vector.

    ``winners_by_trial`` is an array of length K (one winner class id per
    stochastic trial).  Returns a length-``n_classes`` probability vector using
    add-``smooth`` Laplace smoothing, so the entropy is always finite.
    """
    counts = np.bincount(np.asarray(winners_by_trial, dtype=int),
                         minlength=n_classes).astype(float)
    p = counts + smooth
    return p / p.sum()


def decision_with_uncertainty(posterior: np.ndarray) -> dict:
    """Read a decision + two complementary uncertainty measures off a posterior.

    Returns dict with:
      * ``decision``   — arg-max class,
      * ``confidence`` — posterior mass on the winner,
      * ``entropy``    — normalised Shannon entropy in [0,1] (0 = certain),
      * ``margin``     — gap between top-1 and top-2 posterior (0 = tie).
    """
    p = np.asarray(posterior, dtype=float)
    order = np.argsort(p)[::-1]
    ent = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p))
    return {
        "decision": int(order[0]),
        "confidence": float(p[order[0]]),
        "entropy": float(ent),
        "margin": float(p[order[0]] - p[order[1]]) if len(order) > 1 else float(p[order[0]]),
    }
