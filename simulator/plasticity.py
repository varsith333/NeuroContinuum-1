"""Meta-plastic BCM + reward-modulated STDP (the continual-learning core).

Why this is *not* textbook STDP:  vanilla Hebbian/STDP plasticity is
unbounded or fragile — it either saturates every weight to its rail or slowly
*forgets* previously learned patterns when the input statistics drift
(catastrophic forgetting).  Real cortex is believed to use a *sliding*
modification threshold, formalised by the Bienenstock-Cooper-Munro (BCM) rule,
which turns a single synapse into a self-stabilising, bidirectional learner.

BCM rule implemented here (and realised in RTL by ``rtl/bcm_plasticity.v``)::

    dw          = eta * pre * post * (post - theta)      # LTP if post>theta else LTD
    theta      <- theta + eta_theta * (post^2 - theta)   # slow sliding threshold

``theta`` is a slow moving average of the postsynaptic activity squared.  It is
what makes this different from a fixed-threshold Hebbian rule: the *direction*
of every weight change is judged against a threshold that itself drifts with
recent activity.  Sustained strong drive pushes ``theta`` up so only the
strongest responses keep potentiating (competitive learning / decorrelation);
a quiet period drags ``theta`` back down so weak synapses can re-grow.

Two properties follow that plain STDP lacks, and they are what an always-on,
never-retrained edge sensor needs:

  * **directional self-regulation** — activity above/below the sliding
    threshold flips each synapse between LTP and LTD, so responses track the
    current input statistics instead of ratcheting one way forever, and
  * **homeostatic gain control** — combined with synaptic scaling
    (``SynapticHomeostasis`` below) it keeps firing rates in a healthy band
    and prevents both saturation and runaway ("seizure") firing.

(To be precise: the sliding threshold sets the *direction*; the absolute weight
range is held by the hard clip here plus the synaptic-scaling pass in the
network.  ``simulator/examples/test_novel_features.py`` demonstrates each
behaviour in isolation.)

Reward-modulated STDP (R-STDP / three-factor rule) is also provided: a local
STDP *eligibility trace* is accumulated at each synapse and only converted into
a weight change when a global scalar ``reward`` (a "dopamine" broadcast)
arrives.  That yields reward-driven reinforcement learning with purely local
synaptic updates — no backprop, no gradient, no stored replay buffer.
"""

from __future__ import annotations

import numpy as np


class BCMPlasticity:
    """Vectorised BCM metaplastic rule for one synapse population.

    The sliding threshold ``theta`` is the heart of it: it is a slow low-pass of
    recent postsynaptic activity squared, so it *moves* with the input
    statistics.  Weights are pushed up when the post activity is above ``theta``
    and down when below — giving direction control that a fixed-threshold rule
    cannot.  Range safety comes from the ``[w_min, w_max]`` clip here together
    with ``SynapticHomeostasis`` (synaptic scaling) applied by the network.

    Parameters
    ----------
    eta : float
        Learning rate of the weight update.
    eta_theta : float
        Learning rate of the sliding threshold (<< eta, e.g. eta/50).
    theta0 : float
        Initial threshold; adapts immediately to the running activity.
    w_min, w_max : float
        Hard bounds acting as a safety net alongside synaptic scaling.
    """

    def __init__(self, n_pre: int, n_post: int, eta: float = 0.01,
                 eta_theta: float = 0.0002, theta0: float = 0.25,
                 w_min: float = 0.0, w_max: float = 1.0):
        self.eta = float(eta)
        self.eta_theta = float(eta_theta)
        self.w_min, self.w_max = float(w_min), float(w_max)
        self.theta = np.full(n_post, theta0, dtype=float)

    def update(self, W: np.ndarray, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        """One BCM step.  ``W`` is (n_post, n_pre); ``pre``/``post`` are bool/float
        spike vectors of length n_pre / n_post.  Returns the updated ``W`` (in place too)."""
        pre = pre.astype(float)
        post = post.astype(float)
        # outer product: how much each (post,pre) pair is co-active this step
        co = post[:, None] * pre[None, :]
        # bidirectional update around the sliding threshold
        W += self.eta * co * (post[:, None] - self.theta[:, None])
        # slow sliding threshold tracks postsynaptic activity squared
        self.theta += self.eta_theta * (post ** 2 - self.theta)
        np.clip(W, self.w_min, self.w_max, out=W)
        return W


class RewardSTDP:
    """Three-factor (reward-modulated) STDP with an eligibility trace.

    * A classical STDP pair-based rule first deposits a signed *eligibility*
      ``e`` at every synapse, based purely on local pre/post timing.
    * The eligibility trace decays with time constant ``tau_e`` (synapses
      "remember" a causal coincidence for a while).
    * A global scalar ``reward`` (broadcast by a supervisor / critic) multiplies
      the eligibility and only then moves the weight::

            dw = eta * reward * e

    This is the hardware-friendly route to reinforcement learning: each synapse
    only ever sees its own pre spike, its own post spike, and one global
    reward line — exactly the wiring the brain's dopaminergic system provides.
    """

    def __init__(self, n_pre: int, n_post: int, eta: float = 0.05,
                 a_plus: float = 0.01, a_minus: float = 0.012,
                 tau_plus: float = 20.0, tau_minus: float = 20.0,
                 tau_e: float = 200.0, w_min: float = 0.0, w_max: float = 1.0,
                 dt: float = 1.0):
        self.eta = float(eta)
        self.a_plus, self.a_minus = float(a_plus), float(a_minus)
        self.decay_plus = float(np.exp(-dt / tau_plus))
        self.decay_minus = float(np.exp(-dt / tau_minus))
        self.decay_e = float(np.exp(-dt / tau_e))
        self.w_min, self.w_max = float(w_min), float(w_max)
        self.pre_trace = np.zeros(n_pre)
        self.post_trace = np.zeros(n_post)
        self.elig = np.zeros((n_post, n_pre))

    def update(self, W: np.ndarray, pre: np.ndarray, post: np.ndarray,
               reward: float) -> np.ndarray:
        self.pre_trace *= self.decay_plus
        self.post_trace *= self.decay_minus
        self.elig *= self.decay_e
        pre_b, post_b = pre.astype(bool), post.astype(bool)
        if post_b.any():                      # LTP branch (pre shortly before post)
            self.elig[post_b, :] += self.a_plus * self.pre_trace[None, :]
        if pre_b.any():                       # LTD branch (post shortly before pre)
            self.elig[:, pre_b] -= self.a_minus * self.post_trace[:, None]
        self.pre_trace += pre_b
        self.post_trace += post_b
        W += self.eta * float(reward) * self.elig
        np.clip(W, self.w_min, self.w_max, out=W)
        return W

    def accumulate(self, pre: np.ndarray, post: np.ndarray) -> None:
        """Deposit eligibility for one timestep WITHOUT moving any weight.

        Used every timestep during a window to build up the causal trace; the
        reward is applied later in one shot via :meth:`apply_reward` (the
        hardware does exactly this: local eligibility banks, one global reward
        line strobed at the end of an event).
        """
        self.pre_trace *= self.decay_plus
        self.post_trace *= self.decay_minus
        self.elig *= self.decay_e
        pre_b, post_b = pre.astype(bool), post.astype(bool)
        if post_b.any():
            self.elig[post_b, :] += self.a_plus * self.pre_trace[None, :]
        if pre_b.any():
            self.elig[:, pre_b] -= self.a_minus * self.post_trace[:, None]
        self.pre_trace += pre_b
        self.post_trace += post_b

    def apply_reward(self, W: np.ndarray, reward: float) -> np.ndarray:
        """Convert the accumulated eligibility into a weight change now."""
        W += self.eta * float(reward) * self.elig
        np.clip(W, self.w_min, self.w_max, out=W)
        return W


class SynapticHomeostasis:
    """Global homeostatic gain control (a.k.a. synaptic scaling).

    After every window, each postsynaptic weight vector is rescaled so its L1
    norm stays at ``target``.  This is a cheap, purely local safety mechanism
    that cortex is thought to use to keep firing rates in a healthy band.  In
    the self-healing core it also acts as the *first* line of defence before a
    synapse is declared faulty and remapped (see ``simulator/selfhealing.py``).
    """

    def __init__(self, target: float = 1.0, eps: float = 1e-9):
        self.target = float(target)
        self.eps = eps

    def apply(self, W: np.ndarray) -> np.ndarray:
        s = np.abs(W).sum(axis=1, keepdims=True)
        scale = self.target / (s + self.eps)
        np.clip(scale, 0.25, 4.0, out=scale)  # bound the correction
        return W * scale
