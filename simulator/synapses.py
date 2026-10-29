"""Synapse models and spike-timing-dependent plasticity (STDP).

The plasticity rule implemented here is the canonical *trace-based* STDP rule
(Ponulak & Kasinski, 2011).  Each population keeps a low-pass filtered "trace"
of its own spikes:

    pre_trace[j]  <- decay(pre_trace[j])  ; +1 when pre j spikes   (tau_plus)
    post_trace[i] <- decay(post_trace[i]) ; +1 when post i spikes  (tau_minus)

Weight updates fire *with the spike events themselves*:

    on post spike i:  w[i, j] += a_plus  * pre_trace[j]    (pre shortly BEFORE post -> LTP)
    on pre  spike j:  w[i, j] -= a_minus * post_trace[i]   (post shortly BEFORE pre  -> LTD)

This is exactly the rule realised by ``rtl/stdp_synapse.v`` (there in fixed
point, with saturating arithmetic and weight clipping).
"""

from __future__ import annotations

import numpy as np


class STDPRule:
    """Parameters of the trace-based STDP rule."""

    def __init__(
        self,
        a_plus: float = 0.01,
        a_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
        dt: float = 1.0,
    ):
        self.a_plus = float(a_plus)
        self.a_minus = float(a_minus)
        self.tau_plus = float(tau_plus)
        self.tau_minus = float(tau_minus)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        # per-step decay factors of the two traces (forward Euler on exp decay)
        self.decay_plus = float(np.exp(-dt / tau_plus))
        self.decay_minus = float(np.exp(-dt / tau_minus))


class Connection:
    """A weighted, optionally plastic connection between two layers.

    ``W`` is stored post x pre (rows = postsynaptic neurons), so the feed-forward
    drive of a step is simply ``W @ pre_spikes``.  Trace-based STDP updates are
    applied after every timestep in :meth:`learn`.
    """

    def __init__(
        self,
        src,
        dst,
        w,
        rule: STDPRule | None = None,
        normalize: bool = False,
        label: str = "",
    ):
        self.src = src
        self.dst = dst
        self.rule = rule
        self.normalize = normalize
        self.label = label or f"{src.name}->{dst.name}"
        self.learn_enabled = rule is not None

        if np.isscalar(w):
            rng = np.random.default_rng(abs(hash(self.label)) % (2 ** 31))
            self.W = rng.uniform(0.0, float(w), size=(dst.n, src.n))
        else:
            self.W = np.array(w, dtype=float)
            if self.W.shape != (dst.n, src.n):
                raise ValueError(
                    f"weight shape {self.W.shape} != (dst={dst.n}, src={src.n})"
                )

        self.pre_trace = np.zeros(src.n, dtype=float)
        self.post_trace = np.zeros(dst.n, dtype=float)

    def drive(self) -> np.ndarray:
        """Current delivered to the postsynaptic layer for the last step."""
        return self.W @ self.src.spikes.astype(float)

    def learn(self) -> None:
        """Apply one timestep of trace decay + STDP (call once per step)."""
        if self.rule is None:
            return
        # 1. decay traces
        self.pre_trace *= self.rule.decay_plus
        self.post_trace *= self.rule.decay_minus

        if self.learn_enabled:
            post = self.dst.spikes
            pre = self.src.spikes
            # 2a. LTP: postsynaptic spike reads out recent presynaptic activity
            if post.any():
                self.W[post, :] += self.rule.a_plus * self.pre_trace[None, :]
            # 2b. LTD: presynaptic spike reads out recent postsynaptic activity
            if pre.any():
                self.W[:, pre] -= self.rule.a_minus * self.post_trace[:, None]
            np.clip(self.W, self.rule.w_min, self.rule.w_max, out=self.W)
            if self.normalize:
                s = self.W.sum(axis=1, keepdims=True)
                np.divide(self.W, s, out=self.W, where=s > 0)

        # 3. register this step's spikes into the traces
        self.pre_trace += pre.astype(float)
        self.post_trace += post.astype(float)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        plastic = "STDP" if self.rule else "static"
        return f"Connection({self.label}, {plastic}, W={self.W.shape})"
