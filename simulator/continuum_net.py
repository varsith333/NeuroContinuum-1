"""NeuroContinuum-1 — the flagship model fusing all four novel blocks.

Pipeline (one streaming hazard window at a time)::

    raw waveform
        │  featurise()                      (stream.py)
        ▼
    feature vector (real, 0..1)
        │  latency_encode()                 (temporal.py)   ← sparse, 1-spike coding
        ▼
    input spikes ──► [input LIF layer]
                         │  BCM meta-plasticity (plasticity.py)  ← continual adaptation
                         ▼                 + self-healing array (selfhealing.py)
                     [hidden layer, self-repairing]
                         │  R-STDP eligibility + reward broadcast (plasticity.py)
                         ▼
                     [stochastic output layer] (stochastic.py)  ← membrane noise
                         │  K Monte-Carlo trials → posterior
                         ▼
              decision + Bayesian uncertainty  (stochastic.py)

Design intent: every stage is *local*, *event-driven* and *on-chip*, so the
same computation maps one-to-one onto the RTL in ``rtl/``.  The network learns
continually from a drifting stream with a single scalar reward signal — no
backprop, no replay buffer, no cloud retraining — while reporting calibrated
uncertainty and repairing its own failing synapses.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .neurons import LIFLayer
from .plasticity import BCMPlasticity, RewardSTDP, SynapticHomeostasis
from .selfhealing import SelfHealingSynapseArray
from .stochastic import StochasticLIFLayer, posterior_from_trials, decision_with_uncertainty
from .temporal import latency_encode


class NeuroContinuum:
    """End-to-end continual-learning hazard classifier with uncertainty."""

    def __init__(
        self,
        n_feat: int = 9,
        n_hidden: int = 48,
        n_classes: int = 4,
        window_ms: int = 40,
        mc_trials: int = 7,
        noise_sigma: float = 0.10,
        input_gain: float = 6.0,
        output_gain: float = 1.0,
        rstdp_eta: float = 0.05,
        syn_tau: float = 5.0,
        seed: int = 0,
    ):
        self.n_feat, self.n_hidden, self.n_classes = n_feat, n_hidden, n_classes
        self.window_ms = window_ms
        self.mc_trials = mc_trials
        self.input_gain = float(input_gain)
        self.output_gain = float(output_gain)
        self.syn_decay = float(np.exp(-1.0 / syn_tau))  # exponential synapse tail
        rng = np.random.default_rng(seed)

        # --- input layer (deterministic LIF; the "sensor spike encoder") ----
        self.inp = LIFLayer(n_feat, tau=8.0, v_thresh=1.0, t_ref=3.0,
                            name="input")

        # --- hidden layer: stochastic + BCM plasticity + self-healing -------
        self.hid = StochasticLIFLayer(n_hidden, sigma=noise_sigma, seed=seed + 1,
                                      tau=15.0, v_thresh=1.0, t_ref=2.0,
                                      name="hidden")
        W_in = rng.uniform(0.05, 0.6, size=(n_hidden, n_feat))
        self.heal_in = SelfHealingSynapseArray(W_in, spare_frac=0.16)
        self.bcm = BCMPlasticity(n_feat, n_hidden, eta=0.006, eta_theta=0.0004,
                                 theta0=0.30)
        self.homeo = SynapticHomeostasis(target=1.4)

        # --- output layer: stochastic readout, R-STDP from hidden -----------
        self.out = StochasticLIFLayer(n_classes, sigma=noise_sigma, seed=seed + 2,
                                      tau=12.0, v_thresh=1.0, t_ref=1.0,
                                      name="output")
        # Rows unit-normalised so the graded readout is a pure direction match
        # (nearest-class), NOT biased toward whichever random row got largest norm.
        Wout = rng.uniform(0.1, 0.8, size=(n_classes, n_hidden))
        self.W_out = Wout / (np.linalg.norm(Wout, axis=1, keepdims=True) + 1e-9)
        self.rstdp = RewardSTDP(n_hidden, n_classes, eta=rstdp_eta, tau_e=150.0)
        self.rstdp_eta_readout = float(rstdp_eta)  # reward-gated readout rate

        self._reward = 0.0

    # ------------------------------------------------------------------ learn
    def _present(self, feats: np.ndarray, learn: bool,
                 label: int | None = None) -> np.ndarray:
        """Present one encoded feature vector; return per-class spike counts.

        If ``learn`` and a true ``label`` are given, the window ends with a
        reward-modulated update: eligibility is deposited **every timestep**
        (locally, like the RTL eligibility bank), and the single scalar reward
        from the teacher/critic is strobed once at the end.
        """
        lat = latency_encode(feats, t_max=float(self.window_ms - 3), gain=20.0,
                             t_min=1.0)
        valid = np.isfinite(lat)                              # value<=0 -> silent
        spike_t = np.round(np.where(valid, lat, -1)).astype(int)
        T = self.window_ms
        self.inp.reset_state(); self.hid.reset_state(); self.out.reset_state()
        out_spikes = np.zeros(self.n_classes)
        syn_in = np.zeros(self.n_hidden)    # decaying input->hidden synaptic current
        syn_out = np.zeros(self.n_classes)  # decaying hidden->output synaptic current
        out_drive = np.zeros(self.n_classes)  # graded decision signal (Σ current)
        hid_counts = np.zeros(self.n_hidden)   # hidden population code this window

        for t in range(T):
            # true latency code: each input fires EXACTLY ONCE at spike_t[i]
            s_in = valid & (spike_t == t)
            self.inp.spikes = s_in

            # hidden: exponential synaptic current drives the stochastic LIF,
            # then BCM plasticity + self-healing update the live weights
            W_live = self.heal_in.live_view()
            syn_in = syn_in * self.syn_decay + self.input_gain * (W_live @ s_in.astype(float))
            s_hid = self.hid.step(syn_in)
            hid_counts += s_hid
            if learn:
                self.heal_in.update(s_in, s_hid,
                                    lambda W, pre, post: self.bcm.update(W, pre, post))
                self.heal_in.set_live(self.homeo.apply(self.heal_in.live_view()))

            # output: exponential syn current + stochastic membrane, then a
            # winner-take-all race: only the STRONGEST output neuron may emit.
            # (Without WTA all four classes fire together on every burst and the
            # argmax degenerates to a tie broken by noise.)
            syn_out = syn_out * self.syn_decay + self.output_gain * (self.W_out @ s_hid.astype(float))
            out_drive += syn_out            # graded accumulator for the readout
            s_out = self.out.step(syn_out)
            if s_out.sum() > 1:
                keep = int(np.argmax(self.out.v))
                s_out = np.zeros_like(s_out)
                s_out[keep] = True
                self.out.spikes = s_out       # layer state = the winner only
            out_spikes += s_out
            if learn:
                # deposit spike-timing eligibility locally (timing credit) — kept
                # for hardware fidelity; the readout's main update is below.
                self.rstdp.accumulate(self.hid.spikes, self.out.spikes)

        # --- reward-modulated readout learning (three-factor, no backprop) ---
        # factor 1 = local presynaptic activity (hid_counts), factor 2 = a single
        # scalar reward from the teacher/critic, factor 3 routed to the class rows.
        # Pulls the correct row toward the active population code and pushes a
        # wrongly-winning row away.  Purely local per synapse + one reward line.
        if learn and label is not None:
            d = int(np.argmax(out_drive))
            self._reward_readout(hid_counts, d, int(label))
        self._last_drive = out_drive
        self._last_hid_spikes = int(hid_counts.sum())
        return out_spikes

    def _reward_readout(self, hid_counts: np.ndarray, winner: int, label: int) -> None:
        """Reward-gated competitive Hebbian update of ``W_out``.

        The teacher sends one scalar reward; the correct-class row receives a
        positive teaching gate and (on error) the wrongly-winning row a negative
        one.  Each synapse only uses its own hidden input plus that gated reward,
        so it stays hardware-local (a global reward/teaching line + row select).
        """
        h = hid_counts / (np.linalg.norm(hid_counts) + 1e-9)   # unit population code
        eta = self.rstdp_eta_readout
        if winner == label:
            # correct: converge the winning row onto the pattern (reinforce)
            self.W_out[label] += eta * (h - self.W_out[label])
        else:
            # wrong: push the spurious winner off the pattern, pull correct onto it
            self.W_out[winner] -= eta * 0.7 * (h - self.W_out[winner])
            self.W_out[label] += eta * 0.7 * (h - self.W_out[label])
        np.clip(self.W_out, -1.5, 1.5, out=self.W_out)
        # keep rows unit-norm so the graded readout stays a pure direction match
        self.W_out /= (np.linalg.norm(self.W_out, axis=1, keepdims=True) + 1e-9)
        self._reward = 1.0 if winner == label else -1.0

    # --------------------------------------------------------------- inference
    def posterior(self, feats: np.ndarray) -> np.ndarray:
        """Monte-Carlo posterior over K noisy trials.

        Each trial runs the *stochastic* network (membrane noise in hidden +
        output) and the winner is the arg-max of the graded output drive.  The
        hidden layer's trial-to-trial variability makes the winner flip on
        ambiguous inputs -> the vote histogram is the posterior.  Noise is the
        ONLY thing that varies between trials (input is identical), so the
        spread of winners is a genuine uncertainty estimate.
        """
        winners = []
        for _ in range(self.mc_trials):
            self._present(feats, learn=False)
            winners.append(int(np.argmax(self._last_drive)))
        return posterior_from_trials(np.array(winners), self.n_classes)

    def predict(self, feats: np.ndarray, learn: bool = True,
                label: int | None = None) -> dict:
        """One streaming step: classify, optionally learn, return decision+unc.

        Pass the true ``label`` when available so the reward (teacher/critic)
        reflects ground truth; without it the pass is inference-only.
        """
        post = self.posterior(feats)
        info = decision_with_uncertainty(post)
        _ = self._present(feats, learn=learn, label=label)
        info["posterior"] = post
        return info


class DriftTracker:
    """Sliding-window metrics for a drifting stream (accuracy + calibration)."""

    def __init__(self, n_classes: int, window: int = 50):
        self.n_classes = n_classes
        self.yt = deque(maxlen=window)
        self.yp = deque(maxlen=window)
        self.conf = deque(maxlen=window)
        self.acc_hist = []

    def add(self, y: int, info: dict) -> None:
        self.yt.append(y)
        self.yp.append(info["decision"])
        self.conf.append(info["confidence"])

    @property
    def accuracy(self) -> float:
        if not self.yt:
            return float("nan")
        return float(np.mean(np.array(self.yt) == np.array(self.yp)))

    @property
    def mean_confidence(self) -> float:
        return float(np.mean(self.conf)) if self.conf else float("nan")

    def ece(self, bins: int = 8) -> float:
        """Expected calibration error: gap between confidence and accuracy."""
        if len(self.yt) < bins:
            return float("nan")
        conf = np.array(self.conf)
        correct = (np.array(self.yt) == np.array(self.yp)).astype(float)
        edges = np.linspace(0, 1, bins + 1)
        ece = 0.0
        for i in range(bins):
            m = (conf >= edges[i]) & (conf < edges[i + 1])
            if m.sum() > 0:
                ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
        return float(ece)
