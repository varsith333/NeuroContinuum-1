"""Structural-health & machinery monitoring domains: bridge, dam, factory.

``stream.py`` gives a generic seismic hazard stream; this module specialises the
workload to the three target verticals.  The key insight that unifies them: a
bridge, a dam and a factory gearbox are all **"slowly-changing machine that
should vibrate one way but sometimes vibrates another way."**  The sensor is
always a vibration/accelerometer reading; only the *meaning* of the classes
changes.

Each generator produces drifting streams (load grows, bearings wear, moisture
intrudes) precisely so the continual-learning claim stays measurable.

Domain classes
--------------
bridge  : ``traffic`` (normal loaded crossing), ``wind`` (wind buffeting),
          ``crack`` (structural crack acoustic emission), ``impact`` (vehicle
          impact / expansion-joint slam).
dam     : ``normal`` (hydrostatic + slight seepage), ``seepage`` (internal
          erosion channel forming), ``flood`` (extreme load event),
          ``freeze`` (ice / thermal contraction).
factory : ``healthy`` (steady rotational nominal), ``imbalance`` (rotor mass
          imbalance -> 1x running-speed spike), ``bearing`` (bearing defect ->
          periodic impacts), ``looseness`` (structural looseness -> harmonics).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DOMAINS = ("bridge", "dam", "factory")

@dataclass
class DomWindow:
    x: np.ndarray        # raw vibration waveform (zero-mean, unit-ish scale)
    label: int
    t: float
    domain: str
    difficulty: float


class DomainStream:
    """Drifting vibration stream for one of the three target domains."""

    def __init__(self, domain: str = "bridge", fs: int = 512, win: float = 0.5,
                 seed: int = 0, drift: float = 1.0,
                 regime_at: int | None = None):
        if domain not in DOMAINS:
            raise ValueError(f"domain must be one of {DOMAINS}")
        self.domain = domain
        self.fs = int(fs)
        self.win_s = float(win)
        self.n = int(round(self.fs * self.win_s))
        self.rng = np.random.default_rng(seed)
        self.drift = float(drift)
        self.t = 0.0
        self.k = 0
        self.class_names = {
            "bridge": ["traffic", "wind", "crack", "impact"],
            "dam": ["normal", "seepage", "flood", "freeze"],
            "factory": ["healthy", "imbalance", "bearing", "looseness"],
        }[domain]
        # class priors (reversed at the regime change -> hard concept drift)
        priors = {
            "bridge":  [0.55, 0.20, 0.13, 0.12],
            "dam":     [0.60, 0.15, 0.13, 0.12],
            "factory": [0.50, 0.18, 0.17, 0.15],
        }[domain]
        self._priors = np.asarray(priors, dtype=float)
        self._priors = self._priors / self._priors.sum()
        # optional ABRUPT regime change: at window >= regime_at the operating
        # profile flips (rare <-> common classes) and the noise floor steps up,
        # as if a sensor mount loosened or the plant changed shift.  Persistent.
        self.regime_at = regime_at
        self.regime = False
        self._extra_noise = 0.0

    # ------------------------------------------------------------------ physics
    def _noise(self, colour: float = 1.5) -> np.ndarray:
        spec = (self.rng.standard_normal(self.n // 2 + 1)
                + 1j * self.rng.standard_normal(self.n // 2 + 1))
        f = np.linspace(1e-3, 1.0, self.n // 2 + 1)
        spec /= f ** (colour / 2.0)
        x = np.fft.irfft(spec, n=self.n)
        return x / (np.std(x) + 1e-9)

    def _tone(self, f: float, damp: float, phase: float = 0.0) -> np.ndarray:
        """Damped sinusoid — the canonical structural resonance / impact response."""
        t = np.arange(self.n) / self.fs
        return np.exp(-damp * t) * np.sin(2 * np.pi * f * t + phase)

    def _harmonics(self, f0: float, n_h: int, decay: float = 1.0) -> np.ndarray:
        """Sum of running-speed harmonics (the signature of imbalance/looseness)."""
        t = np.arange(self.n) / self.fs
        y = np.zeros(self.n)
        for h in range(1, n_h + 1):
            y += (1.0 / (h ** decay)) * np.sin(2 * np.pi * f0 * h * t
                                               + self.rng.uniform(0, 2 * np.pi))
        return y

    def _impact_train(self, f: float, period_s: float, n_hits: int = 4) -> np.ndarray:
        """Periodic impacts — the classic rolling-element bearing defect signature."""
        y = np.zeros(self.n)
        step = max(1, int(period_s * self.fs))
        for i in range(n_hits):
            start = (i * step + int(self.rng.uniform(0, step // 4))) % self.n
            seg = self._tone(f, damp=40.0)
            end = min(self.n, start + len(seg))
            y[start:end] += seg[:end - start]
        return y

    # ------------------------------------------------------------------ windows
    def _drift_scale(self) -> float:
        """Load/condition slowly increases -> features drift (the continual test)."""
        return 1.0 + self.drift * 0.5 * np.sin(2 * np.pi * self.k / 500.0)

    def next(self) -> DomWindow:
        # ---- abrupt regime change (checked before the priors are used) ------
        if (self.regime_at is not None and not self.regime
                and self.k >= self.regime_at):
            self.regime = True
            self._priors = self._priors[::-1].copy()   # rare <-> common flip
            self._extra_noise = 0.6                    # noise-floor step up

        g = self._drift_scale()
        d = self.domain
        base_noise = 0.5
        w = self._priors
        label = int(self.rng.choice(4, p=w))

        x = self.rng.standard_normal(self.n) * base_noise
        x = self._noise() * base_noise

        if d == "bridge":
            if label == 0:        # traffic: low-freq rumble + slow resonance
                x += 1.2 * g * self._tone(8.0, damp=3.0) + 0.6 * g * self._tone(25.0, damp=6.0)
                diff = 0.3
            elif label == 1:      # wind: broadband buffeting
                x += 0.9 * g * self._noise(colour=0.4)
                diff = 0.45
            elif label == 2:      # crack: sharp high-freq acoustic emission
                x += 2.0 * g * self._tone(140.0, damp=25.0)
                diff = 0.7
            else:                 # impact: strong low slam
                x += 2.2 * g * self._tone(18.0, damp=18.0)
                diff = 0.6
        elif d == "dam":
            if label == 0:        # normal: steady hydrostatic hiss
                x += 0.5 * g * self._noise(colour=1.0)
                diff = 0.3
            elif label == 1:      # seepage: rising narrowband trickle
                x += 1.3 * g * self._tone(60.0, damp=4.0)
                diff = 0.65
            elif label == 2:      # flood: huge low-frequency load
                x += 2.4 * g * self._tone(5.0, damp=1.5)
                diff = 0.55
            else:                 # freeze: ice crack, mid-band snap
                x += 1.7 * g * self._tone(95.0, damp=20.0)
                diff = 0.6
        else:  # factory
            f0 = 24.0 * g         # running speed (~1440 rpm) drifts with load
            if label == 0:        # healthy: clean 1x + faint harmonics
                x += 0.7 * g * self._harmonics(f0, n_h=2)
                diff = 0.3
            elif label == 1:      # imbalance: strong 1x
                x += 2.0 * g * self._harmonics(f0, n_h=1)
                diff = 0.5
            elif label == 2:      # bearing defect: periodic impacts
                x += 1.8 * g * self._impact_train(f=180.0, period_s=1.0 / (f0 * 3.5))
                diff = 0.7
            else:                 # looseness: many harmonics / subharmonics
                x += 1.4 * g * self._harmonics(f0, n_h=6, decay=1.3)
                diff = 0.6

        # noise floor rises over time -> everything gets harder (concept drift)
        x += self.rng.standard_normal(self.n) * 0.15 * (self.k / 900.0) * self.drift
        # ...plus the abrupt step introduced by the regime change
        if self.regime:
            x += self.rng.standard_normal(self.n) * self._extra_noise
        x = x / (np.std(x) + 1e-9)
        diff = float(np.clip(diff + 0.2 * (self.k / 900.0) * self.drift, 0.0, 1.0))
        if self.regime:
            diff = float(np.clip(diff + 0.3, 0.0, 1.0))
        out = DomWindow(x=x, label=label, t=self.t, domain=d, difficulty=diff)
        self.t += self.win_s
        self.k += 1
        return out

    def take(self, count: int) -> list[DomWindow]:
        return [self.next() for _ in range(count)]


def featurise(x: np.ndarray, n_bands: int = 6) -> np.ndarray:
    """Band-power + shape features for a vibration window (shared front-end).

    Deliberately cheap (FFT magnitudes, zero-crossings, crest factor) so it maps
    onto a small fixed-function hardware encoder ahead of the spike encoder.
    """
    spec = np.abs(np.fft.rfft(x)) ** 2
    bands = np.array_split(spec, n_bands)
    bp = np.log1p(np.array([b.sum() for b in bands]))
    bp = bp / (bp.max() + 1e-9)
    zcr = np.mean(np.abs(np.diff(np.sign(x)))) / 2.0
    crest = np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)) + 1e-9) / 4.0
    rms = np.sqrt(np.mean(x ** 2))
    feats = np.concatenate([bp, [zcr, crest, np.clip(rms / 3.0, 0, 1)]])
    return np.clip(feats, 0.0, 1.0)
