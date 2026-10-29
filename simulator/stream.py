"""Non-stationary hazard streaming — the workload that motivates the design.

Why not MNIST?  A benchmark with fixed statistics plays directly into the hands
of offline-trained chips and hides the very problem NeuroContinuum-1 attacks:
*concept drift*.  A real hazard sensor (mine seismicity, structural health,
earthquake early-warning) never stops changing — the geology shifts, seasons
change the noise floor, sensors age, and a blast one day looks nothing like a
rockburst the next.

This module synthesises a streaming, drifting waveform world with four event
classes and *deliberately* non-stationary statistics:

  0 ``background``  — coloured (1/f-ish) microseismic noise,
  1 ``p_wave``      — impulsive compressional onset (fast rise, slow decay),
  2 ``s_wave``      — delayed, lower-frequency, higher-amplitude shear arrival,
  3 ``blast``       — near-instant full-bandwidth detonation (human activity).

Two drift mechanisms are provided so the continual-learning claim can be
measured rather than asserted:

  * ``drift='amplitude'``   — class amplitudes scale slowly over the stream,
  * ``drift='noise'``       — background noise power ramps up over time,
  * ``drift='spectral'``    — the dominant frequency of each class shifts,
  * ``drift='mix'``         — the class prior itself changes (new regime).

Each window also exposes the *true* label and a difficulty score (separation
from the runner-up class) so accuracy, adaptation lag and uncertainty
calibration can all be scored against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASS_NAMES = ["background", "p_wave", "s_wave", "blast"]


@dataclass
class Window:
    x: np.ndarray        # (win,) raw waveform, zero-mean, roughly unit scale
    label: int
    t: float             # stream time in seconds
    difficulty: float    # 0 easy .. 1 hard


class HazardStream:
    """Generates an endless drifting stream of hazard waveforms."""

    def __init__(self, fs: int = 256, win: float = 1.0, seed: int = 0,
                 drift: str = "mix"):
        self.fs = int(fs)
        self.win_s = float(win)
        self.n = int(round(self.fs * self.win_s))
        self.rng = np.random.default_rng(seed)
        self.drift = drift
        self.t = 0.0
        self._step = 0

    # ------------------------------------------------------------------ draw
    def _noise(self, n: int, power: float, colour: float = 1.5) -> np.ndarray:
        """Coloured noise: pink-ish via frequency-domain 1/f^colour shaping."""
        spec = self.rng.standard_normal(n // 2 + 1) + 1j * self.rng.standard_normal(n // 2 + 1)
        f = np.linspace(1e-3, 1.0, n // 2 + 1)
        spec /= f ** (colour / 2.0)
        x = np.fft.irfft(spec, n=n)
        x /= (np.std(x) + 1e-9)
        return np.sqrt(power) * x

    def _envelope(self, n: int, rise: float, decay: float, f: float,
                  phase: float = 0.0) -> np.ndarray:
        t = np.arange(n) / self.fs
        env = np.where(t < rise, t / max(rise, 1e-6),
                       np.exp(-(t - rise) / max(decay, 1e-6)))
        carrier = np.sin(2 * np.pi * f * t + phase)
        return env * carrier

    def _drift_params(self, label: int) -> tuple[float, float, float]:
        """Returns (amplitude, freq_hz, noise_power) for the current stream time."""
        k = self._step
        amp, freq = 1.0, 8.0
        base_noise = 0.4
        if label == 1:      # p-wave: fast, mid band
            amp, freq = 1.2, 14.0
        elif label == 2:    # s-wave: slow, low band, big
            amp, freq = 1.6, 5.0
        elif label == 3:    # blast: huge, wideband impulse
            amp, freq = 2.4, 22.0
        else:               # background
            amp, freq = 0.25, 3.0

        d = self.drift
        if d in ("amplitude", "mix"):
            amp *= 1.0 + 0.6 * np.sin(2 * np.pi * k / 400.0)   # slow amplitude tide
        if d in ("noise", "mix"):
            base_noise *= 1.0 + 0.9 * (k / 800.0) % 1.0        # noise floor climbs
        if d in ("spectral", "mix"):
            freq *= 1.0 + 0.4 * np.sin(2 * np.pi * k / 600.0 + label)  # band drift
        return amp, freq, base_noise

    # ----------------------------------------------------------------- window
    def next(self) -> Window:
        k = self._step
        # non-stationary class prior: early stream is mostly background; later
        # a blast-heavy regime arrives with no warning.
        if self.drift == "mix":
            w = np.array([0.62, 0.14, 0.14, 0.10])
            if k > 500:
                w = np.array([0.30, 0.20, 0.20, 0.30])   # new regime
        else:
            w = np.array([0.62, 0.14, 0.14, 0.10])
        w = w / w.sum()
        label = int(self.rng.choice(4, p=w))

        amp, freq, noise_p = self._drift_params(label)
        n = self.n
        x = self._noise(n, noise_p)
        if label == 0:
            difficulty = 0.15
        elif label == 1:
            x += amp * self._envelope(n, rise=0.02, decay=0.15, f=freq)
            difficulty = 0.55
        elif label == 2:
            lag = int(0.25 * self.fs)
            e = np.zeros(n)
            body = amp * self._envelope(n - lag, rise=0.08, decay=0.40, f=freq)
            e[lag:] = body
            x += e
            difficulty = 0.65
        else:
            x += amp * self._envelope(n, rise=0.004, decay=0.06, f=freq)
            difficulty = 0.75

        # difficulty rises with the noise floor (everything gets harder over time)
        difficulty = float(np.clip(difficulty + 0.25 * (noise_p - 0.4), 0.0, 1.0))
        x = x / (np.std(x) + 1e-9)

        w_out = Window(x=x, label=label, t=self.t, difficulty=difficulty)
        self.t += self.win_s
        self._step += 1
        return w_out

    def take(self, count: int) -> list[Window]:
        return [self.next() for _ in range(count)]


def featurise(x: np.ndarray, n_bands: int = 6) -> np.ndarray:
    """Cheap on-chip-friendly feature extraction (the encoder stage).

    Only integer-friendly operations: banded band-power via rFFT magnitudes,
    zero-crossing rate, peak envelope and variance.  Maps a raw waveform to a
    small real-valued feature vector that the latency encoder then converts to
    spike times — this is the block that would sit in the sensor front-end ASIC.
    """
    n = len(x)
    spec = np.abs(np.fft.rfft(x)) ** 2
    bands = np.array_split(spec, n_bands)
    bp = np.array([b.sum() for b in bands])
    bp = np.log1p(bp)
    bp /= (bp.max() + 1e-9)
    zcr = np.mean(np.abs(np.diff(np.sign(x)))) / 2.0
    env = np.max(np.abs(x)) / (np.std(x) * 4.0 + 1e-9)
    feats = np.concatenate([bp, [zcr, env, np.var(x) / (np.var(x) + 1.0)]])
    return np.clip(feats, 0.0, 1.0)
