"""Noise sweep with the FIXED reward-gated readout: find the accuracy/calibration
tradeoff now that the readout actually learns.  Lower noise -> stronger hidden
representation -> higher accuracy -> (hopefully) naturally calibrated confidence."""
import numpy as np
from simulator.domains import DomainStream, featurise
from simulator.continuum_net import NeuroContinuum, DriftTracker

def run(domain, sigma, seed=0, n=400, **kw):
    stream = DomainStream(domain=domain, seed=seed, drift=1.0)
    net = NeuroContinuum(seed=seed, noise_sigma=sigma, **kw)
    trk = DriftTracker(4, window=60)
    for _ in range(n):
        w = stream.next()
        info = net.predict(featurise(w.x), learn=True, label=w.label)
        trk.add(w.label, info)
    ece = trk.ece()
    return trk.accuracy, (0.0 if np.isnan(ece) else ece), trk.mean_confidence

print(f"{'sigma':>6} | {'acc':>6} {'ECE':>6} {'conf':>6}   (bridge / dam / factory)")
print("-" * 60)
for sig in [0.0, 0.02, 0.05, 0.10, 0.15, 0.25]:
    row = []
    for dom in ["bridge", "dam", "factory"]:
        a, e, c = run(dom, sig)
        row.append(f"{a:.2f}/{e:.2f}/{c:.2f}")
    print(f"{sig:6.2f} | {'  '.join(row)}")
