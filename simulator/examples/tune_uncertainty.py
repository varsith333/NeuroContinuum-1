"""Tune the uncertainty estimator: find a posterior that is actually CALIBRATED.

Sweeps posterior noise (sigma) and Monte-Carlo trial count, reporting
accuracy + ECE (expected calibration error, lower = better) + mean confidence.
Kept as a script so the final configuration in continuum_net.py is justified by
data, not guessed.
"""
import numpy as np
from simulator.domains import DomainStream, featurise
from simulator.continuum_net import NeuroContinuum, DriftTracker


def eval_config(make_net, n=250, seed=0, domain="bridge"):
    net = make_net(seed)
    stream = DomainStream(domain=domain, seed=seed, drift=1.0)
    trk = DriftTracker(4, window=60)
    for _ in range(n):
        w = stream.next()
        info = net.predict(featurise(w.x), learn=True, label=w.label)
        trk.add(w.label, info)
    ece = trk.ece()
    return trk.accuracy, (0.0 if np.isnan(ece) else ece), trk.mean_confidence


rows = []
print("Noise sweep  (mc_trials=11, domain=bridge):")
for sig in [0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80]:
    acc, ece, conf = eval_config(
        lambda s, sig=sig: NeuroContinuum(seed=s, noise_sigma=sig, mc_trials=11))
    rows.append((sig, acc, ece, conf))
    print(f"   sigma={sig:.2f}: acc={acc:.3f}  ECE={ece:.3f}  conf={conf:.3f}")

best = min(rows, key=lambda r: r[2])
print(f"   BEST calibration: sigma={best[0]:.2f}  (ECE={best[2]:.3f}, acc={best[1]:.3f})")

print("\nTrial-count sweep (at chosen sigma, domain=bridge):")
rows2 = []
for mc in [3, 5, 7, 11, 15, 21]:
    acc, ece, conf = eval_config(
        lambda s, mc=mc: NeuroContinuum(seed=s, noise_sigma=best[0], mc_trials=mc))
    rows2.append((mc, acc, ece, conf))
    print(f"   mc={mc:2d}: acc={acc:.3f}  ECE={ece:.3f}  conf={conf:.3f}")

print("\nAll three domains at chosen config:")
for d in ["bridge", "dam", "factory"]:
    acc, ece, conf = eval_config(
        lambda s: NeuroContinuum(seed=s, noise_sigma=best[0], mc_trials=7), domain=d)
    print(f"   {d:8s}: acc={acc:.3f}  ECE={ece:.3f}  conf={conf:.3f}")

print(f"\nRECOMMEND: noise_sigma={best[0]}, mc_trials=7")
