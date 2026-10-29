"""Try to lift the representation ceiling: more hidden units + longer window.
Then lock in the best accuracy/calibration config."""
import numpy as np
from simulator.domains import DomainStream, featurise
from simulator.continuum_net import NeuroContinuum, DriftTracker

def run(domain, seed=0, n=400, **kw):
    stream = DomainStream(domain=domain, seed=seed, drift=1.0)
    net = NeuroContinuum(seed=seed, **kw)
    trk = DriftTracker(4, window=60)
    for _ in range(n):
        w = stream.next()
        info = net.predict(featurise(w.x), learn=True, label=w.label)
        trk.add(w.label, info)
    ece = trk.ece()
    return trk.accuracy, (0.0 if np.isnan(ece) else ece)

configs = [
    ("baseline  hid24 win25", dict(noise_sigma=0.05)),
    ("hid48 win25",          dict(n_hidden=48, noise_sigma=0.05)),
    ("hid48 win40",          dict(n_hidden=48, window_ms=40, noise_sigma=0.05)),
    ("hid64 win40",          dict(n_hidden=64, window_ms=40, noise_sigma=0.05)),
    ("hid64 win40 s0.10",    dict(n_hidden=64, window_ms=40, noise_sigma=0.10)),
    ("hid64 win40 s0.15",    dict(n_hidden=64, window_ms=40, noise_sigma=0.15)),
]
print(f"{'config':26s} | {'bridge':>14} {'dam':>14} {'factory':>14} | {'avgAcc':>7} {'avgECE':>7}")
print("-" * 95)
for name, kw in configs:
    accs, eces = [], []
    for dom in ["bridge", "dam", "factory"]:
        a, e = run(dom, **kw)
        accs.append(a); eces.append(e)
    print(f"{name:26s} | {accs[0]:.2f}/{eces[0]:.2f}     {accs[1]:.2f}/{eces[1]:.2f}     "
          f"{accs[2]:.2f}/{eces[2]:.2f}     | {np.mean(accs):7.3f} {np.mean(eces):7.3f}")
print("\n( format: accuracy/ECE ; want high acc, low ECE )")
