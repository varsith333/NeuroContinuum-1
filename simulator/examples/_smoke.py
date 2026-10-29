"""Smoke test for the NeuroContinuum simulator core (run from repo root)."""
from simulator.stream import HazardStream, featurise
from simulator.continuum_net import NeuroContinuum, DriftTracker

stream = HazardStream(seed=0, drift="mix")
net = NeuroContinuum(seed=0)
trk = DriftTracker(4, window=40)
for i in range(120):
    w = stream.next()
    f = featurise(w.x)
    info = net.predict(f, learn=True, label=w.label)
    trk.add(w.label, info)
    if i in (0, 39, 79, 119):
        print(f"i={i:3d} label={w.label} dec={info['decision']} "
              f"conf={info['confidence']:.2f} H={info['entropy']:.2f} "
              f"acc={trk.accuracy:.3f} ece={trk.ece():.3f}")
print("OK")
