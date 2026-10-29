"""Sweep R-STDP strength with WTA + short output refractory in place."""
import numpy as np
from simulator.domains import DomainStream, featurise
from simulator.continuum_net import NeuroContinuum

def run(domain, learn, seed=0, n=400, **kw):
    stream = DomainStream(domain=domain, seed=seed, drift=1.0)
    net = NeuroContinuum(seed=seed, **kw)
    accs = []
    for _ in range(n):
        w = stream.next()
        info = net.predict(featurise(w.x), learn=learn, label=w.label)
        accs.append(info["decision"] == w.label)
    # accuracy on the LAST half (post-learning)
    return float(np.mean(accs)), float(np.mean(accs[n//2:]))

print("rstdp_eta sweep (factory, WTA, t_ref=1):")
print(f"{'eta':>6} {'accAll':>8} {'acc2ndHalf':>11} {'frozen2nd':>10} {'gap':>7}")
print("-" * 46)
for eta in [0.05, 0.5, 2.0, 8.0, 20.0]:
    aL, aL2 = run("factory", True, rstdp_eta=eta)
    _, aF2 = run("factory", False, rstdp_eta=eta)
    print(f"{eta:6.2f} {aL:8.3f} {aL2:11.3f} {aF2:10.3f} {aL2-aF2:+7.3f}")

print("\nBest eta on bridge + dam:")
best_eta, bestsc = None, -1
for eta in [2.0, 8.0, 20.0]:
    _, b = run("bridge", True, rstdp_eta=eta)
    _, d = run("dam", True, rstdp_eta=eta)
    score = b + d
    print(f"  eta={eta}: bridge2nd={b:.3f} dam2nd={d:.3f} sum={score:.3f}")
    if score > bestsc:
        bestsc, best_eta = score, eta
print(f"\nRECOMMEND rstdp_eta={best_eta}")
