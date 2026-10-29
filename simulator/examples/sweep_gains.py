"""Sweep drive gains + R-STDP rate: find config where (a) layers fire
healthily and (b) trained readout clearly beats the frozen random readout."""
import numpy as np
from simulator.domains import DomainStream, featurise
from simulator.continuum_net import NeuroContinuum

def run(domain, learn, seed=0, n=300, **kw):
    stream = DomainStream(domain=domain, seed=seed, drift=1.0)
    net = NeuroContinuum(seed=seed, **kw)
    accs, hr, orate = [], [], []
    for _ in range(n):
        w = stream.next()
        info = net.predict(featurise(w.x), learn=learn, label=w.label)
        accs.append(info["decision"] == w.label)
        net._present(featurise(w.x), learn=False)
        hr.append(net.hid.spikes.sum()); orate.append(net.out.spikes.sum())
    return float(np.mean(accs)), float(np.mean(hr)), float(np.mean(orate))

print(f"{'igain':>5} {'ogain':>5} {'eta':>5} | {'accL':>6} {'accF':>6} {'gap':>6} | "
      f"{'hid':>5} {'out':>5}")
print("-" * 60)
best = None
for ig in [6, 15, 30]:
    for og in [1, 4, 10]:
        for eta in [0.05, 0.5]:
            kw = dict(input_gain=ig, output_gain=og, rstdp_eta=eta)
            aL, hr, orr = run("factory", True, **kw)
            aF, _, _ = run("factory", False, **kw)
            gap = aL - aF
            print(f"{ig:5d} {og:5d} {eta:5.2f} | {aL:6.3f} {aF:6.3f} {gap:+6.3f} | "
                  f"{hr:5.1f} {orr:5.1f}")
            if best is None or aL > best[0]:
                best = (aL, aF, gap, hr, orr, kw)

print("\nBest by LEARNED accuracy:")
aL, aF, gap, hr, orr, kw = best
print(f"  accLearned={aL:.3f} accFrozen={aF:.3f} gap={gap:+.3f} "
      f"hidden/win={hr:.1f} out/win={orr:.1f}")
print(f"  config={kw}")
