"""Diagnose the SNN readout: are hidden/output neurons firing? Is accuracy
improving over training?  Why doesn't trained beat random (frozen)?"""
import numpy as np
from simulator.domains import DomainStream, featurise
from simulator.continuum_net import NeuroContinuum

def run(domain="factory", n=400, learn=True, seed=0, **kw):
    stream = DomainStream(domain=domain, seed=seed, drift=1.0)
    net = NeuroContinuum(seed=seed, **kw)
    accs, hid_rate, out_rate = [], [], []
    for i in range(n):
        w = stream.next()
        f = featurise(w.x)
        info = net.predict(f, learn=learn, label=w.label)
        accs.append(info["decision"] == w.label)
        # measure firing on a fresh pass (learn off) to see activity
        net._present(f, learn=False)
        hid_rate.append(net.hid.spikes.sum())
        out_rate.append(net.out.spikes.sum())
    seg = lambda a, s: float(np.mean(a[s*100:(s+1)*100]))
    return dict(
        acc_by_100=[round(seg(accs, s), 3) for s in range(n // 100)],
        hidden_spikes_per_window=round(float(np.mean(hid_rate)), 1),
        output_spikes_per_window=round(float(np.mean(out_rate)), 1),
    )

for learn in [True, False]:
    tag = "LEARN " if learn else "FROZEN"
    r = run(learn=learn)
    print(f"[{tag}] acc by 100-window={r['acc_by_100']}  "
          f"hidden/win={r['hidden_spikes_per_window']}  out/win={r['output_spikes_per_window']}")

print(f"\n{'='*60}")
print("Interpretation guide:")
print("  hidden/win ~= 0   -> hidden layer DEAD (drive too weak)")
print("  acc not rising    -> readout not learning")
print("  learn ~= frozen   -> R-STDP adds nothing over random readout")
