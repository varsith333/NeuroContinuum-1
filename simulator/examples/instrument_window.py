"""Instrument ONE window end-to-end to find exactly where the signal dies."""
import numpy as np
from simulator.domains import DomainStream, featurise
from simulator.temporal import latency_encode
from simulator.continuum_net import NeuroContinuum

np.set_printoptions(precision=3, suppress=True, linewidth=120)
stream = DomainStream(domain="factory", seed=0, drift=1.0)
net = NeuroContinuum(seed=0, input_gain=30, output_gain=10)
w = stream.next()
f = featurise(w.x)
print("label:", w.class_names if hasattr(w, 'class_names') else w.label, "=", w.label)
print("features:", f)

lat = latency_encode(f, t_max=net.window_ms - 3, gain=20.0, t_min=1.0)
valid = np.isfinite(lat)
print(f"\nlatencies: {lat}")
print(f"inputs that spike: {valid.sum()} / {len(f)}")

# manual pass to observe internals
spike_t = np.round(np.where(valid, lat, -1)).astype(int)
net.inp.reset_state(); net.hid.reset_state(); net.out.reset_state()
syn_in = np.zeros(net.n_hidden); syn_out = np.zeros(net.n_classes)
hid_fire = []; out_fire = []
syn_in_max = []
for t in range(net.window_ms):
    s_in = valid & (spike_t == t)
    W_live = net.heal_in.live_view()
    syn_in = syn_in * net.syn_decay + net.input_gain * (W_live @ s_in.astype(float))
    syn_in_max.append(syn_in.max())
    s_hid = net.hid.step(syn_in)
    syn_out = syn_out * net.syn_decay + net.output_gain * (net.W_out @ s_hid.astype(float))
    s_out = net.out.step(syn_out)
    hid_fire.append(s_hid.sum()); out_fire.append(s_out.sum())

print(f"\nsyn_in max over time: {syn_in_max}")
print(f"hidden spikes/t: {hid_fire}   total={sum(hid_fire)}")
print(f"hidden v final: max={net.hid.v.max():.3f} (thresh={net.hid.v_thresh})")
print(f"output spikes/t: {out_fire}   total={sum(out_fire)}")
print(f"output v final: {np.round(net.out.v,3)}")

# does W_out distinguish?  drive each output class directly from a stored pattern
print("\n--- direct output-layer drive test ---")
print("W_out (rows=class, cols=hidden):")
print(net.W_out)
