"""Find the ceiling: is the HIDDEN representation linearly separable, and does
the readout approach that ceiling over training?"""
import numpy as np
from simulator.domains import DomainStream, featurise
from simulator.continuum_net import NeuroContinuum

def collect_hidden(domain="factory", n=500, seed=0, learn=True):
    stream = DomainStream(domain=domain, seed=seed, drift=1.0)
    net = NeuroContinuum(seed=seed)
    H, y = [], []
    for _ in range(n):
        w = stream.next()
        f = featurise(w.x)
        net.predict(f, learn=learn, label=w.label)  # advance state/learning
        # capture hidden population code (one clean pass)
        net._present(f, learn=False)
        # integrate hidden activity across window via drive replay
        H.append(net._hid_code() if hasattr(net, "_hid_code") else net.hid.v.copy())
        y.append(w.label)
    return np.array(H), np.array(y)

def softmax_probe(X, y, epochs=400, lr=0.5, split=0.7):
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    d, k = X.shape[1], 4
    s = int(split * len(y)); Xtr, ytr, Xte, yte = X[:s], y[:s], X[s:], y[s:]
    W = np.zeros((d, k)); b = np.zeros(k)
    for _ in range(epochs):
        lg = Xtr @ W + b
        p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        oh = np.zeros_like(p); oh[np.arange(len(ytr)), ytr] = 1
        W -= lr * (Xtr.T @ (p - oh) / len(ytr)); b -= lr * (p - oh).mean(0)
    return (np.argmax(Xtr @ W + b, 1) == ytr).mean(), (np.argmax(Xte @ W + b, 1) == yte).mean()

# Use hidden membrane as the code proxy (rich, per-neuron state)
def collect(domain, learn, n=500, seed=0):
    stream = DomainStream(domain=domain, seed=seed, drift=1.0)
    net = NeuroContinuum(seed=seed)
    X, y, accs = [], [], []
    for _ in range(n):
        w = stream.next(); f = featurise(w.x)
        info = net.predict(f, learn=learn, label=w.label)
        accs.append(info["decision"] == w.label)
        X.append(net.hid.v.copy()); y.append(w.label)
    return np.array(X), np.array(y), np.array(accs)

for dom in ["factory", "bridge"]:
    X, y, accs = collect(dom, learn=True, n=500)
    tr, te = softmax_probe(X, y)
    seg = lambda a, i: float(np.mean(a[i*125:(i+1)*125]))
    print(f"{dom:9s} | readout acc by 125={[round(seg(accs,i),3) for i in range(4)]}")
    print(f"{'':9s} | HIDDEN ceiling (linear probe): train={tr:.3f} test={te:.3f}")
print("\nIf HIDDEN ceiling >> readout acc -> readout learning still bottleneck")
print("If HIDDEN ceiling ~0.5 -> hidden representation itself is the limit")
