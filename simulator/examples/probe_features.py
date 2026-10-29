"""Feature-ceiling diagnostic: how separable ARE the extracted features?

If a plain linear classifier scores high, the SNN readout is the bottleneck
(tune the network).  If it also scores low, the features are the bottleneck
(improve featurise()).  No sklearn needed — tiny numpy softmax regression.
"""
import numpy as np
from simulator.domains import DomainStream, featurise

def linear_probe(domain, n=600, epochs=300, lr=0.5, seed=0):
    s = DomainStream(domain=domain, seed=seed, drift=1.0)
    X, y = [], []
    for _ in range(n):
        w = s.next()
        X.append(featurise(w.x)); y.append(w.label)
    X = np.array(X); y = np.array(y)
    d, k = X.shape[1], 4
    # standardize
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    # train/test split (temporal: first 70% train, last 30% test)
    split = int(0.7 * n)
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
    W = np.zeros((d, k)); b = np.zeros(k)
    for _ in range(epochs):
        logits = Xtr @ W + b
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        onehot = np.zeros_like(p); onehot[np.arange(len(ytr)), ytr] = 1
        gW = Xtr.T @ (p - onehot) / len(ytr)
        gb = (p - onehot).mean(0)
        W -= lr * gW; b -= lr * gb
    # train acc
    ptr = np.argmax(Xtr @ W + b, 1); acc_tr = (ptr == ytr).mean()
    pte = np.argmax(Xte @ W + b, 1); acc_te = (pte == yte).mean()
    return acc_tr, acc_te

print("Linear probe on featurise() features (chance = 0.25):")
print(f"{'domain':10s} {'train_acc':>10s} {'test_acc':>10s}")
for d in ["bridge", "dam", "factory"]:
    tr, te = linear_probe(d)
    print(f"{d:10s} {tr:10.3f} {te:10.3f}")

print("\nInterpretation:")
print("  test_acc high  -> features are fine; SNN readout is the bottleneck")
print("  test_acc ~0.25 -> features are the bottleneck; improve featurise()")
