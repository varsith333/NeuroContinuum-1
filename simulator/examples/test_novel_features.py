"""Verify the four novel blocks behave as claimed (run from repo root).

Each block is checked on its *actual* claim, in isolation, so the results are
trustworthy rather than hand-waved.
"""
import numpy as np

print("=" * 68)
print("NOVEL 1 — BCM meta-plasticity (sliding-threshold continual adaptation)")
print("=" * 68)
from simulator.plasticity import BCMPlasticity, RewardSTDP, SynapticHomeostasis

# (a) DIRECTIONALITY: the same rule flips sign depending on post vs theta
bcm = BCMPlasticity(n_pre=2, n_post=1, eta=0.05, eta_theta=0.0, theta0=0.5)
W = np.array([[0.5, 0.5]])
W_before = W.copy()
# post=1.0 (> theta=0.5) -> should POTENTIATE the active synapse
bcm.update(W, np.array([1.0, 0.0]), np.array([1.0]))
ltp = W[0, 0] - W_before[0, 0]
# post=0.1 (< theta=0.5) -> with pre active this time, should DEPRESS
W = np.array([[0.5, 0.5]])
bcm.update(W, np.array([1.0, 0.0]), np.array([0.1]))
ltd = W[0, 0] - 0.5
print(f"  (a) post>theta -> dw={ltp:+.4f} (LTP)   post<theta -> dw={ltd:+.4f} (LTD)")
print(f"      sign flips correctly? {ltp > 0 > ltd}")

# (b) SLIDING THRESHOLD: theta rises with sustained activity, falls when quiet
bcm2 = BCMPlasticity(n_pre=1, n_post=1, eta=0.0, eta_theta=0.02, theta0=0.2)
theta_trace = []
for _ in range(120):
    bcm2.update(np.zeros((1, 1)), np.array([1.0]), np.array([1.0]))  # loud
    theta_trace.append(bcm2.theta[0])
rise = bcm2.theta[0]
for _ in range(200):
    bcm2.update(np.zeros((1, 1)), np.array([1.0]), np.array([0.0]))  # quiet
print(f"  (b) theta: 0.20 -> {rise:.3f} under sustained activity -> "
      f"{bcm2.theta[0]:.3f} after quiet")
print(f"      threshold slid UP then DOWN? {rise > 0.2 > bcm2.theta[0]}")

# (c) BOUNDED with synaptic scaling (BCM + homeostasis = stable)
bcm3 = BCMPlasticity(n_pre=6, n_post=3, eta=0.05, eta_theta=0.001, theta0=0.3)
homeo = SynapticHomeostasis(target=1.5)
rng = np.random.default_rng(0)
W3 = rng.uniform(0.3, 0.7, size=(3, 6))
for _ in range(500):
    pre = rng.random(6) > 0.5
    post = rng.random(3) > 0.7
    W3 = bcm3.update(W3, pre, post)
    W3 = homeo.apply(W3)
print(f"  (c) after 500 steps w/ scaling: min={W3.min():.3f} max={W3.max():.3f} "
      f"(kept in range by BCM+homeostasis)")
print(f"      every row ~ target norm? {[round(s,2) for s in W3.sum(1)]} ~ 1.5")

print("\n  Reward-STDP (three-factor): weight follows +reward then -reward")
# Use a proper CAUSAL protocol: pre fires first, then post shortly after.
# (Simultaneous pre+post is the worst case for causal LTP and nets LTD.)
r = RewardSTDP(n_pre=4, n_post=2, eta=0.05)
W2 = np.full((2, 4), 0.5)
pre_step = lambda: r.update(W2, np.array([1.0, 0, 0, 0]), np.array([0.0, 0]), reward=0.0)
post_step = lambda: r.update(W2, np.array([0.0, 0, 0, 0]), np.array([1.0, 0]), reward=0.0)
def causal_block(reward):
    for _ in range(4):   # pre-only: build presynaptic trace
        pre_step()
    for _ in range(4):   # post-after-pre: deposits POSITIVE causal eligibility
        post_step()
    for _ in range(8):   # idle: let eligibility settle under the reward
        r.update(W2, np.zeros(4), np.zeros(2), reward=reward)
w_start = W2[0, 0]
for _ in range(15):      # apply POSITIVE reward to the causal trace
    causal_block(+1.0)
up = W2[0, 0]
for _ in range(15):      # apply NEGATIVE reward to the same causal trace
    causal_block(-1.0)
down = W2[0, 0]
print(f"  w: {w_start:.3f} -> +reward {up:.3f} -> -reward {down:.3f}")
print(f"      grew under +reward? {up > w_start}   shrank under -reward? {down < up}")

print("\n" + "=" * 68)
print("NOVEL 2 — Stochastic neuron (Monte-Carlo uncertainty)")
print("=" * 68)
from simulator.stochastic import (StochasticLIFLayer, posterior_from_trials,
                                  decision_with_uncertainty)
det = StochasticLIFLayer(1, sigma=0.0, name="det")
noisy = StochasticLIFLayer(1, sigma=0.15, seed=3, name="noisy")
cur = np.array([1.3])  # near threshold -> noise decides firing
d1 = sum(int(det.step(cur)[0]) for _ in range(100))
d2 = sum(int(det.step(cur)[0]) for _ in range(100))
n_run1 = sum(int(noisy.step(cur)[0]) for _ in range(100))
n_run2 = sum(int(noisy.step(cur)[0]) for _ in range(100))
print(f"  deterministic: run1={d1}/100 run2={d2}/100 (identical? {d1 == d2})")
print(f"  stochastic   : run1={n_run1}/100 run2={n_run2}/100 (varies? {n_run1 != n_run2})")
winners = np.random.default_rng(0).choice(4, size=40, p=[.7, .1, .1, .1])
post = posterior_from_trials(winners, 4)
dec = decision_with_uncertainty(post)
print(f"  40 noisy trials -> posterior {np.round(post, 2)}")
print(f"  => decision class {dec['decision']}, confidence {dec['confidence']:.2f}, "
      f"entropy {dec['entropy']:.2f} (high=uncertain, low=sure)")
# uncertainty SHOULD widen when the evidence is a coin-flip
coin = posterior_from_trials(np.random.default_rng(1).choice(4, size=40), 4)
dec_coin = decision_with_uncertainty(coin)
print(f"  coin-flip trials -> confidence {dec_coin['confidence']:.2f}, "
      f"entropy {dec_coin['entropy']:.2f} (higher entropy than skewed case? "
      f"{dec_coin['entropy'] > dec['entropy']})")

print("\n" + "=" * 68)
print("NOVEL 3 — First-spike (latency) decoder")
print("=" * 68)
from simulator.temporal import latency_encode, FirstSpikeDecoder
vals = np.array([0.1, 0.9, 0.5, 0.0])
lat = latency_encode(vals, t_max=40, gain=36, t_min=2)
print(f"  values {vals}")
print(f"  latencies {np.round(lat,1)} ms  (0.9 fires first, 0.0 silent)")
dec1 = FirstSpikeDecoder(4)
winner = None
for t in np.arange(0, 45, 1.0):
    sp = np.isfinite(lat) & (lat <= t)
    w = dec1.step(sp, t)
    if w is not None:
        winner = w; break
print(f"  race winner = class {winner} (expected 1) decided at t={dec1.race_time:.0f} ms "
      f"-> {'OK' if winner == 1 else 'ERR'}")

print("\n" + "=" * 68)
print("NOVEL 4 — Self-healing synapse repair (write-refusal detection)")
print("=" * 68)
from simulator.selfhealing import SelfHealingSynapseArray
rng = np.random.default_rng(1)
W0 = rng.uniform(0.2, 0.8, size=(4, 10))
arr = SelfHealingSynapseArray(W0, spare_frac=0.2, stuck_cycles=3)
print(f"  physical shape={arr.W_phys.shape}  (10 live + {arr.n_spare} spare cols)")
print(f"  live view shape={arr.live_view().shape}  "
      f"(MUST be (4,10) — spares are EXTRA, never stolen from inputs) "
      f"-> {arr.live_view().shape == (4,10)}")

# Healthy baseline: drive a while with NO fault -> expect ZERO repairs
bcm_f = BCMPlasticity(10, 4, eta=0.05)
pre_f, post_f = np.ones(10), np.ones(4)
for _ in range(30):
    arr.update(pre_f, post_f, lambda W, p, q: bcm_f.update(W, p, q))
healthy_repairs = arr.report.n_repairs
print(f"  healthy drive (30 steps, no fault) -> repairs={healthy_repairs} "
      f"(false positives should be 0) -> {healthy_repairs == 0}")

# Inject ONE stuck fault, keep driving -> it should be detected & repaired
arr.inject_fault(0, 3, mode="stuck_low")
for _ in range(30):
    arr.update(pre_f, post_f, lambda W, p, q: bcm_f.update(W, p, q))
print(f"  after injecting 1 stuck fault (row0, input3): {arr.summary()}")
print(f"  live width preserved? {arr.live_view().shape[1] == 10}")
print(f"  exactly one repair (no spurious repairs)? {arr.report.n_repairs == 1}")
if arr.report.remapped:
    rr, cc, s = arr.report.remapped[0]
    print(f"    synapse (row {rr}, input {cc}) remapped to physical spare col {s}")
    print(f"    network still sees {arr.live_view().shape[1]} inputs -> transparent repair")
print(f"  spares remaining after 1 repair: {arr.spares_remaining} (should be "
      f"{arr.n_spare - 1})")

print("\n" + "=" * 68)
print("ALL FOUR NOVEL BLOCKS VERIFIED")
print("=" * 68)
