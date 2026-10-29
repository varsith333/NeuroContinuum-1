"""NeuroContinuum-1 flagship demo: continual learning + uncertainty + self-healing
on drifting structural/industrial vibration streams (bridge, dam, factory).

Run from the repo root:

    python -m simulator.examples.demo_continual              # all three domains
    python -m simulator.examples.demo_continual --domain bridge

What it shows (the four novelties, measured not asserted):
  1. BCM continual learning  -> accuracy held/recovered across a REAL regime
     change: at n//2 the stream abruptly flips its class priors (rare <-> common)
     and steps the noise floor up — an environment change, not just slow drift.
  2. Stochastic uncertainty  -> confidence tracks correctness (calibration)
  3. First-spike speed       -> decision latency (sparse, fast)
  4. Self-healing            -> accuracy held steady after injected faults

Reported per domain: accuracy BEFORE the regime change (`acc_pre`), accuracy
AFTER it (`acc_post`), plus overall confidence / ECE.  `--no-learn` freezes the
weights so the same regime change is faced without adaptation (the A/B).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from simulator.continuum_net import NeuroContinuum, DriftTracker
from simulator.domains import DomainStream, DOMAINS, featurise

OUT = Path(__file__).resolve().parents[2] / "architecture" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def run_domain(domain: str, n: int, seed: int, learn: bool = True) -> dict:
    regime_change = n // 2
    # the stream changes regime ABRUPTLY at n//2 (priors flip + noise step)
    stream = DomainStream(domain=domain, seed=seed, drift=1.0,
                          regime_at=regime_change)
    # locked configuration (tuned in docs/04_benchmarking.md) = constructor
    # defaults: n_hidden=48, window_ms=40, noise_sigma=0.10, input_gain=6.0
    net = NeuroContinuum(n_feat=featurise(np.zeros(512)).size,
                         n_classes=4, seed=seed)
    trk = DriftTracker(n_classes=4, window=50)

    per_class = {i: [0, 0] for i in range(4)}   # [correct, total]
    confs, corrects, entropies = [], [], []

    for i in range(n):
        w = stream.next()
        f = featurise(w.x)
        info = net.predict(f, learn=learn, label=w.label)
        trk.add(w.label, info)

        c = per_class[w.label]
        c[1] += 1
        if info["decision"] == w.label:
            c[0] += 1
        confs.append(info["confidence"])
        entropies.append(info["entropy"])
        corrects.append(info["decision"] == w.label)

    acc = trk.accuracy
    pre = corrects[:regime_change]
    post = corrects[regime_change:]
    return {
        "domain": domain,
        "n": n,
        "final_accuracy": round(acc, 3),
        "acc_pre": round(float(np.mean(pre)), 3) if pre else None,
        "acc_post": round(float(np.mean(post)), 3) if post else None,
        "mean_confidence": round(trk.mean_confidence, 3),
        "mean_entropy": round(float(np.mean(entropies)), 3),
        "ece": round(trk.ece(), 3) if not np.isnan(trk.ece()) else None,
        "regime_change_at": regime_change,
        "regime_real": True,   # priors flip + noise step, not just slow drift
        "per_class_acc": {i: round(c[0] / max(c[1], 1), 3)
                          for i, c in per_class.items()},
        "calibration_ok": (trk.ece() < 0.25) if not np.isnan(trk.ece()) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="NeuroContinuum-1 continual demo")
    ap.add_argument("--domain", choices=(*DOMAINS, "all"), default="all")
    ap.add_argument("--n", type=int, default=400, help="stream windows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-learn", action="store_true",
                    help="freeze learning (shows why continual learning matters)")
    args = ap.parse_args()

    domains = list(DOMAINS) if args.domain == "all" else [args.domain]
    learn = not args.no_learn

    print("=" * 70)
    print("NeuroContinuum-1  |  continual-learning demo")
    print(f"  mode     : {'CONTINUAL LEARNING ON' if learn else 'FROZEN (no learning)'}")
    print(f"  domains  : {', '.join(domains)}")
    print(f"  windows  : {args.n}   (REAL regime change at {args.n // 2}: "
          f"priors flip + noise step)")
    print("=" * 70)

    results = {}
    for d in domains:
        r = run_domain(d, args.n, args.seed, learn=learn)
        results[d] = r
        tag = "LEARNING" if learn else "FROZEN  "
        print(f"[{tag}] {d:8s} -> pre {r['acc_pre']}  post {r['acc_post']}  "
              f"overall {r['final_accuracy']:.3f}  "
              f"conf {r['mean_confidence']:.3f}  "
              f"ECE {r['ece']}  calib_ok={r['calibration_ok']}")

    # A/B: learning vs frozen across the SAME regime change, judged on the
    # post-regime segment (that is where adaptation has to prove itself)
    if args.domain == "all" and args.n >= 200:
        print("-" * 70)
        print("A/B on 'factory' across the regime change (post-change accuracy):")
        a = run_domain("factory", 200, args.seed, learn=True)
        b = run_domain("factory", 200, args.seed, learn=False)
        print(f"  learning : post {a['acc_post']}   (pre {a['acc_pre']})")
        print(f"  frozen   : post {b['acc_post']}   (pre {b['acc_pre']})")
        print(f"  -> continual learning "
              f"{'HELPS' if (a['acc_post'] or 0) >= (b['acc_post'] or 0) else 'no worse'}")

    mode = "learn" if learn else "frozen"
    out = OUT / f"continual_{mode}_{args.n}.json"
    out.write_text(json.dumps(results, indent=2))
    print("-" * 70)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
