#!/usr/bin/env python3
"""POST-HOC DIAGNOSTIC for TE4-SEQ-v2's failed simulation screen (te4-seq-v2-2026-09-28.md §3.2). Designed after the
screen's result was seen, so it diagnoses and does not re-grade. For each shift rate it compares the sequential rule
with the fixed-N gate it would replace, on the same simulated gates as the registered screen (same seed, same draws).

    python3 -B te4v2_sim_diagnostic.py > te4v2-sim-diagnostic-2026-09-28.txt
"""
import math
import random

import te4v2_replay as v2

G = 4000


def run(scale, label):
    rng = random.Random(v2.SEED)
    worst_seq = worst_fix = 0.0
    excess = (-1.0, None)
    ag = n = used = cap = 0
    for lab, s0, fl, n0 in v2.SIM_NOISE:
        nu0 = v2.prior_df(n0)
        for Q in v2.SIM_Q:
            half = v2.v1.t_inv(0.975, 2 * Q - 1) * s0 / math.sqrt(2 * Q) + fl
            for S, K in v2.SIM_BARS:
                for r in v2.SIM_RATIOS:
                    clear = min(abs(math.log(r) - math.log(S)), abs(math.log(r) - math.log(K))) >= 2 * half
                    ps = pf = a = 0
                    for _ in range(G):
                        bias = rng.uniform(-fl, fl)
                        lvl = {"new": 0.0, "old": 0.0}
                        ds = []
                        for _b in range(2 * Q):
                            for arm in lvl:
                                u = rng.random()
                                if u < 0.01 * scale:
                                    lvl[arm] += rng.choice((-1, 1)) * math.log(1.31)
                                elif u < 0.06 * scale:
                                    lvl[arm] += rng.choice((-1, 1)) * math.log(1.05)
                            ds.append(math.log(r) + bias + rng.gauss(0.0, s0) + lvl["new"] - lvl["old"])
                        vs, q = v2.sequential(ds, Q, s0, nu0, fl, S, K)
                        vf = v2.fixed_n(ds, fl, S, K)
                        ps += vs == "PASS"
                        pf += vf == "PASS"
                        a += vs == vf
                        if clear:
                            used += q
                            cap += Q
                    ag += a
                    n += G
                    if r <= S + 1e-12:
                        worst_seq, worst_fix = max(worst_seq, ps / G), max(worst_fix, pf / G)
                        if (ps - pf) / G > excess[0]:
                            excess = ((ps - pf) / G, f"{lab}, Q {Q}, bars ({S}, {K}), r {r}: seq {ps / G:.3f} fixed-N {pf / G:.3f}")
    print(f"== {label}")
    print(f"   worst P(PASS | r <= S): sequential {worst_seq:.3f}, fixed-N {worst_fix:.3f}")
    print(f"   largest excess of sequential over fixed-N: {excess[0]:+.3f} ({excess[1]})")
    print(f"   agreement with fixed-N: {ag / n:.4f}; blocks saved where clear of the bar: {1 - used / cap:.3f}")


if __name__ == "__main__":
    run(1.0, "registered shift model (a 5% shift per arm per block at p 0.05, 31% at p 0.01)")
    run(0.0, "no shifts: the registry's Gaussian noise and the build bias only")
    run(1 / 7, "shifts at 1/7 the rate (about the registry's observed tail: 3 of 426 nobara peer pairs >= 3.5%)")
