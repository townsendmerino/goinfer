#!/usr/bin/env python3
"""TE4-SEQ-v2: the rule, its simulation screen, and its replay over ABBA-blocked bench_peer.py output.

The rule and both validation parts are pre-registered in te4-seq-v2-2026-09-28.md (5551e9c3), written before this
script existed. Stdlib only; reads records and the noise registry, runs and times nothing.

    python3 te4v2_replay.py selfcheck
    python3 te4v2_replay.py sim [--gates 4000]          # §2.1, the screen
    python3 te4v2_replay.py replay <abba results.json> ...   # §2.2, the held-out night corpus

v1's boundary integrator (te4_replay.py, checked there against gsDesign) and power.py's registry lookup are reused,
not re-implemented.
"""
import importlib.util
import itertools
import json
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v1 = _load("te4_replay", os.path.join(HERE, "te4_replay.py"))
power = _load("power", os.path.join(REPO, "scripts", "power.py"))

ALPHA = 0.05
Q_MIN = 2          # no stop before the second quad (§1.2)
NU0_MAX = 6        # prior df cap (§1.3)
SEED = 20260928
PERM_MAX = 200


# ---------------------------------------------------------------------------------------------------------------
# The rule (§1)
# ---------------------------------------------------------------------------------------------------------------

def prior_df(entry_n):
    return max(1, min(int(entry_n) - 1, NU0_MAX))


def boundaries(Q):
    """z boundaries for looks at q = Q_MIN..Q, information fraction t = q / Q (v1's spending and integrator)."""
    ts = [q / Q for q in range(Q_MIN, Q + 1)]
    return v1.ld_boundaries(ts, ALPHA)


def pooled_se(ds, sigma0, nu0):
    n = len(ds)
    s2 = statistics.variance(ds) if n > 1 else 0.0
    sp2 = ((n - 1) * s2 + nu0 * sigma0 * sigma0) / (n - 1 + nu0)
    return math.sqrt(sp2 / n), n - 1 + nu0


def sequential(ds, Q, sigma0, nu0, floor, S, K):
    """ds: the 2Q block observations d_b = ln(new) - ln(old), in run order. Returns (verdict, quads used), verdict in
    PASS / FAIL / AMBIGUOUS; at the cap the fixed-N verdict (§1.5) applies."""
    assert len(ds) == 2 * Q, (len(ds), Q)
    cs = boundaries(Q)
    for j, q in enumerate(range(Q_MIN, Q + 1)):
        if q == Q:
            break
        obs = ds[:2 * q]
        se, df = pooled_se(obs, sigma0, nu0)
        c = v1.t_inv(v1.Phi(cs[j]), df)
        m = statistics.mean(obs)
        lo, hi = m - c * se - floor, m + c * se + floor
        if lo > math.log(S):
            return "PASS", q
        if hi < math.log(K):
            return "FAIL", q
    return fixed_n(ds, floor, S, K), Q


def fixed_n(ds, floor, S, K):
    """The data-only fixed-N verdict on all blocks (§1.5)."""
    n = len(ds)
    m, s = statistics.mean(ds), statistics.stdev(ds)
    h = v1.t_inv(1.0 - ALPHA / 2.0, n - 1) * s / math.sqrt(n) + floor
    if m - h > math.log(S):
        return "PASS"
    if m + h < math.log(K):
        return "FAIL"
    return "AMBIGUOUS"


# ---------------------------------------------------------------------------------------------------------------
# §2.1 the simulation screen
# ---------------------------------------------------------------------------------------------------------------

SIM_NOISE = [  # (label, sigma0, floor, prior n) -- the three registry cells the pre-registration names
    ("nobara CUDA", 0.0081, 0.0, 20),
    ("CUDA 0.5B cross-build", 0.0164, 0.0144, 6),
    ("Mac CPU", 0.0333, 0.0, 12),
]
SIM_RATIOS = [0.94, 0.97, 1.00, 1.03, 1.06, 1.10]
SIM_BARS = [(0.97, 0.97), (1.03, 0.97)]
SIM_Q = [4, 6]


def sim_blocks(rng, r, sigma0, floor, Q):
    """Per-block paired observations with the registry's noise, a true build bias inside the floor, and persistent
    level shifts in either arm (5% at p 0.05 per arm per block, 31% at p 0.01), signs random."""
    bias = rng.uniform(-floor, floor)
    lvl = {"new": 0.0, "old": 0.0}
    ds = []
    for _ in range(2 * Q):
        for arm in lvl:
            u = rng.random()
            if u < 0.01:
                lvl[arm] += rng.choice((-1, 1)) * math.log(1.31)
            elif u < 0.06:
                lvl[arm] += rng.choice((-1, 1)) * math.log(1.05)
        ds.append(math.log(r) + bias + rng.gauss(0.0, sigma0) + lvl["new"] - lvl["old"])
    return ds


def cmd_sim(argv):
    gates = 4000
    if "--gates" in argv:
        gates = int(argv[argv.index("--gates") + 1])
    rng = random.Random(SEED)
    print(f"TE4-SEQ-v2 simulation screen (§2.1): {gates} gates per combination, seed {SEED}\n")
    print("| noise | Q | bars (S, K) | true r | P(PASS) | agree w/ fixed-N | blocks used / cap | clear of bar |")
    print("|---|---|---|---|---|---|---|---|")
    worst_pass, agree_n, agree_d, clear_used, clear_cap = 0.0, 0, 0, 0, 0
    fails = []
    for label, sigma0, floor, n0 in SIM_NOISE:
        nu0 = prior_df(n0)
        for Q in SIM_Q:
            half = v1.t_inv(1.0 - ALPHA / 2.0, 2 * Q - 1) * sigma0 / math.sqrt(2 * Q) + floor
            for S, K in SIM_BARS:
                for r in SIM_RATIOS:
                    npass = agree = used = 0
                    for _ in range(gates):
                        ds = sim_blocks(rng, r, sigma0, floor, Q)
                        v, q = sequential(ds, Q, sigma0, nu0, floor, S, K)
                        npass += v == "PASS"
                        agree += v == fixed_n(ds, floor, S, K)
                        used += q
                    p = npass / gates
                    dist = min(abs(math.log(r) - math.log(S)), abs(math.log(r) - math.log(K)))
                    clear = dist >= 2 * half
                    if r <= S + 1e-12:
                        worst_pass = max(worst_pass, p)
                        if p > 0.05:
                            fails.append(f"P(PASS) {p:.3f} at r {r} <= S {S} ({label}, Q {Q})")
                    agree_n += agree
                    agree_d += gates
                    if clear:
                        clear_used += used
                        clear_cap += Q * gates
                    print(f"| {label} | {Q} | ({S}, {K}) | {r:.2f} | {p:.3f} | {agree / gates:.3f} | "
                          f"{used / gates:.2f} / {Q} | {'yes' if clear else ''} |")
    agree_rate = agree_n / agree_d
    saving = 1 - clear_used / clear_cap if clear_cap else float("nan")
    print()
    print(f"worst P(PASS) where r <= S: {worst_pass:.3f}  (pass: <= 0.05)")
    print(f"agreement with the fixed-N verdict: {agree_rate:.4f}  (pass: >= 0.95)")
    print(f"blocks saved where the truth clears its nearest bar by >= 2 fixed-N half-widths: {saving:.3f}  (pass: >= 0.25)")
    ok = worst_pass <= 0.05 and agree_rate >= 0.95 and saving >= 0.25
    for f in fails:
        print("  FAIL:", f)
    print(f"\nSCREEN: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------------------
# §2.2 the replay over ABBA-blocked bench_peer.py output
# ---------------------------------------------------------------------------------------------------------------

GATES = [  # (id, backend, model, new engine, old engine, cap Q, instrument)
    (1, "cuda", "0.5B", "goinfer", "goinfer_old", 6, "served-decode"),
    (2, "cuda", "1.5B", "goinfer", "goinfer_old", 4, "served-decode"),
    (3, "cuda", "7B", "goinfer", "goinfer_old", 4, "served-decode"),
    (4, "cuda", "0.5B", "goinfer", "ollama", 4, "served-decode-peer"),
    (5, "cuda", "1.5B", "goinfer", "ollama", 4, "served-decode-peer"),
    (6, "cuda", "7B", "goinfer", "ollama", 4, "served-decode-peer"),
    (7, "cpu", "0.5B", "goinfer", "goinfer_old", 4, "served-decode"),
    (8, "cpu", "1.5B", "goinfer", "ollama", 4, "served-decode-peer"),
]
BAR_SETS = [("NI", 0.97, 0.97), ("peer", 1.03, 0.97)]


def cell_rate(rec):
    rs = (rec.get("counts") or {}).get("completion_rates") or []
    if rs:
        return statistics.mean(rs)
    return statistics.mean(rec["runs"])


def abba_blocks(recs, backend, model, new, old, depth=128, config="greedy"):
    """{block: (new rate, old rate, new secs, old secs)} from records carrying `block` (bench_peer BENCH_ABBA)."""
    by = {}
    for r in recs:
        if not isinstance(r, dict) or r.get("block") is None or not r.get("runs") or r.get("error"):
            continue
        if (r.get("backend"), r.get("model"), r.get("depth"), r.get("config")) != (backend, model, depth, config):
            continue
        by.setdefault(r["block"], {})[r["engine"]] = r
    out = {}
    for b, cells in by.items():
        if new in cells and old in cells:
            out[b] = (cell_rate(cells[new]), cell_rate(cells[old]), cells[new]["secs"], cells[old]["secs"])
    return out


def quad_orders(Q):
    base = list(range(Q))
    if math.factorial(Q) <= PERM_MAX:
        return [list(p) for p in itertools.permutations(base)]
    rng = random.Random(SEED)
    seen, out = set(), [base]
    seen.add(tuple(base))
    while len(out) < PERM_MAX:
        p = base[:]
        rng.shuffle(p)
        if tuple(p) not in seen:
            seen.add(tuple(p))
            out.append(p)
    return out


def cmd_replay(paths):
    recs = []
    for p in paths:
        recs += [r for r in json.load(open(p)) if isinstance(r, dict) and r.get("kind") != "provenance"]
    reg = power.load_registry()
    machine = "nobara"
    rows, agree_all, orders_agree, orders_n, disq, used_sum, cap_sum = [], 0, 0, 0, 0, 0, 0
    sec_used = sec_cap = 0.0
    missing = []
    for gid, be, model, new, old, Q, instrument in GATES:
        blk = abba_blocks(recs, be, model, new, old)
        if sorted(blk) != list(range(2 * Q)):
            missing.append(f"gate {gid}: blocks {sorted(blk)}, want 0..{2 * Q - 1}")
            continue
        e, _ = power.lookup(reg, instrument, {"machine": machine, "backend": be, "model": model, "depth": "d128"})
        sigma0, _why = power.entry_sd(e, True)
        nu0 = prior_df(e.get("n") or 2)
        floor = 0.0  # A/A: one binary, no build effect; peer: registry floors are 0 for these cells (§2.2)
        ds = [math.log(blk[b][0]) - math.log(blk[b][1]) for b in range(2 * Q)]
        cell_secs = [blk[b][2] + blk[b][3] for b in range(2 * Q)]
        for name, S, K in BAR_SETS:
            ref = fixed_n(ds, floor, S, K)
            v, q = sequential(ds, Q, sigma0, nu0, floor, S, K)
            agree_all += v == ref
            used_sum += q
            cap_sum += Q
            sec_used += sum(cell_secs[:2 * q])
            sec_cap += sum(cell_secs)
            ok_orders, bad = 0, 0
            orders = quad_orders(Q)
            for order in orders:
                dd = [ds[2 * k + i] for k in order for i in (0, 1)]
                vo, _ = sequential(dd, Q, sigma0, nu0, floor, S, K)
                ok_orders += vo == ref
                bad += vo == "PASS" and ref != "PASS"
            orders_agree += ok_orders / len(orders)
            orders_n += 1
            disq += bad
            rows.append((gid, be, model, new, old, name, Q, e["id"], sigma0, nu0,
                         math.exp(statistics.mean(ds)), ref, v, q, ok_orders, len(orders), bad))
    print("| gate | cell | arms | bars | prior (sd, df) | ratio | fixed-N | replay | quads | orders agree | disq. |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (gid, be, model, new, old, name, Q, eid, s0, nu0, ratio, ref, v, q, ok, n, bad) in rows:
        print(f"| {gid} | {be} {model} | {new}/{old} | {name} | {s0:.4f}, {nu0} ({eid}) | {ratio:.4f} | {ref} | {v} | "
              f"{q}/{Q} | {ok}/{n} | {bad} |")
    for m in missing:
        print("MISSING:", m)
    if not rows:
        return 2
    n = len(rows)
    saving = 1 - used_sum / cap_sum
    print()
    print(f"1. recorded order: {agree_all}/{n} sub-gates reach their fixed-N verdict")
    print(f"2. orders in which a non-PASS sub-gate reaches PASS: {disq}")
    print(f"3. mean per-sub-gate agreement over quad orders: {orders_agree / orders_n:.4f}")
    print(f"4. cells saved (recorded order): {saving:.3f}; wall saved (measured secs): {1 - sec_used / sec_cap:.3f}")
    ok = agree_all == n and disq == 0 and orders_agree / orders_n >= 0.95 and not missing
    verdict = "PASS" if ok and saving >= 0.25 else ("PARKED" if ok and saving >= 0.15 else "KILL")
    print(f"\nVALIDATION: {verdict}")
    return 0


def selfcheck():
    v1.self_check()
    # The spending boundaries for the looks v2 takes, Q = 4 and 6.
    for Q in (4, 6):
        print(f"v2 looks q = {Q_MIN}..{Q} of {Q}: z boundaries {[round(c, 3) for c in boundaries(Q)]}")


def main(argv):
    if not argv or argv[0] == "selfcheck":
        selfcheck()
        return 0
    if argv[0] == "sim":
        return cmd_sim(argv[1:])
    if argv[0] == "replay":
        return cmd_replay(argv[1:])
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
