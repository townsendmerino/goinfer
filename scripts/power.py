#!/usr/bin/env python3
"""power.py: how many runs resolve a gate's bar, from the noise registry (TE3).

Item TE3 of docs/tasks/task-test-efficiency-2026-09.md. A gate that cannot resolve its bar costs its full wall and
decides nothing, and until now that was found out after the run. This tool answers it before the run, from the A/A
noise the records already hold (docs/measurements/noise-registry.json, explained in noise-registry.md beside it).
Stdlib only; it reads the registry and runs and times nothing.

Speed gates, a paired log-ratio test:

    power.py served-decode nobara/cuda/1.5B/d128 --bar 0.97 --paired
    power.py inproc-token nobara/cpu/7B/d128 --bar 1.03 --expect 1.06 --paired
    power.py served-decode mac/cpu/0.5B/d128 --bar 0.97 --paired --sd 0.0251 --floor 0.0165

  The cell is machine/backend/model/depth (machine mac or nobara; backend cpu, cuda, metal or webgpu; the machine
  fixes the CPU's arch). The registry entry chosen, its statistic and its source are printed with the answer. A
  lookup prefers a measured spread (an A/A or a replicate) over an inferred one however specific, then the
  better-founded provenance (a cross-build A/A first), then the more specific cell (a "*" field matches anything),
  then the larger sd; the other entries covering the cell are listed. --sd (and --floor) replace the registry; --entry ID picks an entry by id.

  The model. Each paired block (one cell per arm in a served pass, one ABBA pair in process, ...; the entry names its
  unit) gives d = ln(new) - ln(old) ~ N(ln(expect) + b, sd^2). The bias b is what re-running the block does not
  re-draw (an order or cell effect), bounded by the entry's floor: |b| <= floor. The decision interval is TE4-SEQ-v1's,
  mean(d) -/+ t * SE -/+ floor, so the gate resolves the bar when the interval clears ln(bar). Power is computed at
  b = 0, which charges the floor once (in the interval); --worst-case charges it twice (b = -floor as well). The
  effect the gate has to be able to show is margin = |ln(expect) - ln(bar)| - floor, and the N is the smallest with
  exact noncentral-t power >= --power (two-sided alpha unless --one-sided). A margin <= 0 cannot be resolved at any N.
  Without --paired the arms are two independent samples of N units each, and the entry must carry an arm-level sd
  (a paired sd cannot be split back into arms: the pairing already removed the common-mode noise).

Evaluation sample sizes:

    power.py binomial --p 0.9 --half-width 0.015          # N for a proportion's interval
    power.py binomial --p 0.9 --n 400                     # SE and 95% half-width at N
    power.py binomial --p 0.9 --stratum 9767:400 --stratum 3219:400 --stratum 72:72
    power.py ece --acc 0.9 --bins 15 --n 872              # plug-in ECE's noise floor at N
    power.py ece --acc 0.9 --bins 4 --floor 0.01          # N for a floor

  binomial is the Wald normal approximation, SE = sqrt(p(1-p)/n); a stratified (weighted) estimate combines
  SE^2 = sum w_k^2 p_k (1-p_k) / n_k with w_k = W_k / sum W. It reproduces the D6a amendment (2852eaa1): 400 noul rows
  near p 0.9 give SE 0.0150, "about +/-1.5 points"; see noise-registry.md.

  ece is an approximation, stated in full in noise-registry.md section 6. For a perfectly calibrated model each
  occupied bin's |accuracy - confidence| is |N(0, p(1-p)/n_b)|, a half-normal with mean sigma*sqrt(2/pi). Summed with
  bin weights n_b/N over B equally filled bins, the plug-in ECE's expected value (its noise floor) is
  sqrt(2/pi) * sqrt(B p(1-p) / N), and B equally filled bins is the largest floor any filling of B bins gives
  (Cauchy-Schwarz), so it is an upper bound for B occupied bins. Its sd lies between sqrt((1-2/pi) p(1-p)/N) (every
  bin calibrated) and sqrt(p(1-p)/N) (every bin far from calibrated). The upward bias of the binned plug-in estimator
  is the effect Kumar, Liang & Ma (NeurIPS 2019, "Verified Uncertainty Calibration") and Roelofs et al. (AISTATS 2022,
  "Mitigating Bias in Calibration Error Estimation") document; the closed form here is only the half-normal mean.

    power.py list [instrument]      # the registry's entries
    power.py verify                 # recompute every entry's sd from its recorded values

Exit status: 0 resolved / printed, 2 usage or registry miss, 3 cannot resolve (at any N, or within --max-n).
"""
import argparse
import json
import math
import os
import statistics
import sys
from statistics import NormalDist

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(os.path.dirname(HERE), "docs", "measurements", "noise-registry.json")
EXIT_OK, EXIT_USAGE, EXIT_UNRESOLVABLE = 0, 2, 3
_N = NormalDist()

# Provenance ranks. A lookup prefers a measured spread (an A/A or a replicate) over an inferred one, however specific
# the inference; then the better-founded provenance (a cross-build A/A first: a new / old gate always compares two
# builds, and a same-binary A/A cannot see build or layout effects); then the more specific cell; then the larger sd.
MEASURED_RANK = 3
PROVENANCE_RANK = {
    "cross-build-aa": 6,     # two builds whose code for this cell the record says is identical
    "direct-aa": 5,          # the same binary in both arms
    "effective-aa": 4,       # the same binary with a switch this cell does not take, or builds called "unchanged"
    "replicate": 3,          # the same A/B (or arm) measured twice: its spread is noise plus any order effect
    "model-inference": 2,    # derived from variance components or from other entries, not a direct A/A
    "single-observation": 1,  # one pair of readings
    "floor-only": 0,         # a floor with no sd (bit-identity, a max, a warning); not usable for N
}


class RegistryMiss(Exception):
    pass


# --------------------------------------------------------------------------------------------------------------
# distributions (stdlib only)

def z(p):
    return _N.inv_cdf(p)


def _betacf(a, b, x):
    """Continued fraction for the regularized incomplete beta (modified Lentz; Numerical Recipes 6.4)."""
    tiny, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 10000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_ppf(p, df):
    """Inverse Student-t CDF by bisection on t_cdf (monotone)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if p == 0.5:
        return 0.0
    lo, hi = -1.0, 1.0
    while t_cdf(lo, df) > p:
        lo *= 2.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def nct_sf(c, df, lam, steps=4000):
    """P(T' > c) for a noncentral t with df degrees of freedom and noncentrality lam.

    T' = (Z + lam) / W with W = sqrt(V/df), V ~ chi-square(df). P(T' > c) = integral of Phi(lam - c w) f_W(w) dw,
    by Simpson's rule over w in (0, wmax]; f_W is the scaled chi density, taken in logs for large df.
    """
    k = df / 2.0
    logc = math.log(2.0) + k * math.log(k) - math.lgamma(k)
    sd_w = 1.0 / math.sqrt(2.0 * df)
    wmax = max(1.0 + 14.0 * sd_w, 4.0) if df > 2 else 30.0
    h = wmax / steps
    total = 0.0
    for i in range(steps + 1):
        w = i * h
        if w == 0.0:
            fw = 0.0 if df > 1 else math.exp(logc)
        else:
            fw = math.exp(logc + (df - 1.0) * math.log(w) - df * w * w / 2.0)
        val = _N.cdf(lam - c * w) * fw
        total += val * (1 if i in (0, steps) else (4 if i % 2 else 2))
    return total * h / 3.0


# --------------------------------------------------------------------------------------------------------------
# the paired log-ratio power calculation

def power_at(n, sd, margin, alpha, paired, one_sided):
    """Exact power of the t test at n (blocks if paired, units per arm if not)."""
    if paired:
        df, se = n - 1, sd / math.sqrt(n)
    else:
        df, se = 2 * n - 2, sd * math.sqrt(2.0 / n)
    if df < 1:
        return 0.0
    lam = margin / se
    crit = t_ppf(1 - alpha if one_sided else 1 - alpha / 2, df)
    p = nct_sf(crit, df, lam)
    if not one_sided:
        p += 1.0 - nct_sf(-crit, df, lam)   # the far tail: negligible at any useful power, kept for exactness
    return p


def n_normal(sd, margin, alpha, power, paired, one_sided):
    """The normal-approximation N, (z_a + z_b)^2 sd^2 / margin^2 (x2 per arm if unpaired): the textbook formula."""
    za = z(1 - alpha if one_sided else 1 - alpha / 2)
    zb = z(power)
    n = ((za + zb) * sd / margin) ** 2
    return 2.0 * n if not paired else n


def solve_n(sd, margin, alpha=0.05, power=0.8, paired=True, one_sided=False, max_n=None):
    """Smallest N with exact t power >= power, or (None, n_z) if it exceeds max_n. margin must be > 0."""
    if margin <= 0 or sd <= 0:
        raise ValueError("margin and sd must be > 0")
    nz = n_normal(sd, margin, alpha, power, paired, one_sided)
    lo_n = 2
    n = max(lo_n, int(math.ceil(nz)))
    if max_n is not None and n > max_n + 50:
        # even the (smaller) normal N is far past the cap: no need to walk the t search up to it
        return None, nz
    while n > lo_n and power_at(n - 1, sd, margin, alpha, paired, one_sided) >= power:
        n -= 1
    while power_at(n, sd, margin, alpha, paired, one_sided) < power:
        n += 1
        if max_n is not None and n > max_n:
            return None, nz
    if max_n is not None and n > max_n:
        return None, nz
    return n, nz


def min_margin_at(n, sd, alpha, power, paired, one_sided):
    """The smallest margin resolvable at N (bisection on power, which rises with the margin)."""
    lo, hi = 0.0, 10.0 * sd
    while power_at(n, sd, hi, alpha, paired, one_sided) < power:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if power_at(n, sd, mid, alpha, paired, one_sided) >= power:
            hi = mid
        else:
            lo = mid
    return hi


# --------------------------------------------------------------------------------------------------------------
# registry

def load_registry(path=REGISTRY):
    with open(path) as f:
        return json.load(f)


def parse_cell(s):
    parts = s.split("/")
    if len(parts) != 4 or not all(parts):
        raise ValueError("cell must be machine/backend/model/depth, e.g. nobara/cuda/1.5B/d128 (got %r)" % s)
    return dict(zip(("machine", "backend", "model", "depth"), parts))


def _match(entry_cell, q):
    """None if the entry does not cover the query, else its specificity (non-wildcard fields that match)."""
    spec = 0
    for k in ("machine", "backend", "model", "depth"):
        ev = entry_cell.get(k, "*")
        if ev == "*":
            continue
        if ev.lower() != q[k].lower():
            return None
        spec += 1
    return spec


def candidates(reg, instrument, cell_q, usable_only=True):
    out = []
    for e in reg["entries"]:
        if e["instrument"] != instrument:
            continue
        spec = _match(e["cell"], cell_q)
        if spec is None:
            continue
        usable = e.get("sd") is not None
        if usable_only and not usable:
            continue
        rank = PROVENANCE_RANK.get(e["provenance"], -1)
        out.append((rank >= MEASURED_RANK, rank, spec, e.get("sd") or 0.0, e))
    out.sort(key=lambda t: t[:4], reverse=True)
    return [t[4] for t in out]


def lookup(reg, instrument, cell_q, entry_id=None):
    if instrument not in reg["instruments"]:
        raise RegistryMiss("unknown instrument %r; the registry has: %s" % (instrument, ", ".join(sorted(reg["instruments"]))))
    if entry_id:
        for e in reg["entries"]:
            if e["id"] == entry_id:
                if e.get("sd") is None:
                    raise RegistryMiss("entry %s carries no sd (%s): a floor, not a noise for N" % (entry_id, e["provenance"]))
                return e, []
        raise RegistryMiss("no registry entry with id %r" % entry_id)
    c = candidates(reg, instrument, cell_q)
    if not c:
        fl = candidates(reg, instrument, cell_q, usable_only=False)
        near = [e["id"] for e in reg["entries"] if e["instrument"] == instrument][:12]
        msg = "registry miss: no entry with an sd for %s %s" % (instrument, "/".join(cell_q[k] for k in ("machine", "backend", "model", "depth")))
        if fl:
            msg += "\n  (floor-only entries cover it: %s)" % ", ".join(e["id"] for e in fl)
        msg += "\n  entries for this instrument: %s" % (", ".join(near) if near else "none")
        msg += "\n  pass --sd (and --floor) to supply the noise, and record where it came from"
        gaps = [g for g in reg.get("gaps", []) if g.get("instrument") == instrument]
        if gaps:
            msg += "\n  the registry lists this instrument's gaps in noise-registry.md section 7"
        raise RegistryMiss(msg)
    return c[0], c[1:]


def entry_sd(e, paired):
    kind = e.get("sd_kind")
    if paired:
        if kind == "paired_log_ratio":
            return e["sd"], "the entry's paired sd"
        if kind == "arm_log_level":
            return math.sqrt(2.0) * e["sd"], "sqrt(2) x the entry's arm-level sd (independent arms, no common-mode credit)"
    else:
        if kind == "arm_log_level":
            return e["sd"], "the entry's arm-level sd"
        if kind == "paired_log_ratio":
            raise RegistryMiss("entry %s is a paired statistic; an unpaired design needs an arm-level sd (pass --sd), because "
                               "the pairing already removed common-mode noise that two independent arms would carry" % e["id"])
    raise RegistryMiss("entry %s has sd_kind %r, which power.py cannot use" % (e["id"], kind))


def recompute_sd(e):
    """Recompute an entry's sd from its recorded values, by its estimator. None if it has no recomputable values."""
    est, vals = e.get("estimator"), e.get("values")
    if not vals or est in (None, "quoted"):
        return None
    if est == "rms_log_about_0":            # A/A ratios: the true log ratio is 0, so sd = RMS of ln(r), n df
        return math.sqrt(sum(math.log(v) ** 2 for v in vals) / len(vals))
    if est == "sd_log":                     # replicate levels or ratios: sample sd of ln, n-1 df
        ls = [math.log(v) for v in vals]
        m = sum(ls) / len(ls)
        return math.sqrt(sum((x - m) ** 2 for x in ls) / (len(ls) - 1))
    if est == "pair_diff":                  # pairs [a, b] of one quantity measured twice: sd of one = sqrt(mean(dln^2)/2)
        ds = [math.log(a) - math.log(b) for a, b in vals]
        return math.sqrt(sum(d * d for d in ds) / (2 * len(ds)))
    if est == "log_sd_pct":                 # values are sds in percent (TE2(b) components): combined per the entry's formula
        return None
    raise ValueError("entry %s: unknown estimator %r" % (e["id"], est))


# --------------------------------------------------------------------------------------------------------------
# commands

def fmt_cell(q):
    return "/".join(q[k] for k in ("machine", "backend", "model", "depth"))


def cmd_gate(args, reg, out):
    try:
        cell_q = parse_cell(args.cell)
    except ValueError as ex:
        print("power.py: %s" % ex, file=sys.stderr)
        return EXIT_USAGE
    if not 0 < args.bar:
        print("power.py: --bar must be a positive ratio", file=sys.stderr)
        return EXIT_USAGE
    inst = reg["instruments"].get(args.instrument) if reg else None
    source = None
    others = []
    if args.sd is not None:
        sd, sd_note = args.sd, "--sd (supplied)"
        floor = args.floor if args.floor is not None else 0.0
        unit = "one paired block" if args.paired else "one unit per arm"
        source = "supplied on the command line"
    else:
        try:
            e, others = lookup(reg, args.instrument, cell_q, args.entry)
            sd, sd_note = entry_sd(e, args.paired)
        except RegistryMiss as ex:
            print("power.py: %s" % ex, file=sys.stderr)
            return EXIT_USAGE
        floor = args.floor if args.floor is not None else (e.get("floor") or 0.0)
        unit = e.get("unit", "?")
        source = e
    max_n = args.max_n if args.max_n is not None else (inst or {}).get("max_n", 30)
    delta = abs(math.log(args.expect) - math.log(args.bar))
    charged = 2.0 * floor if args.worst_case else floor
    margin = delta - charged
    side = "one-sided" if args.one_sided else "two-sided"
    design = "paired blocks" if args.paired else "units per arm (unpaired)"

    p = lambda s="": print(s, file=out)  # noqa: E731
    p("instrument %s, cell %s" % (args.instrument, fmt_cell(cell_q)))
    if isinstance(source, dict):
        p("  registry entry: %s [%s]" % (source["id"], source["provenance"]))
        p("    statistic: %s" % source["statistic"])
        p("    n = %s; %s" % (source.get("n"), source.get("scope", "")))
        for s in source.get("source", [])[:3]:
            p("    source: %s" % _fmt_src(s))
        if others:
            p("    also covering this cell: %s" % ", ".join("%s (sd %.4f, %s)" % (o["id"], o["sd"], o["provenance"]) for o in others[:4]))
    else:
        p("  noise: %s" % source)
    p("  sd = %.4f (log units, per %s): %s" % (sd, unit, sd_note))
    p("  floor = %.4f (does not shrink with N)%s" % (floor, "; charged twice (--worst-case)" if args.worst_case else ""))
    p("  bar %.4g, expected ratio %.4g: |ln(expect/bar)| = %.4f, margin after the floor = %.4f" % (args.bar, args.expect, delta, margin))
    p("  alpha %.3g %s, power %.2f, %s, feasible max N %d" % (args.alpha, side, args.power, design, max_n))
    if margin <= 0:
        p("CANNOT RESOLVE at any N: the expected ratio sits within the non-shrinking floor (%.4f) of the bar. More runs"
          " do not help; change the instrument or the design (a floor comes from order / cell effects that pairing or"
          " counterbalancing inside the pass would re-draw)." % charged)
        return EXIT_UNRESOLVABLE
    n, nz = solve_n(sd, margin, args.alpha, args.power, args.paired, args.one_sided, max_n)
    if n is None:
        mm = min_margin_at(max_n, sd, args.alpha, args.power, args.paired, args.one_sided)
        pw = power_at(max_n, sd, margin, args.alpha, args.paired, args.one_sided)
        lim = math.exp(math.log(args.bar) + (mm + charged) * (1 if args.expect >= args.bar else -1))
        p("CANNOT RESOLVE at a feasible N (max N %d; the normal approximation wants %.1f): at N=%d the power is %.2f, and"
          " the smallest resolvable expected ratio there is %.4f. Change the instrument." % (max_n, nz, max_n, pw, lim))
        return EXIT_UNRESOLVABLE
    pw = power_at(n, sd, margin, args.alpha, args.paired, args.one_sided)
    p("N = %d %s (exact t power %.3f; normal approximation %.1f)" % (n, design, pw, nz))
    return EXIT_OK


def _fmt_src(s):
    if "commit" in s:
        return "commit %s%s" % (s["commit"], (" (%s)" % s["note"]) if s.get("note") else "")
    loc = s["path"] + ((":%s" % s["line"]) if s.get("line") else "")
    return loc + ((' "%s"' % s["quote"]) if s.get("quote") else "")


def _parse_strata(items, default_p):
    out = []
    for it in items:
        parts = it.split(":")
        if len(parts) not in (2, 3):
            raise ValueError("--stratum is WEIGHT:N or WEIGHT:N:P (got %r)" % it)
        w, n = float(parts[0]), int(parts[1])
        pk = float(parts[2]) if len(parts) == 3 else default_p
        if w <= 0 or n <= 0 or not 0 < pk < 1:
            raise ValueError("--stratum %r: weight and n must be > 0 and p in (0, 1)" % it)
        out.append((w, n, pk))
    return out


def binomial_se(p, n):
    return math.sqrt(p * (1 - p) / n)


def stratified_se(strata):
    tw = sum(w for w, _, _ in strata)
    return math.sqrt(sum((w / tw) ** 2 * pk * (1 - pk) / n for w, n, pk in strata))


def binomial_n(p, half_width, conf):
    """Wald N for a half-width at a two-sided confidence; conf=None reads the half-width as one SE."""
    zz = 1.0 if conf is None else z(0.5 + conf / 2)
    return int(math.ceil(zz * zz * p * (1 - p) / (half_width * half_width) - 1e-9))


def cmd_binomial(argv, out):
    ap = argparse.ArgumentParser(prog="power.py binomial", description="sample size / SE for a proportion (Wald)")
    ap.add_argument("--p", type=float, required=True, help="the proportion expected (e.g. 0.9)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--half-width", type=float, help="the half-width wanted, as a proportion (0.015 = 1.5 points)")
    g.add_argument("--n", type=int, help="the sample size: print its SE and interval")
    ap.add_argument("--conf", type=float, default=0.95, help="two-sided confidence for --half-width (default 0.95)")
    ap.add_argument("--stratum", action="append", default=[], metavar="W:N[:P]",
                    help="a stratum of a weighted estimate: population weight W, sample N, proportion P (default --p)")
    a = ap.parse_args(argv)
    if not 0 < a.p < 1:
        print("power.py binomial: --p must be in (0, 1)", file=sys.stderr)
        return EXIT_USAGE
    p = lambda s="": print(s, file=out)  # noqa: E731
    zc = z(0.5 + a.conf / 2)
    if a.stratum:
        try:
            st = _parse_strata(a.stratum, a.p)
        except ValueError as ex:
            print("power.py binomial: %s" % ex, file=sys.stderr)
            return EXIT_USAGE
        tw = sum(w for w, _, _ in st)
        se = stratified_se(st)
        p("weighted proportion over %d strata (weights normalized over %g):" % (len(st), tw))
        for w, n, pk in st:
            p("  weight %.4f  n %d  p %.3f  stratum SE %.4f  contributes %.4f" % (w / tw, n, pk, binomial_se(pk, n), (w / tw) * binomial_se(pk, n)))
        p("SE of the weighted proportion = %.4f (%.2f points); %.0f%% half-width %.4f (%.2f points)" % (se, 100 * se, 100 * a.conf, zc * se, 100 * zc * se))
        return EXIT_OK
    if a.n is not None:
        se = binomial_se(a.p, a.n)
        p("p %.3f, n %d: SE = %.4f (%.2f points); %.0f%% Wald half-width = %.4f (%.2f points)" % (a.p, a.n, se, 100 * se, 100 * a.conf, zc * se, 100 * zc * se))
        if a.n * min(a.p, 1 - a.p) < 10:
            p("  warning: n*min(p,1-p) < 10, the Wald approximation is poor here")
        return EXIT_OK
    if a.half_width is None:
        print("power.py binomial: give --half-width, --n or --stratum", file=sys.stderr)
        return EXIT_USAGE
    n1 = binomial_n(a.p, a.half_width, None)
    nc = binomial_n(a.p, a.half_width, a.conf)
    p("p %.3f, half-width %.4f (%.2f points):" % (a.p, a.half_width, 100 * a.half_width))
    p("  N = %d for it to be one SE (a 68%% interval), which is how the D6a amendment (2852eaa1) read +/-1.5 points" % n1)
    p("  N = %d for it to be the %.0f%% half-width (z = %.3f)" % (nc, 100 * a.conf, zc))
    return EXIT_OK


def ece_floor(p, bins, n):
    """Expected plug-in ECE of a perfectly calibrated model: sqrt(2/pi) * sqrt(B p(1-p) / N) (B equally filled bins)."""
    return math.sqrt(2.0 / math.pi) * math.sqrt(bins * p * (1 - p) / n)


def ece_sd_bounds(p, n):
    v = p * (1 - p) / n
    return math.sqrt((1 - 2 / math.pi) * v), math.sqrt(v)


def ece_n(p, bins, floor):
    return int(math.ceil(2.0 * bins * p * (1 - p) / (math.pi * floor * floor) - 1e-9))


def cmd_ece(argv, out):
    ap = argparse.ArgumentParser(prog="power.py ece", description="noise floor of a binned plug-in ECE (approximation)")
    ap.add_argument("--acc", type=float, required=True, help="accuracy / mean confidence in the occupied bins (e.g. 0.9)")
    ap.add_argument("--bins", type=int, required=True, help="occupied bins (<= the bin count; the full count is the upper bound)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--n", type=int, help="the sample size: print the floor and sd at it")
    g.add_argument("--floor", type=float, help="the floor wanted: print the N that reaches it")
    ap.add_argument("--stratum", action="append", default=[], metavar="W:N[:P]",
                    help="a per-kind ECE combined with weight W over N rows (repeat per kind; instead of --n)")
    a = ap.parse_args(argv)
    if not 0 < a.acc < 1 or a.bins < 1:
        print("power.py ece: --acc in (0, 1) and --bins >= 1", file=sys.stderr)
        return EXIT_USAGE
    if sum(x is not None and x != [] for x in (a.n, a.floor, a.stratum or None)) != 1:
        print("power.py ece: give exactly one of --n, --floor or --stratum", file=sys.stderr)
        return EXIT_USAGE
    p = lambda s="": print(s, file=out)  # noqa: E731
    p("approximation: half-normal mean per bin, %d equally filled bins (an upper bound for %d occupied bins), accuracy %.3f"
      % (a.bins, a.bins, a.acc))
    if a.floor is not None:
        n = ece_n(a.acc, a.bins, a.floor)
        p("N = %d for the plug-in ECE of a calibrated model to average <= %.4f" % (n, a.floor))
        return EXIT_OK
    if a.stratum:
        try:
            st = _parse_strata(a.stratum, a.acc)
        except ValueError as ex:
            print("power.py ece: %s" % ex, file=sys.stderr)
            return EXIT_USAGE
        tw = sum(w for w, _, _ in st)
        fl = sum((w / tw) * ece_floor(pk, a.bins, n) for w, n, pk in st)
        sd_hi = math.sqrt(sum((w / tw) ** 2 * pk * (1 - pk) / n for w, n, pk in st))
        for w, n, pk in st:
            p("  weight %.4f  n %d  floor %.4f" % (w / tw, n, ece_floor(pk, a.bins, n)))
        p("combined ECE floor = %.4f; its sd <= %.4f" % (fl, sd_hi))
        return EXIT_OK
    fl = ece_floor(a.acc, a.bins, a.n)
    lo, hi = ece_sd_bounds(a.acc, a.n)
    p("n %d: a perfectly calibrated model's plug-in ECE averages %.4f; its sd is %.4f-%.4f" % (a.n, fl, lo, hi))
    return EXIT_OK


def cmd_list(argv, reg, out):
    inst = argv[0] if argv else None
    for e in reg["entries"]:
        if inst and e["instrument"] != inst:
            continue
        c = e["cell"]
        sd = ("%.4f" % e["sd"]) if e.get("sd") is not None else "  -   "
        fl = ("%.4f" % e["floor"]) if e.get("floor") else "0"
        print("%-44s %-13s %-34s sd %s floor %-6s n %-4s %s" % (
            e["id"], e["instrument"], "/".join(c.get(k, "*") for k in ("machine", "backend", "model", "depth")),
            sd, fl, e.get("n", "?"), e["provenance"]), file=out)
    return EXIT_OK


def cmd_verify(reg, out):
    bad = 0
    for e in reg["entries"]:
        r = recompute_sd(e)
        if r is None:
            continue
        ok = e.get("sd") is not None and abs(r - e["sd"]) <= 5e-5
        bad += 0 if ok else 1
        print("%-44s recorded %s recomputed %.5f %s" % (e["id"], e.get("sd"), r, "ok" if ok else "MISMATCH"), file=out)
    print("verify: %d mismatch(es)" % bad, file=out)
    return EXIT_OK if bad == 0 else EXIT_UNRESOLVABLE


def fidelity_spreads(path):
    """Per-prompt spreads from a fidelity.WritePositions JSONL (internal/fidelity): the sd of the per-prompt agreement
    difference in points, and the KL residual sd relative to the exact arm's mean (the delta-method term the Go criterion
    bounds). Returns (agree_sd_pts, kl_cv, n_prompts)."""
    per = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            k = (r["cell"], r["prompt"])
            d = per.setdefault(k, {"candidate": [], "exact": []})
            d[r["arm"]].append((1.0 if r["agree"] else 0.0, float(r["kl"])))
    dag, kc, ke = [], [], []
    for d in per.values():
        c, e = d["candidate"], d["exact"]
        if not c or len(c) != len(e):
            continue
        dag.append(100.0 * (sum(x[0] for x in c) / len(c) - sum(x[0] for x in e) / len(e)))
        kc.append(sum(x[1] for x in c) / len(c))
        ke.append(sum(x[1] for x in e) / len(e))
    if len(dag) < 2:
        raise ValueError(f"{path}: fewer than 2 complete prompts")
    me = sum(ke) / len(ke)
    r = (sum(kc) / len(kc)) / me
    resid = [c - r * e for c, e in zip(kc, ke)]
    return statistics.stdev(dag), statistics.stdev(resid) / me, len(dag)


def cmd_fidelity(argv, out):
    """TE12: prompts a fidelity gate needs, per criterion, for internal/fidelity's non-inferiority tests."""
    ap = argparse.ArgumentParser(prog="power.py fidelity",
                                 description="prompts needed per criterion of internal/fidelity's NonInferior (TE12)")
    ap.add_argument("--from-positions", help="a fidelity.WritePositions JSONL to estimate the spreads from")
    ap.add_argument("--agree-sd", type=float, help="sd of the per-prompt agreement difference, points")
    ap.add_argument("--kl-cv", type=float, help="sd of the per-prompt KL residual (c - R e) / mean exact KL")
    ap.add_argument("--agree-margin", type=float, default=1.0, help="points (default 1)")
    ap.add_argument("--kl-margin", type=float, default=0.10, help="fraction of exact's mean KL (default 0.10)")
    ap.add_argument("--alpha", type=float, default=0.05, help="one-sided")
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--max-prompts", type=int, default=400)
    a = ap.parse_args(argv)
    src = "given"
    if a.from_positions:
        a.agree_sd, a.kl_cv, n = fidelity_spreads(a.from_positions)
        src = f"estimated from {n} prompts in {a.from_positions} (each sd is itself uncertain at that n)"
    if a.agree_sd is None or a.kl_cv is None:
        print("power.py fidelity: pass --from-positions, or both --agree-sd and --kl-cv", file=out)
        return EXIT_USAGE
    print(f"spreads ({src}): agreement sd {a.agree_sd:.3f} pts per prompt, KL residual cv {a.kl_cv:.4f}", file=out)
    worst = 0
    for name, sd, m, unit in (("agreement", a.agree_sd, a.agree_margin, "pts"), ("KL ratio", a.kl_cv, a.kl_margin, "")):
        n, nz = solve_n(sd, m, a.alpha, a.power, paired=True, one_sided=True, max_n=a.max_prompts)
        if n is None:
            print(f"  {name:<10} margin {m:g}{unit}: cannot resolve within {a.max_prompts} prompts (normal N {nz:.0f}) "
                  f"— widen the margin only with a reason, or change the instrument", file=out)
            worst = None
        else:
            print(f"  {name:<10} margin {m:g}{unit}: {n} prompts (pooled across the gate's cells)", file=out)
            if worst is not None:
                worst = max(worst, n)
    if worst is not None:
        print(f"prompts for the gate: {worst} pooled (e.g. {math.ceil(worst / 5)} per cell over 5 cells); "
              f"hard flips use the pooled Poisson bound and add no requirement", file=out)
    return EXIT_OK


def cmd_worstcase(argv, out):
    """Zero-failure prompt count for a worst-case claim: (1 - rate)^n <= alpha."""
    ap = argparse.ArgumentParser(prog="power.py worstcase", description="prompts that exclude a bad-prompt rate")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rate", type=float, help="bad-prompt rate to exclude, e.g. 0.10")
    g.add_argument("--prompts", type=int, help="the rate N all-passing prompts exclude")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args(argv)
    if a.rate is not None:
        n = math.ceil(math.log(a.alpha) / math.log(1 - a.rate))
        print(f"{n} prompts, all passing, exclude a bad-prompt rate >= {a.rate:g} at {1 - a.alpha:.0%} confidence", file=out)
    else:
        r = 1 - a.alpha ** (1 / a.prompts)
        print(f"{a.prompts} all-passing prompts exclude a bad-prompt rate >= {r:.3f} at {1 - a.alpha:.0%} confidence",
              file=out)
    return EXIT_OK


def main(argv=None, out=sys.stdout):
    argv = list(sys.argv[1:] if argv is None else argv)
    reg_path = REGISTRY
    if "--registry" in argv:
        i = argv.index("--registry")
        reg_path = argv[i + 1]
        del argv[i:i + 2]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__, file=out)
        return EXIT_OK if argv else EXIT_USAGE
    if argv[0] == "binomial":
        return cmd_binomial(argv[1:], out)
    if argv[0] == "ece":
        return cmd_ece(argv[1:], out)
    if argv[0] == "fidelity":
        return cmd_fidelity(argv[1:], out)
    if argv[0] == "worstcase":
        return cmd_worstcase(argv[1:], out)
    reg = load_registry(reg_path)
    if argv[0] == "list":
        return cmd_list(argv[1:], reg, out)
    if argv[0] == "verify":
        return cmd_verify(reg, out)
    ap = argparse.ArgumentParser(prog="power.py", description="N that resolves a gate's bar (paired log-ratio test)")
    ap.add_argument("instrument", help="a registry instrument: " + ", ".join(sorted(reg["instruments"])))
    ap.add_argument("cell", help="machine/backend/model/depth, e.g. nobara/cuda/1.5B/d128")
    ap.add_argument("--bar", type=float, required=True, help="the ratio the gate must resolve (e.g. 0.97)")
    ap.add_argument("--expect", type=float, default=1.0, help="the true ratio the gate must be able to confirm (default 1.0)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--paired", action="store_true", help="paired blocks (the default design of this repo's gates)")
    ap.add_argument("--one-sided", action="store_true", help="one-sided alpha (default two-sided, as TE4-SEQ-v1)")
    ap.add_argument("--worst-case", action="store_true", help="charge the floor twice: the bias also works against the gate")
    ap.add_argument("--sd", type=float, help="the noise sd in log units, instead of the registry")
    ap.add_argument("--floor", type=float, help="the non-shrinking floor in log units (overrides the entry's)")
    ap.add_argument("--entry", help="use this registry entry id")
    ap.add_argument("--max-n", type=int, help="the feasible cap (default: the instrument's max_n in the registry)")
    a = ap.parse_args(argv)
    if not (0 < a.alpha < 1 and 0 < a.power < 1) or a.expect <= 0:
        print("power.py: alpha and power in (0, 1), expect > 0", file=sys.stderr)
        return EXIT_USAGE
    return cmd_gate(a, reg, out)


if __name__ == "__main__":
    sys.exit(main())
