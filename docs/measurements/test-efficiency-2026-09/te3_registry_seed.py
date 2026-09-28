#!/usr/bin/env python3
"""te3_registry_seed.py: the computed rows of TE3's noise registry, from records already on disk.

Item TE3 of docs/tasks/task-test-efficiency-2026-09.md. Read-only and stdlib-only; it runs, serves and times nothing.
docs/measurements/noise-registry.json carries the values this prints (each entry's `values`, so `scripts/power.py
verify` recomputes its sd), and noise-registry.md explains them.

  usage: python3 docs/measurements/test-efficiency-2026-09/te3_registry_seed.py

1. SERVED DECODE, SAME-BUILD RESTART PAIRS (a direct A/A of a one-cell-per-arm served ratio).
   te2b_variance.py's loader and curation are reused unchanged: its replicated groups are the same host, build,
   model, backend, depth, config, prompt, ctx pin, driver and shape, seen in two or more restarts (cell records). For
   every pair of restarts of one group, d = ln(restart-mean rate i) - ln(restart-mean rate j). Two cells of one build
   are exactly an A/A block of a gate that reads one cell per arm, so the RMS of d about 0 is the A/A sd of that ratio
   (the true log ratio is 0, so the RMS keeps every degree of freedom). The pairs are grouped by separation, as
   te2b_variance.py's section 5 groups them: P same pass (same file), S same session (files < 3 h apart), O other
   session. Pairs inside a group of k > 2 restarts share restarts, so the pairs are not independent: n pairs
   overstates the df, and the table prints the groups (cells) beside them.
   Restart means average all of a restart's runs (2 or 3 runs x 8 completions in these records).

2. THE MAC'S L1 GATE 6, PASS 1 AGAINST PASS 2 (a replicate of one A/B, order reversed; not an A/A).
   cpu-decode-peer-gap-2026-09-27/arm64-speed-served.json (new 5c85f7c0 first) and -pass2.json (the labels swapped,
   so old 3cd62e6d first). new / old per model from the unrounded mean of runs.

3. --json prints the registry entries these become (origin "te3_registry_seed.py"): the rows of section 1 (goinfer
   pooled and per model with >= 3 pairs; peers pooled), the section-2 replicate, the Mac CPU order-effect floor charged
   on every Mac CPU served A/A, and goinfer / peer entries derived from each same-separation goinfer and peer pair of
   rows. A monthly refresh replaces the JSON's entries of that origin with this output; curated entries stay.
"""
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import te2b_variance as T  # noqa: E402

MEAS = T.MEAS


def restart_mean(c):
    ys = [y for r in c["runs"] for y in r]
    return sum(ys) / len(ys)


def served_pairs():
    paths, files, skipped = T.discover()
    cells, cov = T.extract(files)
    G = T.groups_of(cells)
    acc = defaultdict(list)
    for k, v in G.items():
        mu = [restart_mean(c) for c in v]
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                sep = T.pair_class(v[i], v[j]).split()[0]
                d = mu[i] - mu[j]
                c = v[0]
                ec = T.engclass(c["engine"])
                keys = [
                    (c["host"], c["backend"], ec, "*", "*", sep),
                    (c["host"], c["backend"], ec, c["model"], "d%s" % c["depth"], sep),
                    (c["host"], c["backend"], "all", "*", "*", sep),
                ]
                for key in keys:
                    acc[key].append((d, k, v[i]["path"], v[j]["path"]))
    return acc


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def pct(x):
    return 100.0 * (math.exp(abs(x)) - 1.0)


def l1_gate6():
    """The Mac's L1 gate 6: new / old per model, pass 1 (new first) and pass 2 (old first, labels swapped)."""
    import json
    d1 = json.load(open(MEAS / "cpu-decode-peer-gap-2026-09-27" / "arm64-speed-served.json"))
    d2 = json.load(open(MEAS / "cpu-decode-peer-gap-2026-09-27" / "arm64-speed-served-pass2.json"))

    def means(d):
        m = {}
        for r in d[1:]:
            if r.get("engine"):
                m[(r["engine"], r["model"])] = (sum(r["runs"]) / len(r["runs"]), r.get("mean"))
        return m

    m1, m2 = means(d1), means(d2)
    out = []
    for model in ("0.5B", "1.5B", "7B"):
        # pass 1: goinfer = new 5c85f7c0, goinfer_old = old 3cd62e6d; pass 2: the binaries swapped behind the labels
        r1 = m1[("goinfer", model)][0] / m1[("goinfer_old", model)][0]
        r2 = m2[("goinfer_old", model)][0] / m2[("goinfer", model)][0]
        rr1 = m1[("goinfer", model)][1] / m1[("goinfer_old", model)][1]
        rr2 = m2[("goinfer_old", model)][1] / m2[("goinfer", model)][1]
        out.append((model, r1, r2, rr1, rr2))
    return out


SEED = "docs/measurements/test-efficiency-2026-09/te3_registry_seed.py"
SEP_WORDS = {"P": "same pass (one file)", "S": "same session (different passes, files < 3 h apart)",
             "O": "other session (files >= 3 h apart)"}


def registry_entries(acc):
    """The computed registry entries (origin: this script). noise-registry.json carries them verbatim."""
    ents = []
    for key in sorted(acc, key=lambda k: tuple(str(x) for x in k)):
        host, be, eng, model, depth, sep = key
        lst = acc[key]
        if eng == "all":
            continue
        if eng == "goinfer" and model != "*" and len(lst) < 3:
            continue
        if eng == "peer" and model != "*":
            continue
        ds = [d for d, _, _, _ in lst]
        files = sorted({p for _, _, p1, p2 in lst for p in (p1, p2)})
        inst = "served-decode" if eng == "goinfer" and sep != "O" else (
            "served-decode-xsession" if sep == "O" else "served-decode-peer-arm")
        ents.append({
            "id": "seed-%s-%s-%s-%s-%s" % (host, be, eng, model.replace("*", "all"), sep),
            "instrument": inst,
            "cell": {"machine": host, "backend": be, "model": model, "depth": depth},
            "engines": "%s / the same %s build" % (eng, eng),
            "statistic": "RMS about 0 of ln(ratio of two restart means) over same-build %s restart pairs, %s"
                         % (eng, SEP_WORDS[sep]),
            "estimator": "rms_log_about_0",
            "values": [round(math.exp(d), 6) for d in ds],
            "sd": round(rms(ds), 5),
            "sd_kind": "paired_log_ratio",
            "floor": 0.0,
            "n": len(lst),
            "groups": len({k for _, k, _, _ in lst}),
            "unit": "one bench_peer.py cell per arm (a restart: 2-3 runs x 8 completions)",
            "scope": SEP_WORDS[sep],
            "provenance": "direct-aa",
            "source": [{"path": SEED, "note": "section 1, via te2b_variance.py's loader and curation"}]
                      + [{"path": "docs/measurements/" + f} for f in files],
            "origin": "te3_registry_seed.py",
        })
    rows = l1_gate6()
    dls = [math.log(r2) - math.log(r1) for _, r1, r2, _, _ in rows]
    ents.append({
        "id": "seed-mac-cpu-l1gate6-replicate",
        "instrument": "served-decode",
        "cell": {"machine": "mac", "backend": "cpu", "model": "*", "depth": "d128"},
        "engines": "goinfer 5c85f7c0 / goinfer 3cd62e6d (a real A/B, measured twice)",
        "statistic": "sd of one pass's ln(new/old) from two order-reversed passes of the same A/B: "
                     "sqrt(mean(dln^2)/2) over 3 models; floor = mean half-difference (the order-effect point estimate)",
        "estimator": "pair_diff",
        "values": [[round(r1, 6), round(r2, 6)] for _, r1, r2, _, _ in rows],
        "sd": round(math.sqrt(sum(x * x for x in dls) / (2 * len(dls))), 5),
        "sd_kind": "paired_log_ratio",
        "floor": round(sum(dls) / (2 * len(dls)), 5),
        "n": len(rows),
        "unit": "one bench_peer.py pass: one cell per arm, 3 runs x 8 completions, Ollama between the arms",
        "scope": "same session (2026-09-28 by day, BENCH_MAX_LOADAVG=2.5), order reversed between the passes",
        "provenance": "replicate",
        "source": [{"path": SEED, "note": "section 2"},
                   {"path": "docs/measurements/cpu-decode-peer-gap-2026-09-27/arm64-speed-served.json"},
                   {"path": "docs/measurements/cpu-decode-peer-gap-2026-09-27/arm64-speed-served-pass2.json"},
                   {"path": "docs/tasks/task-cpu-decode-peer-gap-2026-09.md", "line": 417,
                    "quote": "moved each ratio by at most 0.022, so the effect is not drift"}],
        "origin": "te3_registry_seed.py",
    })
    # The Mac CPU order effect (a fixed-order design's non-shrinking bias) is charged on every Mac CPU served A/A.
    by_id = {e["id"]: e for e in ents}
    order = by_id["seed-mac-cpu-l1gate6-replicate"]["floor"]
    for e in ents:
        if e["instrument"] == "served-decode" and e["cell"]["machine"] == "mac" and e["cell"]["backend"] == "cpu" \
                and e["provenance"] == "direct-aa":
            e["floor"] = order
            e["floor_source"] = ("seed-mac-cpu-l1gate6-replicate (order-effect point estimate; a counterbalanced "
                                 "design may pass --floor 0)")
    # goinfer / peer ratios, derived from the two same-build A/As of one host, backend and separation.
    for host, be, sep in (("nobara", "cuda", "S"), ("nobara", "cpu", "S"), ("mac", "metal", "P"), ("mac", "cpu", "S")):
        g = by_id.get("seed-%s-%s-goinfer-all-%s" % (host, be, sep))
        p = by_id.get("seed-%s-%s-peer-all-%s" % (host, be, sep))
        if not g or not p:
            continue
        ents.append({
            "id": "derived-%s-%s-goinfer-vs-peer-%s" % (host, be, sep),
            "instrument": "served-decode-peer",
            "cell": {"machine": host, "backend": be, "model": "*", "depth": "*"},
            "engines": "goinfer / a peer (Ollama or llama-server)",
            "statistic": "sqrt((g^2 + p^2) / 2): one goinfer cell's sd (g/sqrt 2) and one peer cell's (p/sqrt 2) "
                         "combined, from the same-build A/A RMS values g = %.5f (%s) and p = %.5f (%s)"
                         % (g["sd"], g["id"], p["sd"], p["id"]),
            "estimator": "quoted", "sd": round(math.sqrt((g["sd"] ** 2 + p["sd"] ** 2) / 2), 5),
            "sd_kind": "paired_log_ratio", "floor": g.get("floor", 0.0), "n": min(g["n"], p["n"]),
            "unit": g["unit"], "scope": g["scope"], "provenance": "model-inference",
            "source": [{"path": SEED, "note": "derived from the two input entries"}],
            "origin": "te3_registry_seed.py",
        })
    return ents


def main():
    if "--json" in sys.argv[1:]:
        import json
        print(json.dumps(registry_entries(served_pairs()), indent=1))
        return
    print("## 1. served decode: same-build restart pairs, d = ln(restart mean i / restart mean j)\n")
    print("RMS(d) about 0 is the A/A sd of a one-cell-per-arm ratio at that separation. 'ratios' are exp(d), the")
    print("values the registry carries (rounded to 6 places).\n")
    acc = served_pairs()
    hdr = "  %-7s %-7s %-8s %-10s %-6s %-4s %6s %6s %9s %9s %9s %8s" % (
        "host", "backend", "engines", "model", "depth", "sep", "pairs", "cells", "RMS(d)%", "med|d|%", "p90|d|%", "max|d|%")
    print(hdr)
    for key in sorted(acc, key=lambda k: tuple(str(x) for x in k)):
        lst = acc[key]
        ds = [d for d, _, _, _ in lst]
        a = sorted(abs(d) for d in ds)
        print("  %-7s %-7s %-8s %-10s %-6s %-4s %6d %6d %9.3f %9.3f %9.3f %8.3f" % (
            key[0], key[1], key[2], key[3], key[4], key[5], len(lst), len({k for _, k, _, _ in lst}),
            100 * rms(ds), pct(a[len(a) // 2]), pct(a[min(len(a) - 1, int(0.9 * len(a)))]), pct(a[-1])))
    for key in (("nobara", "cuda", "all", "*", "*", "O"), ("nobara", "cpu", "all", "*", "*", "O"),
                ("mac", "metal", "all", "*", "*", "O")):
        a = sorted(abs(d) for d, _, _, _ in acc[key])
        big = sum(1 for x in a if x >= math.log(1.035))
        print("  %s/%s other-session pairs: %d of %d reach 3.5%%; p95 %.2f%%, p99 %.2f%%" % (
            key[0], key[1], big, len(a), pct(a[min(len(a) - 1, int(0.95 * len(a)))]),
            pct(a[min(len(a) - 1, int(0.99 * len(a)))])))

    print("\n## 2. Mac L1 gate 6: new / old, pass 1 (new first) vs pass 2 (old first)\n")
    rows = l1_gate6()
    dls = []
    print("  %-5s %9s %9s %11s %14s %9s %10s" % ("model", "pass1", "pass2", "|diff| abs", "(rounded means)", "dln",
                                                "dln / 2"))
    for model, r1, r2, rr1, rr2 in rows:
        dl = math.log(r2) - math.log(r1)
        dls.append(dl)
        print("  %-5s %9.4f %9.4f %11.4f %14.4f %9.4f %10.4f" % (model, r1, r2, abs(r2 - r1), abs(rr2 - rr1), dl, dl / 2))
    s = math.sqrt(sum(x * x for x in dls) / (2 * len(dls)))
    print("\n  pass 2 (old first) reads new/old higher in %d of 3 models" % sum(1 for x in dls if x > 0))
    print("  all-noise reading: sd of one pass's ln(new/old) = sqrt(mean(dln^2)/2) = %.4f (3 df)" % s)
    print("  order-effect point estimate, mean(dln)/2 = %.4f; largest dln/2 = %.4f" % (
        sum(dls) / (2 * len(dls)), max(abs(x) / 2 for x in dls)))
    print("\n## 3. the computed registry entries: rerun with --json")


if __name__ == "__main__":
    main()
