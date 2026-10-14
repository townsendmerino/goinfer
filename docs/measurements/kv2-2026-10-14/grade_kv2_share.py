#!/usr/bin/env python3
"""Grade KV2's G5 (docs/measurements/kv2-2026-10-14.md §2, G5) from kv2_share.py's JSON, by the registered rules.

  grade_kv2_share.py share-<backend>-<name>.json [--graded]   # --graded: the Mac's cell, which carries the 5 s bar

busy: per block s = B1(main) - B1(kv2) and p = B1(main) * hit / B1's prompt tokens (hit: kv2's reused rows); the value
is the median of s / p. DELIVERS [0.80, 1.05]; UNDER [0.50, 0.80); FAILS < 0.50; HELD > 1.05. On the graded cell KV2
goes ahead only when it DELIVERS and the median saving is at least 5 s.
idle: per block r = (B1 + A2)(main) / (B1 + A2)(kv2). PASS: median >= 1.00. FAIL: median < 0.97 and every block < 1.
PARKED otherwise.
Void: an errored cell, a missing arm, A1 ending before B1 was sent, a control outside its band, kv2's busy B1 not
reusing whole blocks of most of its prompt, main or off reusing a block or more, B1's reply differing between arms.
Exit 0: DELIVERS (and, graded, >= 5 s) and idle PASS. 1: any other graded outcome. 2: VOID.
"""
import json, statistics, sys

REPORT_ONLY_SCRIPTS = set()


def main():
    path, graded = sys.argv[1], "--graded" in sys.argv[2:]
    d = json.load(open(path))
    arms = d["args"]["arms"].split(",")
    by, void = {}, []
    for c in d["cells"]:
        if c.get("error"):
            void.append(f"block {c['block']} {c['script']} {c['arm']} errored: {c['error']}")
            continue
        by[(c["script"], c["block"], c["arm"])] = c
    blocks = range(d["args"]["blocks"])
    scripts = d["args"]["scripts"].split(",")

    def get(script, b, arm):
        c = by.get((script, b, arm))
        if c is None:
            void.append(f"block {b} {script}: no {arm} arm")
        return c

    lines, outcome = [], {}
    # identity, free with the run: B1's reply is the same text in every arm
    for script in [s for s in ("busy", "idle") if s in scripts]:
        for b in blocks:
            texts = {arm: by[(script, b, arm)]["b1"]["text"] for arm in arms if (script, b, arm) in by}
            if len(set(texts.values())) > 1:
                void.append(f"block {b} {script}: B1's reply differs between arms: {texts}")
    if "busy" in scripts and "main" in arms and "kv2" in arms:
        ratios, savings, hits, offr = [], [], set(), []
        for b in blocks:
            m, k = get("busy", b, "main"), get("busy", b, "kv2")
            if not m or not k:
                continue
            for c in (m, k):
                if c.get("a1_ended_before_b1_sent"):
                    void.append(f"block {b} busy {c['arm']}: A1 ended before B1 was sent")
            hit, n = k["b1"]["reused"] or 0, k["b1"]["prompt_tokens"] or 0
            hits.add(hit)
            if hit % 64 or hit < 0.85 * n:
                void.append(f"block {b} busy kv2: B1 reused {hit} of {n} tokens, not whole blocks of most of it")
            for c in (m,) + ((get("busy", b, "off"),) if "off" in arms else ()):
                if c and (c["b1"]["reused"] or 0) >= 64:
                    void.append(f"block {b} busy {c['arm']}: B1 reused {c['b1']['reused']} tokens, a block or more")
            s = m["b1"]["ttft"] - k["b1"]["ttft"]
            p = m["b1"]["ttft"] * hit / n if n else 0
            savings.append(s)
            ratios.append(s / p if p else float("nan"))
            if "off" in arms and (o := by.get(("busy", b, "off"))):
                offr.append(o["b1"]["ttft"] / m["b1"]["ttft"])
        if ratios:
            med, sav = statistics.median(ratios), statistics.median(savings)
            v = "DELIVERS" if 0.80 <= med <= 1.05 else "UNDER" if med >= 0.50 and med < 0.80 else "FAILS" if med < 0.50 else "HELD"
            if graded and v == "DELIVERS" and sav < 5:
                v = "DELIVERS, under the 5 s bar"
            outcome["busy"] = v
            lines.append(f"{v}: busy, median saving / projection {med:.3f} over {len(ratios)} blocks; median saving {sav:.2f} s; "
                         f"hits {sorted(hits)}; by block " + " ".join(f"{r:.3f}" for r in ratios))
            ms = [m["b1"]["ttft"] for b in blocks if (m := by.get(("busy", b, "main")))]
            ks = [k["b1"]["ttft"] for b in blocks if (k := by.get(("busy", b, "kv2")))]
            lines.append(f"       B1 time to first token: main median {statistics.median(ms):.2f} s, kv2 median {statistics.median(ks):.2f} s")
        if offr:
            mo = statistics.median(offr)
            lines.append(f"control off / main (busy): median {mo:.3f}")
            if not 0.95 <= mo <= 1.05:
                void.append(f"control: off / main median {mo:.3f} is outside [0.95, 1.05]")
    if "control" in scripts and "main" in arms and "kv2" in arms:
        cr = []
        for b in blocks:
            m, k = get("control", b, "main"), get("control", b, "kv2")
            if not m or not k:
                continue
            if (k["b1"]["reused"] or 0) >= 64:
                void.append(f"block {b} control: kv2's B1 reused {k['b1']['reused']} tokens with the header on")
            cr.append(k["b1"]["ttft"] / m["b1"]["ttft"])
        if cr:
            mc = statistics.median(cr)
            lines.append(f"control header on, kv2 / main: median {mc:.3f}")
            if not 0.95 <= mc <= 1.05:
                void.append(f"control: header-on kv2 / main median {mc:.3f} is outside [0.95, 1.05]")
    if "idle" in scripts and "main" in arms and "kv2" in arms:
        rs, b1r = [], []
        for b in blocks:
            m, k = get("idle", b, "main"), get("idle", b, "kv2")
            if not m or not k:
                continue
            rs.append((m["b1"]["ttft"] + m["a2"]["ttft"]) / (k["b1"]["ttft"] + k["a2"]["ttft"]))
            b1r.append(k["b1"]["ttft"] / m["b1"]["ttft"])
        if rs:
            med = statistics.median(rs)
            v = "PASS" if med >= 1.00 else "FAIL" if med < 0.97 and all(r < 1 for r in rs) else "PARKED"
            outcome["idle"] = v
            lines.append(f"{v}: idle, (B1 + A2) main / kv2 median {med:.3f}; by block " + " ".join(f"{r:.3f}" for r in rs))
            lines.append(f"       idle B1 alone, kv2 / main: median {statistics.median(b1r):.3f} (the copy replaces an in-place take)")
    cps = [ms for c in d["cells"] if c.get("arm") == "kv2" for _, ms in c.get("copies", [])]
    rows = sorted({r for c in d["cells"] if c.get("arm") == "kv2" for r, _ in c.get("copies", [])})
    decl = [x for c in d["cells"] for x in c.get("declines", [])]
    if cps:
        lines.append(f"copy cost (G7, reported): median {statistics.median(cps):.2f} ms over {len(cps)} copies of {rows} rows")
    if decl:
        lines.append(f"declined copies: {decl}")
    print(f"{d['args']['backend']} {d['args']['name']}")
    for l in lines:
        print(l)
    for v in void:
        print("VOID: " + v)
    if void:
        print("VERDICT VOID")
        return 2
    ok = outcome.get("busy") == "DELIVERS" and outcome.get("idle", "PASS") == "PASS"
    print("VERDICT " + (" / ".join(f"{k} {v}" for k, v in outcome.items()) or "nothing graded"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
