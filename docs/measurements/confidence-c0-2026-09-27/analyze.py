#!/usr/bin/env python3
"""C0 of docs/tasks/task-constrained-confidence.md: the pre-registered gates, from the harness's JSONL
(metal/confidence_c0_test.go). Written and committed before any graded run; the rules are the task doc's C0
pre-registration, restated here only as code.

usage: analyze.py <model-a.jsonl> [<model-b.jsonl> ...]
"""
import json, math, re, statistics, sys

KIND = {"category": "enum", "urgent": "boolean", "order_count": "integer", "refund_amount": "number",
        "customer_name": "string", "summary": "string (no gold)"}
GRADED = ["category", "urgent", "order_count", "refund_amount", "customer_name"]
# The pre-registered aggregation per field (primary); the others are reported as secondary.
PRIMARY = {"category": "decision", "urgent": "decision", "order_count": "min", "refund_amount": "min",
           "customer_name": "geomean"}
AGGS = ["decision", "first", "min", "geomean", "product"]


def field_values(output):
    try:
        return json.loads(output)
    except Exception:
        return None


def norm_name(s):
    return re.sub(r"[\s.,;:!]+$", "", s.strip()).lower()


def correct(field, got, gold):
    if field == "refund_amount":
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - gold) < 0.005
    if field == "customer_name":
        return isinstance(got, str) and norm_name(got) == norm_name(gold)
    if field == "urgent":
        return got is gold
    return got == gold


def aggregates(rec, field):
    free = [t["p"] for t in rec["tokens"] if t["field"] == field and not t["forced"]]
    out = {}
    if free:
        out["first"] = free[0]
        out["min"] = min(free)
        out["product"] = math.prod(free)
        out["geomean"] = math.exp(sum(math.log(max(p, 1e-300)) for p in free) / len(free))
    for d in rec["decisions"]:
        if d["field"] == field:
            vals = field_values(rec["output"]) or {}
            chosen = vals.get(field)
            key = str(chosen).lower() if field == "urgent" else chosen
            out["decision"] = d["dist"].get(key, 0.0)
    return out, len(free)


def auroc(pos, neg):
    """P(score of a correct instance > score of a wrong one), ties 1/2 (Mann-Whitney)."""
    if not pos or not neg:
        return None
    s = 0.0
    for a in pos:
        for b in neg:
            s += 1.0 if a > b else 0.5 if a == b else 0.0
    return s / (len(pos) * len(neg))


def main(paths):
    by_model = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            by_model.setdefault(r["model"], []).append(r)
    print("# C0 results\n")
    for model, recs in by_model.items():
        print(f"## {model} ({recs[0]['backend']}), {len(recs)} tickets; complete {sum(r['complete'] for r in recs)}; "
              f"mask-only pass emitted identical ids on {sum(r['same_ids_a_b'] for r in recs)}")
        # C-gate 0 and the forced-fraction table (item 1).
        print("\n| field | kind | instances | value tokens (mean) | free tokens (mean) | free fraction | instances with >= 1 free | gate 0 (>= 80%) |")
        print("|---|---|---:|---:|---:|---:|---:|---|")
        gate0 = {}
        for f, kind in KIND.items():
            ntok, nfree, withfree, n = [], [], 0, 0
            for r in recs:
                toks = [t for t in r["tokens"] if t["field"] == f]
                if not toks:
                    continue
                n += 1
                fr = sum(not t["forced"] for t in toks)
                ntok.append(len(toks)); nfree.append(fr); withfree += fr > 0
            share = withfree / n if n else 0
            gate0[f] = share >= 0.80
            frac = sum(nfree) / sum(ntok) if ntok else 0
            print(f"| {f} | {kind} | {n} | {statistics.mean(ntok) if ntok else 0:.2f} | {statistics.mean(nfree) if nfree else 0:.2f} | "
                  f"{frac:.3f} | {share:.3f} | {'PASS' if gate0[f] else 'FAIL'} |")
        # C-gate 1 (item 4).
        ro = sorted(t["readout_ns"] for r in recs for t in r["tokens"])
        tk = sorted(x for r in recs for x in r["token_ns"])
        med_ro, med_tk = ro[len(ro) // 2], tk[len(tk) // 2]
        # The gate is the MEAN share: every token pays the readout, and it costs several times more inside a free
        # string (nearly the whole vocabulary is legal there) than at a structural position, so a median would
        # understate what an output pays. The median share is reported beside it.
        share = statistics.mean(ro) / statistics.mean(tk)
        q = lambda xs, f: xs[min(len(xs) - 1, int(f * len(xs)))]
        print(f"\nC-gate 1 (cost): readout mean {statistics.mean(ro)/1e3:.1f} us, median {med_ro/1e3:.1f} (p10 {q(ro,.1)/1e3:.1f}, "
              f"p90 {q(ro,.9)/1e3:.1f}); mask-only decode token mean {statistics.mean(tk)/1e6:.2f} ms, median {med_tk/1e6:.2f} "
              f"(p10 {q(tk,.1)/1e6:.2f}, p90 {q(tk,.9)/1e6:.2f}); share of a token: mean {100*share:.2f}% (bar <= 5%): "
              f"{'PASS' if share <= 0.05 else 'FAIL'}; median-over-median {100*med_ro/med_tk:.2f}%")
        # C-gate 2 (item 2/3).
        print("\n| field | aggregation | correct | wrong | excluded | AUROC | gate 2 |")
        print("|---|---|---:|---:|---:|---:|---|")
        for f in GRADED:
            for agg in AGGS:
                pos, neg, excl = [], [], 0
                for r in recs:
                    vals = field_values(r["output"])
                    gold = json.loads(json.dumps(r["gold"]))[f]
                    a, _ = aggregates(r, f)
                    if vals is None or f not in vals or agg not in a:
                        excl += 1
                        continue
                    (pos if correct(f, vals[f], gold) else neg).append(a[agg])
                if agg != PRIMARY[f] and (not pos and not neg):
                    continue
                au = auroc(pos, neg)
                if agg == PRIMARY[f]:
                    if len(pos) < 8 or len(neg) < 8:
                        verdict = "insufficient (needs >= 8 correct and >= 8 wrong)"
                    elif au >= 0.65:
                        verdict = "PASS"
                    elif au < 0.55:
                        verdict = "FAIL"
                    else:
                        verdict = "ambiguous"
                    label = f"**{agg}** (primary)"
                else:
                    verdict, label = "secondary", agg
                print(f"| {f} | {label} | {len(pos)} | {len(neg)} | {excl} | {'n/a' if au is None else f'{au:.3f}'} | {verdict} |")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
