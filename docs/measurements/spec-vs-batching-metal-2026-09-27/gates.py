#!/usr/bin/env python3
"""MC4's spec trigger on Metal (docs/tasks/task-concurrency-2026-09.md MC4, registered in b11bcfc3): S, L and identity
from bench_spec_copy.py's and bench_w7_plain.py's JSON. Keys batchN_i / specN_i, N clients, pair i."""
import json, statistics, sys


def requests(cell):
    """Per-request records: the copy bench's per_client lists, or W7's per_client turns."""
    out = []
    for c in cell["per_client"]:
        turns = c["turns"] if isinstance(c, dict) and "turns" in c else c
        out.append([(t.get("content_sha"), t.get("prompt_tokens"), t.get("prefill_reused_tokens"),
                     t.get("latency_s", (t.get("latency_ms") or 0) / 1000)) for t in turns])
    return out


def p99(xs):
    xs = sorted(xs)
    return xs[max(0, int(round(0.99 * len(xs))) - 1)]


def main(path, name):
    R = json.load(open(path))["results"]
    # W7's JSON nests each key's result under its client count ({"4": {...}}); the copy bench's does not.
    R = {k: (next(iter(v.values())) if isinstance(v, dict) and "aggregate_tok_s" not in v and v else v)
         for k, v in R.items() if v}
    for n in (4, 1):
        agg, lat, same, total, reuse_diff = [], [], 0, 0, 0
        for i in (1, 2, 3):
            b, s = R.get(f"batch{n}_{i}"), R.get(f"spec{n}_{i}")
            if not (b and s):
                print(f"{name} {n} clients pair {i}: missing ({'batch' if not b else 'spec'})")
                continue
            agg.append(s["aggregate_tok_s"] / b["aggregate_tok_s"])
            rb, rs = requests(b), requests(s)
            lat.append(p99([x[3] for c in rs for x in c]) / p99([x[3] for c in rb for x in c]))
            for cb, cs in zip(rb, rs):
                for xb, xs in zip(cb, cs):
                    total += 1
                    same += xb[0] == xs[0]
                    if xb[2] is not None and (xb[1] - xb[2]) != (xs[1] - xs[2]):
                        reuse_diff += 1
            print(f"{name} {n} clients pair {i}: aggregate batch {b['aggregate_tok_s']:.1f} spec {s['aggregate_tok_s']:.1f} "
                  f"= {agg[-1]:.3f}x; p99 request spec/batch {lat[-1]:.3f}x")
        if agg:
            m = "S" if n == 1 else "L"
            print(f"{name} {m} ({n} clients) median {statistics.median(agg):.3f}x; p99 median {statistics.median(lat):.3f}x; "
                  f"identity {same}/{total} replies equal; prefilled-token count differs on {reuse_diff}")


if __name__ == "__main__":
    for p, nm in zip(sys.argv[1::2], sys.argv[2::2]):
        main(p, nm)
