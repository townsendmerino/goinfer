#!/usr/bin/env python3
"""Builds testdata/ids-template/cases.json.gz: G-ids-3's three `decode` turns (ids-carried-2026-10-15.md, "G-ids-3: graded"),
each as the request that generated it, the reply's emitted items, the next request (which carried the reply), and the reply's
generated ids and its prompt's sha256 from serve's --log-token-ids. Reads only a G-ids-3 output directory. The bodies are
kept as the bytes Codex sent: serve renders each tool's parameters verbatim, so a re-serialized body is another prompt.

  python3 -I build_template_fixture.py ~/goinfer-bench/ids3-2026-10/ids3-out testdata/ids-template/cases.json.gz
"""
import gzip, json, sys

CASES = [("ids3-on-edit-r2", 17), ("ids3-on-write-r2", 6), ("ids3-on-edit-r3", 22)]  # (config, the reply's request number)


def raw(out, cfg, n):
    return gzip.open(f"{out}/bodies-{cfg}/{n:05d}.req.json.gz", "rt").read()


def items(out, cfg, n):
    got = []
    for line in gzip.open(f"{out}/bodies-{cfg}/{n:05d}.resp.gz", "rt"):
        if line.startswith("data:"):
            try:
                ev = json.loads(line[5:])
            except ValueError:
                continue
            if ev.get("type") == "response.output_item.done":
                got.append(ev["item"])
    return got


def main():
    out, dst = sys.argv[1], sys.argv[2]
    cases = []
    for cfg, n in CASES:
        prev_raw, next_raw = raw(out, cfg, n), raw(out, cfg, n + 1)
        prev, nxt = json.loads(prev_raw), json.loads(next_raw)
        rec = json.loads(open(f"{out}/logs/token-ids-{cfg}.jsonl").read().splitlines()[n - 1])
        its = items(out, cfg, n)
        k = len(prev["input"])
        assert nxt["input"][:k] == prev["input"], f"{cfg} #{n + 1}: the next request does not extend the previous one"
        echoed = nxt["input"][k:k + len(its)]
        strip = lambda it: {a: b for a, b in it.items() if a != "status"}
        assert [strip(x) for x in echoed] == [strip(x) for x in its], f"{cfg} #{n + 1}: the reply's items are not echoed unchanged"
        cases.append({"id": f"{cfg}#{n}", "prev": prev_raw, "next": next_raw, "n_prev": k, "n_items": len(its), "gen": rec["gen"],
                      "prompt_len": rec["prompt_len"], "prompt_sha256": rec["prompt_sha256"]})
    with gzip.open(dst, "wt") as f:
        json.dump(cases, f)
    print(f"{dst}: {len(cases)} cases")


if __name__ == "__main__":
    main()
