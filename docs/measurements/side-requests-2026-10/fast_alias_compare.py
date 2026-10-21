#!/usr/bin/env python3
"""Tier 1 of the fast alias (side-requests-2026-10-16.md, "Tier 1"): the same opencode tasks without and with it, read from two
matrix.py output directories. Per arm, every request that carried no tools (the side requests) with the model name it named, its
prompt and generated tokens and its times; every run's first main request and its time to first token; and each run's pass/fail and
wall time. Reads only the directories; no server, no GPU.

  python3 -I fast_alias_compare.py OFF_DIR OFF_CONFIG ON_DIR ON_CONFIG
"""
import gzip, json, os, re, sys

RE = re.compile(r"^request: POST (\S+) .*?prompt_tokens=(\d+) completion_tokens=(\d+) ttft=(\d+)ms total=(\d+)ms reused=(\S+) slot=(\S+) div=(\S+)")


def arm(d, cfg):
    wire = [json.loads(l) for l in open(f"{d}/wire-{cfg}.jsonl")]
    log = [RE.match(l) for l in open(f"{d}/logs/serve-{cfg}.log", errors="replace") if l.startswith("request:")]
    log = [m for m in log if m]
    if len(wire) != len(log):
        sys.exit(f"{d}: {len(wire)} wire entries but {len(log)} request lines")
    rows = []
    for i, (w, m) in enumerate(zip(wire, log), 1):
        body = json.load(gzip.open(f"{d}/bodies-{cfg}/{w['bodies']}.req.json.gz"))
        msgs = body.get("messages", [])
        sysp = str(msgs[0].get("content", ""))[:60].replace("\n", " ") if msgs else ""
        rows.append(dict(n=i, model=w["model"], tools=w["tools"], messages=len(msgs), system=sysp, prompt=int(m[2]), completion=int(m[3]),
                         ttft=int(m[4]) / 1000, total=int(m[5]) / 1000, reused=m[6], t=w["t"], max_out=w.get("max_out")))
    recs = [json.loads(l) for l in open(f"{d}/records.jsonl")]
    return rows, recs


def show(name, d, cfg):
    rows, recs = arm(d, cfg)
    t0 = rows[0]["t"]
    print(f"\n== {name}: {len(rows)} requests")
    side = [r for r in rows if r["tools"] == 0]
    print("side requests (no tools):")
    print("   #  model                       prompt  generated  ttft(s)  total(s)  what")
    for r in side:
        print(f"  {r['n']:2d}  {r['model']:26s} {r['prompt']:6d}  {r['completion']:9d}  {r['ttft']:7.1f}  {r['total']:8.1f}  {r['system']}")
    print(f"  side requests: {len(side)}, {sum(r['total'] for r in side):.1f} s of the GPU, {sum(r['completion'] for r in side)} generated tokens")
    firsts = [r for r in rows if r["tools"] > 0 and r["messages"] <= 2]
    print("each run's first main request:")
    for r in firsts:
        print(f"  #{r['n']:2d}  {r['model']:26s} prompt {r['prompt']:5d}  time to first token {r['ttft']:6.1f} s  reused {r['reused']}")
    span = rows[-1]["t"] + rows[-1]["total"] - t0
    print(f"first request to last response: {span:.0f} s")
    print("runs:")
    for r in recs:
        print(f"  {r['task']:7s} passed={r['passed']}  wall {r['wall_seconds']:.0f} s  turns {r['turns']}  tool_calls {r['tool_calls']}  compactions {r['compactions']}")
    return rows, recs


if __name__ == "__main__":
    off_d, off_c, on_d, on_c = sys.argv[1:5]
    show("alias off (the 2026-10-15 run)", off_d, off_c)
    show("alias on", on_d, on_c)
