#!/usr/bin/env python3
"""Replay a recorded Claude Code session through a cold goinfer, twice (A A), and report per request what was reused and what served it (README.md here).

  python3 replay_cc_live.py --serve <bin> --model <path> --bodies <dir of NNNN.json, outside the repository> --out <dir> [--ctx N] [--max-new N]

Writes <out>/requests.jsonl (numbers only: no prompt text) and prints the table and the rule's verdict. Exit 0 no MISS, 1 a MISS, 2 VOID.
Reuses KV8's replay harness (Server, replay, parse). lcp is a lower bound: the tokens of the leading system blocks, tools and whole messages the request
shares with its best earlier sharer, counted by the server's own count_tokens.
"""
import argparse, glob, importlib.util, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rp", os.path.join(HERE, "../kv8-msgboundary-2026-11/replay_kv0h.py"))
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

MISS_MIN_LCP, SHORT_GAP = 2048, 1024


def units(body):
    sysb = body.get("system")
    sysb = sysb if isinstance(sysb, list) else ([{"text": sysb}] if sysb else [])
    return [("S", json.dumps(s, sort_keys=True)) for s in sysb] + [("T", json.dumps(body.get("tools"), sort_keys=True))] + [("M", json.dumps(m, sort_keys=True)) for m in body["messages"]]


def count(model, system, tools, messages):
    body = {"model": model, "messages": messages}
    if system:
        body["system"] = system
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(rp.BASE + "/v1/messages/count_tokens", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))["input_tokens"]


def lcp_bounds(bodies, model):
    """For request i: (k, j, tokens): the units it shares with its best earlier request j, and the tokens of those units."""
    us = [units(b) for b in bodies]
    dummy = [{"role": "user", "content": "x"}]
    base = count(model, None, None, dummy)
    out = []
    for i, u in enumerate(us):
        best_k, best_j = 0, -1
        for j in range(i):
            k = 0
            while k < min(len(u), len(us[j])) and u[k] == us[j][k]:
                k += 1
            if k > best_k:
                best_k, best_j = k, j
        b = bodies[i]
        sysb = b.get("system")
        sysb = sysb if isinstance(sysb, list) else ([{"type": "text", "text": sysb}] if sysb else [])
        nsys = len(sysb)
        tok = 0
        if best_k:
            if best_k <= nsys:
                tok = count(model, sysb[:best_k], None, dummy) - base
            elif best_k == nsys + 1:
                tok = count(model, sysb, b.get("tools"), dummy) - base
            else:
                tok = count(model, sysb, b.get("tools"), b["messages"][:best_k - nsys - 1])
        out.append({"shared_units": best_k, "of_units": len(u), "sharer": best_j + 1 if best_j >= 0 else None, "lcp_lb": tok})
    return out


def one_pass(label, a, files):
    load, avail = rp.readings()
    rec = {"cell": label, "load": load, "avail_mb": avail, "void": load > rp.LOAD_VOID}
    print(f"[{time.strftime('%H:%M:%S')}] pass {label}: load {load:.2f}, {avail} MB available", flush=True)
    if rec["void"]:
        return rec, []
    log, ids = os.path.join(a.out, f"serve-{label}.log"), os.path.join(a.out, f"tokenids-{label}.jsonl")
    s = rp.Server(a.serve, a.model, a.ctx, log, ids)
    try:
        for i, f in enumerate(files):
            rp.replay(s.model, json.load(open(f)), a.max_new)
    finally:
        rec["peak_swap_mb"] = s.peak_swap
        s.close()
    reqs = rp.parse(log, ids)
    assert len(reqs) == len(files), (len(reqs), len(files))
    for i, r in enumerate(reqs):
        r.update(cell=label, idx=i + 1)
    return rec, reqs


def served_by(r, takes):
    """What served request r: a host snapshot (by the request that took it), the device checkpoint, the slot's own rows, or nothing."""
    if r["restored_rows"]:
        who = [f"request {i} (its snapshot {n + 1})" for i, n, rows in takes if rows == r["restored_rows"]]
        return f"snapshot of {r['restored_rows']} rows, taken by " + (who[0] if who else "an earlier request")
    if r["ckpt"]:
        return f"device checkpoint of {r['ckpt']} rows"
    return "the slot's own rows" if r["reused"] else "none"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--serve", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--bodies", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ctx", type=int, default=24576)
    p.add_argument("--max-new", type=int, default=8)
    p.add_argument("--live-log", help="the recording server's log, read as a pass L (its replies were the real ones)")
    p.add_argument("--live-ids")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.bodies, "*.json")))
    bodies = [json.load(open(f))["body"] for f in files]
    allreqs, cells = {}, []
    if a.live_log:
        allreqs["L"] = rp.parse(a.live_log, a.live_ids)
        for i, r in enumerate(allreqs["L"]):
            r.update(cell="L", idx=i + 1)
    for label in ("A1", "A2"):
        rec, reqs = one_pass(label, a, files)
        if rec["void"]:
            print(f"VOID pass {label}; re-run once"); time.sleep(60)
            rec, reqs = one_pass(label + "r", a, files)
            if rec["void"]:
                print("VERDICT: VOID"); return 2
        cells.append(rec); allreqs[label] = reqs
    # lcp bounds, on a fresh server (count_tokens, no generation)
    s = rp.Server(a.serve, a.model, a.ctx, os.path.join(a.out, "serve-lcp.log"), os.path.join(a.out, "tokenids-lcp.jsonl"))
    try:
        lcps = lcp_bounds(bodies, s.model)
    finally:
        s.close()
    verdict, notes = "PASS", []
    with open(os.path.join(a.out, "requests.jsonl"), "w") as f:
        for c in cells:
            f.write(json.dumps(c) + "\n")
        for label, reqs in allreqs.items():
            takes = []
            print(f"\n== pass {label}")
            print(f"{'#':>2} {'ids':>6} {'reused':>6} {'lcp>=':>6} {'first token':>11}  served by")
            for r, l in zip(reqs, lcps):
                sb = served_by(r, takes)
                gap = (l["lcp_lb"] // 64 * 64) - r["reused"]
                tag = ""
                if r["reused"] == 0 and l["lcp_lb"] >= MISS_MIN_LCP and l["sharer"]:
                    tag, verdict = "  MISS", "FAIL"; notes.append(f"{label} request {r['idx']}")
                elif gap >= SHORT_GAP:
                    tag = f"  SHORT by {gap}"
                print(f"{r['idx']:>2} {r['prompt_tokens']:>6} {r['reused']:>6} {l['lcp_lb']:>6} {r['ttft_s']:>10.2f}s  {sb}{tag}")
                for n, t in enumerate(r["took"]):
                    takes.append((r["idx"], n, t["rows"]))
                    print(f"{'':>34}took a snapshot of {t['rows']} rows: {t['mib']} MiB, {t['ms']} ms")
                f.write(json.dumps({**{k: r[k] for k in ("cell", "idx", "prompt_tokens", "ttft_s", "reused", "snap", "ckpt", "restored_rows", "took")}, **l, "served_by": sb, "gap": gap}) + "\n")
    print("VERDICT:", "no MISS in either pass" if verdict == "PASS" else "MISS: " + "; ".join(notes))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
