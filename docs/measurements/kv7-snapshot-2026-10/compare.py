#!/usr/bin/env python3
"""Reads the two arms of run-tier1-*.sh (OUT/off, OUT/on): per replayed session start, the time, the rows reused and restored, whether the
reply and the generated ids are equal on and off; the snapshot counters; the server's peak VmHWM and VmSwap. Exit 1 when a reply differs.

  python3 -I compare.py OUT
"""
import base64, json, os, re, sys

PREFIX = "goinfer-reasoning-v2:"


def strip_key(v):
    """A reply without what is random per request or per server start: the response and call ids, and in a signed reasoning blob (carried
    token ids) the two fields that depend on the server's random signing key: its
    8-byte key id (bytes 2 to 10) and its 32-byte HMAC (the last 32). Everything else, the model identity, the text and the ids, stays."""
    if isinstance(v, dict):  # the response and call ids are random per request
        return {k: ("<id>" if k in ("id", "call_id") and isinstance(x, str) else strip_key(x)) for k, x in v.items()}
    if isinstance(v, list):
        return [strip_key(x) for x in v]
    if isinstance(v, str) and v.startswith(PREFIX):
        raw = base64.b64decode(v[len(PREFIX):])
        return PREFIX + base64.b64encode(raw[:2] + raw[10:-32]).decode()
    return v


def same_reply(a, b):
    """Equal replies, the signing key aside. Chat replies are strings; the Responses and Messages ones are JSON."""
    if a == b:
        return True
    try:
        return strip_key(json.loads(a)) == strip_key(json.loads(b))
    except ValueError:
        return False

LINE = re.compile(r"^request: POST (\S+) .*?prompt_tokens=(\d+) completion_tokens=(\d+) ttft=(\d+)ms total=(\d+)ms reused=(\S+) slot=(\S+) div=(\S+)(.*)$")


def arm(d):
    rep = [json.loads(l) for l in open(f"{d}/replay.jsonl")]
    reqs = []
    for l in open(f"{d}/serve.log", errors="replace"):
        m = LINE.match(l)
        if m:
            snap = re.search(r" snap=(\d+)", m[9])
            reqs.append(dict(prompt=int(m[2]), completion=int(m[3]), ttft=int(m[4]) / 1000, total=int(m[5]) / 1000, reused=m[6], snap=int(snap[1]) if snap else 0))
    ids = []
    p = f"{d}/token-ids.jsonl"
    if os.path.exists(p):
        ids = [json.loads(l).get("gen") for l in open(p)]
    proc = [l.strip().split(",") for l in open(f"{d}/proc.csv")][1:]
    swap = [int(r[3]) for r in proc if len(r) == 4]
    hwm = [int(r[1]) for r in proc if len(r) == 4]
    return rep, reqs, ids, (min(swap or [0]), max(swap or [0]), max(hwm or [0]))


def main():
    out = sys.argv[1]
    arms = {n: arm(f"{out}/{n}") for n in ("off", "on")}
    off, on = arms["off"], arms["on"]
    offn = {r["name"]: i for i, r in enumerate(off[0])}
    bad = 0
    print(f"{'session':10s} {'prompt':>6s} | {'off ttft':>8s} {'on ttft':>8s} {'on reused':>9s} {'restored':>8s} | reply  ids")
    for i, b in enumerate(on[0]):
        rb = on[1][i]
        if b["name"] not in offn:
            print(f"{b['name']:10s} {rb['prompt']:6d} | {'-':>8s} {rb['ttft']:7.1f}s {rb['reused']:>9s} {rb['snap']:8d} | (no off arm)")
            continue
        j = offn[b["name"]]
        a, ra = off[0][j], off[1][j]
        same_r = same_reply(a["reply"], b["reply"])
        same_i = (off[2][j] == on[2][i]) if j < len(off[2]) and i < len(on[2]) else None
        bad += (not same_r) or (same_i is False)
        print(f"{a['name']:10s} {rb['prompt']:6d} | {ra['ttft']:7.1f}s {rb['ttft']:7.1f}s {rb['reused']:>9s} {rb['snap']:8d} | {'equal' if same_r else 'DIFFERENT':6s} {'equal' if same_i else ('n/a' if same_i is None else 'DIFFERENT')}")
    last = on[0][-1].get("kv_snapshots")
    print("\nsnapshot counters after the last on-arm request:", json.dumps(last, sort_keys=True))
    for name, (rep, reqs, ids, (s0, s1, hwm)) in (("off", off), ("on", on)):
        print(f"{name}: VmSwap {s0} kB at the start, peak {s1} kB; peak VmHWM {hwm / 1024:.0f} MiB")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
