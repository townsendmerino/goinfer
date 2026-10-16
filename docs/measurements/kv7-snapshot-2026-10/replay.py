#!/usr/bin/env python3
"""Tier 1 of the prefix snapshot (docs/tasks/task-kv7-prefix-snapshot-2026-10.md): replay recorded session-start requests, in order, against
one running serve, and record each request's reply, usage and wall time, serve's snapshot counters after it, and the server process's
VmHWM and VmSwap once a second. The bodies are the bytes a harness sent; only a few top-level fields are rewritten, by splicing, because a
re-serialized body renders another prompt (serve renders a tool's parameters verbatim).

  python3 -I replay.py --base http://127.0.0.1:18090 --pid PID --out DIR --max-tokens 24 NAME=ROUTE=BODY.json[.gz] ...

ROUTE is chat (/v1/chat/completions), responses (/v1/responses) or messages (/v1/messages). A trailing @MODEL names the model for that session.
"""
import argparse, gzip, json, os, sys, threading, time, urllib.request
from json.decoder import scanstring, JSONDecoder

ROUTES = {"chat": "/v1/chat/completions", "responses": "/v1/responses", "messages": "/v1/messages"}
MAXKEY = {"chat": "max_tokens", "responses": "max_output_tokens", "messages": "max_tokens"}


def top_level(raw):
    """[(key, value_start, value_end)] of the top-level members of the JSON object raw (a str), and the offset of its closing brace."""
    dec = JSONDecoder()
    i = raw.index("{") + 1
    out = []
    while True:
        while raw[i] in " \t\r\n,":
            i += 1
        if raw[i] == "}":
            return out, i
        key, i = scanstring(raw, i + 1, True)
        while raw[i] in " \t\r\n:":
            i += 1
        _, end = dec.raw_decode(raw, i)
        out.append((key, i, end))
        i = end


def splice(raw, set_):
    """raw with each top-level key in set_ (key -> JSON literal) replaced, or appended when absent; every other byte unchanged."""
    members, close = top_level(raw)
    out, pos, done = [], 0, set()
    for key, s, e in members:
        if key in set_:
            out.append(raw[pos:s])
            out.append(set_[key])
            pos = e
            done.add(key)
    out.append(raw[pos:close])
    have = bool(members)
    for key, lit in set_.items():
        if key not in done:
            out.append(("," if have else "") + json.dumps(key) + ":" + lit)
            have = True
    out.append(raw[close:])
    return "".join(out)


def read_body(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return f.read()


def post(url, body, timeout):
    req = urllib.request.Request(url, data=body.encode(), headers={"content-type": "application/json", "authorization": "Bearer goinfer"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return time.time() - t0, json.loads(raw)


def reply_text(route, resp):
    if route == "chat":
        m = resp["choices"][0]["message"]
        return (m.get("reasoning_content") or "") + "␞" + (m.get("content") or "") + "␞" + json.dumps(m.get("tool_calls"), sort_keys=True)
    if route == "responses":
        return json.dumps(resp["output"], sort_keys=True)
    return json.dumps(resp["content"], sort_keys=True)


def sampler(pid, path, stop):
    with open(path, "w") as f:
        f.write("t,VmHWM_kB,VmRSS_kB,VmSwap_kB\n")
        while not stop.is_set():
            vals = {}
            try:
                for line in open(f"/proc/{pid}/status"):
                    k, _, v = line.partition(":")
                    if k in ("VmHWM", "VmRSS", "VmSwap"):
                        vals[k] = int(v.split()[0])
            except OSError:
                break
            f.write(f"{time.time():.1f},{vals.get('VmHWM',0)},{vals.get('VmRSS',0)},{vals.get('VmSwap',0)}\n")
            f.flush()
            stop.wait(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="a top-level field to rewrite in every body (e.g. model='\"qwen3-4b\"')")
    ap.add_argument("sessions", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stop = threading.Event()
    th = threading.Thread(target=sampler, args=(a.pid, os.path.join(a.out, "proc.csv"), stop), daemon=True)
    th.start()
    log = open(os.path.join(a.out, "replay.jsonl"), "w")
    for spec in a.sessions:
        name, route, path = spec.split("=", 2)
        model = None
        if "@" in path:  # NAME=ROUTE=PATH@MODEL: this session names another model (a client's small-model request names the alias)
            path, model = path.rsplit("@", 1)
        raw = read_body(path)
        extra = dict(kv.split("=", 1) for kv in a.set)
        body = splice(raw, {"stream": "false", "temperature": "0", MAXKEY[route]: str(a.max_tokens)} | ({"store": "false"} if route == "responses" else {}) | extra | ({"model": json.dumps(model)} if model else {}))
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {len(body)} bytes to {ROUTES[route]}", flush=True)
        wall, resp = post(a.base + ROUTES[route], body, a.timeout)
        try:
            with urllib.request.urlopen(urllib.request.Request(a.base + "/admin/status", headers={"authorization": "Bearer goinfer"}), timeout=10) as r:
                snaps = json.loads(r.read()).get("kv_snapshots")
        except Exception as e:  # the status route may need an admin key; the serve log carries the same counters
            snaps = {"error": str(e)}
        rec = {"name": name, "route": route, "started": t0, "wall_s": round(wall, 2), "reply": reply_text(route, resp), "usage": resp.get("usage"), "kv_snapshots": snaps}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {wall:.1f} s", flush=True)
    stop.set()
    th.join(timeout=3)


if __name__ == "__main__":
    main()
