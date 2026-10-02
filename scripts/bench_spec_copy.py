#!/usr/bin/env python3
"""A copy-heavy serve workload for n-gram speculation, at 1..N concurrent clients (MC4's spec item,
docs/tasks/task-concurrency-2026-09.md).

W7's chat turns give a prompt-lookup drafter little to copy. Here each request hands the model a section of a Go source
file and asks for it back verbatim, the traffic `--spec ngram` exists for (code edits, RAG, agent loops). Client i sends
`--rounds` requests one after another, for sections i*rounds .. i*rounds+rounds-1, so client 0's requests are the same
at every client count. Greedy. The source is `git show <rev>:decoder/model.go`, pinned so every arm reads the same bytes.

Recorded per cell: each request's latency, prompt and completion tokens (the engine's usage, since speculation emits
bursts) and a hash of its reply; the cell's wall time, and aggregate = completion tokens / wall.

One fresh server per cell. Usage:
  GOINFER_SERVE_CPU=<serve binary> BENCH_W7_MODEL=<model> python3 scripts/bench_spec_copy.py OUT.json --key K \
      --clients 4 [--serve-args=-spec=ngram] [--server-log LOG]
"""
import argparse, hashlib, json, os, subprocess, sys, threading, time, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_w7_plain as w7  # GoinferServer, post, machine_header


def sections(rev, chars, n):
    src = subprocess.run(["git", "-C", os.path.join(HERE, ".."), "show", f"{rev}:decoder/model.go"],
                         capture_output=True, text=True, check=True).stdout
    out, at = [], 0
    for _ in range(n):
        at = src.index("\n", at) + 1  # start each section on a line boundary
        out.append(src[at:at + chars])
        at += chars
    return out


def request(text):
    return {"model": "bench", "temperature": 0, "messages": [{"role": "user", "content":
            "Here is part of a Go file:\n\n```go\n" + text + "\n```\n\n"
            "Rewrite the code above EXACTLY, character for character, with no changes and no commentary."}]}


def cell(url, a, secs):
    # Client i sends rounds[i] requests one after another, starting i*stagger_s after the cell began. With the defaults
    # (every client `--rounds`, no stagger) this is the original shape. A staggered cell (--rounds-list, --stagger-s) is
    # the one that exercises "speculate when alone, batch under load": clients join and leave at different times, so
    # there are stretches with one generation decoding and stretches with several.
    rounds = a.rounds_list or [a.rounds] * a.clients
    first = [sum(rounds[:i]) for i in range(len(rounds))]
    res = [[None] * n for n in rounds]
    t0 = time.perf_counter()

    def client(i):
        time.sleep(i * a.stagger_s)
        for r in range(rounds[i]):
            s = first[i] + r
            body = dict(request(secs[s]), max_tokens=a.max_tokens)
            start = time.perf_counter() - t0
            try:
                lat, resp = w7.post(url, body, timeout=600)
            except urllib.error.HTTPError as e:  # e.g. serve's 413 prefill-memory refusal: recorded, not fatal
                res[i][r] = {"section": s, "http_error": e.code, "completion_tokens": 0}
                continue
            u = resp.get("usage") or {}
            content = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            res[i][r] = {"section": s, "start_s": round(start, 3), "latency_s": round(lat, 3),
                         "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                         "content_sha": hashlib.sha256(content.encode()).hexdigest()[:16]}

    ths = [threading.Thread(target=client, args=(i,)) for i in range(len(rounds))]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    wall = time.perf_counter() - t0
    toks = sum(x["completion_tokens"] or 0 for c in res for x in c)
    errs = [x["http_error"] for c in res for x in c if "http_error" in x]
    return {"clients": len(rounds), "wall_s": round(wall, 3), "completion_tokens": toks, "http_errors": errs,
            "aggregate_tok_s": round(toks / wall, 3), "per_client": res}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--key", required=True)
    ap.add_argument("--clients", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--rounds-list", default="", help="comma list, one request count per client (overrides --clients and "
                    "--rounds), e.g. 8,6,4,2 so clients leave at different times")
    ap.add_argument("--stagger-s", type=float, default=0.0, help="client i starts i*this many seconds after the cell begins")
    ap.add_argument("--chars", type=int, default=3500, help="characters of source per request (~1000 tokens)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--rev", default="cc5f8c2c", help="the commit decoder/model.go is read from")
    ap.add_argument("--backend", default="metal", help="serve's -backend (default metal, so the Mac's own records still reproduce)")
    ap.add_argument("--serve-args", default="")
    ap.add_argument("--server-log", default="")
    a = ap.parse_args()
    a.rounds_list = [int(x) for x in a.rounds_list.split(",") if x]
    if a.rounds_list:
        a.clients = len(a.rounds_list)
    secs = sections(a.rev, a.chars, sum(a.rounds_list) if a.rounds_list else a.clients * a.rounds)
    res = {}
    if os.path.exists(a.out):
        res = json.load(open(a.out)).get("results", {})
    with w7.GoinferServer(a.backend, a.serve_args, a.server_log) as srv:
        # one warm request so the cell does not pay pipeline compile / page-in inside its timing
        w7.post(srv.url, {"model": "bench", "messages": [{"role": "user", "content": "warm"}], "max_tokens": 4, "temperature": 0})
        r = cell(srv.url, a, secs)
    res[a.key] = r
    json.dump({"header": w7.machine_header(), "args": vars(a), "results": res}, open(a.out, "w"), indent=1)
    print(f"[spec-copy] {a.key}: clients={a.clients} wall={r['wall_s']}s aggregate={r['aggregate_tok_s']} tok/s "
          f"({r['completion_tokens']} tokens; HTTP errors {r['http_errors'] or 'none'})", file=sys.stderr)


if __name__ == "__main__":
    main()
