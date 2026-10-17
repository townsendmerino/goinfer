#!/usr/bin/env python3
"""Per-agent figures from a matrix.py output directory (records.jsonl + logs/serve-<config>.log). No server, no GPU.

  python3 -I e2e_metrics.py OUT [OUT ...] [--config REGEX] [--label TEXT]

Reads each run's wire summary (per_request: prompt/completion tokens, tools) and serve's `-log-requests` lines, aligns each
run's wire requests to the next N `request:` lines of its configuration's serve log (N = the run's wire request count; a
wire request matches the serve line with the same prompt and completion tokens), and reports per agent:

  passed            runs graded passed / runs
  session start     time to first token (serve's `ttft=`) of each run's first main request (a request that carried tools)
  follow-up         time to first token of every later main request: n, median, p90, max
  cold restarts     follow-ups that reused nothing (`reused=0`) with no host snapshot restore; the count is of main requests
                    after a run's first. A compaction request is a follow-up.
  compactions       the record's own count (the agent's) and the wire's
  snapshot restores `snap=<rows>` lines: how many, total rows
  carried-id fallbacks  `ids_fallback=` reasons summed over all lines (`-` when the route does not carry ids)
  fast-alias        lines with `fast=1`

Side requests (no tools: titles, summaries on the fast alias) are counted and excluded from the first/follow-up split.
A run whose wire count and serve lines do not align is reported as UNALIGNED and its requests are left out of the split.
"""
import json, os, re, statistics, sys

KV = re.compile(r"(\w+)=(\S+)")


def parse_line(line):
    f = dict(KV.findall(line))
    ms = lambda s: float(s[:-2]) / 1000 if s and s.endswith("ms") else None
    return dict(route=line.split()[2] if line.startswith("request: ") else "", prompt=f.get("prompt_tokens"), comp=f.get("completion_tokens"),
                ttft=ms(f.get("ttft")), reused=None if f.get("reused", "-") == "-" else int(f["reused"]), fast="fast" in f,
                snap=int(f["snap"]) if "snap" in f else 0, fb=f.get("ids_fallback", "-"), used=f.get("ids", "-"), status=f.get("status"))


def serve_lines(path):
    if not os.path.exists(path):
        return []
    return [parse_line(l) for l in open(path, errors="replace") if l.startswith("request: ")]


def labels(r):
    lab = r.get("labels")
    if isinstance(lab, str):
        try:
            lab = json.loads(lab)
        except ValueError:
            lab = {}
    return lab or {}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else None


def load(out, cfg_re):
    recs = []
    for line in open(os.path.join(out, "records.jsonl")):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("void_halted") or r.get("harness_error"):
            continue
        if cfg_re.search(labels(r).get("config", "")):
            recs.append(r)
    by_cfg = {}
    for r in recs:
        by_cfg.setdefault(labels(r).get("config", "?"), []).append(r)
    agents = {}
    for cfg, rs in by_cfg.items():
        lines = serve_lines(os.path.join(out, "logs", f"serve-{cfg}.log"))
        pos = 0
        for r in rs:
            w = r.get("wire") or {}
            if isinstance(w, str):
                w = {}
            per = w.get("per_request") or []
            chunk = lines[pos:pos + len(per)]
            pos += len(per)
            a = agents.setdefault(r["agent"], dict(runs=[], first=[], follow=[], cold=0, snaps=0, snap_rows=0, fb={}, fast=0, side=0,
                                                   comp=0, wcomp=0, unaligned=[], n_follow=0, reqs=0))
            a["runs"].append(r)
            a["comp"] += r.get("compactions") or 0
            a["wcomp"] += w.get("wire_compactions") or 0
            aligned, free = [], list(chunk)
            for p in per:
                hit = next((c for c in free if (p.get("prompt_tokens") is None or c["prompt"] == str(p["prompt_tokens"])) and c["comp"] == str(p.get("completion_tokens"))), None)
                if hit is None and p.get("status") == 200:
                    aligned = None
                    break
                if hit is not None:
                    free.remove(hit)
                aligned.append((p, hit))
            if aligned is None or len(chunk) != len(per):
                a["unaligned"].append(r.get("id"))
                continue
            first = True
            for p, c in aligned:
                if c is None:
                    continue
                a["reqs"] += 1
                a["fast"] += c["fast"]
                if c["snap"]:
                    a["snaps"] += 1
                    a["snap_rows"] += c["snap"]
                for reason in c["fb"].split(","):
                    if reason not in ("-", ""):
                        a["fb"][reason] = a["fb"].get(reason, 0) + 1
                if not p.get("tools"):
                    a["side"] += 1
                    continue
                if c["ttft"] is None:
                    continue
                if first:
                    a["first"].append(c["ttft"])
                    first = False
                else:
                    a["follow"].append(c["ttft"])
                    if c["reused"] == 0 and not c["snap"]:
                        a["cold"] += 1
    return agents


def fmt(xs):
    return "-" if not xs else f"n={len(xs)} median {statistics.median(xs):.1f} s, p90 {pct(xs, .9):.1f} s, max {max(xs):.1f} s"


def report(out, label, cfg_re):
    agents = load(out, cfg_re)
    print(f"\n### {label}\n\n`{out}`\n")
    print("| agent | passed | session start (s) | follow-up TTFT | cold restarts | compactions (agent / wire) | snapshot restores | carried-id fallbacks | fast-alias | side requests | unaligned runs |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in sorted(agents):
        a = agents[name]
        n = len(a["runs"])
        ok = sum(1 for r in a["runs"] if r.get("passed"))
        tasks = ",".join(f"{r['task']}{'+' if r.get('passed') else '-'}" for r in a["runs"])
        fb = ", ".join(f"{k}={v}" for k, v in sorted(a["fb"].items())) or "none"
        sess = ", ".join(f"{x:.1f}" for x in a["first"]) or "-"
        print(f"| {name} | {ok}/{n} ({tasks}) | {sess} | {fmt(a['follow'])} | {a['cold']} of {len(a['follow'])} | {a['comp']} / {a['wcomp']} | "
              f"{a['snaps']} ({a['snap_rows']} rows) | {fb} | {a['fast']} | {a['side']} | {len(a['unaligned'])} |")


def main():
    args, outs, cfg, label = sys.argv[1:], [], ".", None
    i = 0
    while i < len(args):
        if args[i] == "--config":
            cfg = args[i + 1]; i += 2
        elif args[i] == "--label":
            label = args[i + 1]; i += 2
        else:
            outs.append(args[i]); i += 1
    for o in outs:
        report(o, label or os.path.basename(o.rstrip("/")), re.compile(cfg))


if __name__ == "__main__":
    main()
