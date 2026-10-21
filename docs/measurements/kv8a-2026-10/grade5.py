#!/usr/bin/env python3
"""KV8a gate 5, with gate 4(b) read from the same run (README.md here). Reads KV0-H's analyzer output (stats-h-*.json) and serve log of the
KV8a run, and KV0-H's own study statistics as the do-nothing arm.

  python3 grade5.py --base <KV0-H stats json> --stats <run stats json> [--stats ...] --serve-log <run serve log> [--serve-log ...]

Prints one line per check and a VERDICT line: PASS (exit 0), FAIL (1), EXTEND (3: under 10 B turns; the stopping rule adds one chain
session per harness, once), VOID (2).
"""
import argparse, json, statistics, sys

HALF_PROJECTED_S = 17.1  # half of KV0-H's projected 34.2 s median saving per B turn
SAVE_P95_MS = 12.0       # 2% of Codex's 0.6 s A median
MIN_B = 10


def ttfts(stats, cat):
    return [r["ttft_ms"] / 1000 for r in stats["requests"] if r.get("cat") == cat and r.get("ttft_ms")]


def per_agent(stats, cat):
    out = {}
    for r in stats["requests"]:
        if r.get("cat") == cat and r.get("ttft_ms"):
            out.setdefault(r["agent"], []).append(r["ttft_ms"] / 1000)
    return {a: round(statistics.median(v), 3) for a, v in out.items()}


def save_times(paths):
    """ckpt_save_ms of every generating request line, and how many such lines carry none."""
    got, missing = [], 0
    for p in paths:
        for ln in open(p, errors="replace"):
            if not ln.startswith("request: POST") or " status=200 " not in ln:
                continue
            kv = dict(t.split("=", 1) for t in ln.split() if "=" in t)
            if kv.get("completion_tokens", "-") in ("-", "0"):
                continue
            if "ckpt_save_ms" in kv:
                got.append(float(kv["ckpt_save_ms"]))
            else:
                missing += 1
    return got, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--stats", action="append", required=True)
    ap.add_argument("--serve-log", action="append", required=True)
    a = ap.parse_args()
    base = json.load(open(a.base))
    runs = [json.load(open(p)) for p in a.stats]
    run = {"requests": [r for s in runs for r in s["requests"]]}
    lines, fails = [], []
    for s in runs:
        if s.get("excluded"):
            lines.append(f"note: {s['config']}: {s['excluded']} request(s) excluded by KV0-H's own checks")
    b_base, b_run = ttfts(base, "B"), ttfts(run, "B")
    a_base, a_run = ttfts(base, "A"), ttfts(run, "A")
    if len(b_run) < MIN_B:
        print(f"B turns: {len(b_run)}, under {MIN_B}")
        print("VERDICT EXTEND")
        return 3
    mb, mr = statistics.median(b_base), statistics.median(b_run)
    restored = sum(1 for r in run["requests"] if r.get("cat") == "B" and (r.get("reused") or 0) > 0)
    ok5 = mr <= mb - HALF_PROJECTED_S
    lines.append(f"{'ok  ' if ok5 else 'FAIL'} gate 5: B median time to first token {mr:.2f} s over {len(b_run)} turns ({restored} restored), "
                 f"KV0-H {mb:.2f} s over {len(b_base)}: down {mb - mr:.2f} s, bar {HALF_PROJECTED_S} s")
    if not ok5:
        fails.append("gate 5")
    lines.append(f"     A turns (context, gate 4(c)): median {statistics.median(a_run):.2f} s over {len(a_run)}, per agent {per_agent(run, 'A')}; "
                 f"KV0-H {statistics.median(a_base):.2f} s over {len(a_base)}, per agent {per_agent(base, 'A')}")
    lines.append(f"     B turns per agent: {per_agent(run, 'B')}; KV0-H {per_agent(base, 'B')}")
    saves, missing = save_times(a.serve_log)
    if missing or not saves:
        print("\n".join(lines))
        print(f"VERDICT VOID (gate 4(b): {missing} generating request(s) carry no ckpt_save_ms, {len(saves)} do)")
        return 2
    p95 = statistics.quantiles(saves, n=20)[-1] if len(saves) >= 20 else max(saves)
    ok4 = p95 <= SAVE_P95_MS
    lines.append(f"{'ok  ' if ok4 else 'FAIL'} gate 4(b): the save's own time p95 {p95:.3f} ms over {len(saves)} requests (median "
                 f"{statistics.median(saves):.3f}, max {max(saves):.3f}), bar {SAVE_P95_MS} ms")
    if not ok4:
        fails.append("gate 4(b)")
    print("\n".join(lines))
    print(f"VERDICT {'FAIL' if fails else 'PASS'}" + (f" ({', '.join(fails)})" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
