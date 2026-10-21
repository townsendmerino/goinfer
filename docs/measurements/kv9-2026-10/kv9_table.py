#!/usr/bin/env python3
"""Tables for the KV9 step 0 record from kv9_mem.py's output (its JSON-lines log, or the JSON file it writes at the end).

Per cell: the server's footprint after load, after session 1, and after the last session; with settled readings, the
footprint step each later session added once the heap had settled, against the KV that session copied by geometry
(copied rows times the f32 bytes per position). Then what host-block storage would hold instead of today's copies:
the shared preamble's whole 64-row blocks once, plus each session's own rows from the first block it does not share.

  python3 kv9_table.py <run log or json> [...]
"""
import json
import sys

BLOCK = 64  # internal/kvmgr's BlockRows
MB = 1 << 20


def load(path):
    text = open(path).read().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(ln) for ln in text.splitlines() if ln.startswith("{")]


def main():
    recs = [r for p in sys.argv[1:] for r in load(p)]
    cells = {}
    for r in recs:
        cells.setdefault((r["model"], r["target_tokens"], r["pid"]), []).append(r)
    print("| model | prompt | sessions | footprint: loaded / after session 1 / after the last | KV per session (geometry) | "
          "settled step per copy (measured) | today: all sessions' KV | with host blocks | saved |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (model, _, _), rs in cells.items():
        ld = next((r for r in rs if r["event"] == "loaded"), None)
        sess = [r for r in rs if r["event"] == "session"]
        settled = {r["session"]: r for r in rs if r["event"] == "settled"}
        stop = next((r for r in rs if r["event"] == "stopped"), None)
        if not ld or not sess:
            continue
        bpp = ld["bytes_per_position"]
        rows = [s["prompt_tokens"] + s["completion_tokens"] for s in sess]
        reused = [s["reused"] or 0 for s in sess[1:]]
        # the preamble's whole blocks: what host blocks would store once (the prompt's common part; every session's
        # prompt here differs only in its last line, so the longest reuse bounds it from below)
        shared = max(max(reused, default=0), sess[0]["prompt_tokens"] - 32) // BLOCK * BLOCK
        today = sum(rows) * bpp
        blocks = (shared + sum(n - shared for n in rows)) * bpp
        steps = []
        for k in range(2, len(sess) + 1):
            if k in settled and k - 1 in settled:
                steps.append(settled[k]["footprint"] - settled[k - 1]["footprint"])
        step = " / ".join(f"{s/MB:.0f}" for s in steps) + " MB" if steps else "—"
        copy = f"{max(reused, default=0) * bpp / MB:.0f} MB reused by the largest reuse" if any(reused) else "nothing reused"
        n = f"{len(sess)}" + (f" (guard stopped session {stop['session']})" if stop else "")
        print(f"| {model} | {sess[0]['prompt_tokens']} tokens, {shared} in shared blocks | {n} | "
              f"{ld['footprint']/MB:.0f} / {sess[0]['footprint']/MB:.0f} / {sess[-1]['footprint']/MB:.0f} MB | "
              f"{rows[0]*bpp/MB:.0f} MB ({copy}) | {step} | {today/MB:.0f} MB | {blocks/MB:.0f} MB | {(today-blocks)/MB:.0f} MB |")
    print()
    for r in recs:
        if r["event"] == "session":
            print(f"- {r['model']} ~{r['target_tokens']} (pid {r['pid']}): session {r['session']}: {r['prompt_tokens']} prompt "
                  f"tokens, {r['reused']} reused; footprint {r['footprint']/MB:.0f} MB, RSS {r['rss']/MB:.0f} MB; swap grown "
                  f"{r['swap_grown']/MB:.0f} MB since the cell started")
        elif r["event"] == "settled":
            print(f"  - settled after {r['waited_s']} s: footprint {r['footprint']/MB:.0f} MB, RSS {r['rss']/MB:.0f} MB")
        elif r["event"] in ("stopped", "not ready"):
            print(f"- {r['model']} ~{r['target_tokens']}: {r['event']} before session {r.get('session')}: available "
                  f"{r.get('available', 0)/MB:.0f} MB, swap grown {r.get('swap_grown', 0)/MB:.0f} MB")


if __name__ == "__main__":
    main()
