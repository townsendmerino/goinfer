#!/usr/bin/env python3
"""Codex compactions under output caps of 2,048 and 3,072 (against the 4,096 the after run used), from the archive only.

Usage: python3 caps.py [TOKENIZER_JSON]

connect writes model_auto_compact_token_limit as the window less the output cap, so a smaller cap raises Codex's trigger:
16,384 - 4,096 = 12,288 (as run), - 3,072 = 13,312, - 2,048 = 14,336. The model is diagnose_after_compactions.py's (the
recorded trajectories held fixed, its fitted trigger, a compaction restarting at the median post-compaction prompt); only the
trigger changes. It does not model what a truncated reply would have done next, so the truncation counts are printed beside it.
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import diagnose_after_compactions as d  # noqa: E402

WINDOW = 16384


def main():
    runs = d.after_sessions()
    post = [L[i + 1]["inp"] for L in runs.values() for i, r in enumerate(L) if r["comp"]]
    reset = sorted(post)[len(post) // 2] if len(post) % 2 else sum(sorted(post)[len(post) // 2 - 1:len(post) // 2 + 1]) / 2

    def model(limit):
        comps = 0
        for L in runs.values():
            P = L[0]["inp"]
            for n in range(1, len(L)):
                a, b = L[n - 1], L[n]
                if a["comp"] or b["comp"]:
                    continue
                if P + a["out"] + b["new_bytes4"] >= limit:
                    comps, P = comps + 1, reset
                    continue
                P += a["reply_tok"] + (b["inp"] - a["inp"] - a["reply_tok"])
        return comps

    replies = [r for L in runs.values() for r in L]
    plain = [r for r in replies if not r["comp"]]
    summaries = [r for r in replies if r["comp"]]
    print(f"{len(replies)} recorded responses: {len(plain)} replies, {len(summaries)} compaction summaries; "
          f"largest reply {max(r['out'] for r in plain)}, largest summary {max(r['out'] for r in summaries)}")
    for cap in (4096, 3072, 2048):
        limit = WINDOW - cap
        print(f"cap {cap}: trigger {limit}, modelled compactions {model(limit)}; "
              f"replies over the cap {sum(r['out'] > cap for r in plain)} of {len(plain)}, "
              f"summaries over it {sum(r['out'] > cap for r in summaries)} of {len(summaries)}")


if __name__ == "__main__":
    main()
