#!/usr/bin/env python3
"""KV0 traffic study: markdown tables from analyze.py's statistics. Counts, token positions and categories only.

  summarize.py <stats.json> [--rate label=tok_per_s ...] [--block 64] > tables.md
"""
import argparse, collections, json, statistics, sys


def med(xs):
    return statistics.median(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats"); ap.add_argument("--rate", action="append", default=[]); ap.add_argument("--block", type=int, default=64)
    a = ap.parse_args()
    B = a.block
    rates = {k: float(v) for k, v in (r.split("=") for r in a.rate)}
    d = json.load(open(a.stats))
    out = []
    P = out.append
    for cfg, c in d["configs"].items():
        by_agent = collections.defaultdict(list)
        for s in c["sessions"]:
            by_agent[s["agent"]].append(s)
        P(f"\n### Configuration `{cfg}`: {c['model'].split('/')[-1]}, `-ctx {c['ctx']}`\n")
        P("| harness | session | task | requests | main-thread turns | threads | thread switches | first request (tokens) | last request | strictly extends its thread's previous request | other requests | overlapping another |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for ag, sess in by_agent.items():
            for s in sess:
                rq = [r for r in s["requests"] if "n" in r]
                if not rq:
                    continue
                th = collections.Counter(r["thread"] for r in rq)
                main = next((r["thread"] for r in rq if (r.get("tools") or 0) > 0), rq[0]["thread"])
                mr = [r for r in rq if r["thread"] == main]
                follow = [r for r in mr if "extends_prev_thread" in r]
                ext = sum(1 for r in follow if r["extends_prev_thread"])
                switches = sum(1 for a_, b_ in zip(rq, rq[1:]) if a_["thread"] != b_["thread"])
                P(f"| {ag} | {s['id'].split('-')[-1]} | {s['task']} | {len(rq)} | {len(mr)} | {len(th)} | {switches} | {mr[0]['n']:,} | {mr[-1]['n']:,} | "
                  f"{ext} of {len(follow)} | {len(rq) - len(mr)} | {sum(1 for r in rq if r.get('overlaps_another'))} |")
        P("")
        # non-extending requests: where
        P("**Requests that do not strictly extend their thread's previous request** (the divergence and what sits there):\n")
        P("| harness | session | request | tokens | previous in thread | common prefix | first divergence at |")
        P("|---|---|---|---|---|---|---|")
        any_div = False
        for ag, sess in by_agent.items():
            for s in sess:
                for r in s["requests"]:
                    if r.get("extends_prev_thread") is False:
                        any_div = True
                        P(f"| {ag} | {s['id'].split('-')[-1]} {s['task']} | {r['seq']} | {r['n']:,} | {r['prev_thread_tokens']:,} | {r['lcp_prev_thread']:,} | {r.get('divergence_where')} |")
        if not any_div:
            P("| (none) | | | | | | |")
        P("")
        # cross-session: the first main-thread request of each session after the first, per harness
        P("**New conversations: the first main-thread request of each session after the harness's first**\n")
        hdr = "| harness | session | tokens | shared with an earlier session (longest common prefix) | share | the server reused (today) | gain at B=%d (tokens) |" % B
        for lab in rates:
            hdr += f" saved at {lab} (s, projection) |"
        P(hdr)
        P("|---|---|---|---|---|---|---|" + "---|" * len(rates))
        agg = collections.defaultdict(list)
        for ag, sess in by_agent.items():
            for k, s in enumerate(sess):
                rq = [r for r in s["requests"] if "n" in r]
                if not rq:
                    continue
                main = next((r["thread"] for r in rq if (r.get("tools") or 0) > 0), rq[0]["thread"])
                first = next(r for r in rq if r["thread"] == main)
                if k == 0:
                    continue
                g = first[f"gain_{B}"]
                row = f"| {ag} | {s['id'].split('-')[-1]} {s['task']} | {first['n']:,} | {first['lcp_other_session']:,} | {100 * first['lcp_other_session'] / first['n']:.1f}% | {first['today']:,}{'*' if first['today_estimated'] else ''} | {g:,} |"
                for lab, rt in rates.items():
                    row += f" {g / rt:.1f} |"
                P(row)
                agg[ag].append((first["n"], first["lcp_other_session"], g))
        P("")
        P("(* the Responses route does not report reuse; estimated as the overlap with the previous request on the server)\n")
        for ag, v in agg.items():
            P(f"- {ag}: {len(v)} new conversations; shared-with-earlier-session median {med([x[1] for x in v]):,.0f} of {med([x[0] for x in v]):,.0f} tokens; gain at B={B} median {med([x[2] for x in v]):,.0f} (min {min(x[2] for x in v):,}, max {max(x[2] for x in v):,}).")
        P("")
        # per turn: gain over every request that is not a session's first main request
        P("**Per turn: tokens a block-sharing manager would skip beyond what the single-slot server skips today**, over every request:\n")
        P("| harness | requests | with gain > 0 | total gain (tokens) | median / max gain of those | prompt tokens, all requests | gain as a share of all prompt tokens |")
        P("|---|---|---|---|---|---|---|")
        for ag, sess in by_agent.items():
            rq = [r for s in sess for r in s["requests"] if "n" in r]
            gains = [r[f"gain_{B}"] for r in rq]
            pos = [x for x in gains if x > 0]
            tot = sum(r["n"] for r in rq)
            P(f"| {ag} | {len(rq)} | {len(pos)} | {sum(gains):,} | {med(pos) or 0:,.0f} / {max(pos) if pos else 0:,} | {tot:,} | {100 * sum(gains) / tot:.1f}% |")
        P("")
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
