#!/usr/bin/env python3
"""KV0 traffic study, analysis: token-level statistics from the recorded sessions. Reads the token ids the server would
prefill (internal/serveapp/kv0_tokenize_test.go, private), the proxy's wire logs and sessions.jsonl. Writes statistics only:
no request text, no token ids, no piece text leaves this script except short category names.

  analyze.py --sessions sessions.jsonl --wire <dir of wire-<sid>.jsonl> --ids <ids.jsonl> [--ids ...] --out stats.json
             [--block 64 --block 256] [--rate label=tok_per_s ...]

Definitions (registered in docs/measurements/kv0-traffic-2026-10-13.md §1):
  request          a POST to a generation route that the server answered (any status) and whose body rendered to ids
  thread           requests of one session whose first 200 tokens are identical (main loop, a title request, a subagent)
  lcp_prev         longest common token prefix with the same session's previous request (any thread), lcp_prev_thread with the
                   previous request of the same thread
  extends          n > len(prev) and lcp == len(prev): the request is its previous request plus more tokens
  lcp_other        longest common prefix with any earlier request of another session in the same configuration
  divergence       the first position where a request differs from its previous thread request; its message index and role
  overlap          two requests of one session whose [start, end] intervals intersect
  reusable(B)      min(max LCP with any earlier request of the configuration, n-1) rounded down to a multiple of B
  today            prefill_reused_tokens the server itself reported for the request
  gain(B)          max(0, reusable(B) - today)
"""
import argparse, collections, json, os, statistics, sys


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def msg_starts(pieces):
    """(index, role) of every message start: ChatML's <|im_start|>, harmony's <|start|>."""
    out = []
    for i, p in enumerate(pieces):
        if p in ("<|im_start|>", "<|start|>") and i + 1 < len(pieces):
            out.append((i, pieces[i + 1].strip()))
    return out


def where(pieces, pos):
    ms = msg_starts(pieces)
    k = -1
    for j, (i, _) in enumerate(ms):
        if i <= pos:
            k = j
    if k < 0:
        return "before the first message"
    i, role = ms[k]
    end = ms[k + 1][0] if k + 1 < len(ms) else len(pieces)
    return f"{role}#{k} (+{pos - i} of {end - i} tokens)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True); ap.add_argument("--wire", required=True)
    ap.add_argument("--ids", action="append", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--block", action="append", type=int, default=[]); ap.add_argument("--rate", action="append", default=[])
    a = ap.parse_args()
    blocks = a.block or [64, 256]
    rates = {k: float(v) for k, v in (r.split("=") for r in a.rate)}

    ids_by_file = {}
    for p in a.ids:
        for line in open(p):
            j = json.loads(line)
            ids_by_file[j["file"]] = j
    sessions = [json.loads(l) for l in open(a.sessions) if l.strip()]
    sessions.sort(key=lambda s: s["t_start"])

    by_cfg = collections.defaultdict(list)
    for s in sessions:
        by_cfg[s["config"]].append(s)

    out = dict(configs={}, definitions=__doc__)
    for cfg, sess in by_cfg.items():
        earlier = []        # (sid, ids) of every earlier request in the configuration, in order
        cfg_out = dict(model=sess[0]["model"], ctx=sess[0]["ctx"], sessions=[])
        for s in sess:
            wire = []
            wp = os.path.join(a.wire, f"wire-{s['id']}.jsonl")
            for line in open(wp):
                r = json.loads(line)
                if r.get("method") == "POST" and "seq" in r:
                    wire.append(r)
            wire.sort(key=lambda r: r["seq"])
            reqs = []
            for r in wire:
                f = [k for k in ids_by_file if k.endswith(os.path.join(s["id"], f"{r['seq']:04d}.json"))]
                j = ids_by_file[f[0]] if f else None
                reqs.append(dict(w=r, ids=j["ids"] if j and "ids" in j else None, pieces=j.get("pieces") if j else None,
                                 err=(j or {}).get("error", "not tokenized" if not j else None)))
            sres = dict(id=s["id"], agent=s["agent"], task=s["task"], rep=s["rep"], requests=[])
            prev, thread_last, threads = None, {}, {}
            sess_ids = []
            for q in reqs:
                w, ids = q["w"], q["ids"]
                row = dict(seq=w["seq"], route=w["path"].split("?")[0], status=w.get("status"), tools=w.get("tools"),
                           server_prompt_tokens=(w.get("usage") or {}).get("prompt_tokens", (w.get("usage") or {}).get("input_tokens")),
                           server_completion_tokens=(w.get("usage") or {}).get("completion_tokens", (w.get("usage") or {}).get("output_tokens")),
                           today_reused=(w.get("usage") or {}).get("prefill_reused_tokens"), ttfb=w.get("ttfb"),
                           duration=w.get("t_end"), t_start=w.get("t_start"))
                if ids is None:
                    row["tokenize_error"] = q["err"]
                    sres["requests"].append(row)
                    continue
                n = len(ids)
                row["n"] = n
                row["count_matches_server"] = (row["server_prompt_tokens"] == n) if row["server_prompt_tokens"] is not None else None
                tkey = hash(tuple(ids[:200]))
                row["thread"] = threads.setdefault(tkey, len(threads))
                if prev is not None:
                    l = lcp(prev, ids)
                    row["lcp_prev"] = l
                    row["extends_prev"] = n > len(prev) and l == len(prev)
                tl = thread_last.get(row["thread"])
                if tl is not None:
                    l = lcp(tl[0], ids)
                    row["lcp_prev_thread"] = l
                    row["extends_prev_thread"] = n > len(tl[0]) and l == len(tl[0])
                    if not row["extends_prev_thread"]:
                        row["divergence_at"] = l
                        row["divergence_where"] = where(q["pieces"], l) if q["pieces"] else None
                        row["prev_thread_tokens"] = len(tl[0])
                # LCP with the best earlier request of the same session / of another session
                best_same = max((lcp(x, ids) for x in sess_ids), default=0)
                best_other = 0
                for sid, x in earlier:
                    if sid != s["id"]:
                        best_other = max(best_other, lcp(x, ids))
                row["lcp_best_same_session"], row["lcp_other_session"] = best_same, best_other
                best_any = min(max(best_same, best_other), n - 1)
                for B in blocks:
                    row[f"reusable_{B}"] = best_any // B * B
                    row[f"gain_{B}"] = max(0, best_any // B * B - (row["today_reused"] or 0))
                prev = ids
                thread_last[row["thread"]] = (ids, row)
                sess_ids.append(ids)
                sres["requests"].append(row)
            # the session's requests enter the configuration's pool after the session ends
            for q in reqs:
                if q["ids"] is not None:
                    earlier.append((s["id"], q["ids"]))
            # overlap inside the session
            iv = [(r["t_start"], r["t_start"] + (r["duration"] or 0), r["seq"]) for r in sres["requests"] if r.get("t_start")]
            ov = set()
            for i in range(len(iv)):
                for j in range(i + 1, len(iv)):
                    if iv[i][0] < iv[j][1] and iv[j][0] < iv[i][1]:
                        ov.update((iv[i][2], iv[j][2]))
            for r in sres["requests"]:
                r["overlaps_another"] = r["seq"] in ov
            cfg_out["sessions"].append(sres)
        out["configs"][cfg] = cfg_out
    # projections: seconds saved at each box's measured prefill rate, per session-first request and per request
    for cfg, c in out["configs"].items():
        for B in blocks:
            for label, rate in rates.items():
                for sres in c["sessions"]:
                    for r in sres["requests"]:
                        if "n" in r:
                            r[f"gain_{B}_s_{label}"] = r[f"gain_{B}"] / rate
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
