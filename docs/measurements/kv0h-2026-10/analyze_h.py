#!/usr/bin/env python3
"""KV0-H analysis (docs/measurements/kv0h-hybrid-traffic-2026-10.md): classifies every request of a hybrid-model session set as N / A / B / C / D
against its own thread's previous request, from the SERVER's records plus verified prompt ids. Writes statistics only (counts, positions, categories,
seconds): no ids, bodies or text leave this script except short category names.

  analyze_h.py --sessions sessions.jsonl --wire <dir holding wire-<sid>.jsonl> --ids <ids.jsonl> [--ids ...] --tokenids <tokenids-<cfg>.jsonl>
               --serve-log <serve-<cfg>.log> --out stats-h.json [--config <name>]

Inputs, per configuration: the recorded proxy wire logs and sessions.jsonl (kv0_proxy.py, run_sessions.py), the rendered prompt ids of every recorded
body (internal/serveapp/kv0_tokenize_test.go), the server's own --log-token-ids file (per generation: prompt length and SHA-256, generated ids, reuse, slot)
and its --log-requests lines (prompt tokens, time to first token, reused, slot, div).

Ground truth (registration §1.3): a rendering is ACCEPTED only when its length and SHA-256 (little-endian uint32 per id) equal the server's logged
prompt_len and prompt_sha256 for that request; a request that fails is excluded and listed. The committed sequence of a request is its accepted prompt ids
plus the ids the server logged as generated. Invariant (amendment A1): slot= of a request equals the prompt_len plus len(gen) of the generation the
server served immediately before it; every failure is listed with its difference.

Categories (registration §1.4), against the request's own thread's previous request p (C_p = prompt(p) + gen(p), l = lcp(prompt(k), C_p), n = len(prompt(k))):
  N  first request of its thread in its session
  A  l == len(C_p), n > len(C_p), and no request of another thread served between p and k (server order)
  B  len(prompt(p)) <= l < len(C_p), n > l, and no other thread between
  C  another thread served between p and k and k is A- or B-shaped against its own thread (C-A / C-B)
  D  l < len(prompt(p))
"""
import argparse, collections, hashlib, json, os, statistics, struct, sys


def lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def sha(ids):
    return hashlib.sha256(struct.pack("<%dI" % len(ids), *ids)).hexdigest() if ids else hashlib.sha256(b"").hexdigest()


def msg_starts(pieces):
    out = []
    for i, p in enumerate(pieces):
        if p in ("<|im_start|>", "<|start|>") and i + 1 < len(pieces):
            out.append((i, pieces[i + 1].strip()))
    return out


def where(pieces, pos):
    ms = msg_starts(pieces); k = -1
    for j, (i, _) in enumerate(ms):
        if i <= pos:
            k = j
    if k < 0:
        return "before the first message"
    i, role = ms[k]
    end = ms[k + 1][0] if k + 1 < len(ms) else len(pieces)
    return f"{role}#{k} (+{pos - i} of {end - i} tokens)"


def parse_serve_log(path):
    rows = []
    for ln in open(path, errors="replace"):
        if not ln.startswith("request: POST"):
            continue
        d = {}
        for tok in ln.split():
            if "=" in tok:
                k, v = tok.split("=", 1); d[k] = v
        def iv(x):
            try:
                return int(x.rstrip("ms"))
            except (AttributeError, TypeError, ValueError):
                return None
        u, _, c = (d.get("ids") or "").partition("/")
        used, carried = (int(u) if u.isdigit() else 0), (int(c) if c.isdigit() else 0)
        fmt = iv(d.get("ids_fmt")) or 0
        # carried-id class (the owner's per-harness count): the request's carried assistant turns whose ids were used with template formatting replaced,
        # used without, carried but not used (a fallback reason), or no turn carried ids at all
        cls = "fmt" if fmt > 0 else ("nofmt" if used > 0 else ("unused" if carried > 0 else "none"))
        rows.append(dict(route=ln.split()[2], status=iv(d.get("status")), n=iv(d.get("prompt_tokens")), ttft_ms=iv(d.get("ttft")), reused=iv(d.get("reused")),
                         slot=iv(d.get("slot")), div=iv(d.get("div")), canon=d.get("canon"), ids_used=used, ids_carried=carried, ids_fmt=fmt, carried_class=cls,
                         ids_fallback=d.get("ids_fallback")))
    return rows


def pct(x, p):
    x = sorted(x)
    return x[min(len(x) - 1, int(p * len(x)))] if x else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True); ap.add_argument("--wire", required=True); ap.add_argument("--ids", action="append", required=True)
    ap.add_argument("--tokenids", required=True); ap.add_argument("--serve-log", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="")
    a = ap.parse_args()
    ids_by_file = {}
    for p in a.ids:
        for line in open(p):
            j = json.loads(line); ids_by_file[j["file"]] = j
    sessions = [json.loads(l) for l in open(a.sessions) if l.strip()]
    if a.config:
        sessions = [s for s in sessions if s["config"] == a.config]
    sessions.sort(key=lambda s: s["t_start"])
    gens = [json.loads(l) for l in open(a.tokenids) if l.strip()]          # server order (written at the end of each generation)
    slog = parse_serve_log(a.serve_log)
    gen_used = [False] * len(gens)

    # 1. every wire request of every session, with its rendered ids; accept it by (length, SHA-256) against the server's generations
    reqs = []
    for s in sessions:
        wp = os.path.join(a.wire, f"wire-{s['id']}.jsonl")
        wire = sorted((r for r in (json.loads(l) for l in open(wp)) if r.get("method") == "POST" and "seq" in r), key=lambda r: r["seq"])
        for w in wire:
            f = [k for k in ids_by_file if k.endswith(os.path.join(s["id"], f"{w['seq']:04d}.json"))]
            j = ids_by_file[f[0]] if f else None
            reqs.append(dict(s=s, w=w, ids=(j or {}).get("ids"), pieces=(j or {}).get("pieces"), err=(j or {}).get("error", "not tokenized" if not j else None),
                             gi=None, status=None))
    by_sha = collections.defaultdict(list)
    for gi, g in enumerate(gens):
        by_sha[(g["prompt_len"], g["prompt_sha256"])].append(gi)
    for q in sorted((q for q in reqs if q["ids"]), key=lambda q: q["w"].get("t_start") or 0):
        key = (len(q["ids"]), sha(q["ids"]))
        cand = [gi for gi in by_sha.get(key, []) if not gen_used[gi]]
        if cand:
            q["gi"] = cand[0]; gen_used[cand[0]] = True
    excluded = [dict(session=q["s"]["id"], seq=q["w"]["seq"], why=q["err"] or "rendering's length/SHA-256 differs from every logged generation") for q in reqs if q["gi"] is None]
    ok = [q for q in reqs if q["gi"] is not None]
    # server order = generation order; the request line for generation gi is slog[gi] when the logs align (checked below)
    aligned = len(slog) == len(gens)
    for q in ok:
        g = gens[q["gi"]]
        q["gen"] = g["gen"]; q["reused"] = g.get("reused"); q["slot"] = g.get("slot"); q["n"] = g["prompt_len"]
        sl = slog[q["gi"]] if aligned else None
        q["ttft_ms"] = sl["ttft_ms"] if sl else None
        q["carried_class"] = sl["carried_class"] if sl else None; q["ids_fallback"] = sl["ids_fallback"] if sl else None
        q["log_consistent"] = (sl is not None and sl["n"] == q["n"] and sl["reused"] == q["reused"] and sl["slot"] == q["slot"])
    order = sorted(ok, key=lambda q: q["gi"])
    # 2. the invariant: slot == prompt_len + len(gen) of the generation served immediately before (server order, all generations, not only accepted ones)
    inv_fail = []
    for k in range(len(gens)):
        g = gens[k]
        if k == 0 or g.get("slot") is None:
            continue
        pg = gens[k - 1]
        want = pg["prompt_len"] + len(pg["gen"])
        if g["slot"] != want:
            inv_fail.append(dict(gen_index=k, slot=g["slot"], want=want, diff=g["slot"] - want, route=g["route"]))
    # 3. classification, per session, threads by the first 200 prompt ids
    pos_in_order = {id(q): i for i, q in enumerate(order)}
    thread_of = {}; last = {}; rows = []
    for q in order:
        s = q["s"]; ids = q["ids"]; n = len(ids)
        tkey = (s["id"], hash(tuple(ids[:200]))); thread_of[id(q)] = tkey
        row = dict(session=s["id"], agent=s["agent"], task=s["task"], seq=q["w"]["seq"], gi=q["gi"], n=n, reused=q["reused"], slot=q["slot"],
                   ttft_ms=q["ttft_ms"], log_consistent=q["log_consistent"], gen_len=len(q["gen"]),
                   carried_class=q["carried_class"], ids_fallback=q["ids_fallback"])
        p = last.get(tkey)
        if p is None:
            row["cat"] = "N"
            row["lcp_other_session"] = max((lcp(x["ids"], ids) for x in order[:pos_in_order[id(q)]] if x["s"]["id"] != s["id"]), default=0)
        else:
            pq = p; cp = pq["ids"] + pq["gen"]; l = lcp(ids, cp)
            lp = len(pq["ids"])
            between = [x for x in order[pos_in_order[id(pq)] + 1: pos_in_order[id(q)]] if thread_of[id(x)] != tkey]
            row.update(lcp_prev_committed=l, prev_prompt=lp, prev_committed=len(cp), others_between=len(between))
            if l < lp:
                shape = "D"
            elif l == len(cp) and n > len(cp):
                shape = "A"
            elif lp <= l < len(cp) and n > l:
                shape = "B"
            else:
                shape = "D"          # n <= l: the request is a prefix of the committed sequence (not in the registered shapes): reported as D and counted
                row["not_a_shape"] = True
            if shape in ("A", "B"):
                row["shape"] = shape
                row["cat"] = ("C-" + shape) if between else shape
                row["skip_tokens_if_served"] = len(cp) if shape == "A" else lp
                if shape == "B":
                    row["divergence_at"] = l; row["divergence_where"] = where(q["pieces"], l) if q["pieces"] else None
                    row["reprocessed_under_checkpoint"] = n - lp
            else:
                row["cat"] = "D"; row["divergence_at"] = l; row["divergence_where"] = where(q["pieces"], l) if q["pieces"] else None
        last[tkey] = q
        rows.append(row)
    # 4. rates and projections
    cold = [(r["n"], r["ttft_ms"] / 1000) for r in rows if r.get("reused") == 0 and r["n"] >= 4096 and r["ttft_ms"]]
    bands = [(4096, 8191), (8192, 16383), (16384, 10**9)]
    rate = {}
    for lo, hi in bands:
        xs = [n / t for n, t in cold if lo <= n <= hi]
        rate[f"{lo}-{hi if hi < 10**9 else ''}"] = dict(count=len(xs), median_tok_s=statistics.median(xs) if xs else None)
    def band_rate(n):
        key = next((f"{lo}-{hi if hi < 10**9 else ''}" for lo, hi in bands if lo <= n <= hi), None)
        r = rate.get(key, {})
        if r.get("count", 0) >= 5:
            return r["median_tok_s"]
        ok_ = [v["median_tok_s"] for v in rate.values() if v["median_tok_s"] and v["count"] >= 1]
        return statistics.median(ok_) if ok_ else None
    total_today = sum(max(0, r["n"] - (r["reused"] or 0)) for r in rows)
    agg = {}
    for agent in sorted({r["agent"] for r in rows}):
        rs = [r for r in rows if r["agent"] == agent]
        cats = collections.defaultdict(lambda: dict(count=0, tokens_today=0, skip_tokens=0, projected_s=[], ttft_s=[]))
        for r in rs:
            c = cats[r["cat"]]; c["count"] += 1; c["tokens_today"] += max(0, r["n"] - (r["reused"] or 0))
            if r.get("ttft_ms"):
                c["ttft_s"].append(r["ttft_ms"] / 1000)
            if "skip_tokens_if_served" in r:
                sk = max(0, r["skip_tokens_if_served"] - (r["reused"] or 0)); c["skip_tokens"] += sk
                br = band_rate(r["n"]); r["projected_saving_s"] = sk / br if br else None
                if br:
                    c["projected_s"].append(sk / br)
        denom = sum(max(0, r["n"] - (r["reused"] or 0)) for r in rs)
        cc = collections.Counter(r["carried_class"] for r in rs)
        cross = collections.defaultdict(lambda: collections.Counter())
        for r in rs:
            cross[r["cat"]][r["carried_class"]] += 1
        agg[agent] = dict(requests=len(rs), tokens_processed_today=denom, carried_ids=dict(cc), carried_ids_by_category={k: dict(v) for k, v in sorted(cross.items())}, categories={k: dict(count=v["count"], tokens_processed_today=v["tokens_today"],
                          share_of_tokens_today=(v["tokens_today"] / denom if denom else None), skip_tokens=v["skip_tokens"], skip_share=(v["skip_tokens"] / denom if denom else None),
                          median_ttft_s=statistics.median(v["ttft_s"]) if v["ttft_s"] else None, median_projected_saving_s=statistics.median(v["projected_s"]) if v["projected_s"] else None,
                          projected_s_total=sum(v["projected_s"])) for k, v in sorted(cats.items())})
    allc = collections.defaultdict(lambda: dict(skip=0, secs=[]))
    for r in rows:
        if "skip_tokens_if_served" in r:
            c = allc[r["cat"]]; c["skip"] += max(0, r["skip_tokens_if_served"] - (r["reused"] or 0))
            if r.get("projected_saving_s") is not None:
                c["secs"].append(r["projected_saving_s"])
    rule = {}
    for name, cats_ in (("C (parking)", ("C-A", "C-B")), ("B (end-of-prompt checkpoint)", ("B",))):
        skip = sum(allc[c]["skip"] for c in cats_); secs = [s for c in cats_ for s in allc[c]["secs"]]
        share = skip / total_today if total_today else None
        med = statistics.median(secs) if secs else None
        rule[name] = dict(skip_tokens=skip, share_of_tokens_processed_today=share, affected_requests=len(secs), median_projected_saving_s=med,
                          passes_25pct=(share is not None and share >= 0.25), passes_5s=(med is not None and med >= 5.0))
    out = dict(config=a.config or None, definitions=__doc__, requests_total=len(reqs), accepted=len(ok), excluded=excluded,
               logs_aligned=aligned, log_inconsistent=[r["gi"] for r in rows if not r["log_consistent"]],
               slot_invariant=dict(checked=sum(1 for g in gens[1:] if g.get("slot") is not None), failures=inv_fail,
                                   offsets=dict(collections.Counter(f["diff"] for f in inv_fail))),
               cold_rate=rate, tokens_processed_today=total_today, per_agent=agg, rule=rule, requests=rows)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(ok)}/{len(reqs)} requests accepted, {len(inv_fail)} slot-invariant failures")


if __name__ == "__main__":
    main()
