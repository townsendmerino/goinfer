#!/usr/bin/env python3
"""Grade the step-0 replay (id-roundtrip-step0-2026-10-15.md). Reads run.jsonl from the replay; prints every figure in the record.

Usage: python3 -I analyze.py RUN_JSONL

Definitions:
- match: the next request's rendered ids start with the slot's whole id record (the prompt, then every generated id), so the
  next turn would reuse everything;
- divergence: the first position where they differ, as an offset into the reply;
- part: where a generated id sits, by the special tokens before it: reasoning (inside <think>...</think>), tool-call JSON
  (inside <tool_call>...</tool_call>), answer (anything else the model wrote, separators between the parts included);
- canonical run: a maximal run of generated ids between special tokens whose ids equal the tokenizer's own encoding of its text;
- cause: non-canonical ids when the re-rendered text equals the decoded generated text (only the ids differ), a rendering
  difference when the text itself differs.
Reply categories are read from the generated text: escapes = a backslash inside the tool-call JSON; dots = `./...`; non-ASCII =
any byte over 0x7F; code = `func `, `:=`, a code fence, `package ` or a here-document marker.
"""
import json, math, re, sys

THINK, ENDTHINK, CALL, ENDCALL, IM_END = 23, 24, 29, 30, 28


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k} of {n} ({k / n:.0%}, 95% CI {lo:.0%} to {hi:.0%})" if n else "0 of 0"


def part_at(gen, i):
    part = "answer"
    for t in gen[:i]:
        if t == THINK:
            part = "reasoning"
        elif t == ENDTHINK:
            part = "answer"
        elif t == CALL:
            part = "tool-call JSON"
        elif t == ENDCALL:
            part = "answer"
    return part


def categories(text, gen_pieces, gen):
    call_text = ""
    inside = False
    for t, p in zip(gen, gen_pieces):
        if t == CALL:
            inside = True
        elif t == ENDCALL:
            inside = False
        elif inside:
            call_text += p
    return {
        "escapes": "\\" in call_text,
        "dots": "./..." in text,
        "non-ASCII": any(ord(c) > 127 for c in text),
        "code": bool(re.search(r"func |:=|```|package |<<\s*'?EOF", text)),
    }


def local_kind(s):
    if "\\" in s:
        return "JSON escape"
    if "./..." in s or "..." in s:
        return "./..."
    if any(ord(c) > 127 for c in s):
        return "non-ASCII"
    if re.search(r"[{}();=*]|:=", s):
        return "code punctuation"
    if s.strip() == "" or "\n" in s:
        return "whitespace"
    return "other"


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    print(f"== {len(rows)} replies replayed; wall {sum(r['wall_ms'] for r in rows) / 1000:.0f} s")
    bad = [r["id"] for r in rows if not r["prompt_matches_slot"]]
    print(f"validity: slot starts with the route's rendered prompt in {len(rows) - len(bad)} of {len(rows)} {bad or ''}")
    off = {r["usage"]["input"] - r["recorded"]["input"] for r in rows}
    print(f"prompt tokens less the night's: {sorted(off)}")
    lens = {r["slot_len"] - r["P"] - r["usage"]["output"] for r in rows}
    print(f"slot length less prompt less output tokens: {sorted(lens)} (0: the slot holds exactly the generated ids serve counted)")
    errs = [r["id"] for r in rows if "next_error" in r]
    print(f"next-request render errors: {errs}")
    rows = [r for r in rows if "next_error" not in r]
    capped = [r["id"] for r in rows if r["usage"]["output"] >= 4096]
    print(f"replies at the 4,096 guard: {capped}")
    print(f"generated tokens: total {sum(r['usage']['output'] for r in rows)}, median {sorted(r['usage']['output'] for r in rows)[len(rows) // 2]}")

    match = [r for r in rows if r["lcp"] >= r["slot_len"]]
    print(f"\n== overall: the next request re-renders to the generated ids in {fmt(len(match), len(rows))}")

    print("\n== by reply category (a reply can be in several)")
    cats = {r["id"]: categories(r.get("gen_text", ""), r["gen_pieces"], r["gen"]) for r in rows}
    for c in ("escapes", "dots", "non-ASCII", "code"):
        inn = [r for r in rows if cats[r["id"]][c]]
        out = [r for r in rows if not cats[r["id"]][c]]
        print(f"  {c:10s} with: {fmt(sum(r in match for r in inn), len(inn)):40s} without: {fmt(sum(r in match for r in out), len(out))}")
    none = [r for r in rows if not any(cats[r["id"]].values())]
    print(f"  in no category: {fmt(sum(r in match for r in none), len(none))}")
    for lo, hi in ((0, 448), (449, 100000)):
        s = [r for r in rows if lo <= r["usage"]["output"] <= hi]
        print(f"  replies of {lo}-{hi if hi < 100000 else 'more'} tokens: {fmt(sum(r in match for r in s), len(s))}")

    print("\n== by part: runs of generated ids between special tokens that are the tokenizer's own encoding of their text")
    per = {}
    reply_part_bad = {}
    for r in rows:
        gen = r["gen"]
        for run in r["runs"]:
            if "special" in run:
                continue
            p = part_at(gen, run["at"])
            per.setdefault(p, [0, 0])
            per[p][1] += 1
            per[p][0] += run["canonical"]
            if not run["canonical"]:
                reply_part_bad.setdefault(p, set()).add(r["id"])
    for p, (k, n) in per.items():
        print(f"  {p:15s} canonical runs {fmt(k, n)}; replies with a non-canonical run here: {len(reply_part_bad.get(p, ()))}")
    has = {p: [r for r in rows if any(part_at(r['gen'], x['at']) == p for x in r['runs'] if 'special' not in x)] for p in per}
    for p, rs in has.items():
        print(f"  replies with any {p} ids: {len(rs)}; that part canonical throughout in {fmt(sum(r['id'] not in reply_part_bad.get(p, ()) for r in rs), len(rs))}")

    print("\n== each mismatch: where, what, and why")
    causes = {}
    for r in rows:
        if r in match:
            continue
        gen, P = r["gen"], r["P"]
        d = r["lcp"] - P
        tail = r["next_tail"]
        gp, tp = r["gen_pieces"], r["next_tail_pieces"]
        gtext, ttext = r.get("gen_text", ""), r.get("next_tail_text", "")
        same_text = ttext.startswith(gtext)
        noncanon = not all(x.get("canonical", True) for x in r["runs"])
        if same_text:
            cause = "non-canonical ids (text identical)"
        else:
            cause = "rendering difference (text differs)"
        causes.setdefault(cause, []).append(r["id"])
        k = next((i for i in range(min(len(gtext), len(ttext))) if gtext[i] != ttext[i]), min(len(gtext), len(ttext)))
        part = part_at(gen, d) if d < len(gen) else "after the reply"
        window = "".join(gp[max(0, d - 3):d + 3])
        print(f"  {r['id']:22s} reply {len(gen):4d} diverges at {d:4d} ({d - len(gen):+d} from its end) in {part}; "
              f"{local_kind(''.join(gp[max(0, d - 1):d + 2]))}; {cause}; any non-canonical run: {noncanon}")
        print(f"      generated: {[p for p in gp[max(0, d - 3):d + 4]]}")
        print(f"      re-render: {[p for p in tp[max(0, d - 3):d + 4]]}")
        if not same_text:
            print(f"      text differs at char {k}: generated {gtext[max(0, k - 30):k + 30]!r}")
            print(f"                              re-render {ttext[max(0, k - 30):k + 30]!r}")
    print("\n== causes")
    for c, ids in causes.items():
        print(f"  {c}: {len(ids)}")
    print("\n== replies whose ids are all canonical but still mismatch, and the reverse")
    allc = {r["id"]: all(x.get("canonical", True) for x in r["runs"]) for r in rows}
    print(f"  all-canonical replies: {sum(allc.values())} of {len(rows)}; of them matched {sum(allc[r['id']] for r in match)}")
    print(f"  replies with a non-canonical run: {len(rows) - sum(allc.values())}; of them matched {sum(not allc[r['id']] for r in match)}")


if __name__ == "__main__":
    main()
