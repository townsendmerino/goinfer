#!/usr/bin/env python3
"""I3 after run (Codex x Mellum2.1, mellum-16k-codex-f): why it compacted 20 times against job D's once, and why 15 plain
continuations reused nothing. Reads only the archived data beside i3-night-2026-10-13.md; no server, no GPU.

Usage: python3 diagnose_after_compactions.py [TOKENIZER_JSON]
  TOKENIZER_JSON defaults to ~/models/mellum2.1-thinking/tokenizer.json (needs the `tokenizers` package).

Every figure in the section "After run: the compactions and the cold continuations, diagnosed" is printed here.
The re-rendering mirrors Mellum2.1's own template for the shapes Codex sends (one task message, then assistant turns with
reasoning and one call, and tool results grouped into one user turn); its check is that it reproduces the served prompt
growth of all 213 fully reused follow-ups exactly.
"""
import glob, gzip, json, os, re, statistics as st, sys, tarfile, tempfile

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "i3-night-2026-10-13")
TOK = Tokenizer.from_file(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/models/mellum2.1-thinking/tokenizer.json"))
LIMIT = 16384 - 4096  # Codex's model_auto_compact_token_limit as connect writes it: the window less the output limit
ROLLBACK = 448  # the windowed-KV rollback this load printed (no free-VRAM raise)


def nt(s):
    return len(TOK.encode(s, add_special_tokens=False).ids) if s else 0


def think(reason):
    return "<think>\n" + reason.strip("\n") + "\n</think>\n\n" if reason else ""


def reply_text(r):
    s, content = think(r["reason"]), r["msg"].lstrip("\n")
    s += content
    for k, (name, args) in enumerate(r["calls"]):
        if k or content:
            s += "\n"
        s += '<tool_call>\n{"name": "' + name + '", "arguments": ' + args + "}\n</tool_call>"
    return s + "<|im_end|>\n"


def new_text(pend):
    s = ""
    if pend["tool"]:
        s += "<|im_start|>user" + "".join("\n<tool_response>\n" + o + "\n</tool_response>" for o in pend["tool"]) + "<|im_end|>\n"
    s += "".join("<|im_start|>user\n" + u + "<|im_end|>\n" for u in pend["user"])
    return s + "<|im_start|>assistant\n"


def load_session(path):
    """One entry per response, in order: what the model returned, the usage serve reported, and what Codex added before it."""
    out, cur, pend = [], None, {"tool": [], "user": []}

    def fresh():
        return {"reason": "", "calls": [], "msg": ""}

    cur = fresh()
    for line in open(path):
        o = json.loads(line)
        t, p = o["type"], o["payload"]
        if t == "response_item":
            pt = p.get("type")
            text = "".join(c.get("text", "") for c in p.get("content") or [])
            if pt == "reasoning":
                cur["reason"] += text
            elif pt == "function_call":
                cur["calls"].append((p.get("name", ""), p.get("arguments", "")))
            elif pt == "message" and p.get("role") == "assistant":
                cur["msg"] += text
            elif pt == "function_call_output":
                v = p.get("output")
                pend["tool"].append(v if isinstance(v, str) else json.dumps(v))
            elif pt == "message":
                pend["user"].append(text)
        elif t == "token_usage_record":
            u = p["usage"]
            cur.update(inp=u["input_tokens"], out=u["output_tokens"], cached=u["cached_input_tokens"], pend=pend, comp=False)
            cur["reason_tok"], cur["reply_tok"], cur["new_tok"] = nt(think(cur["reason"])), nt(reply_text(cur)), nt(new_text(pend))
            cur["new_bytes4"] = (sum(map(len, pend["tool"])) + sum(map(len, pend["user"]))) // 4
            out.append(cur)
            cur, pend = fresh(), {"tool": [], "user": []}
        elif t == "compacted":
            out[-1]["comp"] = True
    return out


def after_sessions():
    tmp = tempfile.mkdtemp()
    with tarfile.open(os.path.join(DATA, "job-after", "codex-rollouts.tar.gz")) as tf:
        tf.extractall(tmp, filter="data")
    runs = {}
    for f in sorted(glob.glob(tmp + "/runs/*/home/.codex/sessions/*/*/*/*.jsonl")):
        runs[f.split("/runs/")[1].split("/")[0][6:]] = load_session(f)
    return runs


def kind(L, n):
    a, b = L[n - 1], L[n]
    if b["comp"]:
        return "compaction"
    if a["comp"]:
        return "post-compaction"
    if b["cached"] >= a["inp"] + a["out"]:
        return "full"
    return "cold" if b["cached"] == 0 else "partial"


def main():
    runs = after_sessions()
    print("== after run: follow-ups by reuse (from Codex's own cached_input_tokens)")
    kinds = {}
    for L in runs.values():
        for n in range(1, len(L)):
            kinds.setdefault(kind(L, n), []).append((L, n))
    print({k: len(v) for k, v in kinds.items()})

    print("\n== wall each kind cost (serve log, matched in order on prompt and output; cold = time to first token over 20 s)")
    with gzip.open(os.path.join(DATA, "job-after", "serve-mellum-16k-codex-f.log.gz"), "rt") as f:
        sv = [tuple(map(int, m.groups())) for m in
              (re.search(r"prompt_tokens=(\d+) completion_tokens=(\d+) ttft=(\d+)ms total=(\d+)ms", l) for l in f if l.startswith("request:"))]
    with gzip.open(os.path.join(DATA, "job-after", "records.jsonl.gz"), "rt") as f:
        order = [json.loads(l)["id"][6:] for l in f]
    j, cost = 0, {}
    for run in order:
        L = runs[run]
        for n, r in enumerate(L):
            while not (sv[j][0] == r["inp"] and sv[j][1] == r["out"]):
                j += 1
            k = kind(L, n) if n else "start"
            c = cost.setdefault(k, [0, 0, 0.0])
            c[0], c[1], c[2] = c[0] + 1, c[1] + (sv[j][2] > 20000), c[2] + sv[j][3] / 1000
            j += 1
    for k, (cnt, cold, wall) in cost.items():
        print(f"  {k:16s} {cnt:3d} requests, {cold:3d} cold, {wall:6.0f} s")

    print("\n== Q1: growth of the prompt per plain step (no compaction on either side)")
    steps = [(L[n - 1], L[n]) for L in runs.values() for n in range(1, len(L)) if kind(L, n) in ("full", "partial", "cold")]
    d = sum(b["inp"] - a["inp"] for a, b in steps)
    rs = sum(a["reason_tok"] for a, b in steps)
    ans = sum(a["reply_tok"] - a["reason_tok"] for a, b in steps)
    print(f"after: {len(steps)} steps, {d} tokens, mean {d/len(steps):.1f}, median {st.median(b['inp']-a['inp'] for a,b in steps)}")
    print(f"  reasoning {rs} ({rs/d:.1%}), the reply's call or text {ans} ({ans/d:.1%}), new input {d-rs-ans} ({(d-rs-ans)/d:.1%})")
    print(f"  non-reasoning growth per step {(d-rs)/len(steps):.1f}; generated tokens per step {sum(a['out'] for a,b in steps)/len(steps):.1f}")
    dsteps, druns = [], []
    with gzip.open(os.path.join(DATA, "job-d", "records.jsonl.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            pr = r["wire"]["per_request"]
            ps, cs = [x["prompt_tokens"] for x in pr], [x["completion_tokens"] for x in pr]
            # job D's one compaction: edit r3's request 14 (the summary request) and 15 (the restart)
            ss = [(ps[i] - ps[i - 1], cs[i - 1]) for i in range(1, len(ps)) if not (r["compactions"] and i in (14, 15))]
            dsteps += ss
            druns.append((r["id"], ps[0], ss, len(ps)))
    dd = [x for x, _ in dsteps]
    print(f"job D: {len(dd)} steps, {sum(dd)} tokens, mean {sum(dd)/len(dd):.1f}, median {st.median(dd)}, "
          f"generated per step {sum(c for _, c in dsteps)/len(dsteps):.1f}; no reasoning item in any of its 123 responses")

    print("\n== Codex's trigger: last served prompt + its output + bytes/4 of what Codex added since >= LIMIT")
    fired, held = [], []
    for L in runs.values():
        for n in range(len(L) - 1):
            if L[n]["comp"]:
                continue
            v = L[n]["inp"] + L[n]["out"] + L[n + 1]["new_bytes4"]
            (fired if L[n + 1]["comp"] else held).append(v)
    print(f"lowest that fired {min(fired)}, highest that did not {max(held)}, LIMIT {LIMIT}, fired {len(fired)}")
    cyc, post = [], []
    for L in runs.values():
        start = 0
        for i, r in enumerate(L):
            if r["comp"]:
                cyc.append(i - start)
                start = i + 1
                post.append(L[i + 1]["inp"])
    print(f"requests from a (re)start to the compaction: median {st.median(cyc)}, range {min(cyc)}-{max(cyc)}; "
          f"prompt after a compaction: median {st.median(post)}")
    reset = st.median(post)

    def model(drop, rollback=ROLLBACK, limit=LIMIT):
        """Replay the recorded steps with only the replayed reasoning changed. Returns (compactions, colds the rule causes)."""
        comps = colds = 0
        for L in runs.values():
            P = L[0]["inp"]
            for n in range(1, len(L)):
                a, b = L[n - 1], L[n]
                if a["comp"] or b["comp"]:
                    continue
                if P + a["out"] + b["new_bytes4"] >= limit:
                    comps, P = comps + 1, reset
                    continue
                dropped = a["reason_tok"] and drop(a)
                P += (0 if dropped else a["reason_tok"]) + (a["reply_tok"] - a["reason_tok"]) + (b["inp"] - a["inp"] - a["reply_tok"])
                colds += bool(dropped and a["out"] > rollback)
        return comps, colds

    print("\n== the same trajectories with the replayed reasoning changed (model; as-run calibrates it)")
    for name, f, rb in (("keep all (as run)", lambda r: False, ROLLBACK), ("drop all", lambda r: True, ROLLBACK),
                        ("drop all, rollback 1024", lambda r: True, 1024),
                        ("drop a reply's reasoning when the reply <= 448", lambda r: r["out"] <= 448, ROLLBACK),
                        ("drop when the reply <= 1024, rollback 1024", lambda r: r["out"] <= 1024, 1024)):
        print(f"  {name}: compactions, colds it causes = {model(f, rb)}")
    print(f"  keep all at a 32,768 window (trigger 28,672): {model(lambda r: False, ROLLBACK, 32768 - 4096)}")
    simd = 0
    for _, p0, ss, _ in druns:
        P = p0
        for dlt, out in ss:
            if P + out + 150 >= LIMIT:
                simd, P = simd + 1, reset
                continue
            P += dlt + max(0, out - 52)  # 52: the after run's mean reply call/text, the part job D did replay
    print(f"  job D's trajectories with every reply's reasoning replayed (estimated): {simd} compactions")
    rb_latest, rb_none = [], []
    for L in runs.values():
        for n in range(2, len(L) - 1):
            a, b, c = L[n - 1], L[n], L[n + 1]
            if any(x["comp"] for x in (L[n - 2], a, b, c)):
                continue
            rb_none.append(b["out"] if b["reason"] else 0)
            rb_latest.append(a["reason_tok"] + (a["reply_tok"] - a["reason_tok"]) + b["new_tok"] + b["out"] if a["reason"] else 0)
    for name, v in (("drop all (only the newest reply changes)", rb_none), ("keep only the newest reply's (the one before it changes)", rb_latest)):
        print(f"  rollback per step, {name}: median {st.median(v)}, over {ROLLBACK}: {sum(x > ROLLBACK for x in v)} of {len(v)}")

    print("\n== Q3: plain continuations that did not reuse everything")
    print("growth beyond the new input, less the generated count, less the 2 template tokens: 0 means the reply re-tokenized to its own length")
    full0 = sum((b["inp"] - a["inp"] - b["new_tok"]) - a["out"] - 2 == 0 for L, n in kinds["full"] for a, b in [(L[n - 1], L[n])])
    print(f"full reuses with 0: {full0} of {len(kinds['full'])}")
    for k in ("partial", "cold"):
        for L, n in kinds[k]:
            a, b = L[n - 1], L[n]
            run = [r for r, x in runs.items() if x is L][0]
            delta = (b["inp"] - a["inp"] - b["new_tok"]) - a["out"] - 2
            at = b["cached"] - a["inp"]
            where = ""
            if k == "partial" and 0 <= at < a["out"]:
                ids = TOK.encode(reply_text(a), add_special_tokens=False).ids
                where = repr(TOK.decode(ids[max(0, at - 4):at + 3]))
            print(f"  {k:7s} {run:16s} #{n:2d} prev prompt {a['inp']:5d} reply {a['out']:4d} reused {b['cached']:5d} "
                  f"short {a['inp']+a['out']-b['cached']:5d} count change {delta:+d} user msgs {len(b['pend']['user'])} {where}")


if __name__ == "__main__":
    main()
