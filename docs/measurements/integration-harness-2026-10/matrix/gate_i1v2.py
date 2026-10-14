#!/usr/bin/env python3
"""I1-v2's grader (docs/tasks/task-integration-harness-2026-10.md, "I1 live gate, second design (I1-v2)", registered 2026-10-14).

  gate_i1v2.py <records.jsonl> [...]                     grade every `chain` run, then each agent
  gate_i1v2.py --pilot <ok dir> <records.jsonl> [...]    check each pilot run; write <ok dir>/pilot-ok-<agent> for one that passes

Each run is read from the wire (every request goinfer answered, in order, with its prompt size) and from the agent's own events
(chain reads by index and compactions, in the agent's order). The void classes are checked first, in the registered order, and
the first that holds is the run's grade. The registered order puts "never reached" before "agent first", so a run that
compacted by itself and never met an overflow reads as "never reached".
"""
import json, os, statistics, sys

ROUTES = ("/v1/chat/completions", "/v1/messages", "/v1/responses")
SHRINK = 500


def wire(r):
    """The generation requests as (size, overflow, completion): an answered request's prompt, or the size an overflow named.
    Side requests (under half the session's fixed part, such as titles) are left out."""
    reqs = [x for x in r["wire"]["per_request"] if x["path"] in ROUTES]
    fixed = next((x["prompt_tokens"] for x in reqs if x.get("tools") and x.get("prompt_tokens") and not x.get("overflow")), 0)
    out = []
    for x in reqs:
        ovf = bool(x.get("overflow"))
        size = (x.get("overflow_tokens") or int(r["labels"]["ctx"])) if ovf else x.get("prompt_tokens")
        if size and size >= fixed / 2:
            out.append(dict(size=size, overflow=ovf, completion=x.get("completion_tokens") or 0))
    return fixed, out


def steps(seq):
    """Positive growth between consecutive answered requests: one step of the session's history."""
    ans = [x["size"] for x in seq if not x["overflow"]]
    return [b - a for a, b in zip(ans, ans[1:]) if b > a]


def grade(r):
    W = int(r["labels"]["ctx"])
    F, seq = wire(r)
    ch = r.get("chain") or dict(seq=[], last=-1, decoy_content=False)
    w = r["wire"]
    d = dict(overflows=sum(x["overflow"] for x in seq), compactions=r.get("compactions", 0), fixed=F)
    if r.get("harness_error"):
        return "VOID", dict(d, why="harness error: " + r["harness_error"][:120])
    if r.get("void_halted") or w.get("n_halted") or w.get("n_5xx"):
        return "VOID", dict(d, why="box")
    ovf_at = [i for i, x in enumerate(seq) if x["overflow"]]
    if not ovf_at:
        return "VOID", dict(d, why="never reached")
    shrink_at = [i for i in range(1, len(seq)) if seq[i]["size"] < seq[i - 1]["size"] - SHRINK]
    if shrink_at and shrink_at[0] < ovf_at[0]:
        return "VOID", dict(d, why="agent first")
    last_ok = 0
    for x in seq:
        if x["overflow"]:
            if last_ok and x["size"] - last_ok > W - F - 1000:
                return "VOID", dict(d, why=f"incurable step: {x['size'] - last_ok} tokens, room {W - F - 1000}")
        else:
            last_ok = x["size"]
    st = steps(seq)
    med = statistics.median(st) if st else 0
    if ch["decoy_content"] or any(e[0] == "decoy" for e in ch["seq"]):
        return "VOID", dict(d, why="off-chain: a decoy was read")
    if med and max(st) > 2.5 * med:
        return "VOID", dict(d, why=f"off-chain: a step of {max(st)} tokens against a median of {med:.0f}")
    answered_after = lambda i: any(not x["overflow"] for x in seq[i + 1:])
    for i in shrink_at:
        if seq[i]["overflow"] and not answered_after(i):
            return "VOID", dict(d, why="summary too large")
    answered = [x for x in seq if not x["overflow"]]
    if answered and seq[-1] is answered[-1] and answered[-1]["completion"] >= W - answered[-1]["size"] - 16:
        return "VOID", dict(d, why="clamped end")
    reads = [e[1] for e in ch["seq"] if e[0] == "read"]
    end_read = bool(reads) and max(reads) >= ch["last"]
    ended_badly = r.get("exit") != 0 or r.get("ended_on_error") or seq[-1]["overflow"]
    if not end_read and not ended_badly:
        return "VOID", dict(d, why="model stopped")
    # the agent's own order: compactions between consecutive new chain reads, and runs of them with no new read
    gaps, run, worst, top = [0], 0, 0, -1
    for e in ch["seq"]:
        if e[0] == "compact":
            gaps[-1] += 1; run += 1; worst = max(worst, run)
        elif e[0] == "read" and e[1] > top:
            top = e[1]; gaps.append(0); run = 0
    clamped = sum(1 for x in answered if x["completion"] >= W - x["size"] - 16)
    d.update(max_between_reads=max(gaps), consecutive=worst, clamped_replies=clamped, median_step=med)
    if ended_badly:
        return "FAIL", dict(d, why="ended on an overflow or an error")
    if worst >= 3:
        return "FAIL", dict(d, why="three compactions with no new chain read between them")
    if max(gaps) >= 2:
        return "FAIL", dict(d, why="two compactions between consecutive new chain reads")
    if d["compactions"] > d["overflows"]:
        return "FAIL", dict(d, why="more compactions than overflow answers")
    if end_read and r.get("passed") and r.get("exit") == 0:
        return "PASS", d
    return "OTHER", dict(d, why="END read, no code word in the reply (not a registered class; not gradable)")


def verdicts(rows):
    """Per agent, from its graded and planted runs (the control arm is reported only)."""
    out = []
    for agent in sorted({r["agent"] for r, _, _ in rows}):
        g = [v for r, v, _ in rows if r["agent"] == agent and r["labels"].get("arm") == "graded"]
        p = [v for r, v, _ in rows if r["agent"] == agent and r["labels"].get("arm") == "planted"]
        gradable = [v for v in g if v in ("PASS", "FAIL")]
        if "PASS" in p:
            v = "VOID (the planted run passed: the gate cannot go red for this agent)"
        elif gradable.count("FAIL") >= 2:
            v = "FAIL"
        elif len(gradable) >= 2 and all(x == "PASS" for x in gradable):
            v = "PASS" if "FAIL" in p else "PARKED (the planted run did not FAIL: run it once more; VOID again parks the PASS)"
        else:
            v = "PARKED (three more graded runs the next night, then the rule over all gradable runs)"
        out.append(f"AGENT {agent:12} {v}  graded={g} planted={p}")
    return out


def pilot(r, okdir):
    W = int(r["labels"]["ctx"])
    told = r.get("told") or {}
    v, d = grade(r)
    F, seq = wire(r)
    st = steps(seq)
    med = statistics.median(st) if st else 0
    ovf_at = [i for i, x in enumerate(seq) if x["overflow"]]
    shrink_at = [i for i in range(1, len(seq)) if seq[i]["size"] < seq[i - 1]["size"] - SHRINK]
    before = [x["size"] for x in seq[:ovf_at[0]] if not x["overflow"]] if ovf_at else []
    checks = {
        "told window read back is 2 x W": told.get("window") == 2 * W,
        "an overflow answer came": bool(ovf_at),
        "no shrink before the first overflow": bool(ovf_at) and not (shrink_at and shrink_at[0] < ovf_at[0]),
        "history grew to W less two steps before it": bool(before) and med > 0 and max(before) >= W - 2 * med,
        "chain followed": not (r.get("chain") or {}).get("decoy_content") and not (med and st and max(st) > 2.5 * med),
        "median step 700 to 1,300": 700 <= med <= 1300,
    }
    if r["agent"] == "claude-code":
        checks["output cap read back is 4,096"] = told.get("output_cap") == 4096
    if r["agent"] == "codex":
        seen = r.get("context_window_seen") or 0
        checks["Codex took 95% of 2 x W"] = abs(seen - 0.95 * 2 * W) <= 0.01 * 2 * W
    ok = all(checks.values())
    # a diagnostic, not a check: whether the agent's own compaction events show where the wire shows a shrink
    note = f"wire shrinks after the first overflow {sum(1 for i in shrink_at if ovf_at and i > ovf_at[0])}, agent compaction events {r.get('compactions', 0)}"
    if ok and okdir:
        open(os.path.join(okdir, f"pilot-ok-{r['agent']}"), "w").write(json.dumps(dict(id=r["id"], grade=v, checks=checks)) + "\n")
    return ok, dict(checks, **{"(diagnostic, not a check) " + note: True}), (v, d)


def main(argv):
    okdir = None
    if argv[:1] == ["--pilot"]:
        okdir, argv = argv[1], argv[2:]
    rows = []
    for path in argv:
        for line in open(path):
            r = json.loads(line)
            if r.get("task") != "chain":
                continue
            if okdir is not None:
                ok, checks, (v, d) = pilot(r, okdir)
                print(f"PILOT {'OK  ' if ok else 'FAIL'} {r['agent']:12} {r['labels'].get('config')} run grade {v} {json.dumps(d)}")
                for k, c in checks.items():
                    print(f"      {'yes' if c else 'NO '} {k}")
                continue
            v, d = grade(r)
            rows.append((r, v, d))
            print(f"{v:5} {r['agent']:12} {r['labels'].get('arm', '-'):8} config={r['labels'].get('config')} {json.dumps(d)}")
    if okdir is None:
        for line in verdicts(rows):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
