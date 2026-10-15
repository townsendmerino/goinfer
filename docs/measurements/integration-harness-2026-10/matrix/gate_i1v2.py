#!/usr/bin/env python3
"""I1-v2's grader (docs/tasks/task-integration-harness-2026-10.md, "I1 live gate, second design (I1-v2)", registered 2026-10-14,
with its amendment 1 of the same day, made before any run).

  gate_i1v2.py [--fallback agent,...] <records.jsonl> [...]   grade every `chain` run, then each agent
  gate_i1v2.py --pilot <ok dir> <records.jsonl> [...]          check each pilot run; write <ok dir>/pilot-ok-<agent> for one that passes

Graded and planted runs: the void classes are checked first, in the registered order, and the first that holds is the run's
grade; then PASS or FAIL. A control run is not graded: it reports what came first, the agent's own compaction, goinfer's overflow
answer, or neither (amendment 1).

Compactions come from the agent's own events. The wire detector (amendment 1) counts a compaction wherever the history above the
fixed part shrinks by more than DROP between consecutive main-agent requests. It replaces the agent's own count for an agent
named in --fallback (one whose pilot showed its compactions invisible), and is otherwise a cross-check: a run whose PASS or FAIL
would flip on the other count is marked contested, and an agent with a contested deciding run is parked.
"""
import json, os, statistics, sys

ROUTES = ("/v1/chat/completions", "/v1/messages", "/v1/responses")
SHRINK = 500   # tokens: the registered "agent first" shrink
DROP = 0.25    # of the history above the fixed part: amendment 1's wire compaction


def wire(r):
    """The generation requests as (size, overflow, completion): an answered request's prompt, or the size an overflow named.
    Side requests (under half the session's fixed part, such as titles) are left out."""
    reqs = [x for x in r["wire"]["per_request"] if x["path"] in ROUTES]
    fixed = next((x["prompt_tokens"] for x in reqs if x.get("tools") and x.get("prompt_tokens") and not x.get("overflow")), 0)
    out = []
    for i, x in enumerate(reqs):
        ovf = bool(x.get("overflow"))
        size = (x.get("overflow_tokens") or int(r["labels"]["ctx"])) if ovf else x.get("prompt_tokens")
        if size and size >= fixed / 2:
            out.append(dict(size=size, overflow=ovf, completion=x.get("completion_tokens") or 0, pos=i))
    return fixed, out


def main_series(r):
    """The main agent's answered requests, in order, as (position among all generation requests, prompt): those carrying the
    session's first tool count. Side requests, summarisation requests (no tools) and a subagent's (another tool count) are left out."""
    reqs = [x for x in r["wire"]["per_request"] if x["path"] in ROUTES]
    nt = next((x["tools"] for x in reqs if x.get("tools") and x.get("prompt_tokens") and not x.get("overflow")), None)
    return [(i, x["prompt_tokens"]) for i, x in enumerate(reqs) if nt and x.get("tools") == nt and x.get("prompt_tokens") and not x.get("overflow")]


def wire_compactions(r):
    """Amendment 1: positions (among all generation requests) of the main-agent requests that follow a compaction: the history
    above the fixed part F (the first main-agent prompt) fell by more than DROP of itself since the previous main-agent request.
    The runner never sends /compact, so no drop is exempted."""
    s = main_series(r)
    if not s:
        return []
    F = s[0][1]
    return [b[0] for a, b in zip(s, s[1:]) if a[1] > F and a[1] - b[1] > DROP * (a[1] - F)]


def wire_events(r):
    """The wire's own account of the session as agent-style events: ["compact"] at each wire compaction, ["read", k] at each
    main-agent request that grew by at least half the median step (a new file read, k counting them)."""
    s = main_series(r)
    comp = set(wire_compactions(r))
    grow = [b[1] - a[1] for a, b in zip(s, s[1:]) if b[1] > a[1]]
    med = statistics.median(grow) if grow else 0
    ev, k = [], 0
    for a, b in zip(s, s[1:]):
        if b[0] in comp:
            ev.append(["compact"])
        elif med and b[1] - a[1] >= med / 2:
            k += 1
            ev.append(["read", k])
    return ev


def steps(seq):
    """Positive growth between consecutive answered requests: one step of the session's history."""
    ans = [x["size"] for x in seq if not x["overflow"]]
    return [b - a for a, b in zip(ans, ans[1:]) if b > a]


def judge(events, ncomp, novf, end_read, ended_badly, passed):
    """PASS or FAIL from a sequence of compactions and new reads, by the registered rules."""
    gaps, run, worst, top = [0], 0, 0, -1
    for e in events:
        if e[0] == "compact":
            gaps[-1] += 1; run += 1; worst = max(worst, run)
        elif e[0] == "read" and e[1] > top:
            top = e[1]; gaps.append(0); run = 0
    d = dict(max_between_reads=max(gaps), consecutive=worst)
    if ended_badly:
        return "FAIL", dict(d, why="ended on an overflow or an error")
    if worst >= 3:
        return "FAIL", dict(d, why="three compactions with no new chain read between them")
    if max(gaps) >= 2:
        return "FAIL", dict(d, why="two compactions between consecutive new chain reads")
    if ncomp > novf:
        return "FAIL", dict(d, why="more compactions than overflow answers")
    if end_read and passed:
        return "PASS", d
    return "OTHER", dict(d, why="END read, no code word in the reply (not a registered class; not gradable)")


def control_report(r):
    """Amendment 1: a control run is reported, not graded: what came first."""
    F, seq = wire(r)
    reqs_ovf = [x["pos"] for x in seq if x["overflow"]]
    wc = wire_compactions(r)
    own = r.get("compactions", 0)
    if not reqs_ovf and not wc and not own:
        first = "neither"
    elif wc and (not reqs_ovf or wc[0] < reqs_ovf[0]):
        first = "agent first"
    elif not reqs_ovf:
        first = "agent first"   # the agent's own events show a compaction and goinfer never answered an overflow
    else:
        first = "goinfer overflow first"
    return "REPORT", dict(first=first, overflows=len(reqs_ovf), compactions_own=own, compactions_wire=len(wc))


def grade(r, fallback=False):
    if r["labels"].get("arm") == "control":
        return control_report(r)
    W = int(r["labels"]["ctx"])
    F, seq = wire(r)
    ch = r.get("chain") or dict(seq=[], last=-1, decoy_content=False)
    w = r["wire"]
    wc = wire_compactions(r)
    own = r.get("compactions", 0)
    d = dict(overflows=sum(x["overflow"] for x in seq), compactions_own=own, compactions_wire=len(wc), fixed=F)
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
    for i in shrink_at:
        if seq[i]["overflow"] and not any(not x["overflow"] for x in seq[i + 1:]):
            return "VOID", dict(d, why="summary too large")
    answered = [x for x in seq if not x["overflow"]]
    if answered and seq[-1] is answered[-1] and answered[-1]["completion"] >= W - answered[-1]["size"] - 16:
        return "VOID", dict(d, why="clamped end")
    reads = [e[1] for e in ch["seq"] if e[0] == "read"]
    end_read = bool(reads) and max(reads) >= ch["last"]
    ended_badly = bool(r.get("exit") != 0 or r.get("ended_on_error") or seq[-1]["overflow"])
    if not end_read and not ended_badly:
        return "VOID", dict(d, why="model stopped")
    d.update(clamped_replies=sum(1 for x in answered if x["completion"] >= W - x["size"] - 16), median_step=med,
             compactions_used="wire (fallback)" if fallback else "own")
    passed = bool(r.get("passed")) and r.get("exit") == 0
    own_v, own_d = judge(ch["seq"], own, d["overflows"], end_read, ended_badly, passed)
    wire_v, wire_d = judge(wire_events(r), len(wc), d["overflows"], end_read, ended_badly, passed)
    v, dd = (wire_v, wire_d) if fallback else (own_v, own_d)
    other = own_v if fallback else wire_v
    if {v, other} == {"PASS", "FAIL"}:
        dd = dict(dd, contested=f"the {'own' if fallback else 'wire'} count reads {other}")
    return v, dict(d, **dd)


def verdicts(rows):
    """Per agent, from its graded and planted runs (control runs are reported only)."""
    out = []
    for agent in sorted({r["agent"] for r, _, _ in rows}):
        g = [(v, d) for r, v, d in rows if r["agent"] == agent and r["labels"].get("arm") == "graded"]
        p = [v for r, v, _ in rows if r["agent"] == agent and r["labels"].get("arm") == "planted"]
        gradable = [v for v, _ in g if v in ("PASS", "FAIL")]
        contested = any(d.get("contested") for v, d in g if v in ("PASS", "FAIL"))
        if "PASS" in p:
            v = "VOID (the planted run passed: the gate cannot go red for this agent)"
        elif contested:
            v = "PARKED (a deciding run is contested between the agent's own compaction count and the wire's)"
        elif gradable.count("FAIL") >= 2:
            v = "FAIL"
        elif len(gradable) >= 2 and all(x == "PASS" for x in gradable):
            v = "PASS" if "FAIL" in p else "PARKED (the planted run did not FAIL: run it once more; VOID again parks the PASS)"
        else:
            v = "PARKED (three more graded runs the next night, then the rule over all gradable runs)"
        out.append(f"AGENT {agent:12} {v}  graded={[v for v, _ in g]} planted={p}")
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
        cap = int(r["labels"].get("expect_output_cap") or 0)
        checks[f"output cap read back is connect's ({cap})"] = bool(cap) and told.get("output_cap") == cap
    if r["agent"] == "codex":
        seen = r.get("context_window_seen") or 0
        checks["Codex took 95% of 2 x W"] = abs(seen - 0.95 * 2 * W) <= 0.01 * 2 * W
    ok = all(checks.values())
    # amendment 1: the wire detector falls back for an agent whose own compactions do not show where the wire shows them
    wc, own = len(wire_compactions(r)), r.get("compactions", 0)
    fallback = wc > 0 and own == 0
    note = f"wire compactions {wc}, agent compaction events {own}: {'FALLBACK to the wire detector' if fallback else 'own events used, wire cross-checks'}"
    if ok and okdir:
        open(os.path.join(okdir, f"pilot-ok-{r['agent']}"), "w").write(json.dumps(dict(id=r["id"], grade=v, checks=checks, fallback=fallback)) + "\n")
    return ok, dict(checks, **{"(diagnostic, not a check) " + note: True}), (v, d)


def main(argv):
    okdir, fallback = None, set()
    if argv[:1] == ["--pilot"]:
        okdir, argv = argv[1], argv[2:]
    if argv[:1] == ["--fallback"]:
        fallback, argv = set(filter(None, argv[1].split(","))), argv[2:]
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
            v, d = grade(r, fallback=r["agent"] in fallback)
            rows.append((r, v, d))
            print(f"{v:6} {r['agent']:12} {r['labels'].get('arm', '-'):8} config={r['labels'].get('config')} {json.dumps(d)}")
    if okdir is None:
        for line in verdicts(rows):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
