#!/usr/bin/env python3
"""KV8's message-boundary checkpoint, gate T1 (README.md here): serve at main against serve at the change, A B B A, one server per cell, on a chat conversation.

  python3 served_msgboundary.py --main <bin> --change <bin> --model <path> --out <dir> [--fit-only]

Each cell: r1 (a cold conversation of about 8,000 ids), r2 (r1's history, a scripted reply, a short new question: the Keep restore), r3 (r1 with
its last message rewritten from its first words: the case the message-boundary snapshot serves) and, from the second slice's T2, r4 (r3's history, a
scripted reply, a new question: the turn after an edit). Greedy, thinking off, 24 new tokens. T1's bars graded r1 to r3 on rows and replies as below;
T2 (README.md, second slice) grades every reply, r4's rows and the cold request's copy cost too. Prints a VERDICT
line and writes <out>/turns.jsonl. Exit 0 PASS, 1 FAIL, 2 VOID, 3 PARKED.
"""
import argparse, importlib.util, json, os, re, statistics, subprocess, sys, time, urllib.request

spec = importlib.util.spec_from_file_location("two", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../kv8-2026-11/served_two_clients.py"))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
h.CTX = 12288  # --ctx overrides

SYSTEM_LINES, HIST_LINES, LAST_LINES = 105, 61, 49
PASS_BAR, FAIL_BAR, R2_BAR, R1_BAR, R1_FAIL = 0.5, 0.7, 1.10, 1.05, 1.10
COST_MS, COST_MIB, COST_FAIL_MS, COST_FAIL_MIB = 150, 175, 250, 250
REPLY = "Acknowledged. Nothing else to add on that."


def lines(tag, n):
    return "".join(f"Note {i} of {tag}: the {tag} ledger entry {i * 7 % 101} moved to shelf {i * 13 % 97} on day {i % 28 + 1}.\n" for i in range(n))


def conversation(edit):
    """r1's messages; with edit, the last message begins with one more word (a rewrite from its first ids)."""
    last = lines("amber final", LAST_LINES)
    return [
        {"role": "system", "content": "You answer questions about ledgers.\n" + lines("system", SYSTEM_LINES)},
        {"role": "user", "content": lines("first", HIST_LINES) + "Question 1: which shelf?"},
        {"role": "assistant", "content": REPLY},
        {"role": "user", "content": lines("second", HIST_LINES) + "Question 2: which shelf?"},
        {"role": "assistant", "content": REPLY},
        {"role": "user", "content": ("Please " if edit else "") + last + "Question 3: which shelf?"},
    ]


def chat(msgs):
    body = {"model": h.model_id(), "messages": msgs, "max_tokens": h.MAX_NEW, "temperature": 0, "seed": 1, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(h.BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, first, text, usage = time.time(), None, [], {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices") or []:
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    first = first or time.time()
                    text.append(piece)
    return "".join(text), (first or time.time()) - t0, usage


def boundary_estimate():
    """About where r1's last message starts: the tokens of the history before it, by /v1/messages/count_tokens, plus the opening marker. None if the route refuses."""
    msgs = conversation(False)
    hist = [{"role": m["role"], "content": m["content"]} for m in msgs[1:5]]
    body = {"model": h.model_id(), "system": msgs[0]["content"], "messages": hist + [{"role": "user", "content": "x"}]}
    try:
        req = urllib.request.Request(h.BASE + "/v1/messages/count_tokens", json.dumps(body).encode(), {"Content-Type": "application/json"})
        n = json.load(urllib.request.urlopen(req, timeout=30))["input_tokens"]
    except Exception as e:  # noqa: BLE001
        print("boundary estimate unavailable:", e)
        return None
    return n - 4  # "x" and its turn's role and end markers, and the generation prompt, are not history; read as an estimate within a few ids


class A:  # the h.Server's argument holder
    def __init__(self, serve, model):
        self.serve, self.model = serve, model


def cell(label, which, binary, model, out):
    load, avail, vram0 = h.readings()
    rec = {"cell": label, "build": which, "load": load, "avail_mb": avail, "vram_before_mib": vram0, "void": load > h.LOAD_VOID}
    print(f"[{time.strftime('%H:%M:%S')}] cell {label} ({which}): load {load:.2f}, {avail} MB available, {vram0} MiB on the card", flush=True)
    if rec["void"]:
        print(f"VOID CELL {label}: load {load:.2f} over {h.LOAD_VOID}", flush=True)
        return rec, []
    log = os.path.join(out, f"serve-{label}.log")
    s = h.Server(A(binary, model), 1, log)
    rec["load_s"], rec["vram_loaded_mib"] = s.load_s, h.readings()[2]
    turns = []
    rec["boundary_est"] = boundary_estimate()
    r1 = conversation(False)
    r3 = conversation(True)
    plan = [("r1", r1), ("r2", r1 + [{"role": "assistant", "content": REPLY}, {"role": "user", "content": "Question 4: and which shelf after that?"}]), ("r3", r3),
            ("r4", r3 + [{"role": "assistant", "content": REPLY}, {"role": "user", "content": "Question 5: and which shelf after that?"}])]
    try:
        for name, msgs in plan:
            text, ttft, u = chat(msgs)
            turns.append({"cell": label, "build": which, "req": name, "ttft_s": ttft, "reused": u.get("prefill_reused_tokens"), "prompt_tokens": u.get("prompt_tokens"), "reply": text})
            print(f"[{time.strftime('%H:%M:%S')}]   {label} {name}: {u.get('prompt_tokens')} tokens, reused {u.get('prefill_reused_tokens')}, first token {ttft:.2f} s", flush=True)
    finally:
        rec["peak_swap_mb"], rec["peak_hwm_mb"] = s.peak_swap, s.peak_hwm
        s.close()
    took = re.findall(r"kv snapshot: took (\d+) rows \((\d+) MiB host\) from slot \d+ in ([\d.]+) ms", open(log).read())
    rec["snapshots_taken"] = [int(r) for r, _, _ in took]
    rec["snapshot_cost"] = [{"rows": int(r), "mib": int(mib), "ms": float(ms)} for r, mib, ms in took]
    print(f"[{time.strftime('%H:%M:%S')}]   {label}: loaded in {rec['load_s']:.1f} s, peak VmSwap {s.peak_swap} MB, snapshots taken (rows) {rec['snapshots_taken']}", flush=True)
    return rec, turns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main", required=True)
    p.add_argument("--change", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fit-only", action="store_true")
    p.add_argument("--ctx", type=int, default=12288)
    p.add_argument("--no-regression", action="store_true", help="T3b': both arms already serve r3 from the boundary snapshot; grade that nothing moved")
    p.add_argument("--report-cost-only", action="store_true", help="the 9B: the snapshot copy's host MiB and ms are reported, not graded against the 0.8B's bars (the first-token bars still apply)")
    a = p.parse_args()
    h.CTX = a.ctx
    os.makedirs(a.out, exist_ok=True)
    if a.fit_only:
        s = h.Server(A(a.change, a.model), 1, os.path.join(a.out, "fit.log"))
        print(f"loaded in {s.load_s:.1f} s, {h.readings()[2]} MiB on the card")
        s.close()
        for l in open(os.path.join(a.out, "fit.log")):
            if any(k in l for k in ("fit:", "KV plan", "recurrent checkpoint", "KV slots", "resident context", "decline", "snapshot")):
                print("  " + l.rstrip()[:200])
        return 0
    plan = [("A1", "main", a.main), ("B1", "change", a.change), ("B2", "change", a.change), ("A2", "main", a.main)]
    cells, done = [], {}
    for label, which, binary in plan:
        rec, ts = cell(label, which, binary, a.model, a.out)
        if rec["void"]:  # re-run in the same order, once, and the record lists it
            cells.append(rec)
            time.sleep(60)
            rec, ts = cell(label + "r", which, binary, a.model, a.out)
        cells.append(rec)
        done[label] = ts
        time.sleep(3)
    with open(os.path.join(a.out, "turns.jsonl"), "w") as f:
        for c in cells:
            f.write(json.dumps(c) + "\n")
        for ts in done.values():
            for t in ts:
                f.write(json.dumps(t) + "\n")
    if any(c["void"] for c in cells if c["cell"].endswith("r")):
        print("VERDICT: VOID (a re-run cell was also void)")
        return 2
    by = lambda label, req: next(t for t in done[label] if t["req"] == req)
    verdict, notes = "PASS", []

    def worse(v):
        nonlocal verdict
        order = ["PASS", "PARKED", "FAIL"]
        verdict = max(verdict, v, key=order.index)

    rec_by = {c["cell"]: c for c in cells}
    if a.no_regression:
        for pair, (old, new) in {"pair 1": ("A1", "B1"), "pair 2": ("A2", "B2")}.items():
            for r in ("r1", "r2", "r3", "r4"):
                o, n = by(old, r), by(new, r)
                ratio, delta = n["ttft_s"] / o["ttft_s"], n["ttft_s"] - o["ttft_s"]
                print(f"{pair}: {r} {o['ttft_s']:.2f} -> {n['ttft_s']:.2f} s (x{ratio:.3f}, {delta:+.2f} s); rows {o['reused']} -> {n['reused']}")
                if r == "r3":
                    worse("PASS" if 0.98 <= ratio <= 1.02 else "PARKED" if 0.95 <= ratio <= 1.05 else "FAIL")
                    if n["reused"] != o["reused"]:
                        notes.append(f"{pair}: r3 rows {n['reused']} against main's {o['reused']}"); worse("FAIL")
                elif r == "r1":
                    worse("PASS" if ratio <= R1_BAR else "PARKED" if ratio <= R1_FAIL else "FAIL")
                else:
                    worse("PASS" if ratio <= 1.05 or delta <= 0.03 else "PARKED" if ratio <= 1.10 or delta <= 0.06 else "FAIL")
                    if n["reused"] < o["reused"]:
                        notes.append(f"{pair}: {r} rows {n['reused']} against main's {o['reused']}"); worse("FAIL")
                if n["reply"] != o["reply"]:
                    notes.append(f"{pair}: {r}'s reply differs from main's"); worse("FAIL")
            for c in rec_by[new]["snapshot_cost"][:3]:
                print(f"{pair}: cold request's snapshot at {c['rows']} rows: {c['mib']} MiB host, {c['ms']} ms")
        for c in cells:
            if c.get("peak_swap_mb", 0) > h.SWAP_MB:
                notes.append(f"{c['cell']}: VmSwap {c['peak_swap_mb']} MB"); worse("FAIL")
        print("VERDICT:", verdict, "; ".join(notes))
        return {"PASS": 0, "FAIL": 1, "PARKED": 3}[verdict]
    for pair, (old, new) in {"pair 1": ("A1", "B1"), "pair 2": ("A2", "B2")}.items():
        r1, r2, r3 = (by(new, r)["ttft_s"] / by(old, r)["ttft_s"] for r in ("r1", "r2", "r3"))
        r4 = by(new, "r4")["ttft_s"] / by(old, "r4")["ttft_s"]
        print(f"{pair}: r1 {by(old, 'r1')['ttft_s']:.2f} -> {by(new, 'r1')['ttft_s']:.2f} s (x{r1:.3f}); r2 {by(old, 'r2')['ttft_s']:.2f} -> {by(new, 'r2')['ttft_s']:.2f} s (x{r2:.3f}); r3 {by(old, 'r3')['ttft_s']:.2f} -> {by(new, 'r3')['ttft_s']:.2f} s (x{r3:.3f}); r4 {by(old, 'r4')['ttft_s']:.2f} -> {by(new, 'r4')['ttft_s']:.2f} s (x{r4:.3f}, reported)")
        worse("FAIL" if r3 > FAIL_BAR else "PARKED" if r3 > PASS_BAR else "PASS")
        worse("FAIL" if r1 > R1_FAIL else "PARKED" if r1 > R1_BAR else "PASS")
        if r2 > R2_BAR:
            notes.append(f"{pair}: r2 slower than {R2_BAR} of main's"); worse("FAIL")
        if by(new, "r2")["reused"] < by(old, "r2")["reused"]:
            notes.append(f"{pair}: r2 reuses {by(new, 'r2')['reused']}, main {by(old, 'r2')['reused']}"); worse("FAIL")
        if by(new, "r4")["reused"] < by(old, "r4")["reused"]:
            notes.append(f"{pair}: r4 reuses {by(new, 'r4')['reused']}, main {by(old, 'r4')['reused']}: a restored turn lost its Keep point"); worse("FAIL")
        print(f"{pair}: r4 reuse change {by(new, 'r4')['reused']}, main {by(old, 'r4')['reused']}")
        taken = rec_by[new]["snapshots_taken"]
        want = sorted(taken)[-2] if len(taken) >= 2 else None  # the message's start: the second largest of r1's snapshots (Keep's is the largest; a preamble snapshot, when taken, the smallest)
        got = by(new, "r3")["reused"]
        est = rec_by[new].get("boundary_est")
        print(f"{pair}: r3 reuse change {got}, main {by(old, 'r3')['reused']}; the message-boundary snapshot of r1 {want}; boundary estimate {est}; snapshots taken {taken}")
        if want is None or got != want or by(old, "r3")["reused"] or want % 64:
            notes.append(f"{pair}: r3 reuse is not exactly the boundary's block"); worse("FAIL")
        elif est is not None and not (est - 70 <= want <= est + 4):
            notes.append(f"{pair}: the snapshot {want} is not within a block of the estimated boundary {est}"); worse("FAIL")
        for r in ("r1", "r2", "r3", "r4"):
            if by(new, r)["reply"] != by(old, r)["reply"]:
                notes.append(f"{pair}: {r}'s reply differs from main's"); worse("FAIL")
        cost = rec_by[new]["snapshot_cost"][:3]
        for c in cost:
            print(f"{pair}: cold request's snapshot at {c['rows']} rows: {c['mib']} MiB host, {c['ms']} ms")
        if cost and not a.report_cost_only:
            c = cost[0]  # the boundary snapshot, the smaller
            if c["ms"] > COST_FAIL_MS or c["mib"] > COST_FAIL_MIB:
                notes.append(f"{pair}: the boundary snapshot costs {c['ms']} ms and {c['mib']} MiB"); worse("FAIL")
            elif c["ms"] > COST_MS or c["mib"] > COST_MIB:
                notes.append(f"{pair}: the boundary snapshot costs {c['ms']} ms and {c['mib']} MiB: the owner weighs it"); worse("PARKED")
    for c in cells:
        if c.get("peak_swap_mb", 0) > h.SWAP_MB:
            notes.append(f"{c['cell']}: VmSwap {c['peak_swap_mb']} MB"); worse("FAIL")
    print("VERDICT:", verdict, "; ".join(notes))
    return {"PASS": 0, "FAIL": 1, "PARKED": 3}[verdict]


if __name__ == "__main__":
    sys.exit(main())
