#!/usr/bin/env python3
"""KV8's message-boundary checkpoint, gate T1 (README.md here): serve at main against serve at the change, A B B A, one server per cell, on a chat conversation.

  python3 served_msgboundary.py --main <bin> --change <bin> --model <path> --out <dir> [--fit-only]

Each cell: r1 (a cold conversation of about 8,000 ids), r2 (r1's history, a scripted reply, a short new question: the Keep restore) and r3 (r1 with
its last message rewritten from its first words: the case the message-boundary snapshot serves). Greedy, thinking off, 24 new tokens. Prints a VERDICT
line and writes <out>/turns.jsonl. Exit 0 PASS, 1 FAIL, 2 VOID, 3 PARKED.
"""
import argparse, importlib.util, json, os, re, statistics, subprocess, sys, time, urllib.request

spec = importlib.util.spec_from_file_location("two", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../kv8-2026-11/served_two_clients.py"))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
h.CTX = 12288

SYSTEM_LINES, HIST_LINES, LAST_LINES = 105, 61, 49
PASS_BAR, FAIL_BAR, R2_BAR, R1_BAR, R1_FAIL = 0.5, 0.7, 1.10, 1.05, 1.10
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
    r1 = conversation(False)
    plan = [("r1", r1), ("r2", r1 + [{"role": "assistant", "content": REPLY}, {"role": "user", "content": "Question 4: and which shelf after that?"}]), ("r3", conversation(True))]
    try:
        for name, msgs in plan:
            text, ttft, u = chat(msgs)
            turns.append({"cell": label, "build": which, "req": name, "ttft_s": ttft, "reused": u.get("prefill_reused_tokens"), "prompt_tokens": u.get("prompt_tokens"), "reply": text})
            print(f"[{time.strftime('%H:%M:%S')}]   {label} {name}: {u.get('prompt_tokens')} tokens, reused {u.get('prefill_reused_tokens')}, first token {ttft:.2f} s", flush=True)
    finally:
        rec["peak_swap_mb"], rec["peak_hwm_mb"] = s.peak_swap, s.peak_hwm
        s.close()
    rec["snapshots_taken"] = [int(m) for m in re.findall(r"kv snapshot: took (\d+) rows", open(log).read())]
    print(f"[{time.strftime('%H:%M:%S')}]   {label}: loaded in {rec['load_s']:.1f} s, peak VmSwap {s.peak_swap} MB, snapshots taken (rows) {rec['snapshots_taken']}", flush=True)
    return rec, turns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main", required=True)
    p.add_argument("--change", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fit-only", action="store_true")
    a = p.parse_args()
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
    for pair, (old, new) in {"pair 1": ("A1", "B1"), "pair 2": ("A2", "B2")}.items():
        r1, r2, r3 = (by(new, r)["ttft_s"] / by(old, r)["ttft_s"] for r in ("r1", "r2", "r3"))
        print(f"{pair}: r1 {by(old, 'r1')['ttft_s']:.2f} -> {by(new, 'r1')['ttft_s']:.2f} s (x{r1:.3f}); r2 {by(old, 'r2')['ttft_s']:.2f} -> {by(new, 'r2')['ttft_s']:.2f} s (x{r2:.3f}); r3 {by(old, 'r3')['ttft_s']:.2f} -> {by(new, 'r3')['ttft_s']:.2f} s (x{r3:.3f})")
        worse("FAIL" if r3 > FAIL_BAR else "PARKED" if r3 > PASS_BAR else "PASS")
        worse("FAIL" if r1 > R1_FAIL else "PARKED" if r1 > R1_BAR else "PASS")
        if r2 > R2_BAR:
            notes.append(f"{pair}: r2 slower than {R2_BAR} of main's"); worse("FAIL")
        if by(new, "r2")["reused"] < by(old, "r2")["reused"]:
            notes.append(f"{pair}: r2 reuses {by(new, 'r2')['reused']}, main {by(old, 'r2')['reused']}"); worse("FAIL")
        taken = rec_by[new]["snapshots_taken"]
        want = min(taken[:2]) if len(taken) >= 2 else None  # r1's two snapshots: the message's start is the smaller
        got = by(new, "r3")["reused"]
        print(f"{pair}: r3 reuse change {got}, main {by(old, 'r3')['reused']}; the message-boundary snapshot of r1 {want}; snapshots taken {taken}")
        if want is None or got != want or by(old, "r3")["reused"]:
            notes.append(f"{pair}: r3 reuse is not exactly the boundary's rows"); worse("FAIL")
        if by(new, "r3")["reply"] != by(old, "r3")["reply"]:
            notes.append(f"{pair}: r3's reply differs from main's cold reply"); worse("FAIL")
    for c in cells:
        if c.get("peak_swap_mb", 0) > h.SWAP_MB:
            notes.append(f"{c['cell']}: VmSwap {c['peak_swap_mb']} MB"); worse("FAIL")
    print("VERDICT:", verdict, "; ".join(notes))
    return {"PASS": 0, "FAIL": 1, "PARKED": 3}[verdict]


if __name__ == "__main__":
    sys.exit(main())
