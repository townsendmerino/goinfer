#!/usr/bin/env python3
"""KV8's served reading against each client alone (README.md here, the second registration): one slot two clients, two slots two clients, each client alone
on one slot, two slots two clients, one slot two clients, with an edit round that makes both clients restore (the third registration). Uses served_two_clients.py's server,
turns and prompts.

  python3 served_against_alone.py --serve <bin> --model <path> --out <dir>

Prints a VERDICT line and writes <out>/turns.jsonl. Exit 0 PASS, 1 FAIL, 2 VOID, 3 PARKED.
"""
import argparse, json, os, statistics, sys, threading, time, importlib.util

spec = importlib.util.spec_from_file_location("two", os.path.join(os.path.dirname(os.path.abspath(__file__)), "served_two_clients.py"))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


EDIT = "Well,"


def script(prompts, name, r, text, last):
    """Extends client name's prompt after round r's reply: the plain follow-up, or, after round 2, a reply rendered back with a word in front of it."""
    if r == 2:
        last[name] = prompts[name]
        return last[name] + EDIT + text + f"\nQuestion {r}: which shelf next?\nAnswer:"
    return prompts[name] + text + f"\nQuestion {r}: which shelf next?\nAnswer:"


def turns_of(rec, label, slots, client, r, text, ttft, u):
    return {"cell": label, "slots": slots, "round": r, "client": client, "ttft_s": ttft, "reused": u.get("prefill_reused_tokens"), "prompt_tokens": u.get("prompt_tokens"), "reply": text}


def cell(a, label, slots, names, out):
    """One server: names take rounds 0, 1 and 2 in turn, then the edit round. With two clients, both then send at once; alone, the client sends that turn on its own."""
    load, avail, vram0 = h.readings()
    rec = {"cell": label, "slots": slots, "load": load, "avail_mb": avail, "vram_before_mib": vram0, "void": load > h.LOAD_VOID}
    print(f"[{time.strftime('%H:%M:%S')}] cell {label}: {slots} slot(s), clients {names}, load {load:.2f}, {avail} MB available, {vram0} MiB on the card", flush=True)
    if rec["void"]:
        print(f"VOID CELL {label}: load {load:.2f} over {h.LOAD_VOID}", flush=True)
        return rec, []
    s = h.Server(a, slots, os.path.join(out, f"serve-{label}.log"))
    rec["load_s"], rec["vram_loaded_mib"] = s.load_s, h.readings()[2]
    prompts = {n: h.preamble({"A": "amber", "B": "birch"}[n]) for n in names}
    last, turns = {}, []

    def note(r, n, text, ttft, u):
        turns.append(turns_of(rec, label, slots, n, r, text, ttft, u))
        print(f"[{time.strftime('%H:%M:%S')}]   {label} round {r} {n}: {u.get('prompt_tokens')} tokens, reused {u.get('prefill_reused_tokens')}, first token {ttft:.2f} s", flush=True)

    try:
        for r in (0, 1, 2, "edit"):
            for n in names:
                text, ttft, u = h.turn(prompts[n])
                note(r, n, text, ttft, u)
                if r == "edit":
                    prompts[n] += text + "\nQuestion e: which shelf next?\nAnswer:"
                else:
                    prompts[n] = script(prompts, n, r, text, last)
        if len(names) > 1:
            got = {}
            ts = [threading.Thread(target=lambda n=n: got.__setitem__(n, h.turn(prompts[n]))) for n in names]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            for n in names:
                note("both", n, *got[n])
        else:
            note("both", names[0], *h.turn(prompts[names[0]]))
    finally:
        rec["peak_swap_mb"], rec["peak_hwm_mb"] = s.peak_swap, s.peak_hwm
        s.close()
    rec["last_prompt_tokens"] = {n: next(t["prompt_tokens"] for t in turns if t["client"] == n and t["round"] == 2) for n in names}
    return rec, turns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--serve", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plan = [("A1", 1, "AB"), ("B1", 2, "AB"), ("alone-A", 1, "A"), ("alone-B", 1, "B"), ("B2", 2, "AB"), ("A2", 1, "AB")]
    cells, done = [], {}
    for label, slots, names in plan:
        rec, ts = cell(a, label, slots, names, a.out)
        if rec["void"]:  # re-run in the same order, once, and the record lists it
            cells.append(rec)
            time.sleep(60)
            rec, ts = cell(a, label + "r", slots, names, a.out)
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

    def median(ts, rounds):
        return statistics.median(t["ttft_s"] for t in ts if t["round"] in rounds)

    ratios, edits = [], []
    for pair in (1, 2):
        one, two = done[f"A{pair}"], done[f"B{pair}"]
        ratios.append(median(two, (1, 2)) / median(one, (1, 2)))
        edits.append(median(two, ("edit",)) / median(one, ("edit",)))
        print(f"pair {pair}: rounds 1 and 2, one slot {median(one, (1, 2)):.2f} s, two slots {median(two, (1, 2)):.2f} s, ratio {ratios[-1]:.3f}; the edit round, one slot {median(one, ('edit',)):.2f} s, two slots {median(two, ('edit',)):.2f} s, ratio {edits[-1]:.3f}")
    two = done["B1"] + done["B2"]
    unreused = [t for t in two if t["round"] != 0 and not t["reused"]]
    print(f"two slots: {len(unreused)} of {sum(t['round'] != 0 for t in two)} follow-ups reused nothing, the edit and at-once rounds included")
    # The edit round's restores: exactly the client's last prompt's rows.
    unrestored = []
    for label in ("B1", "B2", "alone-A", "alone-B"):
        for t in done[label]:
            if t["round"] == "edit":
                want = next(x["prompt_tokens"] for x in done[label] if x["client"] == t["client"] and x["round"] == 2)
                print(f"{label} client {t['client']}: the edit round reused {t['reused']}, its last prompt was {want} tokens, first token {t['ttft_s']:.2f} s")
                if t["reused"] != want:
                    unrestored.append((label, t["client"], t["reused"], want))
    # Each two-slot client against itself alone, every turn, up to its first difference: past that the prompts differ.
    differ, compared = 0, 0
    for pair in (1, 2):
        for name in "AB":
            for r in (0, 1, 2, "edit", "both"):
                x = next(t for t in done[f"alone-{name}"] if t["round"] == r)
                y = next(t for t in done[f"B{pair}"] if t["client"] == name and t["round"] == r)
                compared += 1
                if x["reply"] != y["reply"]:
                    differ += 1
                    print(f"pair {pair} client {name} round {r}: two slots reused {y['reused']}, alone {x['reused']}: the replies differ: alone {x['reply']!r}, two slots {y['reply']!r}")
                    break
    print(f"{differ} of 4 conversations differ from that client alone; {compared} turns compared")
    # Reported, not graded: the first round's comparison, for what it is.
    cold = sum(next(t for t in done["A1"] if t["client"] == t2["client"] and t["round"] == t2["round"])["reply"] != t2["reply"] for t2 in done["B1"])
    print(f"reported: {cold} of {len(done['B1'])} two-slot turns differ from the cold one-slot arm's, whose prompts diverge after the first")
    swap = max(c.get("peak_swap_mb", 0) for c in cells)
    print(f"peak VmSwap {swap} MB (the bar: {h.SWAP_MB})")
    worst, worst_edit = max(ratios), max(edits)
    if swap > h.SWAP_MB or differ or unreused or unrestored or worst > h.FAIL_BAR or worst_edit > h.FAIL_BAR:
        print(f"VERDICT: FAIL (ratio {worst:.3f}, edit ratio {worst_edit:.3f}, {differ} differ, {len(unreused)} unreused, unrestored {unrestored}, swap {swap} MB)")
        return 1
    if worst > h.PASS_BAR or worst_edit > h.PASS_BAR:
        print(f"VERDICT: PARKED (ratio {worst:.3f}, edit ratio {worst_edit:.3f}, in the band {h.PASS_BAR}-{h.FAIL_BAR})")
        return 3
    print(f"VERDICT: PASS (worst ratio {worst:.3f}, edit ratio {worst_edit:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
