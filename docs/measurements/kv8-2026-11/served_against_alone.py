#!/usr/bin/env python3
"""KV8's served reading against each client alone (README.md here, the second registration): one slot two clients, two slots two clients, each client alone
on one slot, two slots two clients, one slot two clients. Uses served_two_clients.py's server, turns and prompts.

  python3 served_against_alone.py --serve <bin> --model <path> --out <dir>

Prints a VERDICT line and writes <out>/turns.jsonl. Exit 0 PASS, 1 FAIL, 2 VOID, 3 PARKED.
"""
import argparse, json, os, statistics, sys, time, importlib.util

spec = importlib.util.spec_from_file_location("two", os.path.join(os.path.dirname(os.path.abspath(__file__)), "served_two_clients.py"))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def alone(a, label, client, out):
    """One client on one slot: its three turns, then the turn the at-once round sends, on its own."""
    load, avail, vram0 = h.readings()
    rec = {"cell": label, "slots": 1, "load": load, "avail_mb": avail, "vram_before_mib": vram0, "void": load > h.LOAD_VOID}
    print(f"[{time.strftime('%H:%M:%S')}] cell {label}: one slot, client {client} alone, load {load:.2f}, {avail} MB available, {vram0} MiB on the card", flush=True)
    if rec["void"]:
        print(f"VOID CELL {label}: load {load:.2f} over {h.LOAD_VOID}", flush=True)
        return rec, []
    s = h.Server(a, 1, os.path.join(out, f"serve-{label}.log"))
    prompt, turns = h.preamble({"A": "amber", "B": "birch"}[client]), []
    try:
        for r in list(range(h.ROUNDS)) + ["both"]:
            text, ttft, u = h.turn(prompt)
            turns.append({"cell": label, "slots": 1, "round": r, "client": client, "ttft_s": ttft, "reused": u.get("prefill_reused_tokens"), "prompt_tokens": u.get("prompt_tokens"), "reply": text})
            print(f"[{time.strftime('%H:%M:%S')}]   {label} round {r}: {u.get('prompt_tokens')} tokens, reused {u.get('prefill_reused_tokens')}, first token {ttft:.2f} s", flush=True)
            if r != "both":
                prompt += text + f"\nQuestion {r}: which shelf next?\nAnswer:"
    finally:
        rec["peak_swap_mb"], rec["peak_hwm_mb"] = s.peak_swap, s.peak_hwm
        s.close()
    return rec, turns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--serve", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plan = [("A1", 1, 1), ("B1", 2, 1), ("alone-A", "A", 0), ("alone-B", "B", 0), ("B2", 2, 2), ("A2", 1, 2)]
    cells, done = [], {}
    for label, x, pair in plan:
        if label.startswith("alone"):
            run = lambda l=label, c=x: alone(a, l, c, a.out)
        else:
            run = lambda l=label, n=x, q=pair: h.cell(a, l, n, q, a.out)
        rec, ts = run()
        if rec["void"]:  # re-run in the same order, once, and the record lists it
            cells.append(rec)
            time.sleep(60)
            rec, ts = run(label + "r")
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

    def follow(ts):
        return statistics.median(t["ttft_s"] for t in ts if t["round"] in (1, 2))

    ratios = []
    for pair in (1, 2):
        ratios.append(follow(done[f"B{pair}"]) / follow(done[f"A{pair}"]))
        print(f"pair {pair}: follow-up median first token, one slot {follow(done[f'A{pair}']):.2f} s, two slots {follow(done[f'B{pair}']):.2f} s, ratio {ratios[-1]:.3f}")
    two = done["B1"] + done["B2"]
    unreused = [t for t in two if t["round"] != 0 and not t["reused"]]
    print(f"two slots: {len(unreused)} of {sum(t['round'] != 0 for t in two)} follow-ups reused nothing, the at-once round included")
    # Each two-slot client against itself alone, every turn, up to its first difference: past that the prompts differ.
    differ, compared = 0, 0
    for pair in (1, 2):
        for name in "AB":
            for r in (0, 1, 2, "both"):
                x = next(t for t in done[f"alone-{name}"] if t["round"] == r)
                y = next(t for t in done[f"B{pair}"] if t["client"] == name and t["round"] == r)
                compared += 1
                if x["reply"] != y["reply"]:
                    differ += 1
                    print(f"pair {pair} client {name} round {r}: two slots reused {y['reused']}, alone {x['reused']}: the replies differ: alone {x['reply']!r}, two slots {y['reply']!r}")
                    break
    print(f"{differ} of 4 conversations differ from that client alone; {compared} turns compared")
    # Reported, not graded.
    cold = sum(next(t for t in done["A1"] if t["client"] == t2["client"] and t["round"] == t2["round"])["reply"] != t2["reply"] for t2 in done["B1"])
    print(f"reported: {cold} of {len(done['B1'])} two-slot turns differ from the cold one-slot arm's, whose prompts diverge after the first")
    print("reported: first token of each turn that restored its checkpoint:", [f"{t['client']} round {t['round']}: {t['ttft_s']:.2f} s" for t in two if t["reused"] and t["reused"] < t["prompt_tokens"] - h.MAX_NEW - 16])
    swap = max(c.get("peak_swap_mb", 0) for c in cells)
    print(f"peak VmSwap {swap} MB (the bar: {h.SWAP_MB})")
    worst = max(ratios)
    if swap > h.SWAP_MB or differ or unreused or worst > h.FAIL_BAR:
        print(f"VERDICT: FAIL (ratio {worst:.3f}, {differ} differ, {len(unreused)} unreused, swap {swap} MB)")
        return 1
    if worst > h.PASS_BAR:
        print(f"VERDICT: PARKED (ratio {worst:.3f} in the band {h.PASS_BAR}-{h.FAIL_BAR})")
        return 3
    print(f"VERDICT: PASS (worst ratio {worst:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
