"""Split-KV decode attention: goinfer ON vs goinfer OFF, paired per (geometry, depth).

This is the §B6 harness. §B6 exists in docs/benchmarks.md but its harness never did — the original
48 cells were driven ad hoc and their raw output lived in a gitignored scratchpad, so the table
could be read but not reproduced. That is the gap this file closes; the numbers it prints are meant
to REPLACE §B6's after the 2026-08-25 driver/distro re-anchor.

Not bench_peer.py: there is no peer here, both arms are goinfer. Not bench_compare.sh either, which
runs in-process Go benchmarks and by its own design note drives no server — and the whole point of
§B6 was that an in-process loop FLATTERS split-KV. The original TestSplitKVCrossover timed a tight
ForwardArgmax loop and took best-of-3 minimum, which hid the per-token CPU dispatch a real request
exposes and favoured the higher-variance arm (ON's spread is 3.6–6.4 tok/s against OFF's 0.1–0.6).
It reported "break-even at 256" for a geometry whose real crossover is in (512, 1024]. So: a real
served request, client-timed from the first streamed token, decode only.

PAIRED, and adjacent in time. Each (geometry, depth) runs its ON and OFF arms back to back and the
ratio is formed within the pair. Pooling arms across a whole sweep would carry between-cell drift
into a ~10% effect; this repo has measured that difference and made it rule 7.

    python3 scripts/bench_splitkv.py --out docs/measurements/b6-splitkv-<sha>.json
"""
import argparse, json, os, signal, statistics, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_peer as bp  # noqa: E402  — primitives only; nothing here drives a peer

# The four §B6 geometries. Keys match prompts.json, and the comment is the reason each is here:
# together they span the occupancy deficit that decides whether split-KV can ever win.
GEOMETRIES = {
    "0.5B":      (os.path.expanduser("~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"),
                  "nH=14 nKV=2 hd=64 L=24 — crossover ~2560"),
    "1.5B":      (os.path.expanduser("~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
                  "nH=12 nKV=2 hd=128 L=28 — crossover in (512,1024]"),
    "gemma3-1b": (os.path.expanduser("~/models/gemma3-1b-q4_k_m.gguf"),
                  "nH=4 nKV=1 hd=256 L=26, sliding window 512 — the nWin-vs-nKeys case"),
    "phi3-mini": (os.path.expanduser("~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf"),
                  "nH=32 nKV=32 hd=96 L=32 MHA — never crosses over"),
}
DEPTHS = [128, 256, 512, 1024, 2048, 3900]

# TWO DIFFERENT QUESTIONS, and conflating them silently produces a table of 1.000s.
#
# GOINFER_SPLITKV_ATTN=1 does NOT force the split path. It ENABLES the gate, which then decides per
# geometry (cuda/backend.go: `r.splitkvAttn = os.Getenv("GOINFER_SPLITKV_ATTN") != "0"`). Since the
# gate now ships a measured per-geometry table, "ATTN=1 vs ATTN=0" compares the shipped default
# against off — and where the gate correctly declines to split, BOTH arms run the same kernel and
# the ratio is 1.000 by construction. That is a real result, but it is not §B6's.
#
# §B6 characterised split-KV ITSELF at every depth, against a gate that no longer exists. Its
# force-on arm is GOINFER_SPLITKV_MIN_KEYS=0 — "always take the split path" (cuda/resident.go).
MODES = {
    # Reproduces §B6's question: does the split path help, at this geometry and depth?
    "force": {"on": {"GOINFER_SPLITKV_MIN_KEYS": "0"},
              "off": {"GOINFER_SPLITKV_ATTN": "0"}},
    # Validates what actually ships: does the gate's decision cost anything against off?
    # Ratios near 1.000 here are the PASS condition, not a null result.
    "gate":  {"on": {},
              "off": {"GOINFER_SPLITKV_ATTN": "0"}},
}


def run_arm(path, prompt, arm_env, backend, quant):
    """One cell: a FRESHLY started serve, so the arm is a process-level setting.

    Never a mid-process toggle — the shipped gate is read per layer at request time, and flipping it
    inside a live process would measure a state no deployment is ever in."""
    env = dict(os.environ)
    # Clear both knobs first: inheriting one from the caller would silently redefine the arm.
    env.pop("GOINFER_SPLITKV_ATTN", None)
    env.pop("GOINFER_SPLITKV_MIN_KEYS", None)
    env.update(arm_env)
    proc = subprocess.Popen(
        [bp.SERVE[backend], "-model", f"bench={path}", "-backend", backend,
         "-addr", f"127.0.0.1:{bp.GPORT}", "-quant", quant],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid, env=env)
    url = f"http://127.0.0.1:{bp.GPORT}/v1/chat/completions"
    try:
        if not bp.wait_port(bp.GPORT):
            return None, "server did not come up"
        mk = lambda: bp.goinfer_payload(path, prompt, bp.CONFIGS["greedy"])  # noqa: E731
        try:
            bp.post_stream(url, mk(), bp.parse_openai)  # warm: discarded
        except Exception as e:
            return None, f"warmup failed: {e}"
        blocks = []
        for _ in range(bp.NRUNS):
            rates = []
            for _ in range(bp.NCOMP):
                n, tf, tl = bp.post_stream(url, mk(), bp.parse_openai)
                if n >= 2 and tf and tl and tl > tf:
                    rates.append(n / (tl - tf))  # decode-only: from the FIRST streamed token
            if rates:
                blocks.append(statistics.mean(rates))
        return blocks, None
    except Exception as e:
        return None, str(e)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        time.sleep(3)  # let VRAM settle before the next load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--quant", default="int4")
    ap.add_argument("--geometry", action="append", help="repeatable; default all four")
    ap.add_argument("--depth", action="append", type=int, help="repeatable; default all six")
    ap.add_argument("--mode", choices=sorted(MODES), default="force",
                    help="force = split-KV itself (reproduces §B6); gate = the shipped default")
    args = ap.parse_args()

    geoms = {k: v for k, v in GEOMETRIES.items() if not args.geometry or k in args.geometry}
    depths = args.depth or DEPTHS
    missing = [p for p, _ in geoms.values() if not os.path.exists(p)]
    if missing:
        sys.exit("missing model(s):\n  " + "\n  ".join(missing))
    bp.check_bench_disk([p for p, _ in geoms.values()])
    bp.preflight()

    cells, total = {}, len(geoms) * len(depths) * 2
    done, t0 = 0, time.time()
    print(f"# {total} cells: {len(geoms)} geometries x {len(depths)} depths x 2 arms", flush=True)
    for gi, (gkey, (path, note)) in enumerate(geoms.items()):
        for depth in depths:
            key = f"{gkey}:{depth}"
            if key not in bp._PROMPTS:
                print(f"  SKIP {key}: no calibrated prompt — run bench_prompts_calibrate.py", flush=True)
                continue
            prompt = bp.prompt_for_depth(depth, gkey)
            # Alternate which arm goes first, so a systematic first-cell effect cannot land on the
            # same arm every time and masquerade as the split-KV difference.
            arm_env = MODES[args.mode]
            order = ("on", "off") if (gi + depth) % 2 == 0 else ("off", "on")
            arms = tuple((name, arm_env[name]) for name in order)
            pair = {}
            for arm_name, arm_cfg in arms:
                blocks, err = run_arm(path, prompt, arm_cfg, args.backend, args.quant)
                done += 1
                el = time.time() - t0
                eta = (el / done) * (total - done)
                if err:
                    print(f"  [{done}/{total}] {key} {arm_name}: ERROR {err} "
                          f"(elapsed {el/60:.1f}m)", flush=True)
                    pair[arm_name] = {"error": err}
                    continue
                mean = statistics.mean(blocks)
                spread = max(blocks) - min(blocks) if len(blocks) > 1 else 0.0
                pair[arm_name] = {"blocks": blocks, "mean": mean, "spread": spread}
                print(f"  [{done}/{total}] {key} {arm_name}: {mean:.2f} tok/s "
                      f"(spread {spread:.2f}) · elapsed {el/60:.1f}m eta {eta/60:.1f}m", flush=True)
            if "mean" in pair.get("on", {}) and "mean" in pair.get("off", {}):
                pair["ratio_on_over_off"] = pair["on"]["mean"] / pair["off"]["mean"]
                print(f"      -> ratio {pair['ratio_on_over_off']:.3f}", flush=True)
            pair["prompt_tokens"] = bp.prompt_tokens(depth, gkey)
            pair["geometry_note"] = note
            pair["arm_order"] = [a for a, _ in arms]
            cells[key] = pair
            # Written after every pair: a run this long must survive being interrupted.
            json.dump({"provenance": bp.provenance(), "machine": bp.machine_state(),
                       "config": {"mode": args.mode, "arms": MODES[args.mode],
                                  "backend": args.backend, "quant": args.quant,
                                  "ngen": bp.NGEN, "ncomp": bp.NCOMP, "nruns": bp.NRUNS},
                       "cells": cells}, open(args.out, "w"), indent=1, sort_keys=True)

    label = ("split-KV forced on / off" if args.mode == "force"
             else "shipped gate / off — ~1.000 is the PASS, not a null result")
    print(f"\n# ratios ({label})  [{(time.time()-t0)/60:.1f} min]")
    hdr = "| geometry | " + " | ".join(str(d) for d in depths) + " |"
    print(hdr + "\n|" + "---|" * (len(depths) + 1))
    for gkey in geoms:
        row = [f"| {gkey} "]
        for d in depths:
            c = cells.get(f"{gkey}:{d}", {})
            r = c.get("ratio_on_over_off")
            row.append(f"| {r:.3f} " if r else "| — ")
        print("".join(row) + "|")
    print(f"\nraw cells: {args.out}")


if __name__ == "__main__":
    main()
