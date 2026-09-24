"""Speculative decoding x the CUDA flash-decode lane: paired arms (R6 spec-decode story, option B).

Pre-registered in docs/measurements/attn-decode-fa-verify-PREREGISTERED.md. Arms, each a FRESH `serve` (never a mid-process toggle):
  A  exact + `-spec ngram`               (today's behaviour: speculation on the exact attention tree)
  B  lane  + `-spec ngram`               (GOINFER_CUDA_FLASH_DECODE=16; verify rows on the multi-row lane)
  AA a second A, for the A/A floor
  P0 plain exact / PL plain lane          (context only)
R = tok/s(B) / tok/s(A), formed within each PAIR, arm order alternating across pairs. Greedy, decode-only client timing from the first
streamed token, token counts from the engine's usage (speculative decoding emits bursts, so counting chunks would be wrong).

  GOINFER_SERVE_CUDA=/path/serve-cuda python3 scripts/bench_spec_lane.py --out r.json --cell 1.5B:3900 --cell 7B:7500
"""
import argparse, json, os, signal, statistics, subprocess, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_peer as bp  # noqa: E402  (primitives only)

MODELS = {
    "1.5B": os.path.expanduser("~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
    "7B": os.path.expanduser("~/models/qwen2.5-7b-instruct-q4_k_m.gguf"),
}
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "decoder", "model.go")
PORT = 18123
ARMS = {
    "A": ({}, ["-spec", "ngram"]),
    "B": ({"GOINFER_CUDA_FLASH_DECODE": "16"}, ["-spec", "ngram"]),
    "AA": ({}, ["-spec", "ngram"]),
    "P0": ({}, []),
    "PL": ({"GOINFER_CUDA_FLASH_DECODE": "16"}, []),
}


def prompt(chars):
    src = open(SRC).read()[:chars]
    return ("Here is a Go file:\n\n" + src +
            "\n\nRewrite the file above EXACTLY, character for character, with no changes and no commentary.")


def run_arm(model, chars, arm, ntok, reps):
    env = dict(os.environ)
    for k in ("GOINFER_CUDA_FLASH_DECODE", "GOINFER_CUDA_FLASH_DECODE_MIN_KEYS", "GOINFER_CUDA_FLASH_DECODE_VERIFY"):
        env.pop(k, None)
    # Default ON since 2026-09-23: the "exact" arms (A, AA, P0) must say so explicitly, or they would run the lane too.
    env["GOINFER_CUDA_FLASH_DECODE"] = "0"
    aenv, extra = ARMS[arm]
    env.update(aenv)
    ctx = 8192
    proc = subprocess.Popen([bp.SERVE["cuda"], "-model", "bench=" + MODELS[model], "-backend", "cuda", "-quant", "int4",
                             "-addr", f"127.0.0.1:{PORT}", "-ctx", str(ctx)] + extra,
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=1)
                break
            except Exception:
                if proc.poll() is not None:
                    return None, "server exited at startup"
                time.sleep(1)

        def one():
            body = {"model": "bench", "stream": True, "max_tokens": ntok, "temperature": 0,
                    "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": prompt(chars)}]}
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            t_first = t_last = None
            comp = ptok = 0
            text = ""
            for line in urllib.request.urlopen(req, timeout=1800):
                line = line.decode().strip()
                if not line.startswith("data:") or "[DONE]" in line:
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    comp, ptok = d["usage"].get("completion_tokens", 0), d["usage"].get("prompt_tokens", 0)
                if d.get("choices") and d["choices"][0]["delta"].get("content"):
                    now = time.time()
                    t_first = t_first or now
                    t_last = now
                    text += d["choices"][0]["delta"]["content"]
            return ((comp - 1) / (t_last - t_first) if t_first and t_last > t_first and comp > 1 else 0.0), ptok, text

        one()  # warm, discarded
        rates, ptok, txt = [], 0, ""
        for _ in range(reps):
            r, ptok, txt = one()
            rates.append(r)
        return {"rates": rates, "mean": statistics.mean(rates), "prompt_tokens": ptok, "text_head": txt[:80]}, None
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            pass
        time.sleep(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", action="append", required=True, help="MODEL:CHARS, e.g. 1.5B:13000 (chars of decoder/model.go in the prompt)")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--ntok", type=int, default=192)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--context", action="store_true", help="also run the plain exact / plain lane context arms once per cell")
    args = ap.parse_args()
    bp.preflight()
    res = {"provenance": bp.provenance(), "cells": {}}
    for cell in args.cell:
        model, chars = cell.split(":")
        chars = int(chars)
        key = f"{model}:{chars}"
        c = {"pairs": [], "aa": [], "context": {}}
        for p in range(args.pairs):
            order = ("A", "B") if p % 2 == 0 else ("B", "A")
            pair = {}
            for arm in order:
                out, err = run_arm(model, chars, arm, args.ntok, args.reps)
                pair[arm] = out if not err else {"error": err}
                print(f"  {key} pair {p} {arm}: {out['mean']:.1f} tok/s (prompt {out['prompt_tokens']} tok) {out['rates']}" if out else f"  {key} {arm} ERROR {err}", flush=True)
            if "mean" in pair.get("A", {}) and "mean" in pair.get("B", {}):
                pair["ratio_B_over_A"] = pair["B"]["mean"] / pair["A"]["mean"]
                print(f"      -> B/A {pair['ratio_B_over_A']:.3f}", flush=True)
            c["pairs"].append(pair)
        # A/A floor: two more exact+spec arms, adjacent, order irrelevant
        aa = []
        for _ in range(2):
            out, err = run_arm(model, chars, "A", args.ntok, args.reps)
            aa.append(out["mean"] if out else None)
        if all(aa):
            c["aa_ratio"] = aa[1] / aa[0]
            print(f"  {key} A/A floor: {aa[1]/aa[0]:.3f}", flush=True)
        if args.context:
            for arm in ("P0", "PL"):
                out, err = run_arm(model, chars, arm, args.ntok, args.reps)
                c["context"][arm] = out
                print(f"  {key} context {arm}: {out['mean']:.1f} tok/s" if out else f"  {key} {arm} ERROR {err}", flush=True)
        rs = [p["ratio_B_over_A"] for p in c["pairs"] if "ratio_B_over_A" in p]
        c["median_ratio_B_over_A"] = statistics.median(rs) if rs else None
        res["cells"][key] = c
        json.dump(res, open(args.out, "w"), indent=1, sort_keys=True)
    print("\n# B/A (lane+spec / exact+spec), median of pairs")
    for k, c in res["cells"].items():
        print(f"  {k}: {c['median_ratio_B_over_A']}  (A/A floor {c.get('aa_ratio')})")


if __name__ == "__main__":
    main()
