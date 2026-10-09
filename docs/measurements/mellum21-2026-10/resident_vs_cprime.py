#!/usr/bin/env python3
"""Follow-up C of docs/tasks/task-mellum21-2026-10.md: what does Mellum2.1's expert streaming (C') cost in decode, against the same model fully resident?

    python3 resident_vs_cprime.py <serve-binary> <model-dir> <out-dir> [--smoke]

Two arms at the SAME ctx (2048: probed 2026-10-09, fully resident loads at 2048 and 1024 but declines at 3072 and 4096 on the 8 GB card with 6.5 GB of int4mix weights):
  R  serve --model <dir> --backend cuda --ctx 2048                      (decode path: cuda-resident, every expert on the GPU)
  S  serve --model <dir> --backend cuda --ctx 2048 --moe-cache-experts  (C': 45 of 64 expert slots per layer, the rest streamed)
Order ABBA by session (R S S R), one cold server per session, one discarded warm request each, then 3 prompts x 3 reps, greedy, thinking off,
streamed; decode rate = (n-1) / (t_last - t_first) over the content deltas (the rule `serve check` uses). Greedy C' output is documented as
bit-identical to resident, so the texts are compared too (a mismatch is reported, not hidden). Pre-registered reading, per prompt ratio
R/S of the median rates, overall = the median of the three: >= 1.15 worth building (windowed KV for residency); < 1.05 park; between is ambiguous
and parks. A resident arm that did not load cuda-resident, or an S arm without the C' line, voids the job.
"""
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

PORT = 18931
PROMPTS = [
    "Explain in detail how a hash table handles collisions, covering chaining and open addressing, with a short example of each.",
    "Write a clear, step by step description of how TCP establishes a connection and why each step of the handshake is needed.",
    "Describe how a B-tree keeps itself balanced during insertions and deletions, and why databases prefer it to a binary tree.",
]
REPS = 3
MAXTOK = 128


def http(path, body=None, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", None if body is None else json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def vram():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode().split()[0])


def one(model, prompt, maxtok):
    body = {"model": model, "stream": True, "temperature": 0, "max_tokens": maxtok, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    t_first = t_last = None
    n, text = 0, []
    for line in http("/v1/chat/completions", body):
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        ev = json.loads(line[5:])
        for ch in ev.get("choices", []):
            d = ch.get("delta", {})
            c = d.get("content") or d.get("reasoning_content") or ""
            if c:
                now = time.time()
                if t_first is None:
                    t_first = now
                t_last = now
                n += 1
                text.append(c)
    rate = (n - 1) / (t_last - t_first) if n > 1 and t_last > t_first else float("nan")
    return rate, n, "".join(text)


def session(binary, model_dir, arm, outdir, tag, prompts, reps, maxtok):
    flags = ["--model", model_dir, "--backend", "cuda", "--ctx", "2048", "--addr", f"127.0.0.1:{PORT}"] + (["--moe-cache-experts"] if arm == "S" else [])
    log = open(os.path.join(outdir, f"serve-{tag}.log"), "w")
    p = subprocess.Popen([binary] + flags, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        t0 = time.time()
        while True:
            try:
                http("/v1/models", timeout=3).read()
                break
            except Exception:
                if p.poll() is not None:
                    raise SystemExit(f"VOID: {tag} server exited during load (see serve-{tag}.log)")
                if time.time() - t0 > 900:
                    raise SystemExit(f"VOID: {tag} not ready in 15 min")
                time.sleep(2)
        banner = open(os.path.join(outdir, f"serve-{tag}.log")).read()
        resident = "decode path: cuda-resident" in banner
        cprime = "C′ cache" in banner
        if arm == "R" and (not resident or cprime):
            raise SystemExit(f"VOID: arm R is not fully resident (resident={resident} C'={cprime}); see serve-{tag}.log")
        if arm == "S" and not (resident and cprime):
            raise SystemExit(f"VOID: arm S has no C' line (resident={resident} C'={cprime}); see serve-{tag}.log")
        model = json.load(http("/v1/models"))["data"][0]["id"]
        one(model, "Say hello.", 16)  # warm, discarded
        v_idle = vram()
        rows = []
        for pi, pr in enumerate(prompts):
            for r in range(reps):
                rate, n, text = one(model, pr, maxtok)
                rows.append({"arm": arm, "tag": tag, "prompt": pi, "rep": r, "rate": rate, "n": n, "text": text})
                print(f"  {tag} {arm} prompt {pi} rep {r}: {n} tokens, {rate:.2f} tok/s", flush=True)
        v_load = vram()
        return rows, {"tag": tag, "arm": arm, "vram_mib_idle": v_idle, "vram_mib_after": v_load}
    finally:
        p.terminate()
        try:
            p.wait(30)
        except Exception:
            p.kill()
        time.sleep(5)


def main():
    binary, model_dir, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    smoke = "--smoke" in sys.argv
    os.makedirs(outdir, exist_ok=True)
    try:
        http("/v1/models", timeout=2)
        raise SystemExit(f"FATAL: something already serves on {PORT}")
    except SystemExit:
        raise
    except Exception:
        pass
    order = ["R", "S"] if smoke else ["R", "S", "S", "R"]
    prompts, reps, maxtok = (PROMPTS[:1], 2, 32) if smoke else (PROMPTS, REPS, MAXTOK)
    rows, meta = [], []
    for i, arm in enumerate(order):
        r, m = session(binary, model_dir, arm, outdir, f"s{i + 1}{arm}", prompts, reps, maxtok)
        rows += r
        meta.append(m)
    json.dump({"rows": rows, "sessions": meta}, open(os.path.join(outdir, "rows.json"), "w"), indent=1)
    out = []
    ratios = []
    for pi in range(len(prompts)):
        R = [x["rate"] for x in rows if x["arm"] == "R" and x["prompt"] == pi]
        S = [x["rate"] for x in rows if x["arm"] == "S" and x["prompt"] == pi]
        ratios.append(statistics.median(R) / statistics.median(S))
        out.append(f"prompt {pi}: resident median {statistics.median(R):.2f} tok/s (min {min(R):.2f} max {max(R):.2f}, n={len(R)}); C' median {statistics.median(S):.2f} (min {min(S):.2f} max {max(S):.2f}, n={len(S)}); ratio {ratios[-1]:.3f}")
    overall = statistics.median(ratios)
    same = all(
        len({x["text"] for x in rows if x["prompt"] == pi}) == 1 for pi in range(len(prompts)))
    # session drift: the two sessions of each arm against each other (ABBA)
    for arm in ("R", "S"):
        ss = sorted({x["tag"] for x in rows if x["arm"] == arm})
        if len(ss) == 2:
            a, b = (statistics.median([x["rate"] for x in rows if x["tag"] == t]) for t in ss)
            out.append(f"arm {arm} session medians {a:.2f} vs {b:.2f} tok/s (drift {100 * (b / a - 1):+.1f}%)")
    out.append("VRAM MiB (idle/after): " + "; ".join(f"{m['tag']} {m['vram_mib_idle']}/{m['vram_mib_after']}" for m in meta))
    out.append(f"greedy texts identical across arms and sessions: {same}")
    verdict = "WORTH BUILDING" if overall >= 1.15 else ("PARK" if overall < 1.05 else "AMBIGUOUS -> parked")
    out.append(f"OVERALL resident/C' decode ratio (median of per-prompt ratios) = {overall:.3f}  -> {verdict}" + ("  [SMOKE, not a result]" if smoke else ""))
    text = "\n".join(out)
    print(text)
    open(os.path.join(outdir, "result.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
