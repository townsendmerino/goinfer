#!/usr/bin/env python3
"""G-W2 and G-W3 of docs/tasks/task-cuda-windowed-kv-2026-10.md: Mellum2.1 on the 8 GB card, serve with and without --windowed-kv.

    python3 windowed_kv_gate.py <serve-binary> <model-dir> <out-dir> [--smoke]

Arms at ctx 2048 (the largest context the full-KV plan loads fully resident on this card):
  F  serve --model <dir> --backend cuda --ctx 2048
  W  serve --model <dir> --backend cuda --ctx 2048 --windowed-kv        (windowed layers hold 1024+512 = 1536 positions)
Sessions run F W W F (ABBA), one cold server each, one discarded warm request, then per session:
  short  3 prompts x 3 reps, 128 tokens           (decode never leaves the window: the view with base 0, so this is the view's pure overhead)
  long   1 prompt of >= 1500 tokens x 3 reps, 384 tokens   (decode runs from ~1500 to ~1900, past the 1536 a windowed layer holds: compaction runs)
Greedy, thinking off, streamed; rate = (n-1)/(t_last-t_first) over content deltas, the rule `serve check` uses.
Two last sessions are the load claim, at ctx 4096 (follow-up C probed 2026-10-09: full KV declines at 3072 and 4096 on this card; the windowed price at 4096 is
21 x 1536 + 7 x 4096 = 60,928 layer-positions x ~4 KB = ~244 MB against ~276 MB usable beside the weights and the 384 MB reserve): F4096 (full KV) must
decline to the CPU path, and W4096 (--windowed-kv) must be fully resident and decode the short prompts. ctx 16384 was in the first registration and was
withdrawn before any graded run: the seven full-attention layers alone price 7 x 16384 x ~4 KB = ~0.46 GB against ~0.28 GB usable, so no windowed plan
holds it on this card (the smoke of 2026-10-10 declined it, serve-s5W16.log); the ceiling is about ctx 5400.

G-W2 (correctness), pre-registered: every text of one prompt is identical across both arms and all four sessions (greedy, resident = deterministic). Any
difference FAILS. The W4096 load must be cuda-resident without the C' line and decode non-empty text, and the F4096 control must NOT load resident (W4096's text is compared to W's
and reported, not graded: a different context cap is a different launch geometry).
G-W3 (no speed cost), pre-registered: per class (short, long) the ratio W/F of the medians over the ABBA sessions; PASS in [0.98, 1.02]; FAIL below 0.97;
0.97-0.98 AMBIGUOUS -> parked; above 1.02 is reported as faster and passes. The job's verdict is the worse of the two classes. A resident arm that did not
load cuda-resident, a W arm without the windowed line, or a long prompt under 1500 tokens VOIDS the job.
"""
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

PORT = 18932
SHORT = [
    "Explain in detail how a hash table handles collisions, covering chaining and open addressing, with a short example of each.",
    "Write a clear, step by step description of how TCP establishes a connection and why each step of the handshake is needed.",
    "Describe how a B-tree keeps itself balanced during insertions and deletions, and why databases prefer it to a binary tree.",
]
PARA = ("The harbour town kept its ledgers in a stone room behind the customs house, and every spring the clerks counted the barrels again, "
        "because the tide tables and the tallies never quite agreed and nobody wanted to be the one to explain the difference to the magistrate. ")


def long_prompt(k):
    return (f"Read the following notes and then answer the question at the end.\n\n" + "".join(f"[{i}] " + PARA for i in range(k)) +
            "\nQuestion: in two or three paragraphs, describe what the clerks did each spring and why, and invent a plausible reason the tallies disagreed.")


def http(path, body=None, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", None if body is None else json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def vram():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode().split()[0])


def prompt_tokens(model, prompt):
    body = {"model": model, "stream": False, "temperature": 0, "max_tokens": 1, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    return json.load(http("/v1/chat/completions", body))["usage"]["prompt_tokens"]


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


def session(binary, model_dir, arm, outdir, tag, ctx, windowed, workloads, require_resident=True):
    flags = ["--model", model_dir, "--backend", "cuda", "--ctx", str(ctx), "--addr", f"127.0.0.1:{PORT}"] + (["--windowed-kv"] if windowed else [])
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
        win = "windowed KV: sliding-window layers hold" in banner
        if not require_resident:
            return [], {"tag": tag, "arm": arm, "resident": resident and not cprime, "vram_mib_idle": vram(), "vram_mib_after": vram(), "long_prompt_tokens": None}
        if not resident or cprime:
            raise SystemExit(f"VOID: {tag} is not fully resident (resident={resident} C'={cprime}); see serve-{tag}.log")
        if windowed and not win:
            raise SystemExit(f"VOID: {tag} asked for --windowed-kv but the build did not engage it; see serve-{tag}.log")
        if not windowed and win:
            raise SystemExit(f"VOID: {tag} is a control arm but windowed KV engaged")
        model = json.load(http("/v1/models"))["data"][0]["id"]
        one(model, "Say hello.", 16)  # warm, discarded
        v_idle = vram()
        rows, ptoks = [], {}
        for cls, prompts, reps, maxtok in workloads:
            for pi, pr in enumerate(prompts):
                if cls == "long":
                    ptoks[pi] = prompt_tokens(model, pr)
                    if ptoks[pi] < 1500:
                        raise SystemExit(f"VOID: the long prompt is {ptoks[pi]} tokens, under the 1500 that puts decode past the {1024 + 512} a windowed layer holds")
                for r in range(reps):
                    rate, n, text = one(model, pr, maxtok)
                    rows.append({"arm": arm, "tag": tag, "cls": cls, "prompt": pi, "rep": r, "rate": rate, "n": n, "text": text})
                    print(f"  {tag} {arm} {cls} prompt {pi} rep {r}: {n} tokens, {rate:.2f} tok/s", flush=True)
        return rows, {"tag": tag, "arm": arm, "vram_mib_idle": v_idle, "vram_mib_after": vram(), "long_prompt_tokens": ptoks.get(0)}
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
    lp = long_prompt(27)
    work = [("short", SHORT[:1], 2, 32), ("long", [lp], 1, 64)] if smoke else [("short", SHORT, 3, 128), ("long", [lp], 3, 384)]
    order = [("F", "s1F"), ("W", "s2W")] if smoke else [("F", "s1F"), ("W", "s2W"), ("W", "s3W"), ("F", "s4F")]
    rows, meta = [], []
    for arm, tag in order:
        r, m = session(binary, model_dir, arm, outdir, tag, 2048, arm == "W", work)
        rows += r
        meta.append(m)
    # the load claim at ctx 4096: the full-KV plan declines (control), the windowed one is fully resident and decodes (speed not graded)
    _, metaC = session(binary, model_dir, "F4096", outdir, "s5F4096", 4096, False, [], require_resident=False)
    r, m = session(binary, model_dir, "W4096", outdir, "s6W4096", 4096, True, [("short", SHORT[:1] if smoke else SHORT, 1, 64 if smoke else 128)])
    rows16, meta16 = r, m
    json.dump({"rows": rows, "rows4096": rows16, "sessions": meta + [metaC, meta16]}, open(os.path.join(outdir, "rows.json"), "w"), indent=1)
    out, ratios, identical = [], {}, True
    for cls in ("short", "long"):
        prompts = sorted({x["prompt"] for x in rows if x["cls"] == cls})
        for pi in prompts:
            texts = {x["text"] for x in rows if x["cls"] == cls and x["prompt"] == pi}
            same = len(texts) == 1
            identical &= same
            F = [x["rate"] for x in rows if x["arm"] == "F" and x["cls"] == cls and x["prompt"] == pi]
            W = [x["rate"] for x in rows if x["arm"] == "W" and x["cls"] == cls and x["prompt"] == pi]
            out.append(f"{cls} prompt {pi}: full median {statistics.median(F):.2f} tok/s (min {min(F):.2f} max {max(F):.2f}, n={len(F)}); windowed median {statistics.median(W):.2f} "
                       f"(min {min(W):.2f} max {max(W):.2f}, n={len(W)}); texts identical across arms and sessions: {same} ({len(texts)} distinct)")
        F = [x["rate"] for x in rows if x["arm"] == "F" and x["cls"] == cls]
        W = [x["rate"] for x in rows if x["arm"] == "W" and x["cls"] == cls]
        ratios[cls] = statistics.median(W) / statistics.median(F)
        out.append(f"{cls}: windowed/full ratio of medians = {ratios[cls]:.3f}")
    for arm in ("F", "W"):
        ss = sorted({x["tag"] for x in rows if x["arm"] == arm})
        if len(ss) == 2:
            a, b = (statistics.median([x["rate"] for x in rows if x["tag"] == t]) for t in ss)
            out.append(f"arm {arm} session medians {a:.2f} vs {b:.2f} tok/s (drift {100 * (b / a - 1):+.1f}%)")
    out.append("VRAM MiB (idle/after): " + "; ".join(f"{m['tag']} {m['vram_mib_idle']}/{m['vram_mib_after']}" for m in meta + [metaC, meta16]))
    ok16 = all(x["n"] > 8 and x["text"].strip() for x in rows16)
    same16 = [x["text"] == next(y["text"] for y in rows if y["cls"] == "short" and y["prompt"] == x["prompt"] and y["arm"] == "W") for x in rows16 if x["cls"] == "short"]
    ctrl = not metaC["resident"]
    out.append(f"G-W2 ctx 4096 load: full-KV control declined (not resident): {ctrl}; windowed resident and decoded {len(rows16)} requests non-empty: {ok16}; text equals the ctx-2048 windowed text: {same16} (reported, not graded)")
    worst = min(ratios.values())
    g3 = "PASS" if worst >= 0.98 else ("FAIL" if worst < 0.97 else "AMBIGUOUS -> parked")
    out.append(f"G-W2 (texts identical across arms and sessions, 4096 windowed load decodes, full-KV 4096 declines): {'PASS' if identical and ok16 and ctrl else 'FAIL'}")
    out.append(f"G-W3 OVERALL worst class ratio {worst:.3f} -> {g3}" + ("  [SMOKE, not a result]" if smoke else ""))
    text = "\n".join(out)
    print(text)
    open(os.path.join(outdir, "result.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
