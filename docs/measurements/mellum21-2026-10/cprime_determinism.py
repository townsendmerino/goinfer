#!/usr/bin/env python3
"""Follow-up C2 of docs/tasks/task-mellum21-2026-10.md: is Mellum2.1's expert streaming (C') deterministic, and does it track the resident build?

    python3 cprime_determinism.py <serve-binary> <model-dir> <out-dir>
    python3 cprime_determinism.py --selftest

Follow-up C (2026-10-09) found the resident arm giving ONE text per prompt across two cold sessions while C' gave a different text per prompt in each of its two cold
sessions, neither equal to the resident text. C' is documented bit-identical to resident. Sessions (ctx 2048, one cold server each, greedy, thinking off, non-streamed with
top-3 log-probabilities, 96 tokens; a discarded warm request first, then the three prompts once each, in the order shown):
  R1   resident                              history P0 P1 P2
  S1a  --moe-cache-experts (C')              history P0 P1 P2
  S1b  --moe-cache-experts                   history P0 P1 P2     (identical to S1a: only a nondeterminism can tell them apart)
  S2   --moe-cache-experts                   history P2 P1 P0     (a different cache history)
  S3   --moe-cache-experts, CUDA_LAUNCH_BLOCKING=1, history P0 P1 P2   (serialises launches: a diagnostic; if S3 differs from S1a/b the streams are involved)
Reads, per prompt, then the verdict (rule pre-registered in the task doc, "C2"):
  D1  S1a vs S1b identical on all three prompts            -> DETERMINISTIC at a fixed history; else NONDETERMINISTIC
  D2  S1a vs S2 identical on all three                     -> HISTORY-INDEPENDENT; else HISTORY-DEPENDENT
  D3  S1a vs R1: identical, or the first differing token is a near-tie (R1's probability of S1a's token >= half of R1's top token's) -> BENIGN; else NOT NEAR-TIE
  D4  S3 vs S1a identical on all three                     -> SYNC-INERT; else SYNC-SENSITIVE
VOID: a resident arm that did not load cuda-resident, or an S arm without the C' line.
"""
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

PORT = 18933
PROMPTS = [
    "Explain in detail how a hash table handles collisions, covering chaining and open addressing, with a short example of each.",
    "Write a clear, step by step description of how TCP establishes a connection and why each step of the handshake is needed.",
    "Describe how a B-tree keeps itself balanced during insertions and deletions, and why databases prefer it to a binary tree.",
]
MAXTOK = 96


def http(path, body=None, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", None if body is None else json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def ask(model, prompt, maxtok=MAXTOK):
    body = {"model": model, "stream": False, "temperature": 0, "max_tokens": maxtok, "logprobs": True, "top_logprobs": 3,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": prompt}]}
    resp = json.load(http("/v1/chat/completions", body))
    c = resp["choices"][0]
    toks = [{"token": x["token"], "logprob": x["logprob"], "top": [(t["token"], t["logprob"]) for t in x["top_logprobs"]]} for x in (c.get("logprobs") or {}).get("content") or []]
    return {"text": c["message"].get("content") or "", "toks": toks, "reused": resp.get("usage", {}).get("prefill_reused_tokens", 0)}


def ask_stream(model, prompt, maxtok):
    """Follow-up C's own request shape: streamed, greedy, thinking off; the text is the concatenated content deltas."""
    body = {"model": model, "stream": True, "temperature": 0, "max_tokens": maxtok, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    text = []
    for line in http("/v1/chat/completions", body):
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        for ch in json.loads(line[5:]).get("choices", []):
            d = ch.get("delta", {})
            text.append(d.get("content") or d.get("reasoning_content") or "")
    return {"text": "".join(text), "toks": []}


def session(binary, model_dir, tag, flags, env, order, outdir, stream=False, by_request=False):
    log = open(os.path.join(outdir, f"serve-{tag}.log"), "w")
    cmd = [binary, "--model", model_dir, "--backend", "cuda", "--ctx", "2048", "--addr", f"127.0.0.1:{PORT}"] + flags
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, env={**os.environ, **env})
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
        resident, cprime = "decode path: cuda-resident" in banner, "C′ cache" in banner
        if "--moe-cache-experts" in flags and not (resident and cprime):
            raise SystemExit(f"VOID: {tag} asked for C' but the banner lacks it (resident={resident} C'={cprime})")
        if "--moe-cache-experts" not in flags and (not resident or cprime):
            raise SystemExit(f"VOID: {tag} is the resident arm but is not fully resident (resident={resident} C'={cprime})")
        model = json.load(http("/v1/models"))["data"][0]["id"]
        ask(model, "Say hello.")  # warm, discarded
        out = {}
        for k, pi in enumerate(order):
            r = ask_stream(model, PROMPTS[pi], 128) if stream else ask(model, PROMPTS[pi])
            out[(pi, k) if stream else (k if by_request else pi)] = r
            print(f"  {tag} request {k} prompt {pi}: {len(r['text'])} chars", flush=True)
        return out
    finally:
        p.terminate()
        try:
            p.wait(30)
        except Exception:
            p.kill()
        time.sleep(5)


def first_diff(a, b):
    n = min(len(a["toks"]), len(b["toks"]))
    for i in range(n):
        if a["toks"][i]["token"] != b["toks"][i]["token"]:
            return i
    return None if len(a["toks"]) == len(b["toks"]) and a["text"] == b["text"] else n


def near_tie(ref, other, i):
    """ref's probability of other's token at position i against ref's own top token's."""
    if i >= len(ref["toks"]) or i >= len(other["toks"]):
        return False, None
    top = ref["toks"][i]["top"]
    p_top = math.exp(top[0][1])
    p_o = [math.exp(lp) for t, lp in top if t == other["toks"][i]["token"]]
    return (bool(p_o) and p_o[0] >= 0.5 * p_top), (p_o[0] / p_top if p_o else 0.0)


def grade(res):
    """res: tag -> {prompt index -> {text, toks}}. Returns (lines, verdict dict)."""
    lines, v = [], {}
    def same(a, b):
        return [first_diff(res[a][i], res[b][i]) is None for i in range(3)]
    for key, (a, b), yes, no in (("D1", ("S1a", "S1b"), "DETERMINISTIC at a fixed history", "NONDETERMINISTIC"),
                                  ("D2", ("S1a", "S2"), "HISTORY-INDEPENDENT", "HISTORY-DEPENDENT"),
                                  ("D4", ("S1a", "S3"), "SYNC-INERT", "SYNC-SENSITIVE")):
        s = same(a, b)
        v[key] = yes if all(s) else no
        lines.append(f"{key} {a} vs {b}: identical per prompt {s} -> {v[key]}")
        for i in range(3):
            if not s[i]:
                d = first_diff(res[a][i], res[b][i])
                lines.append(f"    prompt {i}: first differing token {d}")
    ok = True
    for i in range(3):
        d = first_diff(res["R1"][i], res["S1a"][i])
        if d is None:
            lines.append(f"D3 prompt {i}: S1a identical to R1")
            continue
        nt, ratio = near_tie(res["R1"][i], res["S1a"][i], d)
        ok &= nt
        lines.append(f"D3 prompt {i}: first differing token {d}, R1 gives S1a's token {ratio if ratio is None else round(ratio, 3)} of its top: near-tie {nt}")
    v["D3"] = "BENIGN" if ok else "NOT NEAR-TIE"
    lines.append(f"D3 -> {v['D3']}")
    lines.append("C2 VERDICT: " + "; ".join(f"{k} {v[k]}" for k in ("D1", "D2", "D3", "D4")))
    return lines, v


def selftest():
    def toks(ts):
        return [{"token": t, "logprob": math.log(0.6), "top": [(t, math.log(0.6)), ("zz", math.log(0.3)), ("yy", math.log(0.1))]} for t in ts]
    def mk(ts, text=None):
        return {"text": text or "".join(ts), "toks": toks(ts)}
    base = [mk(list("abcdef")), mk(list("ghijkl")), mk(list("mnopqr"))]
    res = {"R1": base, "S1a": base, "S1b": base, "S2": base, "S3": base}
    _, v = grade(res)
    assert v == {"D1": "DETERMINISTIC at a fixed history", "D2": "HISTORY-INDEPENDENT", "D4": "SYNC-INERT", "D3": "BENIGN"}, v
    # a planted nondeterminism: S1b differs from S1a at one token of one prompt
    bad = [mk(list("abcdef")), mk(list("ghiXkl")), mk(list("mnopqr"))]
    _, v = grade({**res, "S1b": bad})
    assert v["D1"] == "NONDETERMINISTIC", v
    # a history dependence only
    _, v = grade({**res, "S2": bad})
    assert v["D1"] == "DETERMINISTIC at a fixed history" and v["D2"] == "HISTORY-DEPENDENT", v
    # S1a differs from R1 where R1 gave the other token 0.3 of 0.6 = 0.5 (a near-tie) vs 0.1/0.6 (not)
    near = [mk(list("abcdef")), mk(list("ghizkl")), mk(list("mnopqr"))]   # 'z' is not R1's 'zz' token: ratio 0 -> not near-tie
    _, v = grade({**res, "S1a": near, "S1b": near, "S2": near, "S3": near})
    assert v["D3"] == "NOT NEAR-TIE", v
    r1 = [mk(list("abcdef")), {"text": "ghizkl", "toks": [dict(t) for t in toks(list("ghijkl"))]}, mk(list("mnopqr"))]
    r1[1]["toks"][3]["top"] = [("j", math.log(0.6)), ("z", math.log(0.3)), ("yy", math.log(0.1))]
    _, v = grade({**res, "R1": r1, "S1a": near, "S1b": near, "S2": near, "S3": near})
    assert v["D3"] == "BENIGN", v
    print("selftest ok")


def grade_c4(res):
    """C4: which in-session state makes C' differ from itself? Sessions R (resident) and S (C'), requests A B A A (A = prompt 0, B = prompt 1), non-streamed.
    res: tag -> {k -> {text, reused}} for k = 0..3. Returns (lines, verdict)."""
    lines = []
    def t(tag, k):
        return res[tag][k]["text"]
    ru = [res["S"][k]["reused"] for k in range(4)]
    lines.append(f"C4 S prefill_reused_tokens per request (A B A A): {ru}")
    if not (ru[0] == 0 and ru[2] == 0 and ru[3] > 0):
        return lines + ["C4 VERDICT: VOID: the reuse pattern is not the one the design needs (A#1 and A#3 cold, A#4 reused)"], "VOID"
    r_all = t("R", 0) == t("R", 2) == t("R", 3)
    a13, a34 = t("S", 0) == t("S", 2), t("S", 2) == t("S", 3)
    lines.append(f"C4 R resident: A#1 == A#3 == A#4: {r_all}")
    lines.append(f"C4 S: A#1 == A#3 (both cold, different expert-cache state): {a13}; A#3 == A#4 (cold against reused): {a34}")
    if a13 and not a34:
        v = "REUSE-DEPENDENT: the repeated prompt (prefix reuse) differs from the cold one, whatever the expert cache holds"
    elif not a13:
        v = "CACHE-STATE-DEPENDENT: two cold requests of one prompt differ"
    else:
        v = "NOT REPRODUCED: A#1, A#3 and A#4 are all equal in C' here"
    lines.append(f"C4 VERDICT: {v}" + ("" if r_all else " [resident control NOT constant]"))
    return lines, v


def selftest_c4():
    mk = lambda text, reused=0: {"text": text, "reused": reused}
    same = {0: mk("a"), 1: mk("b"), 2: mk("a"), 3: mk("a", 100)}
    assert grade_c4({"R": same, "S": same})[1].startswith("NOT REPRODUCED")
    reuse = {0: mk("a"), 1: mk("b"), 2: mk("a"), 3: mk("a2", 100)}
    assert grade_c4({"R": same, "S": reuse})[1].startswith("REUSE-DEPENDENT")
    cache = {0: mk("a"), 1: mk("b"), 2: mk("a3"), 3: mk("a3", 100)}
    assert grade_c4({"R": same, "S": cache})[1].startswith("CACHE-STATE-DEPENDENT")
    assert grade_c4({"R": same, "S": {0: mk("a"), 1: mk("b"), 2: mk("a", 5), 3: mk("a", 100)}})[1] == "VOID"
    print("selftest c4 ok")


def grade_c3(res):
    """Follow-up C's own history: P0 x3, P1 x3, P2 x3, streamed, text only. res: tag -> {(prompt, k) -> {text}}."""
    lines = []
    def same(a, b):
        keys = sorted(res[a])
        return [res[a][k]["text"] == res[b][k]["text"] for k in keys]
    d1, dr = same("S1a", "S1b"), same("R1", "S1a")
    rep = {tag: [len({res[tag][k]["text"] for k in res[tag] if k[0] == pi}) for pi in range(3)] for tag in res}
    lines.append(f"C3 S1a vs S1b identical per request {d1}")
    lines.append(f"C3 R1 vs S1a identical per request {dr}")
    lines.append(f"C3 distinct texts per prompt within a session (3 reps each): {rep}")
    v1 = "DETERMINISTIC at C's history" if all(d1) else "NONDETERMINISTIC at C's history"
    vr = "MATCHES RESIDENT" if all(dr) else "DIFFERS FROM RESIDENT"
    lines.append(f"C3 VERDICT: {v1}; {vr}")
    return lines


def selftest_c3():
    mk = lambda t: {"text": t, "toks": []}
    keys = [(p, k) for p in range(3) for k in range(3)]
    base = {k: mk(f"t{k[0]}") for k in keys}
    assert grade_c3({"R1": base, "S1a": base, "S1b": base})[-1] == "C3 VERDICT: DETERMINISTIC at C's history; MATCHES RESIDENT"
    bad = {**base, (1, 2): mk("other")}
    assert "NONDETERMINISTIC" in grade_c3({"R1": base, "S1a": base, "S1b": bad})[-1]
    assert "DIFFERS FROM RESIDENT" in grade_c3({"R1": base, "S1a": bad, "S1b": bad})[-1]
    print("selftest c3 ok")


def main():
    if "--selftest" in sys.argv:
        selftest()
        selftest_c3()
        return selftest_c4()
    if "--c4" in sys.argv:
        binary, model_dir, outdir = [a for a in sys.argv[1:] if not a.startswith("--")][:3]
        os.makedirs(outdir, exist_ok=True)
        res = {}
        for tag, flags in (("R", []), ("S", ["--moe-cache-experts"])):
            res[tag] = session(binary, model_dir, tag, flags, {}, [0, 1, 0, 0], outdir, by_request=True)
        json.dump({t: {str(k): v for k, v in r.items()} for t, r in res.items()}, open(os.path.join(outdir, "sessions.json"), "w"))
        lines, _ = grade_c4(res)
        print("\n".join(lines))
        open(os.path.join(outdir, "result.txt"), "w").write("\n".join(lines) + "\n")
        return
    if "--c-history" in sys.argv:
        binary, model_dir, outdir = [a for a in sys.argv[1:] if not a.startswith("--")][:3]
        os.makedirs(outdir, exist_ok=True)
        order = [p for p in range(3) for _ in range(3)]
        C = ["--moe-cache-experts"]
        res = {}
        for tag, flags in (("R1", []), ("S1a", C), ("S1b", C)):
            res[tag] = session(binary, model_dir, tag, flags, {}, order, outdir, stream=True)
        json.dump({t: {f"{k[0]}.{k[1]}": r for k, r in v.items()} for t, v in res.items()}, open(os.path.join(outdir, "sessions.json"), "w"))
        lines = grade_c3(res)
        print("\n".join(lines))
        open(os.path.join(outdir, "result.txt"), "w").write("\n".join(lines) + "\n")
        return
    binary, model_dir, outdir = sys.argv[1:4]
    os.makedirs(outdir, exist_ok=True)
    try:
        http("/v1/models", timeout=2)
        raise SystemExit(f"FATAL: something already serves on {PORT}")
    except SystemExit:
        raise
    except Exception:
        pass
    C = ["--moe-cache-experts"]
    plan = [("R1", [], {}, [0, 1, 2]), ("S1a", C, {}, [0, 1, 2]), ("S1b", C, {}, [0, 1, 2]), ("S2", C, {}, [2, 1, 0]), ("S3", C, {"CUDA_LAUNCH_BLOCKING": "1"}, [0, 1, 2])]
    res = {}
    for tag, flags, env, order in plan:
        res[tag] = session(binary, model_dir, tag, flags, env, order, outdir)
    json.dump({t: {str(i): r for i, r in v.items()} for t, v in res.items()}, open(os.path.join(outdir, "sessions.json"), "w"))
    lines, _ = grade(res)
    print("\n".join(lines))
    open(os.path.join(outdir, "result.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
