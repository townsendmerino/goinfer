#!/usr/bin/env python3
"""bench_tool_failure.py — T0 of docs/tasks/task-tool-grammar-union-2026-09.md: how often does a
model that has 12 tools and tool_choice "auto" emit a tool call goinfer cannot parse?

METHOD. For one (model, quant) it starts one `serve`, then:

  1. HISTORY. Replays scripts/w4_transcript_base.json (12 tools, 10 agent turns) with a NAMED
     tool_choice per turn at temperature 0, exactly as bench_peer_transcript.py does. A named
     choice rides constrained decoding, so every history call is well-formed and the arguments
     are the model's own — the conversation stays on-distribution for THIS model, and the
     history is identical across every sample of the same turn.
  2. MEASURE. For each turn k, sends the history up to (not including) turn k's assistant reply
     with tool_choice "auto" — the unconstrained decode the doc is about — greedy once, and at
     temperature T for --samples seeds. Teacher-forced: each measured request sees the
     constrained history, never its own earlier output, so one bad sample cannot compound.

CLASSIFICATION (per sample, from the HTTP reply only):
  parsed        tool_calls present, every name is in the tool list
  unknown_name  tool_calls present, some name is NOT in the tool list
  unparsed      NO tool_calls, but the content carries the family's call opener — the model
                started a call and the body did not parse (the failure T1 exists to remove)
  truncated     as unparsed, but finish_reason == "length" (max_tokens ate the call)
  unwrapped     (opener families only) NO tool_calls and NO opener, but the content is bare call
                JSON — starts with "{" or "[" and carries a "name" key. The model MEANT a call and
                dropped the wrapper. NOT the doc's headline failure and NOT fixed by T1: a union
                grammar armed on the wrapper never arms here. Reported as its own column.
  prose         no tool_calls, no opener, not bare call JSON — a legal answer under auto
Attempted calls = parsed + unknown_name + unparsed + truncated. The headline rate is
(unparsed + truncated + unknown_name) / attempted, reported with a 95% Wilson interval, next to
the same failures over ALL samples so a model that mostly answers in prose cannot hide behind
its denominator. INFORMATIONAL, not in the headline: args_invalid (a parsed call whose
arguments miss a required key or have the wrong JSON type — T1's union would prevent these too),
and on_intended (parsed name == the transcript's intended tool).

BLIND SPOT, stated: the server drops the raw text when at least one call parsed, so a SECOND
call in a parallel turn that failed to parse is invisible here. This measures the first-order
failure (a turn that yields no usable call); it is a floor on the true rate.

  python3 scripts/bench_tool_failure.py out.json --model 1.5B --quant int4 --samples 30 --temp 0.7
"""
import argparse, json, math, os, platform, signal, socket, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8097
SERVE = os.environ.get("GOINFER_SERVE_CUDA", os.path.expanduser("~/bench-cur/serve-cuda"))
M = os.path.expanduser("~/models/")
# key -> (path, family opener the parser keys on). llama3 has no opener: its call is a bare JSON
# object, so an unparsed llama3 call is "content that starts with '{' and mentions a name key".
MODELS = {
    "0.5B": (M + "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", "<tool_call>"),
    "1.5B": (M + "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "<tool_call>"),
    "7B": (M + "qwen2.5-7b-instruct-q4_k_m.gguf", "<tool_call>"),
    "llama3-1B": (M + "llama-3.2-1b-instruct-q4_k_m.gguf", None),
    "MoE-35B-A3B": (M + "qwen3.6-35b-a3b-q4_k_m.gguf", "<tool_call>"),
}


def wait_port(port, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), 1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


class Server:
    def __init__(self, key, quant):
        self.path = MODELS[key][0]
        self.quant = quant

    def __enter__(self):
        argv = [SERVE, "-model", f"t0={self.path}", "-backend", "cuda", "-addr", f"127.0.0.1:{PORT}", "-quant", self.quant]
        self.proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        if not wait_port(PORT):
            raise RuntimeError("server did not come up")
        self.url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
        return self

    def __exit__(self, *a):
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            self.proc.wait(timeout=60)
        except Exception:
            self.proc.kill()
        time.sleep(3)


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


_JT = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def args_ok(schema, args_json):
    """required keys present and top-level property types right. Deliberately shallow."""
    try:
        a = json.loads(args_json) if isinstance(args_json, str) else args_json
    except Exception:
        return False
    if not isinstance(a, dict):
        return False
    for r in schema.get("required", []):
        if r not in a:
            return False
    for k, v in a.items():
        t = (schema.get("properties", {}).get(k) or {}).get("type")
        if t in _JT:
            if t in ("integer", "number") and isinstance(v, bool):
                return False
            if not isinstance(v, _JT[t]):
                return False
    return True


def classify(body, tool_names, schemas, opener, intended):
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    calls = msg.get("tool_calls") or []
    content = msg.get("content") or ""
    fin = ch.get("finish_reason")
    r = {"finish": fin, "completion_tokens": (body.get("usage") or {}).get("completion_tokens")}
    if calls:
        names = [(c.get("function") or {}).get("name") for c in calls]
        r["cls"] = "parsed" if all(n in tool_names for n in names) else "unknown_name"
        r["names"] = names
        r["on_intended"] = names[0] == intended
        r["args_invalid"] = any(
            n in schemas and not args_ok(schemas[n], (c.get("function") or {}).get("arguments"))
            for n, c in zip(names, calls))
        return r
    started = (opener in content) if opener else (content.lstrip().startswith("{") and '"name"' in content)
    if started:
        r["cls"] = "truncated" if fin == "length" else "unparsed"
        r["raw_head"] = content[:300]
    elif opener and content.lstrip()[:1] in ("{", "[") and '"name"' in content:
        r["cls"] = "unwrapped"
        r["raw_head"] = content[:300]
    else:
        r["cls"] = "prose"
        r["content_head"] = content[:120]
    return r


def run(key, quant, temps, samples, max_tokens, fixture, progress):
    opener = MODELS[key][1]
    tools = fixture["tools"]
    names = {t["function"]["name"] for t in tools}
    schemas = {t["function"]["name"]: t["function"].get("parameters") or {} for t in tools}
    out = {"model": key, "quant": quant, "turns": []}
    with Server(key, quant) as s:
        msgs = []
        total = len(fixture["turns"]) * (1 + sum(1 if t == 0 else samples for t in temps))
        done = 0
        t0 = time.time()
        for turn in fixture["turns"]:
            if turn["user"] is not None:
                msgs.append({"role": "user", "content": turn["user"]})
            base = {"model": "t0", "stream": False, "messages": msgs, "tools": tools}
            cells = {}
            for temp in temps:
                n = 1 if temp == 0 else samples
                rows = []
                for i in range(n):
                    p = {**base, "tool_choice": "auto", "temperature": temp, "max_tokens": max_tokens}
                    if temp > 0:
                        p["seed"] = 1000 * turn["n"] + i
                    rows.append(classify(post(s.url, p), names, schemas, opener, turn["tool_choice"]))
                    done += 1
                    if done % 10 == 0 or done == total:
                        el = time.time() - t0
                        progress(f"[{key}/{quant}] turn {turn['n']}/{len(fixture['turns'])} {done}/{total} samples "
                                 f"elapsed {el/60:.1f}m")
                cells[str(temp)] = rows
            out["turns"].append({"n": turn["n"], "intended": turn["tool_choice"], "cells": cells})
            # advance the history with the CONSTRAINED, named, greedy reply (identical for every sample above)
            h = post(s.url, {**base, "temperature": 0, "max_tokens": 256,
                              "tool_choice": {"type": "function", "function": {"name": turn["tool_choice"]}}})
            m = h["choices"][0]["message"]
            c = (m.get("tool_calls") or [{}])[0]
            msgs.append(m)
            msgs.append({"role": "tool", "name": (c.get("function") or {}).get("name"),
                         "tool_call_id": c.get("id"), "content": turn["tool_result"]})
    return out


def summarise(out):
    res = {}
    for temp in sorted({t for tr in out["turns"] for t in tr["cells"]}, key=float):
        rows = [r for tr in out["turns"] for r in tr["cells"].get(temp, [])]
        c = {k: sum(1 for r in rows if r["cls"] == k) for k in ("parsed", "unknown_name", "unparsed", "truncated", "unwrapped", "prose")}
        attempted = c["parsed"] + c["unknown_name"] + c["unparsed"] + c["truncated"]
        fail = c["unknown_name"] + c["unparsed"] + c["truncated"]
        lo, hi = wilson(fail, attempted)
        alo, ahi = wilson(fail, len(rows))
        # any_unusable: every intended call that did not reach the harness as a tool_call, wrapper or not
        unus = fail + c["unwrapped"]
        ulo, uhi = wilson(unus, attempted + c["unwrapped"])
        res[temp] = {"n": len(rows), **c, "attempted": attempted, "failures": fail,
                     "any_unusable": unus, "any_unusable_rate": unus / (attempted + c["unwrapped"]) if attempted + c["unwrapped"] else None,
                     "any_unusable_ci95": [ulo, uhi],
                     "fail_rate_of_attempted": (fail / attempted) if attempted else None,
                     "ci95_of_attempted": [lo, hi],
                     "fail_rate_of_all": fail / len(rows) if rows else None, "ci95_of_all": [alo, ahi],
                     "args_invalid": sum(1 for r in rows if r.get("args_invalid")),
                     "on_intended": sum(1 for r in rows if r.get("on_intended")),
                     "prose_rate": c["prose"] / len(rows) if rows else None}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--quant", default="int4")
    ap.add_argument("--samples", type=int, default=30, help="samples per turn at each nonzero temperature")
    ap.add_argument("--temp", type=float, nargs="*", default=[0.7], help="nonzero temperatures (greedy always runs)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--fixture", default=os.path.join(HERE, "w4_transcript_base.json"))
    a = ap.parse_args()
    fixture = json.load(open(a.fixture))
    sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    hdr = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": platform.node(),
           "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
           "commit": sh(f"git -C {HERE} rev-parse --short HEAD"), "dirty": bool(sh(f"git -C {HERE} status --porcelain -- ':!docs' ':!scripts'")),
           "serve": SERVE, "serve_mtime": sh(f"stat -c %y {SERVE}"), "loadavg": sh("cat /proc/loadavg"),
           "model_path": MODELS[a.model][0], "fixture": os.path.basename(a.fixture), "samples": a.samples,
           "temps": [0.0] + a.temp, "max_tokens": a.max_tokens}
    prog = lambda s: print(s, file=sys.stderr, flush=True)
    out = run(a.model, a.quant, [0.0] + a.temp, a.samples, a.max_tokens, fixture, prog)
    out["header"] = hdr
    out["summary"] = summarise(out)
    json.dump(out, open(a.out, "w"), indent=1)
    for t, s in out["summary"].items():
        ci = s["ci95_of_attempted"]
        print(f"{a.model}/{a.quant} T={t}: n={s['n']} parsed={s['parsed']} unparsed={s['unparsed']} truncated={s['truncated']} "
              f"unknown={s['unknown_name']} unwrapped={s['unwrapped']} prose={s['prose']} | fail/attempted={s['failures']}/{s['attempted']}"
              + (f" = {100*s['fail_rate_of_attempted']:.2f}% [{100*ci[0]:.2f},{100*ci[1]:.2f}]" if s['attempted'] else "")
              + f" | args_invalid={s['args_invalid']} on_intended={s['on_intended']}")


if __name__ == "__main__":
    main()
