#!/usr/bin/env python3
"""bench_peer_prefill.py — the goinfer-vs-peer PREFILL harness.

WHY THIS EXISTS AS A SEPARATE SCRIPT. bench_peer.py is decode-only *by
construction*: it times the inter-token rate from the FIRST streamed token
onward, so prefill is excluded on both sides and neither engine's self-reported
accounting is trusted. That is a deliberate property of the decode anchor
(§B8) and is not being changed here — this script is additive, so a bug in the
prefill method cannot disturb the decode rows.

WHAT IT MEASURES

    prefill tok/s = prompt_tokens / TTFT

where TTFT is client-timed from just before the request is written to the first
streamed content event. TTFT includes prefill plus one sampling step plus HTTP
and scheduling overhead. That overhead is common-mode — both engines are driven
identically over their own HTTP server — so the RATIO is the defensible
quantity and the absolute rate is an upper-bound-flavoured figure. Stated here
rather than left for a reader to infer.

THE TRAP THIS EXISTS TO AVOID, MEASURED BEFORE THE HARNESS WAS WRITTEN.
bench_peer.py sends the SAME prompt for every completion in a cell. Harmless for
decode. Fatal for prefill, because **Ollama caches the prompt and goinfer does
not**. Measured 2026-09-01, 1.5B q4_k_m, 2060-token prompt, RTX 2070 SUPER:

    goinfer  repeat 2142 ms   new 2151 ms   ratio 1.00  -> no caching
    ollama   repeat  337 ms   new  640 ms   ratio 0.53  -> CACHES

Reusing one prompt would therefore have compared goinfer's real prefill against
Ollama's cache lookup: ~6.3x instead of the ~3.4x that cell actually shows —
overstating the gap by nearly 2x, in the direction that flatters nobody and
misdirects the roadmap.

So EVERY REQUEST GETS A UNIQUE PREFIX. `Session NNNN. ` is a fixed token length
that varies in content, so it defeats prefix-keyed reuse while keeping the token
count identical across the requests within a cell.

AND THE DEFEAT IS ASSERTED, NOT ASSUMED -- BY TWO CHECKS, BECAUSE THE OBVIOUS
ONE IS AMBIGUOUS.

  1. repeat-vs-fresh, per engine, recorded per cell. Read it correctly: a LOW
     ratio is HEALTHY. It means the engine caches AND the fresh prompts are
     missing that cache, which is exactly what the unique prefix is for. Ollama
     reads ~0.56 and goinfer ~1.00 (it caches nothing), and BOTH are fine. The
     first version of this script refused on a low ratio -- it fired on the
     healthy case, the "guard that inverts under the condition it exists for"
     failure this repo has already recorded once.

     A ratio near 1.0 is the genuinely ambiguous reading: it means either the
     engine does not cache (fine) or the prefix failed to defeat a cache that
     is not prefix-keyed (fatal). The ratio ALONE cannot separate those, which
     is why there is a second check.

  2. DEPTH SCALING, which is absolute rather than relative. A real prefill grows
     with prompt length; a cache lookup does not. Sweeping >= 2 depths and
     asserting TTFT rises with depth catches a hit masquerading as prefill even
     when check 1 reads 1.0. --require-scaling makes it a refusal.

  GOINFER_SERVE_CUDA=~/bench-cur/serve-cuda OLLAMA_BIN=~/ollama-0325/bin/ollama \
    python3 scripts/bench_peer_prefill.py out.json --models 0.5B,1.5B --depths 512,1024,2048
"""
import argparse, json, os, platform, signal, socket, statistics, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
GPORT, OPORT = 8098, 11498
SERVE_CUDA = os.environ.get("GOINFER_SERVE_CUDA", os.path.expanduser("~/bench-cur/serve-cuda"))
OLLAMA = os.environ.get("OLLAMA_BIN", os.path.expanduser("~/ollama-0325/bin/ollama"))
OLLAMA_MODELS = os.environ.get("OLLAMA_MODELS", os.path.expanduser("~/ollama-0325/models"))
MODELS = {
    "0.5B": (os.path.expanduser("~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"), "q05"),
    "1.5B": (os.path.expanduser("~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"), "q15"),
    "7B":   (os.path.expanduser("~/models/qwen2.5-7b-instruct-q4_k_m.gguf"), "q7b"),
}
PROMPTS = json.load(open(os.path.join(HERE, "prompts.json")))


def uniq(i):
    """Fixed-token-length, content-varying prefix. Constant length matters: it
    keeps prompt_tokens identical across a cell's requests, so the rate's
    numerator does not drift with the counter."""
    return f"Session {i:04d}. "


def wait_port(port, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), 1):
                return True
        except OSError:
            time.sleep(0.4)
    return False


def parse_openai(line):
    if not line.startswith("data: "):
        return None, None
    body = line[6:]
    if body == "[DONE]":
        return None, None
    try:
        d = json.loads(body)
    except Exception:
        return None, None
    usage = d.get("usage") or {}
    ptok = usage.get("prompt_tokens")
    ch = (d.get("choices") or [{}])[0]
    return (ch.get("delta") or {}).get("content"), ptok


def parse_ollama(line):
    try:
        d = json.loads(line)
    except Exception:
        return None, None
    return (d.get("message") or {}).get("content"), d.get("prompt_eval_count")


def ttft(url, payload, parse):
    """Return (ttft_seconds, prompt_tokens_reported)."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t_send = time.perf_counter()
    first, ptok = None, None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            txt, p = parse(line)
            if p:
                ptok = p
            if txt and first is None:
                first = time.perf_counter() - t_send
                # keep draining: the usage/prompt_eval_count record arrives at the end
    return first, ptok


class Engine:
    def __init__(self, name, model_key):
        self.name, self.model_key = name, model_key
        self.path, self.tag = MODELS[model_key]
        self.proc = None

    def __enter__(self):
        if self.name == "goinfer":
            self.proc = subprocess.Popen(
                [SERVE_CUDA, "-model", f"bench={self.path}", "-backend", "cuda",
                 "-addr", f"127.0.0.1:{GPORT}", "-quant", "int4"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
            self.port = GPORT
            self.url = f"http://127.0.0.1:{GPORT}/v1/chat/completions"
            self.parse = parse_openai
        else:
            env = dict(os.environ, OLLAMA_MODELS=OLLAMA_MODELS, OLLAMA_HOST=f"127.0.0.1:{OPORT}")
            self.proc = subprocess.Popen([OLLAMA, "serve"], env=env,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                         preexec_fn=os.setsid)
            self.port = OPORT
            self.url = f"http://127.0.0.1:{OPORT}/api/chat"
            self.parse = parse_ollama
        if not wait_port(self.port):
            raise RuntimeError(f"{self.name}: server did not come up")
        return self

    def __exit__(self, *a):
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            self.proc.wait(timeout=45)
        except Exception:
            self.proc.kill()
        time.sleep(2)

    def payload(self, text):
        # max_tokens/num_predict = 1: only the FIRST token is needed, and generating
        # more would add decode time to a wall clock that is never read past t_first
        # anyway. Keeping it at 1 shortens the run without touching what is measured.
        if self.name == "goinfer":
            return {"model": "bench", "stream": True, "max_tokens": 1, "temperature": 0,
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": text}]}
        return {"model": self.tag, "stream": True,
                "options": {"temperature": 0, "num_predict": 1, "num_gpu": 99},
                "messages": [{"role": "user", "content": text}]}

    def measure(self, base, n, warm_prefix=9000):
        # Warm-up on its OWN prompt. It must load weights without priming the prompt
        # under test -- the first version of the cache probe warmed with the measured
        # prompt, which made the first timed request a cache hit and reported "no
        # caching" for an engine that was caching.
        ttft(self.url, self.payload(uniq(warm_prefix) + base), self.parse)
        out, ptoks = [], []
        for i in range(n):
            t, p = ttft(self.url, self.payload(uniq(i) + base), self.parse)
            if t:
                out.append(t)
            if p:
                ptoks.append(p)
        return out, (statistics.median(ptoks) if ptoks else None)

    def cache_check(self, base):
        """repeat-vs-new, after a warm-up on a third prompt. Returns the ratio."""
        ttft(self.url, self.payload(uniq(8000) + base), self.parse)
        fresh = [ttft(self.url, self.payload(uniq(8100 + i) + base), self.parse)[0] for i in range(2)]
        rep_text = self.payload(uniq(8200) + base)
        ttft(self.url, rep_text, self.parse)              # prime
        rep = [ttft(self.url, rep_text, self.parse)[0] for _ in range(2)]
        fresh = [f for f in fresh if f]
        rep = [r for r in rep if r]
        if not fresh or not rep:
            return None
        return statistics.median(rep) / statistics.median(fresh)


def machine_header():
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader"),
        "kernel": platform.release(),
        "distro": sh(". /etc/os-release 2>/dev/null && echo $PRETTY_NAME"),
        "goinfer_commit": sh("git -C %s rev-parse --short HEAD" % HERE),
        "goinfer_dirty": bool(sh("git -C %s status --porcelain" % HERE)),
        "peer_version": sh("%s --version 2>/dev/null | head -1" % OLLAMA),
        "serve_mtime": sh("stat -c %%y %s 2>/dev/null" % SERVE_CUDA),
        "loadavg": sh("cat /proc/loadavg"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--models", default="0.5B,1.5B")
    ap.add_argument("--depths", default="512,1024,2048")
    ap.add_argument("--n", type=int, default=6, help="distinct prompts per cell")
    ap.add_argument("--verify-nocache", action="store_true", default=True)
    ap.add_argument("--cache-bar", type=float, default=0.80,
                    help="repeat/fresh below this = engine caches and fresh prompts miss it (healthy)")
    ap.add_argument("--require-scaling", action="store_true", default=True,
                    help="refuse if TTFT does not rise with depth (a cache hit does not scale)")
    a = ap.parse_args()

    hdr = machine_header()
    print(json.dumps(hdr, indent=2))
    results, cachechk = [], []

    for mk in a.models.split(","):
        for depth in [int(d) for d in a.depths.split(",")]:
            key = f"{mk}:{depth}"
            if key not in PROMPTS:
                print(f"skip {key}: not in prompts.json")
                continue
            base = PROMPTS[key]["text"]
            cell = {"model": mk, "depth": depth, "calibrated_tokens": PROMPTS[key]["tokens"]}
            # INTERLEAVED: both engines measured for this cell before moving on, each
            # with its own server start/stop, so drift between cells cannot land on one
            # engine (CLAUDE.md: peer comparisons must be same-session interleaved).
            for name in ("goinfer", "ollama"):
                with Engine(name, mk) as e:
                    if a.verify_nocache:
                        r = e.cache_check(base)
                        cachechk.append({"model": mk, "depth": depth, "engine": name, "repeat_over_new": r})
                        # Recorded, NOT refused on. A low ratio is the healthy signal
                        # (engine caches; our fresh prompts miss it). See the header.
                        if r is not None:
                            note = "caches; fresh prompts miss it (healthy)" if r < a.cache_bar \
                                   else "no caching detected on identical prompts"
                            print(f"  [cache-check] {name:8s} {key} repeat/fresh={r:.2f}  {note}")
                    ts, ptok = e.measure(base, a.n)
                    if not ts:
                        cell[name] = {"error": "no timings"}
                        continue
                    med = statistics.median(ts)
                    tok = ptok or PROMPTS[key]["tokens"]
                    cell[name] = {"ttft_ms_median": round(med * 1000, 1),
                                  "ttft_ms_all": [round(t * 1000, 1) for t in ts],
                                  "prompt_tokens": tok,
                                  "prefill_tok_s": round(tok / med, 1),
                                  "spread_pct": round(100 * (max(ts) - min(ts)) / med, 1)}
            if "goinfer" in cell and "ollama" in cell and "error" not in cell["goinfer"] and "error" not in cell["ollama"]:
                cell["peer_over_goinfer"] = round(
                    cell["ollama"]["prefill_tok_s"] / cell["goinfer"]["prefill_tok_s"], 2)
            results.append(cell)
            g = cell.get("goinfer", {}).get("prefill_tok_s")
            o = cell.get("ollama", {}).get("prefill_tok_s")
            print(f"{mk:6s} K={depth:<6d} goinfer {g!s:>9} tok/s   ollama {o!s:>9} tok/s   "
                  f"ratio {cell.get('peer_over_goinfer')}")

    # CHECK 2 -- absolute, and the one that catches what the ratio cannot. Real
    # prefill work grows with prompt length; a cache lookup is flat. If an engine's
    # median TTFT does not rise across the swept depths, its numbers are lookups.
    scaling = {}
    for name in ("goinfer", "ollama"):
        for mk in a.models.split(","):
            pts = [(c["depth"], c[name]["ttft_ms_median"]) for c in results
                   if c["model"] == mk and name in c and "error" not in c[name]]
            if len(pts) < 2:
                continue
            pts.sort()
            lo, hi = pts[0], pts[-1]
            grow = hi[1] / lo[1]
            depth_ratio = hi[0] / lo[0]
            scaling[f"{name}:{mk}"] = {"depths": [lo[0], hi[0]], "ttft_ms": [lo[1], hi[1]],
                                       "ttft_growth": round(grow, 2), "depth_growth": round(depth_ratio, 2)}
            # Bar deliberately loose (1.5x TTFT growth over a >=2x depth growth):
            # prefill is superlinear-ish but there is fixed overhead in TTFT, so the
            # test is "does it move at all", not "is it exactly linear".
            if a.require_scaling and depth_ratio >= 2 and grow < 1.5:
                print(f"REFUSING: {name} {mk} TTFT barely moved across depths "
                      f"({lo[0]}->{hi[0]} tokens, {lo[1]:.0f}->{hi[1]:.0f} ms, {grow:.2f}x). "
                      f"Prefill scales with length and a cache lookup does not, so these "
                      f"are lookups, not prefills.")
                sys.exit(2)
    for k, v in scaling.items():
        print(f"  [scaling] {k:16s} {v['depths'][0]}->{v['depths'][1]} tok  "
              f"{v['ttft_ms'][0]:.0f}->{v['ttft_ms'][1]:.0f} ms  ({v['ttft_growth']}x over {v['depth_growth']}x depth)")

    json.dump({"header": hdr, "cache_check": cachechk, "scaling_check": scaling, "cells": results},
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
