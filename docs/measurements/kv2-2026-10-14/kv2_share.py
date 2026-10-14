#!/usr/bin/env python3
"""KV2's G5, the sharing gate (docs/measurements/kv2-2026-10-14.md §2, G5), and G7's served copy cost.

Three arms: main (the merge base), off (this branch, -kv-share off), kv2 (this branch). Three scripts, each on its own
freshly started server, the arms in a rotating order per block (ABC, BCA, CAB, ...):
  busy     A1 streams; at A1's first content token B1 is sent; B1's time to first token is timed.
  idle     A1 runs to completion, then B1, then A2 (A1's messages, A1's reply, a short turn); B1 and A2 are timed.
  control  busy, with the header on: each conversation's preamble carries its own three-hex token near its start.
The prompts have a header-off Claude Code first request's shape (docs/measurements/kv0-traffic-2026-10/header-probe.txt):
a shared system preamble of about 18,213 tokens, the conversation's own first message, and a fixed tail of about 1,940
tokens after it. The server's own token counts are recorded; the preamble's words are fixed, not its token count.

  kv2_share.py OUT.json --backend metal --new BIN --old BIN --model PATH --name 1.5B [--blocks 6] [--arms main,off,kv2]
      [--scripts busy,idle,control] [--log-dir DIR]
"""
import argparse, json, os, re, signal, socket, subprocess, sys, threading, time, urllib.request

PORT = 8098
WORDS_PER_TOKEN = 2500 / 3070  # KV1's 2,500-word prompts were 3,069-3,070 tokens on Qwen2.5's tokenizer
SECTIONS = ["tool use", "file edits", "shell commands", "git workflow", "tests", "code review", "planning", "memory",
            "permissions", "output style", "subagents", "hooks", "settings", "MCP servers", "notebooks", "web fetch"]
PARA = ("When the {s} instructions apply, read the relevant files before changing them, prefer small reviewable edits, "
        "state what you verified and what you did not, and keep the user's working tree as you found it unless asked. "
        "Rule {i} of the {s} section: a command that touches shared state is confirmed first; a long output is "
        "summarised with its exit status; a failure is diagnosed before it is retried. ")


def words(n, salt):
    out, i = [], 0
    while len(" ".join(out).split()) < n:
        out.append(PARA.format(s=SECTIONS[(i + salt) % len(SECTIONS)], i=i))
        i += 1
    return " ".join(" ".join(out).split()[:n])


SIZE = {"pre": 18213, "tail": 1940}  # target tokens; --preamble-tokens / --tail-tokens change them (a smoke only)


def preamble(header_hex=None):
    head = "You are an interactive CLI tool that helps users with software engineering tasks.\n"
    if header_hex is not None:
        head = f"x-anthropic-billing-header: cc_version=2.1.296.{header_hex}; cc_entrypoint=sdk-cli;\n" + head
    return head + words(int(SIZE["pre"] * WORDS_PER_TOKEN), 0)


def first_message(conv):
    asks = {"A": "Find where the rate limiter's refill is computed and explain the off-by-one in one paragraph.",
            "B": "List the three slowest tests in this repository and suggest one fix for each of them."}
    tail = "\n\n<env>\nWorking directory: /work/project\nIs a git repository: true\nPlatform: darwin\n</env>\n" + words(
        int(SIZE["tail"] * WORDS_PER_TOKEN), 7) + "\nToday's date is 2026-10-14."
    return asks[conv] + tail


def wait_port(proc, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
            return True
        except OSError:
            time.sleep(0.25)
    return False


def stream(messages, max_tokens, on_first=None):
    """Streams one chat request. Returns a dict: ttft (s), text, prompt_tokens, reused, done (s after the send)."""
    body = {"model": "bench", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = {"ttft": None, "text": "", "prompt_tokens": None, "reused": None, "sent": time.time()}
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            try:
                ev = json.loads(line[5:])
            except ValueError:
                continue
            if ev.get("usage"):
                out["prompt_tokens"] = ev["usage"].get("prompt_tokens")
                out["reused"] = ev["usage"].get("prefill_reused_tokens")
            piece = ((ev.get("choices") or [{}])[0].get("delta") or {}).get("content")
            if piece:
                if out["ttft"] is None:
                    out["ttft"] = time.perf_counter() - t0
                    if on_first:
                        on_first()
                out["text"] += piece
    out["done"] = time.perf_counter() - t0
    out["ended"] = time.time()
    return out


def conv_messages(conv, header_hex=None):
    return [{"role": "system", "content": preamble(header_hex)}, {"role": "user", "content": first_message(conv)}]


def run_busy(header):
    hx = ("a3f", "7c2") if header else (None, None)
    b, box = {}, {}

    def send_b():
        box["b_sent"] = time.time()
        b.update(stream(conv_messages("B", hx[1]), 8))

    th = threading.Thread(target=send_b)
    a = stream(conv_messages("A", hx[0]), 128, on_first=th.start)
    th.join()
    return {"a1": a, "b1": b, "a1_ended_before_b1_sent": a["ended"] < box.get("b_sent", 0)}


def run_idle():
    a1 = stream(conv_messages("A"), 128)
    b1 = stream(conv_messages("B"), 8)
    a2 = stream(conv_messages("A") + [{"role": "assistant", "content": a1["text"]},
                                      {"role": "user", "content": "Continue with the fix."}], 8)
    return {"a1": a1, "b1": b1, "a2": a2}


def copies_in(log, offset):
    """The 'kv share' lines a server wrote after offset: rows and milliseconds of each copy, and any declines."""
    with open(log) as f:
        f.seek(offset)
        text = f.read()
    cp = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"kv share: copied (\d+) rows from another conversation in ([\d.]+) ms", text)]
    dec = re.findall(r"kv share: a copy was declined \(([^)]*)\)", text)
    return cp, dec


def cell(a, arm, script, block, log):
    binary = a.old if arm == "main" else a.new
    argv = [binary, "-model", f"bench={a.model}", "-backend", a.backend, "-require-backend", "-addr", f"127.0.0.1:{PORT}",
            "-quant", a.quant, "-ctx", str(a.ctx), "-kv-sessions", str(a.slots)] + a.serve_args.split()
    if arm == "off":
        argv += ["-kv-share", "off"]
    with open(log, "a") as sink:
        sink.write(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} block {block} {script} {arm}: {' '.join(argv)}\n")
        sink.flush()
        start = os.path.getsize(log)
        proc = subprocess.Popen(argv, stdout=sink, stderr=sink, preexec_fn=os.setsid)
        try:
            if not wait_port(proc, a.load_timeout):
                return {"error": "the server did not come up"}
            stream([{"role": "user", "content": "warm"}], 4)
            r = run_busy(script == "control") if script in ("busy", "control") else run_idle()
            sink.flush()
            r["copies"], r["declines"] = copies_in(log, start)
            return r
        except Exception as e:  # recorded, never retried: the grader voids the cell
            return {"error": repr(e)}
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=60)
            except Exception:
                proc.kill()
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--backend", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--arms", default="main,off,kv2")
    ap.add_argument("--scripts", default="busy,idle,control")
    ap.add_argument("--ctx", type=int, default=24576)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--serve-args", default="")
    ap.add_argument("--load-timeout", type=float, default=420)
    ap.add_argument("--log-dir", default=".")
    ap.add_argument("--preamble-tokens", type=int, default=18213)
    ap.add_argument("--tail-tokens", type=int, default=1940)
    ap.add_argument("--quant", default="int4")
    a = ap.parse_args()
    SIZE["pre"], SIZE["tail"] = a.preamble_tokens, a.tail_tokens
    arms, scripts = a.arms.split(","), a.scripts.split(",")
    cells, t_start, total = [], time.time(), a.blocks * len(arms) * len(scripts)
    for b in range(a.blocks):
        order = arms[b % len(arms):] + arms[:b % len(arms)]
        for script in scripts:
            for arm in order:
                r = cell(a, arm, script, b, os.path.join(a.log_dir, f"share-{a.backend}-{a.name}-{arm}.log"))
                r.update(arm=arm, script=script, block=b, model=a.name, backend=a.backend, at=time.strftime("%H:%M:%S"))
                cells.append(r)
                json.dump({"args": vars(a), "cells": cells}, open(a.out, "w"), indent=1)
                done, el = len(cells), time.time() - t_start
                b1 = r.get("b1", {})
                print(f"[kv2-share {a.backend} {a.name}] cell {done}/{total} block {b} {script} {arm}: "
                      f"B1 ttft {b1.get('ttft')} reused {b1.get('reused')} of {b1.get('prompt_tokens')}; copies {r.get('copies')} "
                      f"{r.get('error', '')} elapsed {el / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
