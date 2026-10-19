#!/usr/bin/env python3
"""Does prefix reuse work for Qwen3.5-9B (a Gated-DeltaNet hybrid) when each request only APPENDS to the previous one?

Tack's "reused" cell changes the last 512 tokens of the previous prompt, which a recurrent model cannot resume (decoder/cachestate.go, recurrentCells).
Its loop appends the model's reply and the next user turn instead. This probe plays that: turn 1 is a cold prompt of about N tokens, turn 2 and turn 3 append the
previous reply and a new user message through the chat endpoint, and a cold control of the turn-3 length (a nonce at the start of the system prompt) follows.
Time to first streamed token for each, plus usage.prompt_tokens.

  python3 probe.py --serve <binary> --model <gguf> --out <dir> [--sizes 4096,8192] [--ctx 16384]
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request

PORT = 18199
BASE = f"http://127.0.0.1:{PORT}"
SYS = "You are a careful assistant in a long-running coding session. Answer in one short sentence."
WORDS = ("the build queue drains while a cache warms and the scheduler picks the next task from a list of pending jobs, each with its own "
         "deadline and a note about which files it reads before it writes anything back to the shared tree").split()


def filler(chars):
    out, i = [], 0
    while sum(map(len, out)) + len(out) < chars:
        out.append(WORDS[(i * 7 + i // 13) % len(WORDS)] + (f"-{i % 997}" if i % 11 == 0 else ""))
        i += 1
    return " ".join(out)


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def ttft(messages):
    """Seconds to the first streamed token, the reply text, and the prompt token count."""
    body = {"model": "probe", "messages": messages, "stream": True, "max_tokens": 24, "temperature": 0,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0, first, text, ptoks = time.time(), None, "", None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                ptoks = d["usage"].get("prompt_tokens")
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece and first is None:
                    first = time.time() - t0
                text += delta.get("content") or ""
    return first, text, ptoks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="4096,8192")
    ap.add_argument("--ctx", default="16384")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "serve.log"), "w")
    proc = subprocess.Popen([a.serve, "-model", f"probe={a.model}", "-backend", "cuda", "-addr", f"127.0.0.1:{PORT}", "-ctx", a.ctx],
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    rows = []
    try:
        t0 = time.time()
        while time.time() - t0 < 600:
            try:
                urllib.request.urlopen(BASE + "/v1/models", timeout=1).read()
                break
            except Exception:
                if proc.poll() is not None:
                    sys.exit("server exited during load; see serve.log")
                time.sleep(0.5)
        print(f"server up in {time.time() - t0:.1f}s", flush=True)
        # chars per token from the server's own count, once
        probe_text = filler(4000)
        n = post("/v1/messages/count_tokens", {"model": "probe", "messages": [{"role": "user", "content": probe_text}]}).get("input_tokens")
        cpt = len(probe_text) / n
        print(f"{cpt:.2f} chars per token", flush=True)
        ttft([{"role": "user", "content": "warm up"}])
        for size in [int(x) for x in a.sizes.split(",")]:
            fill = filler(int(size * cpt))
            sysmsg = SYS + "\n\nReference notes:\n" + fill
            msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": "Say one word about the notes."}]
            results = []
            for turn, follow in ((1, None), (2, "And what is one more word?"), (3, "One last word, please.")):
                if follow:
                    msgs = msgs + [{"role": "assistant", "content": reply}, {"role": "user", "content": follow}]
                first, reply, ptoks = ttft(msgs)
                results.append({"turn": turn, "ttft_s": round(first, 3), "prompt_tokens": ptoks})
                print(f"  size {size} turn {turn}: ttft {first:.2f}s, {ptoks} prompt tokens", flush=True)
            cold_msgs = [dict(msgs[0], content=f"run {time.time_ns()}\n\n" + msgs[0]["content"])] + msgs[1:]
            first, _, ptoks = ttft(cold_msgs)
            results.append({"turn": "cold control (turn-3 length)", "ttft_s": round(first, 3), "prompt_tokens": ptoks})
            print(f"  size {size} cold control: ttft {first:.2f}s, {ptoks} prompt tokens", flush=True)
            rows.append({"size": size, "results": results})
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            os.killpg(proc.pid, signal.SIGKILL)
        log.close()
    json.dump({"serve": a.serve, "model": a.model, "rows": rows}, open(os.path.join(a.out, "probe.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
