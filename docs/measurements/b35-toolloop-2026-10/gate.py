#!/usr/bin/env python3
"""B35-TL (README.md here): does B35's kept state serve a new question after a tool loop? Qwen3.5-9B through /v1/chat/completions, a
conversation of three questions, each answered after a tool loop, every assistant turn's reasoning sent back. Writes gate.json; grade.py
reads it.

  r1  Q1                         r4  ... + answer + Q2 (graded)
  r2  ... + call + result        r5  ... + call + result
  r3  ... + call + result        r6  ... + answer + Q3 (graded)

  A     one server, streamed: r1-r6 in order (time to first token, prompt tokens, reused tokens), then a cold control of r6's length.
  B     one server, logprobs: r1-r6 in order, the warm path.
  cold  a fresh server each, logprobs: r4, then r6.

  python3 gate.py --serve <binary> --model <gguf> --out <dir> [--size 4096] [--result 1500] [--ctx 16384]
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request

PORT = 18207
BASE = f"http://127.0.0.1:{PORT}"
SYS = "You are a careful assistant in a long-running coding session. Use the tools when they help. Answer in one short sentence."
WORDS = ("the build queue drains while a cache warms and the scheduler picks the next task from a list of pending jobs, each with its own "
         "deadline and a note about which files it reads before it writes anything back to the shared tree").split()
MAX_TOKENS = 48
THINK = {"chat_template_kwargs": {"enable_thinking": True}}
TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
    for n, d in (("read_file", "Read a file from the repository."), ("list_dir", "List a directory of the repository."))]


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def filler(chars, seed):
    out, i = [], seed
    while sum(map(len, out)) + len(out) < chars:
        out.append(WORDS[(i * 7 + i // 13) % len(WORDS)] + (f"-{i % 997}" if i % 11 == 0 else ""))
        i += 1
    return " ".join(out)


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


class Server:
    def __init__(self, binary, model, ctx, log):
        self.log = open(log, "w")
        self.proc = subprocess.Popen([binary, "-model", f"m={model}", "-backend", "cuda", "-addr", f"127.0.0.1:{PORT}", "-ctx", ctx],
                                     stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < 600:
            try:
                urllib.request.urlopen(BASE + "/v1/models", timeout=1).read()
                self.up_s = round(time.time() - t0, 1)
                return
            except Exception:
                if self.proc.poll() is not None:
                    sys.exit(f"server exited during load; see {log}")
                time.sleep(0.5)
        self.close()
        sys.exit(f"server not up in 600 s; see {log}")

    def close(self):
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=30)
        except Exception:
            os.killpg(self.proc.pid, signal.SIGKILL)
        self.log.close()
        time.sleep(2)  # the card's memory is released before the next load


def body(messages, **kw):
    return {"model": "m", "messages": messages, "tools": TOOLS, "max_tokens": MAX_TOKENS, "temperature": 0, **THINK, **kw}


def streamed(messages):
    """Seconds to the first streamed token (content, reasoning or a call), the reply, the usage."""
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body(messages, stream=True,
                                 stream_options={"include_usage": True})).encode(), headers={"Content-Type": "application/json"})
    t0, first, text, use = time.time(), None, "", {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                use = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "") + json.dumps(delta.get("tool_calls") or "")
                if piece.strip('"') and first is None:
                    first = time.time() - t0
                text += (delta.get("reasoning_content") or "") + (delta.get("content") or "")
    return first, text, use


def with_logprobs(messages):
    """The reply (reasoning, content and the tool calls' JSON), its tokens with their logprobs exactly as served (the JSON text of each
    number; the tools route serves none, so the list is empty there), and the usage."""
    raw = urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body(messages, logprobs=True)).encode(),
                                 headers={"Content-Type": "application/json"}), timeout=900).read().decode()
    d = json.loads(raw, parse_float=str)  # keep each logprob's exact text: bit for bit means the same digits
    ch = d["choices"][0]
    toks = [[t["token"], t["logprob"]] for t in (ch.get("logprobs") or {}).get("content") or []]
    msg = ch["message"]
    return (msg.get("reasoning_content") or "") + (msg.get("content") or "") + json.dumps(msg.get("tool_calls") or []), toks, d.get("usage", {})


def requests(sysmsg, result_chars):
    """The six requests. Assistant turns are the client's own history, scripted, each with its reasoning sent back."""
    def call(i, name, path, why):
        return {"role": "assistant", "content": "", "reasoning_content": why,
                "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps({"path": path})}}]}

    def result(i, name, seed):
        return {"role": "tool", "tool_call_id": f"call_{i}", "name": name, "content": filler(result_chars, seed)}

    def answer(text, why):
        return {"role": "assistant", "content": text, "reasoning_content": why}

    c = [{"role": "system", "content": sysmsg}, {"role": "user", "content": "Which file sets the scheduler's deadline?"}]
    ends = [2]
    c += [call(1, "list_dir", "sched/", "List the scheduler package first."), result(1, "list_dir", 101)]
    ends.append(len(c))
    c += [call(2, "read_file", "sched/queue.go", "The queue file is the likely place."), result(2, "read_file", 202)]
    ends.append(len(c))
    c += [answer("sched/queue.go sets it.", "The queue file names the deadline."), {"role": "user", "content": "Does it read the shared tree?"}]
    ends.append(len(c))
    c += [call(3, "read_file", "sched/tree.go", "Check the tree reader."), result(3, "read_file", 303)]
    ends.append(len(c))
    c += [answer("Yes, before it writes.", "The tree reader runs first."), {"role": "user", "content": "One word for the whole design?"}]
    ends.append(len(c))
    return [c[:e] for e in ends]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=4096, help="system prompt tokens")
    ap.add_argument("--result", type=int, default=1500, help="tokens per tool result")
    ap.add_argument("--ctx", default="16384")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rec = {"serve": a.serve, "model": a.model, "ctx": a.ctx, "size": a.size, "result": a.result, "max_tokens": MAX_TOKENS}
    t_all = time.time()

    say("A: streamed r1-r6 on one server, then a cold control of r6's length")
    s = Server(a.serve, a.model, a.ctx, os.path.join(a.out, "serve-A.log"))
    probe = filler(4000, 0)
    cpt = len(probe) / post("/v1/messages/count_tokens", {"model": "m", "messages": [{"role": "user", "content": probe}]})["input_tokens"]
    reqs = requests(SYS + "\n\nReference notes:\n" + filler(int(a.size * cpt), 0), int(a.result * cpt))
    rec["messages"] = reqs
    streamed([{"role": "user", "content": "warm up"}])
    rows = []
    for i, msgs in enumerate(reqs):
        first, text, use = streamed(msgs)
        rows.append({"r": i + 1, "ttft_s": round(first, 3), "prompt_tokens": use.get("prompt_tokens"), "reused": use.get("prefill_reused_tokens"),
                     "reply": text})
        say(f"  r{i + 1}: ttft {first:.2f}s, {use.get('prompt_tokens')} prompt tokens, {use.get('prefill_reused_tokens')} reused "
            f"({time.time() - t_all:.0f}s elapsed)")
    cold = [dict(reqs[5][0], content=f"run {time.time_ns()}\n\n" + reqs[5][0]["content"])] + reqs[5][1:]
    first, _, use = streamed(cold)
    rows.append({"r": "cold control (r6 length)", "ttft_s": round(first, 3), "prompt_tokens": use.get("prompt_tokens"),
                 "reused": use.get("prefill_reused_tokens")})
    say(f"  cold control: ttft {first:.2f}s, {use.get('prefill_reused_tokens')} reused")
    rec["A"], rec["A_load_s"] = rows, s.up_s
    s.close()

    say("B: logprobs r1-r6 on one server (warm)")
    s = Server(a.serve, a.model, a.ctx, os.path.join(a.out, "serve-B.log"))
    rec["B"] = []
    for i, msgs in enumerate(reqs):
        text, toks, use = with_logprobs(msgs)
        rec["B"].append({"r": i + 1, "reply": text, "tokens": toks, "usage": use})
    s.close()

    rec["cold"] = []
    for r in (4, 6):
        say(f"cold: r{r} on a fresh server")
        s = Server(a.serve, a.model, a.ctx, os.path.join(a.out, f"serve-cold-{r}.log"))
        text, toks, use = with_logprobs(reqs[r - 1])
        rec["cold"].append({"r": r, "reply": text, "tokens": toks, "usage": use})
        s.close()
    json.dump(rec, open(os.path.join(a.out, "gate.json"), "w"), indent=1)
    say(f"done in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
