#!/usr/bin/env python3
"""B35's end-to-end gate (README.md here): Qwen3.5-9B through /v1/chat/completions, a three-turn conversation that appends each reply and a
new question. Writes one JSON record; grade.py reads it.

Per size, on the B35 build:
  A  one server, streamed: turns 1, 2, 3 (time to the first token, prefill_reused_tokens), then a cold control of turn 3's length
     (a nonce first in the system prompt).
  B  one server, logprobs: turns 1, 2, 3, the warm path, recording each token and its logprob.
  C, D  a fresh server each, logprobs: turn 2, then turn 3, cold (the same messages as B's).
And on the base build (main without B35), each on a fresh server, logprobs: turns 1, 2, 3 cold, so cold output is compared across builds.

  python3 gate.py --b35 <serve> --base <serve> --model <gguf> --out <dir> [--sizes 4096,8192] [--ctx 16384]
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request

PORT = 18203
BASE = f"http://127.0.0.1:{PORT}"
SYS = "You are a careful assistant in a long-running coding session. Answer in one short sentence."
WORDS = ("the build queue drains while a cache warms and the scheduler picks the next task from a list of pending jobs, each with its own "
         "deadline and a note about which files it reads before it writes anything back to the shared tree").split()
FOLLOW = (None, "And what is one more word?", "One last word, please.")
MAX_TOKENS = 24


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def filler(chars):
    out, i = [], 0
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
    return {"model": "m", "messages": messages, "max_tokens": MAX_TOKENS, "temperature": 0, **kw}


def streamed(messages):
    """Seconds to the first streamed token, the reply, the usage."""
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
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece and first is None:
                    first = time.time() - t0
                text += delta.get("content") or ""
    return first, text, use


def with_logprobs(messages):
    """The reply, its tokens with their logprobs exactly as served (the JSON text of each number), and the usage."""
    raw = urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body(messages, logprobs=True)).encode(),
                                 headers={"Content-Type": "application/json"}), timeout=900).read().decode()
    d = json.loads(raw, parse_float=str)  # keep each logprob's exact text: bit for bit means the same digits
    ch = d["choices"][0]
    toks = [[t["token"], t["logprob"]] for t in (ch.get("logprobs") or {}).get("content") or []]
    return ch["message"].get("content") or "", toks, d.get("usage", {})


def conversation(sysmsg, replies):
    """The messages of turns 1-3, given the replies of turns 1 and 2."""
    msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": "Say one word about the notes."}]
    out = [msgs]
    for t in (1, 2):
        msgs = msgs + [{"role": "assistant", "content": replies[t - 1]}, {"role": "user", "content": FOLLOW[t]}]
        out.append(msgs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b35", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="4096,8192")
    ap.add_argument("--ctx", default="16384")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rec = {"b35": a.b35, "base": a.base, "model": a.model, "ctx": a.ctx, "max_tokens": MAX_TOKENS, "sizes": []}
    t_all = time.time()
    n_sizes = len(a.sizes.split(","))
    for si, size in enumerate(int(x) for x in a.sizes.split(",")):
        say(f"size {size} ({si + 1}/{n_sizes}), {time.time() - t_all:.0f}s elapsed")
        row = {"size": size}

        say("  A: B35, streamed turns 1-3 and a cold control")
        s = Server(a.b35, a.model, a.ctx, os.path.join(a.out, f"serve-{size}-A.log"))
        probe = filler(4000)
        cpt = len(probe) / post("/v1/messages/count_tokens", {"model": "m", "messages": [{"role": "user", "content": probe}]})["input_tokens"]
        sysmsg = SYS + "\n\nReference notes:\n" + filler(int(size * cpt))
        streamed([{"role": "user", "content": "warm up"}])
        replies, a_rows = [], []
        for t in range(3):
            msgs = conversation(sysmsg, (replies + ["", ""])[:2])[t]
            first, text, use = streamed(msgs)
            replies.append(text)
            a_rows.append({"turn": t + 1, "ttft_s": round(first, 3), "prompt_tokens": use.get("prompt_tokens"),
                           "reused": use.get("prefill_reused_tokens"), "reply": text})
            say(f"    turn {t + 1}: ttft {first:.2f}s, {use.get('prompt_tokens')} prompt tokens, {use.get('prefill_reused_tokens')} reused")
        cold = conversation(sysmsg, replies)[2]
        cold = [dict(cold[0], content=f"run {time.time_ns()}\n\n" + cold[0]["content"])] + cold[1:]
        first, _, use = streamed(cold)
        a_rows.append({"turn": "cold control (turn 3 length)", "ttft_s": round(first, 3), "prompt_tokens": use.get("prompt_tokens"),
                       "reused": use.get("prefill_reused_tokens")})
        say(f"    cold control: ttft {first:.2f}s, {use.get('prefill_reused_tokens')} reused")
        row["A"], row["A_load_s"] = a_rows, s.up_s
        s.close()
        convo = conversation(sysmsg, replies)
        row["messages"] = convo

        say("  B: B35, logprobs, turns 1-3 on one server (warm)")
        s = Server(a.b35, a.model, a.ctx, os.path.join(a.out, f"serve-{size}-B.log"))
        row["B"] = []
        for t in range(3):
            text, toks, use = with_logprobs(convo[t])
            row["B"].append({"turn": t + 1, "reply": text, "tokens": toks, "usage": use})
        s.close()

        for arm, binary, turns in (("cold", a.b35, (2, 3)), ("base", a.base, (1, 2, 3))):
            row[arm] = []
            for t in turns:
                say(f"  {arm}: turn {t} cold on a fresh server")
                s = Server(binary, a.model, a.ctx, os.path.join(a.out, f"serve-{size}-{arm}-{t}.log"))
                text, toks, use = with_logprobs(convo[t - 1])
                row[arm].append({"turn": t, "reply": text, "tokens": toks, "usage": use})
                s.close()
        rec["sizes"].append(row)
        json.dump(rec, open(os.path.join(a.out, "gate.json"), "w"), indent=1)
    say(f"done in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
