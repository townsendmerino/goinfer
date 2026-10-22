#!/usr/bin/env python3
"""KV4's G3b at tier 1 (kv4-cuda-2026-10-21.md §2, as amended 2026-10-22): four conversations whose prompts share a preamble of about
16,000 tokens (the 1.5B's depth-16,000 bench prompt as the system message, a different short user turn each), on one server at serve's
defaults. Serve copies a prefix only when the slot holding it is busy (an idle holder is reused in place), so each conversation is sent
once the one before it is streaming: conversation 0 prefills the preamble, and 1 to 3 each copy it (KV2), which under KV4a-2 shares the
blocks. The pool's count is read from serve's log line after each copy.

Pass: (1) three shared copies are logged, and the full pool's blocks in use at the last copy are at most the shared blocks plus, per
conversation, the blocks of its own tail and its generated tokens plus one; (2) each reply equals that conversation's reference: on a
fresh server, conversation 0 alone, then conversation i (which reuses slot 0's rows in place and decodes solo; MC3's batched step is
per-row identical to solo decode). The reply on a fresh server with nothing before it is reported, not graded.

  g3b_served.py <serve binary> <out dir>
"""
import json, os, re, signal, subprocess, sys, threading, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../scripts"))
import bench_peer_prefill as bp

binary, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
MODEL = os.path.expanduser("~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
PRE = bp.PROMPTS["1.5B:16000"]["text"]
TURNS = ["Summarise the passage in one sentence.", "List three words from the passage.", "What is the passage about?",
         "Name the passage's first topic."]
GEN = 128  # long enough that each conversation is still decoding when the next arrives


def server(log):
    p = subprocess.Popen([binary, "-model", f"bench={MODEL}", "-backend", "cuda", "-addr", f"127.0.0.1:{bp.GPORT}", "-quant", "int4",
                          "-require-backend"], stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    assert bp.wait_port(bp.GPORT, 300), "serve did not come up"
    return p


def body(turn, stream):
    return json.dumps({"model": "bench", "max_tokens": GEN, "temperature": 0, "stream": stream,
                       "stream_options": {"include_usage": True} if stream else None,
                       "messages": [{"role": "system", "content": PRE}, {"role": "user", "content": turn}]}).encode()


def ask(turn):
    req = urllib.request.Request(f"http://127.0.0.1:{bp.GPORT}/v1/chat/completions", body(turn, False), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return r["choices"][0]["message"]["content"], r.get("usage", {})


def stream(turn, started, result):
    """Streams one conversation; sets started at its first content chunk, and stores (reply, usage) in result."""
    req = urllib.request.Request(f"http://127.0.0.1:{bp.GPORT}/v1/chat/completions", body(turn, True), {"Content-Type": "application/json"})
    text, usage = [], {}
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for c in ev.get("choices", []):
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    text.append(piece)
                    started.set()
    started.set()
    result.append(("".join(text), usage))


def stop(p):
    os.killpg(p.pid, signal.SIGTERM)
    time.sleep(5)


log = os.path.join(out, "shared.log")
p = server(log)
try:
    results, threads = [[] for _ in TURNS], []
    for i, t in enumerate(TURNS):
        started = threading.Event()
        th = threading.Thread(target=stream, args=(t, started, results[i]))
        th.start()
        threads.append(th)
        started.wait(900)
    for th in threads:
        th.join()
    together = [r[0] for r in results]
finally:
    stop(p)

refs, alone = [], []
for i, t in enumerate(TURNS):
    p = server(os.path.join(out, f"ref-{i}.log"))
    try:
        first = ask(TURNS[0])
        refs.append(first if i == 0 else ask(t))
    finally:
        stop(p)
    if i == 0:
        alone.append(first)
        continue
    p = server(os.path.join(out, f"alone-{i}.log"))
    try:
        alone.append(ask(t))
    finally:
        stop(p)

copies = [l for l in open(log) if "cuda: kv copy:" in l and "shares rows" in l]
for l in copies:
    print(l.rstrip())
m = re.search(r"rows \[(\d+), (\d+)\), (\d+) blocks; (\d+) of (\d+) full-pool blocks in use", copies[-1]) if copies else None
ok = True
if not m or len(copies) < 3:
    print(f"VOID: {len(copies)} shared copies logged, want 3")
    ok = False
else:
    shared = int(m.group(2)) // 64
    prompts = [u.get("prompt_tokens", 0) for _, u in together]
    gens = [u.get("completion_tokens", GEN) for _, u in together]
    own = [-(-(n + g - shared * 64) // 64) for n, g in zip(prompts, gens)]  # each conversation's own blocks past the shared ones
    in_use = int(m.group(4))
    limit = shared + sum(o + 1 for o in own)
    print(f"preamble shared: {shared} blocks; prompts {prompts} tokens, generated {gens}; full pool in use at the last copy {in_use} "
          f"(limit {limit}; {sum(-(-(n + g) // 64) for n, g in zip(prompts, gens))} if each stored its own)")
    ok &= in_use <= limit
for i, ((a, ua), (b, _), (c, _)) in enumerate(zip(together, refs, alone)):
    same = a == b
    ok &= same
    print(f"conversation {i}: reused {ua.get('prefill_reused_tokens')} tokens; reply {'equals' if same else 'DIFFERS FROM'} its reference; "
          f"{'equals' if a == c else 'differs from'} its reply on a fresh server (reported)")
json.dump({"together": together, "refs": refs, "alone": alone}, open(os.path.join(out, "replies.json"), "w"), indent=1)
print("G3b served:", "PASS" if ok else "FAIL")
