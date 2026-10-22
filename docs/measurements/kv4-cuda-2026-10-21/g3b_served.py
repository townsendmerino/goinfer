#!/usr/bin/env python3
"""KV4's G3b at tier 1 (kv4-cuda-2026-10-21.md §2): four conversations whose prompts share a preamble of about 16,000 tokens (the 1.5B's
depth-16,000 bench prompt as the system message, a different short user turn each), sent one after another to one server at serve's
defaults. Serve copies the preamble into each later conversation's slot (KV2), which under KV4a-2 shares the blocks; the pool's own count
is read from serve's log line after each copy. Pass: the full pool's blocks in use once all four have prefilled are at most the shared
blocks plus 4 x (the blocks of a conversation's own rows + 1), and each reply equals that conversation's reply on a fresh server alone.

  g3b_served.py <serve binary> <out dir>
"""
import json, os, re, signal, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../scripts"))
import bench_peer_prefill as bp

binary, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
MODEL = os.path.expanduser("~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
PRE = bp.PROMPTS["1.5B:16000"]["text"]
TURNS = ["Summarise the passage in one sentence.", "List three words from the passage.", "What is the passage about?",
         "Name the passage's first topic."]


def server(log):
    p = subprocess.Popen([binary, "-model", f"bench={MODEL}", "-backend", "cuda", "-addr", f"127.0.0.1:{bp.GPORT}", "-quant", "int4",
                          "-require-backend"], stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    assert bp.wait_port(bp.GPORT, 300), "serve did not come up"
    return p


def ask(turn):
    body = json.dumps({"model": "bench", "max_tokens": 16, "temperature": 0,
                       "messages": [{"role": "system", "content": PRE}, {"role": "user", "content": turn}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{bp.GPORT}/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return r["choices"][0]["message"]["content"], r.get("usage", {})


def stop(p):
    os.killpg(p.pid, signal.SIGTERM)
    time.sleep(5)


log = os.path.join(out, "shared.log")
p = server(log)
try:
    together = [ask(t) for t in TURNS]
finally:
    stop(p)
alone = []
for i, t in enumerate(TURNS):
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
    own = [-(-(n - shared * 64) // 64) for n in prompts]  # each conversation's own blocks past the shared ones
    # the last copy's count is before the fourth conversation's own rows; add them, and the generated tokens' blocks are within the +1
    in_use = int(m.group(4)) + own[3]
    limit = shared + sum(o + 1 for o in own)
    print(f"preamble shared: {shared} blocks; prompts {prompts} tokens; full pool in use after the fourth prefill <= {in_use} "
          f"(limit {limit}; {sum(-(-n // 64) for n in prompts)} if each stored its own)")
    ok &= in_use <= limit
for i, ((a, ua), (b, _)) in enumerate(zip(together, alone)):
    same = a == b
    ok &= same
    print(f"conversation {i}: reused {ua.get('prefill_reused_tokens')} tokens; reply {'equals' if same else 'DIFFERS FROM'} its reply alone")
print("G3b served:", "PASS" if ok else "FAIL")
