#!/usr/bin/env python3
"""G-S14c5's requests (docs/tasks/task-multimodal-support-2026-10.md): input_audio chat requests for the three LibriSpeech files against a running serve of Qwen3-ASR.
Usage: s14c5_served.py <port> <out dir>   (from the repo root). Prints each reply verbatim, the usage, and the checks (a)-(c)."""
import base64, json, re, sys, time, urllib.request
port, out = sys.argv[1:3]
S = "testdata/speech/"
FILES = [("original16k", "librispeech-1272-128104-0000.wav"), ("resample44k1", "librispeech-1272-128104-0000-44k1.wav"), ("stereo48k", "librispeech-1272-128104-0000-48k-stereo.wav")]
WANT = "language English<asr_text>Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
REF = "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel".split()
def norm(t):
    t = t.replace("Mr.", "mister").replace("Mr", "mister")
    return re.sub(r"[^a-z' ]", "", t.lower()).split()
def wer(ref, hyp):
    d = [[i + j if i * j == 0 else 0 for j in range(len(hyp) + 1)] for i in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + (ref[i-1] != hyp[j-1]))
    return d[-1][-1] / len(ref)
def ask(msgs, max_tokens=64):
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": 0}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=900))
    return r, time.time() - t
ok = True
for name, f in FILES:
    b64 = base64.b64encode(open(S + f, "rb").read()).decode()
    r, el = ask([{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}])
    text = r["choices"][0]["message"]["content"]; u = r["usage"]
    open(f"{out}/{name}.txt", "w").write(text)
    print(f"[{name}] {el:.1f}s prompt_tokens {u['prompt_tokens']} completion_tokens {u['completion_tokens']}\n    reply: {text!r}")
    ctrl = re.findall(r"<\|[^|>]*\|>", text)
    if ctrl: print(f"    NOTE: control token(s) in the raw reply: {ctrl}")
    body = text.split("<asr_text>", 1)[-1]
    w = wer(REF, norm(body))
    if name == "original16k":
        ok &= (text == WANT); print(f"    (a) byte-equal to the library path: {text == WANT}; (c) prompt_tokens == 91: {u['prompt_tokens'] == 91}"); ok &= u["prompt_tokens"] == 91
    else:
        ok &= (w == 0); print(f"    (b) word error rate {w:.3f}")
r, _ = ask([{"role": "user", "content": "Say hello."}], 8)
print("[text only] answers:", repr(r["choices"][0]["message"]["content"])[:80])
print("G-S14c5:", "PASS" if ok else "NOT PASSED (see the lines above)")
