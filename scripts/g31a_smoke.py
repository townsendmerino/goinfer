#!/usr/bin/env python3
"""G-31a1's requests (docs/tasks/task-multimodal-support-2026-10.md, "G-31a"): three text chat requests and one image request, 48 greedy tokens each, against a running serve.
Applies the registered automatic bars (non-empty, >= 8 distinct tokens, no <unused / <pad>) and writes every reply verbatim for the reader's coherence call.
Usage: g31a_smoke.py <port> <out dir> <repo root>   (exit 0 when every automatic bar holds; the coherence call is NOT made here)"""
import base64, json, os, sys, time, urllib.request
port, out, root = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
TEXT = ["Explain in two sentences why the sky is blue.", "Write a Python function that returns the n-th Fibonacci number.", "What is the capital of Australia? Answer in one sentence."]
img = base64.b64encode(open(os.path.join(root, 'testdata/glm_ocr/table.png'), 'rb').read()).decode()
reqs = [(f"text{i + 1}", [{"role": "user", "content": p}]) for i, p in enumerate(TEXT)]
reqs.append(("image", [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}}, {"type": "text", "text": "What does this image show? Answer briefly."}]}]))
ok = True
for name, msgs in reqs:
    body = {"messages": msgs, "max_tokens": 48, "temperature": 0, "logprobs": True, "top_logprobs": 1}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}), timeout=3600))
    c = r['choices'][0]
    text = c['message']['content'] or ''
    toks = [x['token'] for x in (c.get('logprobs') or {}).get('content') or []]
    bars = {'non_empty': bool(text.strip()), 'distinct_tokens>=8': len(set(toks)) >= 8, 'no_unused_or_pad': not any(s in text for s in ('<unused', '<pad>')), 'tokens': len(toks)}
    held = bars['non_empty'] and bars['distinct_tokens>=8'] and bars['no_unused_or_pad']
    ok &= held
    open(os.path.join(out, f'{name}.txt'), 'w').write(text)
    json.dump({'usage': r.get('usage'), 'bars': bars, 'seconds': time.time() - t}, open(os.path.join(out, f'{name}.json'), 'w'))
    print(f"[g31a1 {name}] {time.time() - t:6.1f}s prompt_tokens {r.get('usage', {}).get('prompt_tokens')} bars {'HELD' if held else 'NOT HELD'} {bars}\n    reply: {text[:300]!r}", flush=True)
print("G-31a1 AUTOMATIC BARS:", "HELD" if ok else "NOT HELD")
sys.exit(0 if ok else 1)
