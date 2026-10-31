#!/usr/bin/env python3
"""llama.cpp's greedy continuation of run F's exact prompt ids against run F's own output (reasoning, recipient, arguments)."""
import json, re, sys, os
L, F = sys.argv[1], sys.argv[2]
l = json.load(open(os.path.join(L, "llama-completion.json")))
raw = l.get("content", "")
f = [json.loads(x) for x in open(F)][0]
fr, fc = f["reasoning"], f["calls"][0] if f["calls"] else {"name": "", "arguments": ""}
m = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>", raw, re.S)
lr = m.group(1) if m else None
c = re.search(r"to=functions\.([A-Za-z0-9_]+)\s*(?:<\|constrain\|>\s*json)?\s*<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>|$)", raw, re.S)
ln, la = (c.group(1), c.group(2)) if c else (None, None)
def first_diff(a, b):
    return next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
print("llama.cpp raw (first 700):", repr(raw[:700]))
print("tokens predicted", l.get("tokens_predicted"), "stop_type", l.get("stop_type"))
print("reasoning: run F", repr(fr), "| llama.cpp", repr(lr), "| equal", fr == lr)
print("recipient: run F", fc["name"], "| llama.cpp", ln, "| equal", fc["name"] == ln)
if la is not None:
    k = first_diff(fc["arguments"], la)
    print(f"arguments: run F {len(fc['arguments'])} chars, llama.cpp {len(la)} chars; identical for the first {k}")
    print("  run F    :", repr(fc["arguments"][:300]))
    print("  llama.cpp:", repr(la[:300]))
    reps = la.count('"-lc"')
    print(f"  llama.cpp's arguments hold '\"-lc\"' {reps} times (run F's: {fc['arguments'].count(chr(34)+'-lc'+chr(34))})")
