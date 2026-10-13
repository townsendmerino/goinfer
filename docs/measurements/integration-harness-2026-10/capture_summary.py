#!/usr/bin/env python3
"""Print the shape of each request capture.py recorded: capture_summary.py <dir>. Bodies are not printed."""
import json, sys, glob, os
def size(x): return len(json.dumps(x)) if x else 0
for f in sorted(glob.glob(os.path.join(sys.argv[1], "req-*.json")), key=lambda p: int(p.split("-")[-1].split(".")[0])):
    r = json.load(open(f)); b = r["body"]; h = {k.lower(): v for k, v in r["headers"].items()}
    if not isinstance(b, dict):
        print(f"{os.path.basename(f)} {r['method']} {r['path']} (no JSON body) ua={h.get('user-agent','')[:60]}"); continue
    tools = b.get("tools") or []
    kinds = sorted({str(t.get("type", "-")) for t in tools})
    conv = b.get("messages") if "messages" in b else b.get("input")
    roles = [m.get("role", m.get("type")) for m in conv] if isinstance(conv, list) else type(conv).__name__
    rest = sorted(k for k in b if k not in ("messages", "input", "system", "instructions", "tools", "model", "stream"))
    print(f"{os.path.basename(f)} {r['method']} {r['path']} model={b.get('model')} stream={b.get('stream')} tools={len(tools)} kinds={kinds} "
          f"tools={size(tools)}ch system={size(b.get('system') or b.get('instructions'))}ch conversation={size(conv)}ch roles={roles}")
    print(f"   other keys: {rest}")
    print(f"   caps: max_tokens={b.get('max_tokens')} max_completion_tokens={b.get('max_completion_tokens')} max_output_tokens={b.get('max_output_tokens')}")
    print(f"   ua={h.get('user-agent','')[:70]} beta={h.get('anthropic-beta','')[:160]}")
