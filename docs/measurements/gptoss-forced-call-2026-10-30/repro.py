#!/usr/bin/env python3
"""Reproduce BIG1's empty forced tool call on gpt-oss (README.md, this directory).

Usage: repro.py BASE MODEL SRC_REPO OUT.jsonl CASES

CASES is a comma list of: short128, short24, long128, long1024. The prompts are BIG1's own
(docs/measurements/big01-moe-serve-2026-10-30/probe.py, imported from SRC_REPO, which is a checkout at 5df3b780 so the 12k
prompt is byte for byte the failing one). Every SSE frame is kept: reasoning, content, tool-call deltas, finish reasons,
usage and error frames.
"""
import importlib.util
import json
import sys
import time
import urllib.request

BASE, MODEL, SRC, OUT, CASES = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(",")
DEADLINE = 900

# probe.py reads its argv at import; give it one, then take SYSTEM, TOOLS and source_text from it unchanged.
argv = sys.argv
sys.argv = ["probe.py", BASE, MODEL, SRC, "/dev/null"]
spec = importlib.util.spec_from_file_location("probe", SRC + "/docs/measurements/big01-moe-serve-2026-10-30/probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
sys.argv = argv


def stream(body):
    body = dict(body, model=MODEL, stream=True, temperature=0, stream_options={"include_usage": True})
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    rec = {"frames": 0, "reasoning": "", "content": "", "calls": {}, "finish": [], "usage": None, "errors": [],
           "first_s": None, "other": []}
    with urllib.request.urlopen(req, timeout=DEADLINE) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            rec["frames"] += 1
            ev = json.loads(data)
            if "error" in ev:
                rec["errors"].append(ev["error"])
            if ev.get("usage"):
                rec["usage"] = ev["usage"]
            if not ev.get("choices") and "error" not in ev and not ev.get("usage"):
                rec["other"].append(ev)
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if ch.get("finish_reason"):
                    rec["finish"].append(ch["finish_reason"])
                txt = (d.get("reasoning_content") or d.get("reasoning") or "")
                rec["reasoning"] += txt
                rec["content"] += d.get("content") or ""
                for tc in d.get("tool_calls") or []:
                    c = rec["calls"].setdefault(str(tc.get("index", 0)), {"name": "", "arguments": ""})
                    f = tc.get("function") or {}
                    c["name"] += f.get("name") or ""
                    c["arguments"] += f.get("arguments") or ""
                if rec["first_s"] is None and (txt or d.get("content") or d.get("tool_calls")):
                    rec["first_s"] = time.monotonic() - t0
    rec["total_s"] = time.monotonic() - t0
    rec["calls"] = list(rec["calls"].values())
    return rec


SHORT = [{"role": "system", "content": probe.SYSTEM},
         {"role": "user", "content": "The build fails in cuda/pack_stream.go. Start by reading that file."}]
LONG = [{"role": "system", "content": probe.SYSTEM},
        {"role": "user", "content": "Here is part of the repository.\n\n" + probe.source_text(["decoder"], 39000)
         + "\n\nThe build fails in cuda/pack_stream.go. Read it."}]
PLAN = {"short128": (SHORT, 128), "short24": (SHORT, 24), "long128": (LONG, 128), "long1024": (LONG, 1024)}

with open(OUT, "a") as out:
    for case in CASES:
        msgs, mx = PLAN[case]
        rec = stream({"messages": msgs, "tools": probe.TOOLS, "tool_choice": "required", "max_tokens": mx})
        rec.update(case=case, max_tokens=mx, at=time.strftime("%F %T %Z"))
        out.write(json.dumps(rec) + "\n")
        out.flush()
        u = rec["usage"] or {}
        print(f"{case}: {rec['total_s']:.1f} s, first {rec['first_s']}, calls {rec['calls']}, finish {rec['finish']}, "
              f"usage prompt {u.get('prompt_tokens')} completion {u.get('completion_tokens')}, errors {rec['errors']}, "
              f"reasoning {len(rec['reasoning'])} chars, content {rec['content'][:160]!r}", flush=True)
