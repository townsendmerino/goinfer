#!/usr/bin/env python3
"""A short serve check of one loaded goinfer-serve model: a forced tool call, decode rate, cold first-token
time at about 4k and 12k prompt tokens, and an append-only follow-up to the 12k conversation.

Usage: probe.py <base-url> <model-name> <source-root> <out.jsonl>

Every request streams with temperature 0 and has its own deadline. The long prompts are Go source from
<source-root>, a different set of files for each, so neither shares a prefix with an earlier request beyond
the chat template's opening tokens. Prints one line per step and writes one JSON object per step. Exits 1
when a forced call returns no call, or any call that is not a listed tool with arguments that parse as a JSON
object; every step still runs, and the failures are listed at the end.
"""
import json
import pathlib
import sys
import time
import urllib.request

BASE, MODEL, ROOT, OUT = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4]
DEADLINE = 900  # seconds per request

SYSTEM = "You are a coding agent working in a Go repository. Use the tools to read and change files."

# A coding agent's tool list: the shape a harness sends (several tools, nested JSON-schema parameters).
TOOLS = [
    {"type": "function", "function": {"name": n, "description": d, "parameters": {
        "type": "object", "properties": p, "required": r, "additionalProperties": False}}}
    for n, d, p, r in [
        ("read_file", "Read a file from the repository.",
         {"path": {"type": "string", "description": "repo-relative path"},
          "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["path"]),
        ("write_file", "Create or overwrite a file.",
         {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
        ("edit_file", "Replace one exact string in a file.",
         {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
         ["path", "old_string", "new_string"]),
        ("run_command", "Run a shell command in the repository root and return its output.",
         {"command": {"type": "string"}, "timeout_seconds": {"type": "integer"}}, ["command"]),
        ("list_dir", "List a directory.", {"path": {"type": "string"}}, ["path"]),
        ("search", "Search file contents with a regular expression.",
         {"pattern": {"type": "string"}, "glob": {"type": "string"}}, ["pattern"]),
        ("finish", "Report that the task is done.", {"summary": {"type": "string"}}, ["summary"]),
    ]
]


# The stream's error events and finish reasons for the request in flight; record() files them with its step.
STREAM_NOTES = []


def stream(body):
    """POST a streaming chat completion; return (ttft_s, total_s, text, tool_calls, usage, times), times being when
    each non-empty delta arrived, in seconds from the request."""
    body = dict(body, model=MODEL, stream=True, temperature=0, stream_options={"include_usage": True})
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    STREAM_NOTES.clear()
    t0 = time.monotonic()
    ttft, text, calls, usage, times = None, [], {}, None, []
    with urllib.request.urlopen(req, timeout=DEADLINE) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("error"):
                STREAM_NOTES.append({"error": ev["error"]})
            for ch in ev.get("choices") or []:
                if ch.get("finish_reason"):
                    STREAM_NOTES.append({"finish_reason": ch["finish_reason"]})
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                got = False
                if d.get("content"):
                    text.append(d["content"])
                    got = True
                if d.get("reasoning_content") or d.get("reasoning"):
                    got = True  # a reasoning model's first token is its first reasoning token
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    c["id"] = tc.get("id") or c["id"]
                    f = tc.get("function") or {}
                    c["name"] += f.get("name") or ""
                    c["arguments"] += f.get("arguments") or ""
                    got = True
                if got:
                    now = time.monotonic() - t0
                    times.append(now)
                    if ttft is None:
                        ttft = now
    return ttft, time.monotonic() - t0, "".join(text), list(calls.values()), usage, times


def source_text(dirs, target_chars):
    parts, n = [], 0
    for d in dirs:
        for p in sorted((ROOT / d).glob("*.go")):
            if p.name.endswith("_test.go"):
                continue
            s = f"// file: {p.relative_to(ROOT)}\n{p.read_text()}\n"
            parts.append(s)
            n += len(s)
            if n >= target_chars:
                return "".join(parts)[:target_chars]
    return "".join(parts)


FAILURES = []
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


def call_ok(c):
    """Whether one returned call names a listed tool and carries arguments that parse as a JSON object."""
    try:
        args = json.loads(c["arguments"])
    except ValueError:
        return False
    return c["name"] in TOOL_NAMES and isinstance(args, dict)


def forced_ok(step, calls):
    """Records a failure unless a forced call returned at least one call and every call is valid."""
    if not calls:
        FAILURES.append(f"{step}: no tool call returned")
        return False
    bad = [c for c in calls if not call_ok(c)]
    if bad:
        FAILURES.append(f"{step}: invalid call(s) {bad}")
        return False
    return True


def record(out, step, **kw):
    kw["step"] = step
    kw["stream_notes"] = list(STREAM_NOTES)
    out.write(json.dumps(kw) + "\n")
    out.flush()
    print(step, json.dumps({k: v for k, v in kw.items() if k not in ("text", "messages")})[:600], flush=True)


def main():
    out = open(OUT, "a")
    models = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=30))
    record(out, "models", models=models)

    # 1. Forced tool call: tool_choice "required" (goinfer constrains the call form with its grammar).
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "The build fails in cuda/pack_stream.go. Start by reading that file."}]
    ttft, tot, text, calls, usage, _ = stream({"messages": msgs, "tools": TOOLS, "tool_choice": "required",
                                               "max_tokens": 128})
    record(out, "forced_tool_call", ttft_s=ttft, total_s=tot, tool_calls=calls, text=text, usage=usage,
           calls_ok=forced_ok("forced_tool_call", calls))

    # 2. Decode rate: a short prompt, 256 tokens out.
    msgs = [{"role": "user", "content": "Explain in detail, step by step, how a mixture-of-experts transformer "
                                         "layer routes each token to its experts and combines their outputs."}]
    ttft, tot, text, _, usage, times = stream({"messages": msgs, "max_tokens": 256})
    n = (usage or {}).get("completion_tokens") or len(times)
    # Deltas after the first over the time they took: usage counts the stop token too, which arrives with no text.
    rate = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else None
    record(out, "decode", ttft_s=ttft, total_s=tot, completion_tokens=n, decode_tok_s=rate, usage=usage,
           text=text)

    # 3. Cold first token at ~4k and ~12k prompt tokens (Go source at ~3.2 chars per token).
    long_msgs = {}
    for name, dirs, chars in [("cold_4k", ["internal/serveapp"], 13000),
                              ("cold_12k", ["cuda", "cuda/internal/gpudev"], 39000)]:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Here is part of the repository.\n\n" + source_text(dirs, chars)
                 + "\n\nIn two sentences, what does this code do?"}]
        ttft, tot, text, _, usage, _ = stream({"messages": msgs, "max_tokens": 48})
        record(out, name, ttft_s=ttft, total_s=tot, prompt_tokens=(usage or {}).get("prompt_tokens"),
               usage=usage, text=text)
        long_msgs[name] = (msgs, text)

    # 4. Append-only follow-up to the 12k conversation (the most recent one, so its KV is the live GPU cache).
    msgs, reply = long_msgs["cold_12k"]
    msgs = msgs + [{"role": "assistant", "content": reply},
                   {"role": "user", "content": "Which function in that code allocates pinned host memory?"}]
    ttft, tot, text, _, usage, _ = stream({"messages": msgs, "max_tokens": 48})
    record(out, "append_12k", ttft_s=ttft, total_s=tot, prompt_tokens=(usage or {}).get("prompt_tokens"),
           usage=usage, text=text)

    # 5. A harness's turn pair at ~12k: a cold forced call over a 12k context (other files again, so cold), then
    # the tool result appended and the next forced call, timed to its first token.
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Here is part of the repository.\n\n"
             + source_text(["decoder"], 39000) + "\n\nThe build fails in cuda/pack_stream.go. Read it."}]
    ttft, tot, text, calls, usage, _ = stream({"messages": msgs, "tools": TOOLS, "tool_choice": "required",
                                               "max_tokens": 128})
    record(out, "harness_cold_12k_tool_call", ttft_s=ttft, total_s=tot,
           prompt_tokens=(usage or {}).get("prompt_tokens"), tool_calls=calls, usage=usage, text=text,
           calls_ok=forced_ok("harness_cold_12k_tool_call", calls))
    if not calls:
        FAILURES.append("harness_append_tool_call: not run (the cold call returned none)")
    else:
        c = calls[0]
        cid = c["id"] or "call_0"
        result = (ROOT / "cuda/pack_stream.go").read_text()[:1200]
        msgs = msgs + [{"role": "assistant", "content": text or None, "tool_calls": [
                           {"id": cid, "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}]},
                       {"role": "tool", "tool_call_id": cid, "content": result}]
        ttft, tot, text, calls, usage, _ = stream({"messages": msgs, "tools": TOOLS, "tool_choice": "required",
                                                   "max_tokens": 128})
        record(out, "harness_append_tool_call", ttft_s=ttft, total_s=tot,
               prompt_tokens=(usage or {}).get("prompt_tokens"), tool_calls=calls, usage=usage, text=text,
               calls_ok=forced_ok("harness_append_tool_call", calls))

if __name__ == "__main__":
    main()
    if FAILURES:
        print("FAILED: " + "; ".join(FAILURES), flush=True)
        sys.exit(1)
