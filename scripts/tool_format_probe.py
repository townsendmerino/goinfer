#!/usr/bin/env python3
"""How a served model answers tool-using prompts: a call, a call in the wrong format (dropped as prose), or prose.

    python3 scripts/tool_format_probe.py http://127.0.0.1:18432 [--samples 3] [--max-tokens 900] [--thinking-budget 400] [--json out.json]

Eight prompts over a handful of tools, each asked --samples times (the first greedy, the rest sampled at T=0.7 with a fixed seed), through
/v1/chat/completions. Every reply is classified, never judged by eye:

  call         finish_reason "tool_calls" and a parsed call (checked below: right tool, arguments of the right types)
  xml_dropped  no parsed call, but the text holds <function=...> — Qwen3.5/Qwen3-Coder's own call format, unparsed
  json_dropped no parsed call, but the text holds <tool_call> (a Hermes-JSON call the parser could not read)
  prose        an answer in plain text (right for prompt 8, which needs no tool)
  empty        no content, no call (a reply cut off inside its reasoning)

With --loop, every reply that was an ok call is followed up: the call and a canned tool result are replayed and the model is asked again,
which is where the history format matters. The second reply is `answered` when it is prose that mentions the result, `call_again` when it
calls another tool, `ignored` when it is prose that does not use the result, and `dropped` when it holds an unparsed <function=/<tool_call>.

and a call is `ok` only when it names the expected tool and every argument has the expected JSON type — so a call with brightness "70"
instead of 70 is a call, not an ok call. Exit status is always 0: this reports, it does not gate.
"""
import argparse, json, sys, time, urllib.request, collections

TOOLS = {
 "get_weather": {"description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}},
 "get_time": {"description": "Get the current time in a timezone", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "IANA timezone, e.g. Asia/Tokyo"}}, "required": ["timezone"]}},
 "calculator": {"description": "Evaluate an arithmetic expression", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
 "search": {"description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "description": "How many results"}}, "required": ["query"]}},
 "read_file": {"description": "Read a file from disk", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
 "write_file": {"description": "Write text to a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
 "set_light": {"description": "Control a smart light", "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "on": {"type": "boolean"}, "brightness": {"type": "number", "description": "0 to 100"}}, "required": ["room", "on"]}},
}
# (id, user prompt, tools offered, expected tool or None, expected argument types)
PROMPTS = [
 ("weather",   "What is the weather in Paris right now?", ["get_weather", "get_time"], "get_weather", {"city": str}),
 ("calc",      "What is 1234 * 5678? Use the calculator.", ["calculator", "get_weather"], "calculator", {"expression": str}),
 ("search",    "Find recent news about the Go 1.24 release, 3 results please.", ["search", "get_weather"], "search", {"query": str, "max_results": int}),
 ("read",      "Show me the contents of /etc/hostname.", ["read_file", "write_file"], "read_file", {"path": str}),
 ("pick",      "What time is it in Tokyo?", ["get_weather", "get_time", "calculator"], "get_time", {"timezone": str}),
 ("multiline", "Create a file hello.py that prints \"hello\" and then prints the numbers 1 to 3, one per line.", ["write_file", "read_file"], "write_file", {"path": str, "content": str}),
 ("typed",     "Turn on the kitchen light at 70% brightness.", ["set_light", "get_weather"], "set_light", {"room": str, "on": bool, "brightness": (int, float)}),
 ("control",   "What is the capital of France?", ["get_weather", "calculator"], None, {}),
]

# What the tool "returns" in --loop mode, and a marker the final answer should carry if it used the result.
RESULTS = {
 "get_weather": ("It is 18 degrees Celsius and sunny in Paris.", ["18"]),
 "get_time": ("21:30 JST on 2026-10-01", ["21:30", "9:30"]),
 "calculator": ("7006652", ["7006652", "7,006,652"]),
 "search": ("1. Go 1.24 adds generic type aliases. 2. Go 1.24 speeds up maps.", ["alias", "map"]),
 "read_file": ("myhost.example.com", ["myhost"]),
 "write_file": ("ok: wrote 62 bytes to hello.py", ["62", "wrote", "hello.py", "created", "file"]),
 "set_light": ("light set: room=kitchen on=true brightness=70", ["kitchen", "70", "on"]),
}

def post(base, body):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))

def classify(msg, finish, want, types):
    calls = msg.get("tool_calls") or []
    text = (msg.get("content") or "")
    if calls:
        c = calls[0]["function"]
        try:
            args = json.loads(c["arguments"])
        except Exception:
            return "call", False, "arguments not JSON"
        if want is None:
            return "call", False, "called %s on a prompt that needs no tool" % c["name"]
        if c["name"] != want:
            return "call", False, "called %s, wanted %s" % (c["name"], want)
        for k, t in types.items():
            if k not in args:
                if k in ("max_results",):  # optional in the schema
                    continue
                return "call", False, "missing %s" % k
            v = args[k]
            if t is bool and not isinstance(v, bool): return "call", False, "%s=%r is not a boolean" % (k, v)
            if t is not bool and isinstance(v, bool): return "call", False, "%s=%r is a boolean" % (k, v)
            if not isinstance(v, t): return "call", False, "%s=%r is %s" % (k, v, type(v).__name__)
        return "call", True, ""
    if "<function=" in text: return "xml_dropped", False, ""
    if "<tool_call>" in text: return "json_dropped", False, ""
    if not text.strip(): return "empty", False, ""
    return "prose", want is None, ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=900); ap.add_argument("--thinking-budget", type=int, default=0)
    ap.add_argument("--json"); ap.add_argument("--model", default="m"); ap.add_argument("--loop", action="store_true")
    a = ap.parse_args()
    rows = []
    for pid, user, offered, want, types in PROMPTS:
        tools = [{"type": "function", "function": dict(name=n, **TOOLS[n])} for n in offered]
        for s in range(a.samples):
            body = {"model": a.model, "messages": [{"role": "user", "content": user}], "tools": tools, "max_tokens": a.max_tokens,
                    "temperature": 0 if s == 0 else 0.7, "seed": 1000 + s}
            if a.thinking_budget: body["thinking_token_budget"] = a.thinking_budget
            t = time.time()
            try:
                r = post(a.base, body); ch = r["choices"][0]
                kind, ok, why = classify(ch["message"], ch["finish_reason"], want, types)
                detail = (ch["message"].get("tool_calls") or [{}])[0].get("function") or (ch["message"].get("content") or "")[:120]
            except Exception as e:
                kind, ok, why, detail = "error", False, str(e)[:100], ""
            row = dict(prompt=pid, sample=s, kind=kind, ok=ok, why=why, detail=detail, secs=round(time.time() - t, 1))
            loop = ""
            if a.loop and kind == "call" and ok:
                try:
                    call = ch["message"]["tool_calls"][0]
                    result, markers = RESULTS[call["function"]["name"]]
                    asst = {"role": "assistant", "content": ch["message"].get("content") or "", "tool_calls": ch["message"]["tool_calls"]}
                    if ch["message"].get("reasoning_content"): asst["reasoning_content"] = ch["message"]["reasoning_content"]
                    body2 = dict(body, messages=body["messages"] + [asst, {"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"], "content": result}])
                    r2 = post(a.base, body2); m2 = r2["choices"][0]["message"]; t2 = (m2.get("content") or "")
                    if m2.get("tool_calls"): loop = "call_again"
                    elif "<function=" in t2 or "<tool_call>" in t2: loop = "dropped"
                    elif any(mk.lower() in t2.lower() for mk in markers): loop = "answered"
                    else: loop = "ignored"
                    row["loop"], row["loop_text"] = loop, t2[:120]
                except Exception as e:
                    loop = "error"; row["loop"] = loop; row["loop_text"] = str(e)[:100]
            rows.append(row)
            print("%-9s s%d  %-12s %-4s %-10s %s %s" % (pid, s, kind, "ok" if ok else "", loop, why, json.dumps(detail)[:100]), flush=True)
    kinds = collections.Counter(r["kind"] for r in rows)
    needs = [r for r in rows if r["prompt"] != "control"]
    print("\n== %d replies" % len(rows))
    print("   by kind:", dict(kinds))
    print("   tool prompts: %d/%d correct calls (%.0f%%); control prompt answered in prose: %d/%d" % (
        sum(r["ok"] for r in needs), len(needs), 100.0 * sum(r["ok"] for r in needs) / max(1, len(needs)),
        sum(r["ok"] for r in rows if r["prompt"] == "control"), sum(1 for r in rows if r["prompt"] == "control")))
    if a.loop:
        lk = collections.Counter(r["loop"] for r in rows if "loop" in r)
        n_ok = sum(1 for r in rows if r["prompt"] != "control" and r["ok"])
        print("   loops (after %d ok calls):" % n_ok, dict(lk), " answered: %d/%d" % (lk.get("answered", 0), n_ok))
    if a.json: json.dump(rows, open(a.json, "w"), indent=1)

if __name__ == "__main__":
    main()
