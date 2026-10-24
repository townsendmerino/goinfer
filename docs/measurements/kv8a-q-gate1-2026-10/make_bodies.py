#!/usr/bin/env python3
"""KV8a-Q gate 1 fixture: one conversation as the /v1/chat/completions requests a client sends, three questions, each answered
after a tool loop.

  r1  user Q1
  r2  ... + assistant (call get_weather Paris) + tool result
  r3  ... + assistant (call get_weather Lyon)  + tool result      <- Q1's last tool-loop request
  r4  ... + assistant (final answer) + user Q2                    <- the second question
  r5  ... + assistant (call get_wind Lyon) + tool result          <- Q2's tool-loop request
  r6  ... + assistant (final answer) + user Q3                    <- the third question

MODE on sets chat_template_kwargs.enable_thinking true and sends each assistant turn's reasoning back (reasoning_content), as
Codex, Claude Code and Flint's card do. MODE off sets it false and sends no reasoning (a model with thinking off returns none).
MODE default sets neither: the template's own default, and no reasoning.

Writes OUTDIR/rN.json in the shape TestKV0_tokenizeBodies reads ({"path", "body"}). Usage: make_bodies.py OUTDIR MODE
"""
import json, os, sys

out, mode = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)


def tool(name, desc):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


# Two tools: with one tool under auto, a request ending in a user message forces that call (forcedTool), which renders thinking
# off; agent clients send many tools.
tools = [tool("get_weather", "Current weather for a city."), tool("get_wind", "Current wind speed for a city.")]


def asst(content, reasoning, call=None):
    m = {"role": "assistant", "content": content}
    if mode == "on":
        m["reasoning_content"] = reasoning
    if call:
        cid, name, city = call
        m["tool_calls"] = [{"id": cid, "type": "function",
                            "function": {"name": name, "arguments": json.dumps({"city": city})}}]
    return m


def result(cid, name, payload):
    return {"role": "tool", "tool_call_id": cid, "name": name, "content": json.dumps(payload)}


sysmsg = {"role": "system", "content": "You are a helpful assistant."}
q1 = {"role": "user", "content": "Is it warmer in Paris or in Lyon right now?"}
a1 = asst("", "I need both temperatures. Start with Paris.", ("call_1", "get_weather", "Paris"))
t1 = result("call_1", "get_weather", {"city": "Paris", "temp_c": 14})
a2 = asst("", "Paris is 14 C. Now Lyon.", ("call_2", "get_weather", "Lyon"))
t2 = result("call_2", "get_weather", {"city": "Lyon", "temp_c": 17})
a3 = asst("Lyon is warmer: 17 C against 14 C in Paris.", "Lyon 17 C beats Paris 14 C.")
q2 = {"role": "user", "content": "Is it windy in Lyon?"}
a4 = asst("", "Check the wind in Lyon.", ("call_3", "get_wind", "Lyon"))
t3 = result("call_3", "get_wind", {"city": "Lyon", "wind_kmh": 9})
a5 = asst("Not much: 9 km/h in Lyon.", "9 km/h is light.")
q3 = {"role": "user", "content": "Thanks. Should I take a jacket?"}

convo = [sysmsg, q1, a1, t1, a2, t2, a3, q2, a4, t3, a5, q3]
ends = {"r1": 2, "r2": 4, "r3": 6, "r4": 8, "r5": 10, "r6": 12}
for name, n in ends.items():
    body = {"model": "qwen3.5", "messages": convo[:n], "tools": tools, "stream": False}
    if mode != "default":
        body["chat_template_kwargs"] = {"enable_thinking": mode == "on"}
    with open(os.path.join(out, name + ".json"), "w") as f:
        json.dump({"path": "/v1/chat/completions", "body": json.dumps(body)}, f, indent=1)
        f.write("\n")
