#!/usr/bin/env python3
"""turn8_probe.py: replays the 12-tool transcript to one turn and prints what that turn writes, with a large token budget.
  turn8_probe.py <serve binary> <backend> <model> <port> <turn> <max_tokens>"""
import json, os, signal, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toolcall_smoke as ts

serve, backend, model, port, upto, budget = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
fixture = json.load(open(sys.argv[7]))
log = open(f"{os.path.dirname(os.path.abspath(__file__))}/turn8-{backend}.serve.log", "ab")
proc = subprocess.Popen([serve, "-model", f"smoke={model}", "-backend", backend, "-addr", f"127.0.0.1:{port}"],
                        stdout=log, stderr=log, preexec_fn=os.setsid)
try:
    assert ts.wait_port(port, proc), "server did not come up"
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    msgs = []
    for turn in fixture["turns"]:
        if turn["user"] is not None:
            msgs.append({"role": "user", "content": turn["user"]})
        last = turn["n"] == upto
        t0 = time.time()
        st, body, dt = ts.post(url, {"model": "smoke", "stream": False, "messages": msgs, "tools": fixture["tools"],
                                     "temperature": 0, "max_tokens": budget if last else ts.REPLAY_MAX_TOKENS,
                                     "tool_choice": {"type": "function", "function": {"name": turn["tool_choice"]}}})
        ch = body["choices"][0]; m = ch["message"]; calls = m.get("tool_calls") or []
        used = (body.get("usage") or {}).get("completion_tokens")
        print(f"turn {turn['n']} ({turn['tool_choice']}): finish {ch.get('finish_reason')}, {used} tokens, {dt:.1f}s, calls {len(calls)}", flush=True)
        if last:
            text = calls[0]["function"]["arguments"] if calls else (m.get("content") or "")
            print(f"--- {len(text)} chars; head:\n{text[:700]}\n--- tail:\n{text[-900:]}")
            # how repetitive is the tail: the share of the last 2000 chars covered by its most common 60-char window
            tail = text[-2000:]
            seen = {}
            for i in range(0, max(0, len(tail) - 60)):
                seen[tail[i:i + 60]] = seen.get(tail[i:i + 60], 0) + 1
            if seen:
                w, c = max(seen.items(), key=lambda kv: kv[1])
                print(f"--- most repeated 60-char window in the last 2000 chars: {c} times: {w!r}")
            break
        c = calls[0]
        msgs += [m, {"role": "tool", "name": c["function"]["name"], "tool_call_id": c.get("id"), "content": turn["tool_result"]}]
finally:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM); proc.wait(timeout=60)
    except Exception:
        proc.kill()
