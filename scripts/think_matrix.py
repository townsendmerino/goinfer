#!/usr/bin/env python3
"""Client matrix for thinking on the wire (docs/tasks/task-qwen35-think-prompt-2026-09.md).

    python3 scripts/think_matrix.py --url http://127.0.0.1:8099 [--model NAME] [--max 200] [--json OUT]

Drives a RUNNING goinfer-serve with the requests each client family sends — OpenAI chat (plain, streaming, tools, with the
vLLM/llama.cpp kwargs), Anthropic Messages (with and without `thinking`, streaming and not), the Responses API, count_tokens —
across thinking unset / on / off, and asserts the INVARIANT on every cell, not just "a reply came back":

  * `content` never carries think markup (unless the request asked for reasoning_format none / deepseek-legacy);
  * an empty `content` is only ever legal when the reply was cut off (finish_reason length / max_tokens);
  * streaming and non-streaming agree byte for byte at temperature 0;
  * Anthropic thinking blocks appear iff the request asked for thinking, come first, and the stream's block indices, start/stop
    pairs and signature_delta are well-formed;
  * count_tokens equals the prompt the generation actually used;
  * the prompt half: usage.prompt_tokens moves by exactly the think-block token counts between modes.

A cell that passes only because the reply was empty is a FAIL (the same trap as a skipped test). Stdlib only, so it runs
unattended on the night queue. Exit status: 0 all pass, 1 a cell failed, 2 all that ran passed but some path was NOT EXERCISED. Progress is printed per cell with elapsed time.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

TAGS = ("<think>", "</think>")   # the family's think delimiters; --tags overrides (Gemma 4: "<|channel>,<channel|>")
Q = "What is 2 + 3? Answer in one short sentence."
TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
ATOOL = {"name": "get_weather", "description": "Weather for a city",
         "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}

results = []
t0 = time.time()
MAX_THINK = 1500
OFF_DELTA, ON_DELTA = 4, 2   # prompt tokens the closed block / open `<think>\n` add over the server default (--thinking asis)


def post(url, body, stream=False, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    # A 503 is the server's swap guard (another job is squeezing the box) or a halt: wait and retry instead of dying, but say so.
    for attempt in range(60):
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
            break
        except urllib.error.HTTPError as e:
            if e.code != 503 or attempt == 59:
                raise
            print(f"[{time.time() - t0:6.0f}s] 503 from the server (swap guard / halt?), retry {attempt + 1}/60 in 10s", flush=True)
            time.sleep(10)
    if not stream:
        return json.loads(r.read())
    events = []
    ev = None
    for raw in r:
        line = raw.decode().rstrip("\n")
        if line.startswith("event: "):
            ev = line[7:]
        elif line.startswith("data: "):
            d = line[6:]
            if d == "[DONE]":
                break
            events.append((ev, json.loads(d)))
            ev = None
    return events


UNEXERCISED = []


def cell(name, ok, why=""):
    results.append((name, ok, why))
    print(f"[{time.time() - t0:6.0f}s] {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n        -> {why}"), flush=True)


def unexercised(name, why):
    """A cell whose path the model did not take (e.g. it never finished thinking). NOT a pass: it is listed at the end and
    makes the exit status 2, so a matrix that proved less than it says is visible in the morning record."""
    UNEXERCISED.append((name, why))
    print(f"[{time.time() - t0:6.0f}s] NOT EXERCISED  {name}\n        -> {why}", flush=True)


def kw(think):
    return {} if think == "unset" else {"chat_template_kwargs": {"enable_thinking": think == "on"}}


def chat_nonstream(base, model, think, mx, extra=None):
    body = {"model": model, "messages": [{"role": "user", "content": Q}], "max_tokens": mx, "temperature": 0, **kw(think)}
    body.update(extra or {})
    return post(base + "/v1/chat/completions", body)


def chat_stream(base, model, think, mx, extra=None):
    body = {"model": model, "messages": [{"role": "user", "content": Q}], "max_tokens": mx, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, **kw(think)}
    body.update(extra or {})
    ev = post(base + "/v1/chat/completions", body, stream=True)
    content = reasoning = ""
    finish = usage = None
    order = []
    for _, d in ev:
        if d.get("usage"):
            usage = d["usage"]
        for c in d.get("choices", []):
            dl = c.get("delta", {})
            if dl.get("reasoning_content"):
                reasoning += dl["reasoning_content"]
                order.append("r")
            if dl.get("content"):
                content += dl["content"]
                order.append("c")
            if c.get("finish_reason"):
                finish = c["finish_reason"]
    return content, reasoning, finish, usage, order


def invariant(content, reasoning, finish, allow_tags=False):
    if not allow_tags and any(t in content for t in TAGS):
        return f"think markup leaked into content: {content[:120]!r}"
    if content.strip() == "" and finish not in ("length", "max_tokens"):
        return f"empty content on a reply that ended {finish!r} (reasoning={reasoning[:60]!r})"
    return ""


def run(base, model, mx):
    # ---- OpenAI chat, every thinking setting, streaming vs not ---------------------------------------------------
    ptoks = {}
    for think in ("unset", "off", "on"):
        r = chat_nonstream(base, model, think, mx)
        ch = r["choices"][0]
        msg = ch["message"]
        content, reasoning = msg["content"] or "", msg.get("reasoning_content", "")
        ptoks[think] = r["usage"]["prompt_tokens"]
        cell(f"chat/{think}/non-stream: invariant", not invariant(content, reasoning, ch["finish_reason"]),
             invariant(content, reasoning, ch["finish_reason"]))
        sc, sr, sf, su, order = chat_stream(base, model, think, mx)
        cell(f"chat/{think}/stream==non-stream", (sc, sr, sf) == (content, reasoning, ch["finish_reason"]),
             f"stream {(sc, sr, sf)!r} vs non-stream {(content, reasoning, ch['finish_reason'])!r}")
        import re
        cell(f"chat/{think}/stream: all reasoning deltas precede all content deltas", re.fullmatch(r"r*c*", "".join(order)) is not None,
             f"delta order {''.join(order)[:60]}")
        cell(f"chat/{think}/stream: usage matches", su is not None and su["prompt_tokens"] == r["usage"]["prompt_tokens"],
             f"stream usage {su} vs {r['usage']}")
        print(f"         ({think}: prompt_tokens={ptoks[think]}, content={content[:60]!r}, reasoning={reasoning[:40]!r})", flush=True)

    # ---- the prompt half, end to end: the think block's token counts --------------------------------------------
    cell(f"prompt: off adds exactly {OFF_DELTA} tokens over unset (the closed block)", ptoks["off"] - ptoks["unset"] == OFF_DELTA,
         f"unset={ptoks['unset']} off={ptoks['off']} on={ptoks['on']}")
    cell(f"prompt: on adds exactly {ON_DELTA} tokens over unset", ptoks["on"] - ptoks["unset"] == ON_DELTA,
         f"unset={ptoks['unset']} off={ptoks['off']} on={ptoks['on']}")
    print(f"         (prompt_tokens unset={ptoks['unset']} off={ptoks['off']} on={ptoks['on']})", flush=True)

    # ---- truncated inside the think block: no answer, reasoning only, finish length -----------------------------
    r = chat_nonstream(base, model, "on", 6)
    ch = r["choices"][0]
    msg = ch["message"]
    if msg.get("reasoning_content"):
        cell("chat/on/truncated in reasoning: empty content + reasoning + length",
             (msg["content"] or "") == "" and ch["finish_reason"] == "length",
             f"content={msg['content']!r} finish={ch['finish_reason']}")
    else:
        unexercised("chat/on/truncated in reasoning", "the model wrote no reasoning in 6 tokens, so the truncated-inside-the-block path did not run")

    # ---- the full path: a reply that thinks AND then answers (needs room: --max-think) -------------------------------
    # Sampled, not greedy: greedy thinking on a small model often loops forever (the Qwen card says so), which would leave this
    # path unexercised for a reason that has nothing to do with the server. A seed keeps it repeatable.
    r = chat_nonstream(base, model, "on", MAX_THINK, {"temperature": 0.6, "top_p": 0.95, "seed": 7})
    ch = r["choices"][0]
    msg = ch["message"]
    content, reasoning = msg["content"] or "", msg.get("reasoning_content", "")
    if ch["finish_reason"] == "length":
        unexercised(f"chat/on/completes within {MAX_THINK} tokens",
                    f"the model was still {'thinking' if not content else 'writing'} at the limit (reasoning {len(reasoning)} chars, content {len(content)}); the think-then-answer path did not run on this model")
    elif not reasoning:
        unexercised(f"chat/on/completes within {MAX_THINK} tokens",
                    "thinking was requested but the model wrote no reasoning (finish stop); nothing to separate on this model/prompt")
    else:
        cell(f"chat/on/completes within {MAX_THINK} tokens: reasoning, then a clean non-empty answer, finish stop",
             content.strip() != "" and not any(t in content for t in TAGS),
             f"finish={ch['finish_reason']} reasoning={len(reasoning)}ch content={content[:80]!r}")
        print(f"         (answer after thinking: {content[:80]!r}; reasoning {len(reasoning)} chars)", flush=True)

    # ---- reasoning_format ---------------------------------------------------------------------------------------
    r = chat_nonstream(base, model, "on", mx, {"reasoning_format": "none"})
    c = r["choices"][0]["message"]["content"] or ""
    cell("chat/on/reasoning_format none: no reasoning_content field", "reasoning_content" not in r["choices"][0]["message"],
         str(r["choices"][0]["message"].keys()))
    r = chat_nonstream(base, model, "on", mx, {"reasoning_format": "deepseek-legacy"})
    m = r["choices"][0]["message"]
    cell("chat/on/reasoning_format legacy: reasoning_content filled AND tags kept when the model thought",
         (not m.get("reasoning_content")) or (TAGS[1] in (m["content"] or "") or r["choices"][0]["finish_reason"] == "length"),
         f"content={m['content'][:80]!r}")
    try:
        chat_nonstream(base, model, "unset", 8, {"reasoning_format": "bogus"})
        cell("chat: bad reasoning_format is a 400", False, "accepted")
    except urllib.error.HTTPError as e:
        cell("chat: bad reasoning_format is a 400", e.code == 400, str(e.code))

    # ---- reasoning_effort (OpenAI's spelling): none turns thinking off, anything else leaves the default alone ----
    r = chat_nonstream(base, model, "unset", mx, {"reasoning_effort": "none"})
    cell("chat/reasoning_effort none: no reasoning", not r["choices"][0]["message"].get("reasoning_content"),
         str(r["choices"][0]["message"].get("reasoning_content"))[:80])
    r = chat_nonstream(base, model, "unset", mx, {"reasoning_effort": "high"})
    cell("chat/reasoning_effort high: prompt unchanged vs unset (a bare effort must not flip a client)",
         r["usage"]["prompt_tokens"] == ptoks["unset"], f"{r['usage']['prompt_tokens']} vs {ptoks['unset']}")

    # ---- tools ----------------------------------------------------------------------------------------------------
    for think in ("unset", "on", "off"):
        body = {"model": model, "messages": [{"role": "user", "content": "What's the weather in Paris?"}], "tools": [TOOL],
                "max_tokens": mx, "temperature": 0, **kw(think)}
        r = post(base + "/v1/chat/completions", body)
        ch = r["choices"][0]
        msg = ch["message"]
        why = invariant(msg.get("content") or "", msg.get("reasoning_content", ""), ch["finish_reason"]) if not msg.get("tool_calls") else ""
        if msg.get("tool_calls"):
            why = why or ("think markup in content alongside a tool call" if any(t in (msg.get("content") or "") for t in TAGS) else "")
        cell(f"chat/tools/{think}: invariant (calls={len(msg.get('tool_calls') or [])})", not why, why)

    # ---- Anthropic Messages ---------------------------------------------------------------------------------------
    def anth(thinking, stream=False, tools=False):
        body = {"model": model, "max_tokens": mx, "temperature": 0, "messages": [{"role": "user", "content": Q}]}
        if thinking:
            body["thinking"] = {"type": thinking, "budget_tokens": 1024} if thinking == "enabled" else {"type": thinking}
        if tools:
            body["tools"] = [ATOOL]
            body["messages"] = [{"role": "user", "content": "What's the weather in Paris?"}]
        if stream:
            body["stream"] = True
            return post(base + "/v1/messages", body, stream=True)
        return post(base + "/v1/messages", body)

    for thinking in (None, "enabled", "disabled"):
        r = anth(thinking)
        types = [b["type"] for b in r["content"]]
        texts = "".join(b.get("text", "") for b in r["content"] if b["type"] == "text")
        ok = True
        why = ""
        if thinking != "enabled" and "thinking" in types:
            ok, why = False, f"thinking block without a request for one: {types}"
        if "thinking" in types and types[0] != "thinking":
            ok, why = False, f"thinking block not first: {types}"
        if any(t in texts for t in TAGS):
            ok, why = False, f"think markup in text block: {texts[:100]!r}"
        if texts.strip() == "" and r["stop_reason"] != "max_tokens" and "tool_use" not in types:
            ok, why = False, f"empty text on stop_reason {r['stop_reason']}"
        cell(f"anthropic/{thinking}/non-stream: blocks {types}", ok, why)

        ev = anth(thinking, stream=True)
        order, starts, stops, idxs = [], {}, set(), []
        bad = ""
        sig_before_stop = {}
        for name, d in ev:
            if name == "content_block_start":
                starts[d["index"]] = d["content_block"]["type"]
                idxs.append(d["index"])
            elif name == "content_block_delta":
                if d["index"] not in starts or d["index"] in stops:
                    bad = f"delta for block {d['index']} that is not open"
                if d["delta"]["type"] == "signature_delta":
                    sig_before_stop[d["index"]] = True
            elif name == "content_block_stop":
                stops.add(d["index"])
                if starts.get(d["index"]) == "thinking" and not sig_before_stop.get(d["index"]):
                    bad = "thinking block closed without a signature_delta"
        if idxs != list(range(len(idxs))):
            bad = bad or f"block indices not 0..n-1 in order: {idxs}"
        if set(starts) != stops:
            bad = bad or f"unclosed/unopened blocks: started {sorted(starts)} stopped {sorted(stops)}"
        kinds = [starts[i] for i in sorted(starts)]
        if thinking != "enabled" and "thinking" in kinds:
            bad = bad or "streamed a thinking block nobody asked for"
        if "thinking" in kinds and kinds[0] != "thinking":
            bad = bad or f"thinking not first: {kinds}"
        stext = "".join(d["delta"].get("text", "") for n, d in ev if n == "content_block_delta" and d["delta"]["type"] == "text_delta")
        if any(t in stext for t in TAGS):
            bad = bad or f"think markup in streamed text: {stext[:100]!r}"
        if stext != texts:
            bad = bad or f"stream text {stext[:60]!r} != non-stream {texts[:60]!r}"
        cell(f"anthropic/{thinking}/stream: well-formed, kinds {kinds}", not bad, bad)

    r = anth("enabled", tools=True)
    types = [b["type"] for b in r["content"]]
    cell(f"anthropic/enabled/tools non-stream: blocks {types}", "thinking" not in types[1:] and not any(
        t in "".join(b.get("text", "") for b in r["content"] if b["type"] == "text") for t in TAGS), str(types))
    ev = anth("enabled", stream=True, tools=True)
    starts = [(d["index"], d["content_block"]["type"]) for n, d in ev if n == "content_block_start"]
    cell(f"anthropic/enabled/tools stream: indices {starts}", [i for i, _ in starts] == list(range(len(starts))), str(starts))

    # count_tokens must render exactly as generation does
    for thinking in (None, "enabled", "disabled"):
        body = {"model": model, "max_tokens": mx, "messages": [{"role": "user", "content": Q}]}
        if thinking:
            body["thinking"] = {"type": thinking, "budget_tokens": 1024} if thinking == "enabled" else {"type": thinking}
        ct = post(base + "/v1/messages/count_tokens", body)["input_tokens"]
        gen = post(base + "/v1/messages", {**body, "max_tokens": 4, "temperature": 0})["usage"]["input_tokens"]
        cell(f"anthropic/{thinking}: count_tokens == generation input_tokens ({ct})", ct == gen, f"{ct} vs {gen}")

    # ---- Responses API --------------------------------------------------------------------------------------------
    for eff in (None, "none", "high"):
        body = {"model": model, "input": Q, "max_output_tokens": mx, "temperature": 0}
        if eff:
            body["reasoning"] = {"effort": eff}
        r = post(base + "/v1/responses", body)
        txt = "".join(c.get("text", "") for o in r.get("output", []) if o.get("type") == "message" for c in o.get("content", []))
        bad = any(t in txt for t in TAGS)
        cell(f"responses/effort={eff}: output_text clean", not bad and (txt.strip() != "" or r.get("status") == "incomplete"),
             f"text={txt[:100]!r} status={r.get('status')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--model", default="")
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--json", default="")
    ap.add_argument("--max-think", type=int, default=1500, help="max_tokens for the cell that must think AND finish with an answer")
    ap.add_argument("--tags", default="<think>,</think>", help="comma-separated think delimiters that must never reach content")
    ap.add_argument("--off-delta", type=int, default=4, help="prompt tokens thinking=off adds over unset (closed block: 4)")
    ap.add_argument("--on-delta", type=int, default=2, help="prompt tokens thinking=on adds over unset (open <think>\\n: 2; Qwen3 writes nothing: 0)")
    a = ap.parse_args()
    global OFF_DELTA, ON_DELTA, MAX_THINK, TAGS
    TAGS = tuple(a.tags.split(","))
    OFF_DELTA, ON_DELTA, MAX_THINK = a.off_delta, a.on_delta, a.max_think
    model = a.model
    if not model:
        model = json.load(urllib.request.urlopen(a.url + "/v1/models"))["data"][0]["id"]
    print(f"think_matrix: {a.url} model={model} max_tokens={a.max}", flush=True)
    run(a.url, model, a.max)
    bad = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(bad)}/{len(results)} cells pass in {time.time() - t0:.0f}s", flush=True)
    if UNEXERCISED:
        print(f"{len(UNEXERCISED)} cell(s) NOT EXERCISED (a path the model did not take — not a pass):", flush=True)
        for n, w in UNEXERCISED:
            print(f"  - {n}: {w}", flush=True)
    if a.json:
        json.dump([{"cell": n, "pass": ok, "why": w} for n, ok, w in results] +
                  [{"cell": n, "pass": None, "why": w} for n, w in UNEXERCISED], open(a.json, "w"), indent=1)
    sys.exit(1 if bad else (2 if UNEXERCISED else 0))


if __name__ == "__main__":
    main()
