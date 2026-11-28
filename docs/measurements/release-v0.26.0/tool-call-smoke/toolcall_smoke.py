#!/usr/bin/env python3
"""toolcall_smoke.py: a release smoke for tool calls, after tool parameters began to enforce `$ref`, `type` lists,
`anyOf` and `oneOf` (a tool's parameters compile leniently: what the compiler cannot enforce is any JSON value).

It starts one `serve` on one model and sends four groups of requests, then checks every reply from the HTTP body alone:

  replay    the 12-tool agent transcript (scripts/w4_transcript_base.json), 10 turns, a named tool_choice per turn,
            temperature 0. These tools use only `type`, `properties`, `required` and `items`: the control.
  keywords  five tools whose parameters use the newly enforced keywords (a `$ref` into `$defs`, a nullable `anyOf`,
            a `type` list, a `oneOf` discriminated by a `const`, a recursive `$ref`), two prompts each, named
            tool_choice, temperature 0.
  sampled   the keywords group again at temperature 0.8, three seeds each.
  auto      all 17 tools offered with tool_choice "auto", one prompt per keyword tool, temperature 0.
  required  all 17 tools offered with tool_choice "required" (the model picks the tool, the grammar is the union of every
            tool's), the keyword tools' prompts, temperature 0. The call must satisfy the schema of the tool it names.
  auto5     the five keyword tools alone with tool_choice "auto", their prompts, temperature 0.

A reply PASSES when the status is 200, finish_reason is "tool_calls", the call names the chosen tool (named groups),
its arguments parse as JSON and they satisfy the tool's schema under the validator below (the keywords these tools
use, and no others). Under "auto" (the auto and auto5 groups) a prose answer is legal and is counted apart. A reply the request's token budget
cut (finish_reason "length") is counted apart as truncated: it is not a pass, and it says nothing about the schema.
The exit code is 0 when every named request passed and no "auto" reply carried a call that failed the check; the JSON
written to <out> has every row.

  toolcall_smoke.py <serve binary> <backend> <model file> <out.json> [--quant Q] [--port N] [--transcript FILE]
                    [--groups replay,keywords,sampled,auto,required,auto5]

Not a timing: the wall time per request is recorded as context only.
"""
import argparse, json, os, signal, socket, subprocess, sys, time, urllib.error, urllib.request

# Token budgets per request. A replay turn can write a whole source file as one argument (write_file), so its budget is
# the larger one; at 256 that turn was cut on the CPU and the replay could not go on.
REPLAY_MAX_TOKENS = 1024
KEYWORD_MAX_TOKENS = 400


def validate(schema, value, root, path="$"):
    """Errors of value against schema, for the keywords the smoke's tools use. An unknown keyword is an error, so a
    tool added here with a keyword this does not read cannot pass by being unchecked."""
    known = {"type", "properties", "required", "items", "enum", "const", "anyOf", "oneOf", "$ref", "$defs",
             "additionalProperties", "description"}
    errs = []
    extra = set(schema) - known
    if extra:
        return [f"{path}: the validator does not read {sorted(extra)}"]
    if "$ref" in schema:
        ref = schema["$ref"]
        target = root
        if ref != "#":
            for part in ref[2:].split("/"):
                target = target[part]
        return validate(target, value, root, path)
    if "const" in schema and value != schema["const"]:
        errs.append(f"{path}: {value!r} is not the const {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} is not in the enum")
    for kw in ("anyOf", "oneOf"):
        if kw in schema:
            ok = [i for i, alt in enumerate(schema[kw]) if not validate(alt, value, root, path)]
            if not ok:
                errs.append(f"{path}: matches no {kw} alternative")
            elif kw == "oneOf" and len(ok) > 1:
                errs.append(f"{path}: matches {len(ok)} oneOf alternatives")
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]

        def is_type(name):
            if name == "null":
                return value is None
            if name == "boolean":
                return isinstance(value, bool)
            if name == "integer":
                return isinstance(value, int) and not isinstance(value, bool)
            if name == "number":
                return isinstance(value, (int, float)) and not isinstance(value, bool)
            return isinstance(value, {"string": str, "array": list, "object": dict}[name])

        if not any(is_type(n) for n in types):
            return errs + [f"{path}: {type(value).__name__} is not {t}"]
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for r in schema.get("required", []):
            if r not in value:
                errs.append(f"{path}: missing required {r!r}")
        for k, v in value.items():
            if k in props:
                errs += validate(props[k], v, root, f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}: {k!r} is not a declared property")
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            errs += validate(schema["items"], v, root, f"{path}[{i}]")
    return errs


def fn(name, description, parameters):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


KEYWORD_TOOLS = [
    fn("create_event", "Create a calendar event.", {
        "type": "object",
        "$defs": {"Attendee": {"type": "object", "properties": {
            "name": {"type": "string"}, "email": {"type": "string"},
            "role": {"type": "string", "enum": ["required", "optional"]}}, "required": ["name", "role"]}},
        "properties": {
            "title": {"type": "string"},
            "start": {"type": "string", "description": "ISO 8601 date and time"},
            "duration_minutes": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "location": {"type": ["string", "null"]},
            "attendees": {"type": "array", "items": {"$ref": "#/$defs/Attendee"}}},
        "required": ["title", "start", "duration_minutes", "location", "attendees"]}),
    fn("apply_change", "Apply one change to a file in the repository.", {
        "type": "object",
        "properties": {"change": {"oneOf": [
            {"type": "object", "properties": {"kind": {"const": "rename"}, "path": {"type": "string"},
                                              "new_path": {"type": "string"}},
             "required": ["kind", "path", "new_path"]},
            {"type": "object", "properties": {"kind": {"const": "delete"}, "path": {"type": "string"}},
             "required": ["kind", "path"]},
            {"type": "object", "properties": {"kind": {"const": "write"}, "path": {"type": "string"},
                                              "content": {"type": "string"}},
             "required": ["kind", "path", "content"]}]}},
        "required": ["change"]}),
    fn("save_outline", "Save a nested outline of a document.", {
        "type": "object",
        "$defs": {"Node": {"type": "object", "properties": {
            "title": {"type": "string"},
            "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}}, "required": ["title", "children"]}},
        "properties": {"root": {"$ref": "#/$defs/Node"}},
        "required": ["root"]}),
    fn("search_orders", "Search customer orders.", {
        "type": "object",
        "$defs": {"DateRange": {"type": "object", "properties": {
            "from": {"type": "string"}, "to": {"type": "string"}}, "required": ["from", "to"]}},
        "properties": {
            "customer": {"type": "string"},
            "placed": {"anyOf": [{"$ref": "#/$defs/DateRange"}, {"type": "null"}]},
            "status": {"type": ["string", "null"], "enum": ["open", "shipped", "cancelled", None]},
            "limit": {"type": ["integer", "null"]}},
        "required": ["customer", "placed", "status", "limit"]}),
    fn("set_threshold", "Set an alert threshold, as a number or as the name of a preset.", {
        "type": "object",
        "properties": {
            "metric": {"type": "string"},
            "threshold": {"anyOf": [{"type": "number"}, {"type": "string", "enum": ["low", "medium", "high"]}]},
            "notify": {"type": "array", "items": {"type": ["string", "null"]}}},
        "required": ["metric", "threshold", "notify"]}),
]

KEYWORD_PROMPTS = {
    "create_event": [
        "Put a design review on the calendar for 2026-12-02 at 14:00, 45 minutes, in room Cedar. Ana Ruiz "
        "(ana@example.com) must attend and Bo Chen is optional.",
        "Schedule 'Quarterly planning' for 2026-12-10T09:00. I don't know how long it will run or where yet. "
        "Just me for now: Dee Park, required."],
    "apply_change": [
        "Rename docs/setup.md to docs/install.md.",
        "Delete the file tmp/scratch.txt, we no longer need it."],
    "save_outline": [
        "Save this outline: 'Guide' has two sections, 'Install' and 'Usage'. 'Usage' has the subsections 'CLI' "
        "and 'API'.",
        "Save an outline with a single top-level item called 'Notes' and nothing under it."],
    "search_orders": [
        "Find up to 5 shipped orders for customer Acme placed between 2026-10-01 and 2026-10-31.",
        "List every order for the customer Globex, whatever its status or date, with no limit."],
    "set_threshold": [
        "Alert on cpu_load above 0.85 and notify ops@example.com.",
        "Set the error_rate alert to the 'high' preset. Nobody needs to be notified."],
}


def wait_port(port, proc, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), 1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, {"error_body": e.read().decode("utf-8", "replace")[:600]}, time.time() - t0


def check(status, body, schemas, named=None):
    """One row: what came back and whether it passes. named is the tool a named tool_choice asked for."""
    row = {"status": status, "named": named}
    if status != 200:
        row.update(cls="http_error", ok=False, detail=body.get("error_body", "")[:300])
        return row
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    calls = msg.get("tool_calls") or []
    row["finish"] = ch.get("finish_reason")
    row["completion_tokens"] = (body.get("usage") or {}).get("completion_tokens")
    if row["finish"] == "length":
        row.update(cls="truncated", ok=False, content_head=(msg.get("content") or "")[:200])
        return row
    if not calls:
        row.update(cls="prose", ok=named is None, content_head=(msg.get("content") or "")[:200])
        return row
    f = calls[0].get("function") or {}
    row["name"], row["arguments"] = f.get("name"), f.get("arguments")
    problems = []
    if named is not None and row["name"] != named:
        problems.append(f"called {row['name']!r}, not the named {named!r}")
    if row["finish"] != "tool_calls":
        problems.append(f"finish_reason {row['finish']!r}")
    try:
        args = json.loads(row["arguments"]) if isinstance(row["arguments"], str) else row["arguments"]
    except Exception as e:
        args, problems = None, problems + [f"arguments do not parse: {e}"]
    if args is not None:
        if row["name"] in schemas:
            problems += validate(schemas[row["name"]], args, schemas[row["name"]])
        else:
            problems.append(f"unknown tool {row['name']!r}")
    row.update(cls="call", ok=not problems, problems=problems)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("serve"), ap.add_argument("backend"), ap.add_argument("model"), ap.add_argument("out")
    ap.add_argument("--quant", default=""), ap.add_argument("--port", type=int, default=8197)
    ap.add_argument("--transcript", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "w4_transcript_base.json"))
    ap.add_argument("--groups", default="replay,keywords,sampled,auto,required,auto5")
    a = ap.parse_args()
    want = a.groups.split(",")
    unknown = set(want) - {"replay", "keywords", "sampled", "auto", "required", "auto5"}
    if unknown:
        sys.exit(f"unknown group(s): {sorted(unknown)}")
    fixture = json.load(open(a.transcript))
    all_tools = fixture["tools"] + KEYWORD_TOOLS
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in all_tools}
    argv = [a.serve, "-model", f"smoke={a.model}", "-backend", a.backend, "-addr", f"127.0.0.1:{a.port}"]
    if a.quant:
        argv += ["-quant", a.quant]
    log = open(a.out + ".serve.log", "ab")
    t_start = time.time()
    proc = subprocess.Popen(argv, stdout=log, stderr=log, preexec_fn=os.setsid)
    out = {"serve": a.serve, "backend": a.backend, "model": a.model, "quant": a.quant, "argv": argv,
           "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "groups": {}}

    def say(s):
        print(f"[smoke {time.time() - t_start:6.1f}s] {s}", file=sys.stderr, flush=True)

    try:
        if not wait_port(a.port, proc):
            say("the server did not come up; see " + a.out + ".serve.log")
            out["fatal"] = "server did not come up"
            return 2
        say(f"server up on {a.backend}; groups: {a.groups}")
        url = f"http://127.0.0.1:{a.port}/v1/chat/completions"

        def named(name):
            return {"type": "function", "function": {"name": name}}

        rows, msgs = [], []
        for turn in fixture["turns"] if "replay" in want else []:
            if turn["user"] is not None:
                msgs.append({"role": "user", "content": turn["user"]})
            st, body, dt = post(url, {"model": "smoke", "stream": False, "messages": msgs, "tools": fixture["tools"],
                                      "temperature": 0, "max_tokens": REPLAY_MAX_TOKENS,
                                      "tool_choice": named(turn["tool_choice"])})
            r = check(st, body, schemas, turn["tool_choice"])
            r.update(turn=turn["n"], seconds=round(dt, 2))
            rows.append(r)
            if st != 200 or not (body["choices"][0]["message"].get("tool_calls")):
                say(f"replay turn {turn['n']}: {r['cls']}, no call to advance the history with; the replay stops here")
                break
            m = body["choices"][0]["message"]
            c = m["tool_calls"][0]
            msgs += [m, {"role": "tool", "name": c["function"]["name"], "tool_call_id": c.get("id"),
                         "content": turn["tool_result"]}]
        if "replay" in want:
            out["groups"]["replay"] = rows
            say(f"replay: {sum(r['ok'] for r in rows)}/{len(rows)} pass of {len(fixture['turns'])} turns")

        def keyword_rows(temp, seeds):
            rows = []
            for tool in KEYWORD_TOOLS:
                name = tool["function"]["name"]
                for pi, prompt in enumerate(KEYWORD_PROMPTS[name]):
                    for seed in seeds:
                        p = {"model": "smoke", "stream": False, "messages": [{"role": "user", "content": prompt}],
                             "tools": KEYWORD_TOOLS, "temperature": temp, "max_tokens": KEYWORD_MAX_TOKENS,
                             "tool_choice": named(name)}
                        if seed is not None:
                            p["seed"] = seed
                        st, body, dt = post(url, p)
                        r = check(st, body, schemas, name)
                        r.update(tool=name, prompt=pi, seed=seed, seconds=round(dt, 2))
                        rows.append(r)
            return rows

        if "keywords" in want:
            out["groups"]["keywords"] = rows = keyword_rows(0, [None])
            say(f"keywords: {sum(r['ok'] for r in rows)}/{len(rows)} pass")
        if "sampled" in want:
            out["groups"]["sampled"] = rows = keyword_rows(0.8, [11, 12, 13])
            say(f"sampled: {sum(r['ok'] for r in rows)}/{len(rows)} pass")

        def chosen_rows(tools, choice, prompts_per_tool):
            """The model picks the tool: "auto" (prose is legal) or "required" (a call is forced)."""
            rows = []
            for tool in KEYWORD_TOOLS:
                name = tool["function"]["name"]
                for pi, prompt in enumerate(KEYWORD_PROMPTS[name][:prompts_per_tool]):
                    st, body, dt = post(url, {"model": "smoke", "stream": False, "tools": tools, "temperature": 0,
                                              "max_tokens": KEYWORD_MAX_TOKENS, "tool_choice": choice,
                                              "messages": [{"role": "user", "content": prompt}]})
                    r = check(st, body, schemas, None)
                    if choice == "required" and r["cls"] == "prose":
                        r["ok"] = False  # a forced call that came back as prose
                    r.update(intended=name, prompt=pi, seconds=round(dt, 2))
                    rows.append(r)
            return rows

        def say_chosen(g, rows):
            calls = [r for r in rows if r["cls"] == "call"]
            say(f"{g}: {len(calls)} calls of {len(rows)} replies, {sum(r['ok'] for r in calls)} of them pass, "
                f"{sum(r.get('name') == r['intended'] for r in calls)} to the intended tool; "
                f"{sum(r['cls'] == 'prose' for r in rows)} prose, {sum(r['cls'] == 'truncated' for r in rows)} cut")

        if "auto" in want:
            out["groups"]["auto"] = rows = chosen_rows(all_tools, "auto", 1)
            say_chosen("auto", rows)
        if "required" in want:
            out["groups"]["required"] = rows = chosen_rows(all_tools, "required", 2)
            say_chosen("required", rows)
        if "auto5" in want:
            out["groups"]["auto5"] = rows = chosen_rows(KEYWORD_TOOLS, "auto", 2)
            say_chosen("auto5", rows)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            proc.kill()
        out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        json.dump(out, open(a.out, "w"), indent=1)

    bad = [(g, r) for g, rs in out["groups"].items() for r in rs if not r["ok"]]
    for g, r in bad:
        say(f"FAIL {g}: {r.get('tool') or r.get('intended') or r.get('turn')}: {r.get('problems') or r.get('detail') or r.get('cls')}")
    n = sum(len(rs) for rs in out["groups"].values())
    short = "replay" in want and len(out["groups"].get("replay", [])) < len(fixture["turns"])
    cut = sum(r["cls"] == "truncated" for _, r in bad)
    verdict = "PASS" if not bad and not short else "FAIL"
    print(f"VERDICT: {verdict} — tool-call smoke on {a.backend}: {n - len(bad)} of {n} replies pass"
          + (f"; {cut} cut by the token budget" if cut else "") + ("; the replay stopped early" if short else ""))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
