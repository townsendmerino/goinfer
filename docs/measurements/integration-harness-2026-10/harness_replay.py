#!/usr/bin/env python3
"""I4, replay phase: post each request capture.py recorded to a running goinfer, with the output capped at one token and
streaming off, and print what goinfer answered: the status, the prompt's size in tokens, and any tools it dropped. The
harness is not in the loop, so nothing a model writes is executed.

  harness_replay.py <capture dir with one subdirectory a harness> <base url> [harness ...]

REPLAY_MODEL=<id> replaces each body's model name (the recorders were told "coder"). With harness names, only those subdirectories are replayed. A request whose body is byte-identical to an earlier one in
the same directory (a harness retrying after the recorder's 400) is skipped and said so.
"""
import glob, json, os, sys, urllib.request, urllib.error
root, base = sys.argv[1], sys.argv[2].rstrip("/")
only = set(sys.argv[3:])
def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer goinfer"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, dict(r.headers), json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try: j = json.loads(e.read() or b"{}")
        except Exception: j = {}
        return e.code, dict(e.headers), j
for d in sorted(glob.glob(os.path.join(root, "*"))):
    if not os.path.isdir(d) or (only and os.path.basename(d) not in only): continue
    seen = {}
    for f in sorted(glob.glob(os.path.join(d, "req-*.json")), key=lambda p: int(p.split("-")[-1].split(".")[0])):
        r = json.load(open(f)); b = r["body"]; path = r["path"].split("?")[0]
        if r["method"] != "POST" or not isinstance(b, dict): continue
        key = json.dumps(b, sort_keys=True)
        if key in seen:
            print(f"{os.path.basename(d)} {os.path.basename(f)} {path}: identical to {seen[key]}, skipped", flush=True); continue
        seen[key] = os.path.basename(f)
        b = dict(b)
        if os.environ.get("REPLAY_MODEL"): b["model"] = os.environ["REPLAY_MODEL"]
        b["stream"] = False; b.pop("stream_options", None)
        if path == "/v1/chat/completions": b["max_tokens"] = 1
        elif path == "/v1/responses": b["max_output_tokens"] = 1
        elif path == "/v1/messages": b["max_tokens"] = 1
        st, hdr, j = post(path, b)
        u = j.get("usage") or {}
        tok = u.get("prompt_tokens", u.get("input_tokens"))
        dropped = hdr.get("X-Goinfer-Tools-Dropped") or hdr.get("x-goinfer-tools-dropped")
        err = (j.get("error") or {}).get("message") if isinstance(j.get("error"), dict) else j.get("error")
        print(f"{os.path.basename(d)} {os.path.basename(f)} {path}: status {st}, prompt tokens {tok}, tools sent {len(b.get('tools') or [])}"
              + (f", dropped: {dropped}" if dropped else "") + (f", error: {str(err)[:300]}" if err else ""), flush=True)
