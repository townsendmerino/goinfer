#!/usr/bin/env python3
"""Step 0 of the id round trip (id-roundtrip-step0-2026-10-15.md): rebuild the after run's Codex requests as Codex sent them.

Usage: python3 -I prep.py OUT_DIR

Reads the archived after run (../i3-night-2026-10-13/job-after/codex-rollouts.tar.gz) and one request body Codex 0.162.1
sent on 2026-10-14 (LIVE_REQ, outside the repository; it supplies the first developer message and the tool list, which the
rollouts do not hold). Writes OUT_DIR/requests.jsonl, one line per model request before each session's first compaction:

  id, run, n            the session and the request's index in it
  body                  the /v1/responses body: Codex's own fields, input = [developer base] + the rollout's items so far
  follow                the items Codex appended after this response and before its next request (tool outputs, inserted
                        messages); the next request's render puts the echoed reply between body.input and these
  recorded              the usage serve reported for this request on the night (input, output, cached tokens)
  archived_reply        the night's reply items (reasoning, calls, message), for the client-edit comparison

The wire shape is the rollout item less Codex's `internal_chat_message_metadata_passthrough` key, which is what the live
capture shows Codex sending (prep's own check, printed at the end, is against that capture).
"""
import glob, json, os, sys, tarfile, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.join(HERE, "..", "i3-night-2026-10-13", "job-after", "codex-rollouts.tar.gz")
LIVE = os.path.expanduser("~/goinfer-bench/responses-reasoning-live-2026-10-14")
LIVE_REQ = os.path.join(LIVE, "req-3.json")

OUTPUT_TYPES = {"reasoning", "function_call"}


def wire(item):
    return {k: v for k, v in item.items() if k != "internal_chat_message_metadata_passthrough"}


def is_output(p):
    return p.get("type") in OUTPUT_TYPES or (p.get("type") == "message" and p.get("role") == "assistant")


def requests_of(path, dev0, template):
    """Requests before the session's first compaction (the compaction request itself excluded)."""
    hist, cur, snap, out = [], [], None, []
    lines = [json.loads(l) for l in open(path)]
    for i, o in enumerate(lines):
        t, p = o["type"], o["payload"]
        if t == "compacted":
            if out:
                out.pop()  # the response before a `compacted` entry is the compaction request
            break
        if t == "response_item":
            if is_output(p):
                if snap is None:
                    snap = list(hist)
                cur.append(wire(p))
            else:
                hist.append(wire(p))
        elif t == "token_usage_record":
            if snap is None:
                snap = list(hist)
            u = p["usage"]
            body = dict(template)
            body["input"] = [dev0] + snap
            out.append({"n": len(out), "body": body, "archived_reply": cur, "follow_start": len(hist) + len(cur),
                        "recorded": {"input": u["input_tokens"], "output": u["output_tokens"], "cached": u["cached_input_tokens"]}})
            hist += cur
            cur, snap = [], None
    # follow: the items after each reply up to the next request's snapshot
    for k, r in enumerate(out):
        nxt = out[k + 1]["body"]["input"][1:] if k + 1 < len(out) else None
        r["follow"] = nxt[r["follow_start"]:] if nxt is not None else None
        del r["follow_start"]
    return out


def main():
    od = sys.argv[1]
    os.makedirs(od, exist_ok=True)
    live = json.load(open(LIVE_REQ))
    dev0 = live["input"][0]
    template = {k: v for k, v in live.items() if k not in ("input", "prompt_cache_key", "client_metadata")}
    tmp = tempfile.mkdtemp()
    with tarfile.open(ARCH) as tf:
        tf.extractall(tmp, filter="data")
    n_req = 0
    with open(os.path.join(od, "requests.jsonl"), "w") as f:
        for path in sorted(glob.glob(tmp + "/runs/*/home/.codex/sessions/*/*/*/*.jsonl")):
            run = path.split("/runs/")[1].split("/")[0][6:]
            for r in requests_of(path, dev0, template):
                r["run"], r["id"] = run, f"{run}#{r['n']}"
                f.write(json.dumps(r) + "\n")
                n_req += 1
    # check against the live capture: rebuilding the live session's third request must give its body's input exactly
    lp = glob.glob(LIVE + "/home/.codex/sessions/*/*/*/*.jsonl")[0]
    lr = requests_of(lp, dev0, template)
    print(f"{n_req} requests written; live check: request 3 input rebuilt == captured: {lr[2]['body']['input'] == live['input']}")


if __name__ == "__main__":
    main()
