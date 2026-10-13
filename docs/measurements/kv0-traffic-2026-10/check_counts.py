#!/usr/bin/env python3
"""KV0 traffic study, validity check: the token count the tokenizer rendered for each request must equal the prompt_tokens the
server logged for it (`-log-requests` lines "request: POST <route> ... status=200 prompt_tokens=N"). The server's usage block is
parsed by the proxy for the chat and Responses routes but not for the Messages stream, so the log is the common source.

  check_counts.py <stats.json> <serve log> [<serve log> ...]     (one log per configuration, named serve-<config>.log)

Compares multisets per configuration: requests the server answered with 200, rendered n against logged prompt_tokens.
"""
import collections, json, re, sys

stats = json.load(open(sys.argv[1]))
bad = 0
for cfg, c in stats["configs"].items():
    log = [p for p in sys.argv[2:] if f"serve-{cfg}.log" in p]
    if not log:
        print(f"{cfg}: no serve log given"); bad += 1; continue
    logged = collections.Counter()
    for line in open(log[0]):
        m = re.match(r"request: POST (\S+) model=\S+ status=200 prompt_tokens=(\d+)", line)
        if m:
            logged[int(m.group(2))] += 1
    rendered = collections.Counter(r["n"] for s in c["sessions"] for r in s["requests"] if "n" in r and r.get("status") == 200)
    missing = rendered - logged
    extra = logged - rendered
    print(f"{cfg}: {sum(rendered.values())} rendered 200-requests, {sum(logged.values())} logged; rendered not in the log: {sum(missing.values())}; logged not rendered: {sum(extra.values())}")
    bad += sum(missing.values()) + sum(extra.values())
sys.exit(1 if bad else 0)
