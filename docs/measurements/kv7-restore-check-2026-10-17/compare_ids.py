#!/usr/bin/env python3
"""compare_ids.py DIR ARM... — the last generation of each arm's serve-ARM.log.ids against the first arm's: prompt hash, reused rows, generated ids identical or not."""
import json, sys
d, arms = sys.argv[1], sys.argv[2:]
rows = {a: [json.loads(l) for l in open(f"{d}/serve-{a}.log.ids")][-1] for a in arms}
base = rows[arms[0]]["gen"]
for a, g in rows.items():
    print(a, "prompt", g["prompt_len"], g["prompt_sha256"][:12], "reused", g["reused"], "gen", len(g["gen"]),
          "identical" if g["gen"] == base else "DIFFERS at %d" % next(i for i, (x, y) in enumerate(zip(g["gen"], base)) if x != y))
