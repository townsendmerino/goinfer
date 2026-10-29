#!/usr/bin/env python3
"""Cut the gpu gate's Metal cell groups' sections out of a live `gate gpu` log (SB12 step 8's C5).

    python3 live_sections.py GATE_LOG > live.txt

The sections are suite (2.), parity (2c.), lifecycle (4.) and prefill (4b.), each from the blank line before its header to the
blank line before the next header, in the order the gate ran them. The runner's heartbeat lines (stderr, merged into the log:
"   ... <cell>: <time> elapsed, ...") are dropped; nothing else is changed, colour codes included.
"""
import re, sys

HEADERS = ("2. Metal suite", "2c. resident PARITY gates", "4. lifecycle", "4b. prefill")
head = re.compile(r"^\x1b\[1m== (.*) ==\x1b\[0m$")
beat = re.compile(r"^   \.\.\. \S+: .* elapsed, ")

lines = open(sys.argv[1], encoding="utf-8").read().split("\n")
out, keep = [], False
for i, ln in enumerate(lines):
    m = head.match(ln)
    if m:
        keep = m.group(1).startswith(HEADERS)
        if keep:
            out.append("")  # the blank line hdr prints before the header
    if keep and not (m is None and ln == "" and i + 1 < len(lines) and head.match(lines[i + 1])):
        if not beat.match(ln):
            out.append(ln)
sys.stdout.write("\n".join(out) + "\n")
