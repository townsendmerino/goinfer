#!/usr/bin/env python3
"""matrix/run_cell.py with one more task, `chain`, for KV0's traffic study (the stall remedy registered in
docs/measurements/kv0-traffic-2026-10-13.md §1b). The matrix files themselves are untouched.

`chain` forces one tool round per file: a harness whose model asks for every file in one parallel tool-call round produces a
two-request session however the task is phrased. Here each file ends with the name of the next one, the names are unguessable,
and only the last file holds the answer. The answer is the last file's secret word.
"""
import os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(HERE, "..", "integration-harness-2026-10", "matrix")
sys.path.insert(0, MATRIX)
import tasks  # noqa: E402

STEPS = 7
SECRET = "heliotrope"
TOPICS = ["tide tables", "a ledger of small tools", "the maintenance log of a lighthouse", "seed catalogue notes",
          "a timetable for a ferry", "notes on a bell foundry", "inventory of a map room"]


def make_chain(work):
    rng = random.Random(20261013)
    names = ["".join(rng.choice("0123456789abcdef") for _ in range(6)) for _ in range(STEPS)]
    os.makedirs(os.path.join(work, "chain"))
    for i, n in enumerate(names):
        lines = [f"Record {i + 1} of {STEPS}: {TOPICS[i]}."]
        for j in range(1, 11):
            lines.append(f"  entry {j:02d}: item-{rng.randrange(100, 999)} counted {rng.randrange(10, 900)} units at stall {chr(97 + rng.randrange(26))}{rng.randrange(1, 40)}.")
        lines.append(f"NEXT: chain/{names[i + 1]}.txt" if i + 1 < STEPS else f"END: the secret word is {SECRET}.")
        open(os.path.join(work, "chain", f"{n}.txt"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(work, "START.txt"), "w").write(f"The chain starts at chain/{names[0]}.txt\n")


tasks.TASKS["chain"] = dict(
    prompt="Read START.txt. It names the first file of a chain. Read that file with your file-reading tool; its last line names the next file to read, "
           "or says END. Follow the chain one file at a time, until a file says END. Then answer with the secret word.",
    setup=make_chain, answer=lambda a: SECRET in a.lower())

import run_cell  # noqa: E402

sys.exit(run_cell.main())
