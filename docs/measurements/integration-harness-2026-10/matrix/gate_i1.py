#!/usr/bin/env python3
"""I1's live gate (docs/tasks/task-integration-harness-2026-10.md, "I3 run plan, registered 2026-10-13"): read the `bulk` and `bulkbig`
runs in records.jsonl and grade each against the rule registered in I1.

  gate_i1.py <records.jsonl>

Per run, from the wire log (every request goinfer answered, in order) and the harness's own output:
  overflows      requests answered with the dialect's overflow (a 400 naming the prompt or context, or response.failed)
  in_a_row       the longest run of overflow answers with none answered normally between them. REPORTED, NOT GRADED: Codex
                 answers an overflowing compaction request by dropping its oldest item and asking again, so four in a row
                 can be one compaction
  compactions    the harness's own count, or else what the wire shows (a tool-less request of 1,000 tokens or more after the first request with tools, or a prompt under 60% of the largest so far)
  continued      the harness exited 0 with its final answer, not on an error, and the last request was not an overflow

  PASS  at least one overflow, compactions <= overflows and fewer than 3, and the session continued
  FAIL  the session ended on the overflow or an error, or compactions >= 3 (a loop), or compactions > overflows
  VOID  no overflow and no compaction (the window was never reached), or an incurable jump (one step added more than fits
        beside the fixed part once everything older is compacted): either way the run says nothing about the wording
  NOTE  no overflow but the harness compacted by itself first (a pass for the session, no evidence about the wording)
"""
import json, sys


def wire_compactions(reqs):
    """What the wire shows of a compaction: a request with no tools, after the session has sent one with tools, carrying
    1,000 tokens or more (a summarisation of the history; a title or a side request is smaller or comes first); or the next
    prompt falling under 60% of the largest so far."""
    n, seen_tools, peak = 0, False, 0
    for x in reqs:
        p = x.get("prompt_tokens") or 0
        if x.get("tools"):
            seen_tools = True
            if peak and p and p < 0.6 * peak:
                n += 1
                peak = p
            peak = max(peak, p)
        elif seen_tools and p >= 1000 and not x.get("overflow"):
            n += 1
    return n


def grade(r):
    reqs = [x for x in r["wire"]["per_request"] if x["path"] in ("/v1/chat/completions", "/v1/messages", "/v1/responses")]
    over = [bool(x.get("overflow")) for x in reqs]
    run = best = 0
    for o in over:
        run = run + 1 if o else 0
        best = max(best, run)
    n_over = sum(over)
    comp = r.get("compactions") or wire_compactions(reqs)
    ended_on_overflow = bool(over) and over[-1]
    continued = (not r.get("ended_on_error")) and not ended_on_overflow and r.get("exit") == 0
    d = dict(overflows=n_over, in_a_row=best, compactions=comp, continued=continued, ended_on_overflow=ended_on_overflow)
    # A step whose own new content cannot fit once everything older is compacted away is no test of the wording: no answer
    # could cure it. New content = the rejected prompt less the last one answered before it; room = the window less the
    # session's fixed part (its first request with tools) and 1,000 tokens for the summary.
    window = int(r["labels"].get("ctx") or 0)
    fixed = next((x["prompt_tokens"] for x in reqs if x.get("tools") and x.get("prompt_tokens")), 0)
    last_ok = 0
    for x in reqs:
        if x.get("overflow"):
            n = x.get("overflow_tokens")
            if n and window and fixed and n - last_ok > window - fixed - 1000:
                return "VOID", dict(d, note=f"incurable jump: one step added {n - last_ok} tokens, the room after compaction is {window - fixed - 1000}")
        elif x.get("prompt_tokens") and x.get("tools"):
            last_ok = x["prompt_tokens"]
    if n_over == 0 and comp == 0:
        return "VOID", dict(d, note="the window was never reached")
    if n_over == 0:
        return "NOTE", dict(d, note="the harness compacted before any overflow")
    if not continued or comp >= 3 or comp > n_over:
        return "FAIL", d
    return "PASS", d


if __name__ == "__main__":
    for line in open(sys.argv[1]):
        r = json.loads(line)
        if not (r["task"].startswith("bulk") or "-ovf" in (r["labels"].get("config") or "")):
            continue
        v, d = grade(r)
        print(f"{v:5} {r['agent']:12} {r['task']:8} config={r['labels'].get('config')} ctx={r['labels'].get('ctx')} {json.dumps(d)}")
