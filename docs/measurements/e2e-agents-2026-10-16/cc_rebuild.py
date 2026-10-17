#!/usr/bin/env python3
"""Rebuild Claude Code requests from the transcripts the agent wrote (runs/<id>/home/.claude/projects/-work/*.jsonl), for an e2e run that kept no request bodies.

  python3 -I cc_rebuild.py E2E_OUT BASE_BODY.req.json.gz OUT_DIR

Each transcript attachment carries `rendered`: the exact reminder text and its role. A role-`system` reminder goes to the trailing system-role message (joined by a
blank line, the <system-reminder> wrapper removed); a role-`user` one to the first user message, before the task. The skeleton (system prompt, the 21 tools) comes
from BASE_BODY, a recorded Claude Code 2.1.296 first request. Writes read/search/edit (the session starts) and post1A/B, post2A/B (the request after each
compaction, with the summary before or after the attribution reminder; B is the order that reproduces the logged divergence). The rebuilt prompts are 158 ids
longer than the logged ones, the same 158 in every session: the skeleton's system prompt is not the run's exactly. Render them with
TestSideRequests_ccDiag (internal/serveapp) and read lengths and divergences relative to that constant.
"""
import copy, glob, gzip, json, os, re, sys

e2e, base_path, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
base = json.load(gzip.open(base_path))


def inner(c):
    m = re.match(r'^<system-reminder>\n(.*)\n</system-reminder>$', c, re.S)
    return m.group(1) if m else c


def entries(run):
    f = glob.glob(f"{e2e}/runs/{run}/home/.claude/projects/-work/*.jsonl")[0]
    return [json.loads(l) for l in open(f)]


def put(name, user_blocks, sys_blocks):
    b = copy.deepcopy(base)
    b['messages'][0]['content'] = [{'type': 'text', 'text': t} for t in user_blocks]
    b['messages'][1]['content'] = [{'type': 'text', 'text': "\n\n".join(inner(t) for t in sys_blocks)}]
    json.dump(b, gzip.open(f"{out}/{name}.req.json.gz", "wt"))


def split(ents):
    sysb, usr, task, summ = [], {}, None, None
    for o in ents:
        if o.get('type') == 'user' and o.get('isCompactSummary'):
            summ = o['message']['content']
        elif o.get('type') == 'user' and task is None:
            task = o['message']['content']
        elif o.get('type') == 'attachment' and o.get('rendered'):
            for c in o['rendered']:
                if o['renderedRole'] == 'system':
                    sysb.append(c['content'])
                else:
                    usr[o['attachment']['type']] = c['content']
    return sysb, usr, task, summ


for name, run in (("read", "claude-code-read-r1-175711"), ("search", "claude-code-search-r1-180033"), ("edit", "claude-code-edit-r1-180358")):
    sysb, usr, task, _ = split(entries(run)[:14])
    put(name, [usr['session_context'], usr['remote_session_change'], task], sysb)
ents = entries("claude-code-edit-r1-180358")
bounds = [i for i, o in enumerate(ents) if o.get('subtype') == 'compact_boundary']
for tag, lo in zip(("post1", "post2"), bounds):
    sysb, usr, _, summ = split(ents[lo:lo + 10])
    put(tag + "A", [usr['session_context'], usr['remote_session_change'], summ], sysb)
    put(tag + "B", [usr['session_context'], summ, usr['remote_session_change']], sysb)
