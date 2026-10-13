#!/usr/bin/env python3
"""Wall-clock attribution of Claude Code session transcripts (docs/measurements/test-wait-2026-10.md).

test-efficiency-2026-09/transcript_miner.py sums Bash call durations. This instead walks each transcript's events in
time order and gives every gap between two consecutive events to exactly one bucket:
  tool:<class>     a foreground tool call is still running (the highest-priority class wins when calls overlap)
  model            the model is working: after a tool result, a prompt or a notification, or between its own blocks
  bgwait:<class>   the turn ended and the next event is a background job's notification (gap < 3 h)
  sched-wakeup-wait  the turn ended and the next event is a scheduled wake-up prompt (gap < 3 h)
  away(...)        the turn ended and the next event is the owner's prompt, an AskUserQuestion, or a wait over 3 h
  stall            a model gap of 30 min or more (API errors, compaction, an abandoned turn)

Usage:
  session_wall.py collect OUT.json TRANSCRIPT.jsonl [...]   per-day buckets, local time, one or many transcripts
  session_wall.py report [--since YYYY-MM-DD] OUT.json [...]  grouped totals and the weekly trend
"""
import collections, datetime as dt, json, re, sys

def cat_bash(c):
    if re.search(r'\bgit push\b', c): return 'push'
    if re.search(r'night\.py', c): return 'night-admin'
    if re.search(r'cmd/gate|\bgate (quick|gpu|release|parity|kill)|refresh_parity_hashes', c): return 'gate'
    if re.search(r'\bgh run\b|\bgh pr checks\b', c): return 'ci'
    if re.search(r'\bgo test\b', c) and re.search(r'-bench\b|-bench=', c): return 'bench'
    if re.search(r'\bgo test\b|\bpytest\b', c): return 'test'
    if re.search(r'scripts/bench|bench_|hyperfine|/usr/bin/time|goinfer-chat|goinfer-serve|ollama |llama-bench|tok/s', c): return 'bench'
    if re.search(r'^\s*(until|while)\b|\bsleep \d|\bpgrep\b|\bps (aux|-p)|caffeinate -w|launchctl list', c):
        for pat, k in ((r'gh run|\bci\b', 'poll-ci'), (r'night|nobara', 'poll-night'), (r'gate', 'poll-gate'),
                       (r'bench|tok/s|peer', 'poll-bench'), (r'test', 'poll-test')):
            if re.search(pat, c): return k
        return 'poll'
    if re.search(r'\bgo (build|vet|install|run)\b|gofmt|staticcheck|_lint\.py|govulncheck|go mod', c): return 'build-lint'
    if re.search(r"open\([^)]*['\"]w['\"]|\.write\(|sed -i|cat >|tee |patch ", c): return 'edit'
    if re.search(r'\bgit (commit|add|merge|rebase|checkout|branch|stash|worktree|cherry-pick|reset|tag|pull|fetch)\b', c): return 'git'
    if re.search(r'\bssh\b', c): return 'remote-other'
    return 'read'

TOOLCAT = {'Read': 'read', 'Grep': 'read', 'Glob': 'read', 'WebFetch': 'read', 'WebSearch': 'read', 'ToolSearch': 'read',
           'Edit': 'edit', 'Write': 'edit', 'MultiEdit': 'edit', 'NotebookEdit': 'edit',
           'Agent': 'subagent', 'Task': 'subagent', 'Workflow': 'subagent', 'Monitor': 'poll', 'TaskOutput': 'poll',
           'ScheduleWakeup': 'sched', 'AskUserQuestion': 'human', 'EnterPlanMode': 'human', 'ExitPlanMode': 'human',
           'TaskStop': 'git'}
PRI = ['gate', 'test', 'push', 'ci', 'bench', 'poll-gate', 'poll-test', 'poll-ci', 'poll-bench', 'poll-night', 'poll',
       'subagent', 'build-lint', 'human', 'night-admin', 'remote-other', 'git', 'edit', 'read', 'sched', 'other']
CAP_MODEL, CAP_WAIT = 1800, 3 * 3600

def T(s): return dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()

def kind(d):
    if d['type'] == 'assistant':
        return 'a_tool' if any(b.get('type') == 'tool_use' for b in d['message'].get('content', [])) else 'a_text'
    if d['type'] == 'system': return 'sys'
    c = d['message']['content']
    if isinstance(c, list) and any(isinstance(b, dict) and b.get('type') == 'tool_result' for b in c): return 'u_result'
    if (d.get('origin') or {}).get('kind') == 'human': return 'u_human'
    s = c if isinstance(c, str) else ' '.join(b.get('text', '') for b in c if isinstance(b, dict))
    return 'u_notif' if s.startswith('<task-notification>') else 'u_other'

def collect(path, daily):
    ev = []
    for line in open(path):
        try: d = json.loads(line)
        except ValueError: continue
        if d.get('type') in ('assistant', 'user', 'system') and 'timestamp' in d: ev.append(d)
    ev.sort(key=lambda d: d['timestamp'])
    pending, bg_by_tool, running_bg = {}, {}, []
    prev = prevkind = None
    for d in ev:
        ts, k = T(d['timestamp']), kind(d)
        if prev is not None and ts > prev:
            g = ts - prev
            if pending:
                b = 'tool:' + min(pending.values(), key=PRI.index)
                if b == 'tool:human': b = 'away(human)'
            elif prevkind in ('u_result', 'u_human', 'u_notif', 'u_other', 'sys') or k in ('a_text', 'a_tool'):
                b = 'model' if g < CAP_MODEL else 'stall'
            elif k == 'u_human': b = 'away(human)'
            elif g >= CAP_WAIT: b = 'away(overnight-bg)'
            elif k == 'u_notif': b = 'bgwait:' + (running_bg[-1][1] if running_bg else 'unknown')
            else: b = 'sched-wakeup-wait'
            daily[dt.datetime.fromtimestamp(prev).strftime('%Y-%m-%d')][b] += g
        if k == 'a_tool':
            for b in d['message']['content']:
                if b.get('type') != 'tool_use': continue
                if b['name'] == 'Bash':
                    c = cat_bash(b['input'].get('command', ''))
                    if b['input'].get('run_in_background'): bg_by_tool[b['id']], c = c, 'read'
                else: c = TOOLCAT.get(b['name'], 'other')
                pending[b['id']] = c
        elif k == 'u_result':
            for b in d['message']['content']:
                if not isinstance(b, dict) or b.get('type') != 'tool_result': continue
                i = b.get('tool_use_id'); pending.pop(i, None)
                if i in bg_by_tool:
                    m = re.search(r'with ID: (\w+)', json.dumps(b.get('content')))
                    if m: running_bg.append((m.group(1), bg_by_tool[i]))
        elif k == 'u_notif':
            m = re.search(r'<task-id>(\w+)</task-id>', d['message']['content'] if isinstance(d['message']['content'], str) else '')
            if m: running_bg = [r for r in running_bg if r[0] != m.group(1)]
        elif k == 'u_human':
            pending.clear()
        prev, prevkind = ts, k

GROUPS = {
    'wait: tests & gates (local)': ['tool:test', 'tool:gate', 'tool:push', 'tool:poll-gate', 'tool:poll-test',
                                    'bgwait:test', 'bgwait:gate', 'bgwait:poll-gate', 'bgwait:poll-test', 'bgwait:push'],
    'wait: CI after push': ['tool:ci', 'bgwait:ci', 'sched-wakeup-wait', 'tool:poll-ci', 'bgwait:poll-ci'],
    'wait: benchmarks/measurement': ['tool:bench', 'tool:poll-bench', 'bgwait:bench', 'bgwait:poll-bench'],
    'wait: nobara/night jobs': ['tool:poll-night', 'bgwait:poll-night', 'tool:night-admin', 'bgwait:night-admin',
                                'tool:remote-other', 'bgwait:remote-other'],
    'wait: other polling': ['tool:poll', 'bgwait:poll', 'bgwait:read', 'bgwait:unknown', 'bgwait:edit', 'bgwait:git',
                            'bgwait:sched', 'bgwait:subagent', 'bgwait:other'],
    'build/vet/lint': ['tool:build-lint', 'bgwait:build-lint'],
    'tool work (read/edit/git)': ['tool:read', 'tool:edit', 'tool:git', 'tool:subagent', 'tool:sched', 'tool:other'],
    'model thinking/writing': ['model'],
}
EXCLUDED = {'away(human)', 'away(overnight-bg)', 'stall'}

def report(paths, since):
    tot, weeks = collections.Counter(), collections.defaultdict(collections.Counter)
    for p in paths:
        for day, c in json.load(open(p)).items():
            if day < since: continue
            d = dt.date.fromisoformat(day); w = (d - dt.timedelta(days=d.weekday())).isoformat()
            for g, ks in GROUPS.items():
                v = sum(c.get(k, 0) for k in ks); tot[g] += v; weeks[w][g] += v
            for k in c:
                if k not in EXCLUDED and not any(k in ks for ks in GROUPS.values()):
                    print('unclassified bucket:', k, file=sys.stderr)
    act = sum(tot.values())
    print(f'active {act / 3600:.0f} h')
    for g, v in tot.most_common(): print(f'  {g:30s} {v / 3600:6.1f} h  {100 * v / act:5.1f}%')
    print('week of     active  tests+gates+CI  all waiting')
    for w in sorted(weeks):
        c = weeks[w]; a = sum(c.values()) or 1
        tg = c['wait: tests & gates (local)'] + c['wait: CI after push']
        aw = sum(v for k, v in c.items() if k.startswith(('wait', 'build')))
        print(f'{w}  {a / 3600:5.0f} h  {100 * tg / a:5.1f}%  {100 * aw / a:5.1f}%')

if __name__ == '__main__':
    if len(sys.argv) >= 4 and sys.argv[1] == 'collect':
        daily = collections.defaultdict(collections.Counter)
        for p in sys.argv[3:]: collect(p, daily)
        json.dump({k: dict(v) for k, v in daily.items()}, open(sys.argv[2], 'w'), indent=0, sort_keys=True)
    elif len(sys.argv) >= 3 and sys.argv[1] == 'report':
        args, since = sys.argv[2:], '0000'
        if args[0] == '--since': since, args = args[1], args[2:]
        report(args, since)
    else:
        sys.exit(__doc__)
