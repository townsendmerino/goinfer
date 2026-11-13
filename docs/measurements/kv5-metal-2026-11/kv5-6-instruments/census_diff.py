#!/usr/bin/env python3
"""MM1's decision census on main 4db6cd19 and on the branch, dumped to text and diffed: which groups and lines KV5-6's table pricing
moves on this host class."""
import difflib, os, subprocess
D = '/Users/francistownsend-merino/goinfer-work/kv5-6'
WT = '/Users/francistownsend-merino/tmcode/goinfer/.claude/worktrees/op4m-c-2026-10-31'
out = {}
for name, tree in [('main', f'{D}/main-4db6cd19'), ('branch', WT)]:
    dump = f'{D}/census-{name}.txt'
    r = subprocess.run(['go', 'test', '-count=1', '-run', '^TestDecisionCensus$', './decoder/', '-args', f'-census.dump={dump}'],
                       cwd=tree, capture_output=True, text=True, env=dict(os.environ, GOMAXPROCS='4'))
    print(name, 'rc', r.returncode, r.stdout.strip().splitlines()[-1:] if r.stdout.strip() else r.stderr[-300:])
    out[name] = open(dump).read().splitlines() if os.path.exists(dump) else []
diff = list(difflib.unified_diff(out['main'], out['branch'], 'main', 'branch', n=0, lineterm=''))
open(f'{D}/census-diff.txt', 'w').write('\n'.join(diff) + '\n')
print(f"lines: main {len(out['main'])}, branch {len(out['branch'])}; diff lines {len(diff)}")
