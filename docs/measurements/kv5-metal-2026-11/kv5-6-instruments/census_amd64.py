#!/usr/bin/env python3
"""MM1's decision census for the amd64 host class, under Rosetta (GOARCH=amd64), on main 4db6cd19 and the branch: dumped and diffed as
census_diff.py does for arm64, then classified."""
import difflib, os, subprocess
D = '/Users/francistownsend-merino/goinfer-work/kv5-6'
WT = '/Users/francistownsend-merino/tmcode/goinfer/.claude/worktrees/op4m-c-2026-10-31'
out = {}
for name, tree in [('main', f'{D}/main-4db6cd19'), ('branch', WT)]:
    dump = f'{D}/census-amd64-{name}.txt'
    r = subprocess.run(['go', 'test', '-count=1', '-run', '^TestDecisionCensus$', './decoder/', '-args', f'-census.dump={dump}'],
                       cwd=tree, capture_output=True, text=True, env=dict(os.environ, GOMAXPROCS='4', GOARCH='amd64'))
    print(name, 'rc', r.returncode, r.stdout.strip().splitlines()[-1:])
    out[name] = open(dump).read().splitlines()
diff = list(difflib.unified_diff(out['main'], out['branch'], 'main', 'branch', n=0, lineterm=''))
open(f'{D}/census-diff.txt', 'w').write('\n'.join(diff) + '\n')
print(f"amd64 lines: main {len(out['main'])}, branch {len(out['branch'])}; diff lines {len(diff)}")
