#!/usr/bin/env python3
"""gemma3-1b at depth 128, each KV5 merge against f8632013 (main just before KV5-1), to place G2's FAIL there: context, not a gate.
Builds the bisect binaries under the shared timing lock, then runs each comparison (gemma_bisect.py ab, 3 quads) as its own hold of the
exclusive lock, each started only on AC with the 1-minute load under 3. Rungs: 031feff2 (KV5-2's merge), 0730d410 (KV5-4's),
c1182f99 (KV5-5's), 8b7e9941 (main now), e2762eb0 (KV5-6 final). Log: gemma-ladder.log. A refusal waits 10 min, until 19:30."""
import datetime as dt, subprocess, sys, time
D = '/Users/francistownsend-merino/goinfer-work/kv5-6'
WT = '/Users/francistownsend-merino/tmcode/goinfer/.claude/worktrees/op4m-c-2026-10-31'
LOG = f'{D}/gemma-ladder.log'
deadline = dt.datetime.now().replace(hour=19, minute=30, second=0, microsecond=0)
RUNGS = sys.argv[1:] or ['031feff2', '0730d410', 'c1182f99', '8b7e9941', 'e2762eb0']
LOCK = ['python3', 'scripts/timing_lock.py', 'run']


def say(s):
    with open(LOG, 'a') as f:
        f.write(time.strftime('%H:%M:%S ') + s + '\n')


def quiet():
    batt = subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True).stdout.splitlines()[0]
    load = float(subprocess.run(['sysctl', '-n', 'vm.loadavg'], capture_output=True, text=True).stdout.split()[1])
    return 'AC Power' in batt and load < 3, batt, load


def held(label, cmd, shared=False):
    while True:
        if not shared:
            ok, batt, load = quiet()
            if not ok:
                if dt.datetime.now() > deadline:
                    say(f'{label}: not started by 19:30 ({batt}; load {load})')
                    sys.exit(3)
                time.sleep(60)
                continue
            say(f'{label}: on AC, load {load}: taking the lock')
        else:
            say(f'{label}: taking the shared lock')
        with open(f'{D}/gemma-ladder-run.log', 'a') as f:
            rc = subprocess.run(['caffeinate', '-i'] + LOCK + (['--shared'] if shared else []) + ['--wait', '3600', '--label', f'kv5-6-{label}', '--'] + cmd,
                                cwd=WT, stdout=f, stderr=subprocess.STDOUT).returncode
        say(f'{label}: rc={rc}')
        if rc != 75:
            return rc
        if dt.datetime.now() > deadline:
            say(f'{label}: refused, and past 19:30: stopping')
            sys.exit(75)
        time.sleep(600)


build = [r for r in RUNGS if r not in ('8b7e9941', 'e2762eb0')]
if build and held('gemma-bisect-build', ['python3', '-u', f'{D}/gemma_bisect.py', 'prep'] + build, shared=True):
    sys.exit(1)
for r in RUNGS:
    held(f'gemma-{r}', ['python3', '-u', f'{D}/gemma_bisect.py', 'ab', f'f86-vs-{r}', 'f8632013', r, '3'])
say('done')
