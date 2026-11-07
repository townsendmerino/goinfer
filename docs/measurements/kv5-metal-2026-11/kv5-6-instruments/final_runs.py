#!/usr/bin/env python3
"""KV5-6's timed runs on the final build (serve-new at 0b2c9bd0, serve-main at main 4db6cd19), one after another, each its own hold of
the exclusive timing lock (so another session can take a turn between them), each started only on AC with the 1-minute load under 3:
  1. G3c (docs/measurements/kv5-metal-2026-11/g3c.py), about 12 min;
  2. the gemma3-1b depth-128 A/B, main against this build (gemma_ab.py), about 5 min;
  3. G2's MC3 and TTFT cells (run_g2_kv5_6_day.py, whose decode steps are already done), about 40 min.
A refusal (exit 75) waits 10 min and retries, until 19:30. Log: final-runs.log."""
import datetime as dt, subprocess, sys, time
D = '/Users/francistownsend-merino/goinfer-work/kv5-6'
WT = '/Users/francistownsend-merino/tmcode/goinfer/.claude/worktrees/op4m-c-2026-10-31'
LOG = f'{D}/final-runs.log'
deadline = dt.datetime.now().replace(hour=19, minute=30, second=0, microsecond=0)
STEPS_ALL = [
    ('g3c', ['python3', '-u', 'docs/measurements/kv5-metal-2026-11/g3c.py', f'{D}/serve-main', f'{D}/serve-new', f'{D}/g3c-final.json'], f'{D}/g3c-final-run.log'),
    ('gemma-ab', ['python3', '-u', f'{D}/gemma_ab.py'], f'{D}/gemma-ab-run.log'),
    ('g2-mc3-ttft', ['python3', '-u', 'docs/measurements/kv5-metal-2026-11/run_g2_kv5_6_day.py'], f'{D}/g2-day-run2.log'),
    ('mc3s4', ['python3', '-u', 'docs/measurements/kv5-metal-2026-11/run_mc3_kv5_6.py'], f'{D}/mc3s4-run.log'),
] + [(f'final-{x}', ['python3', '-u', 'docs/measurements/kv5-metal-2026-11/run_g2_kv5_6_final.py', x], f'{D}/final-{x}-run.log')
     for x in ('g3c', 'decode', 'mc3', 'ttft')] + [('mc3-attrib', ['python3', '-u', f'{D}/mc3_attrib.py'], f'{D}/mc3-attrib-run.log'),
     ('mc3-cache', ['python3', '-u', f'{D}/mc3_attrib.py', 'C'], f'{D}/mc3-cache-run.log'),
     ('mc3-final-attrib', ['python3', '-u', f'{D}/mc3_attrib.py', 'D', 'E'], f'{D}/mc3-final-attrib-run.log'),
     ('gemma-direct', ['python3', '-u', f'{D}/gemma_bisect.py', 'ab', 'main-vs-e2762eb0', '8b7e9941', 'e2762eb0', '5'], f'{D}/gemma-direct-run.log')]
STEPS = [st for st in STEPS_ALL if st[0] in (sys.argv[1:] or [st[0] for st in STEPS_ALL])]


def say(s):
    with open(LOG, 'a') as f:
        f.write(time.strftime('%H:%M:%S ') + s + '\n')


def quiet():
    batt = subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True).stdout.splitlines()[0]
    load = float(subprocess.run(['sysctl', '-n', 'vm.loadavg'], capture_output=True, text=True).stdout.split()[1])
    return 'AC Power' in batt and load < 3, batt, load


for name, cmd, out in STEPS:
    while True:
        ok, batt, load = quiet()
        if not ok:
            if dt.datetime.now() > deadline:
                say(f'{name}: not started by 19:30 ({batt}; load {load})')
                sys.exit(3)
            time.sleep(60)
            continue
        say(f'{name}: on AC, load {load}: taking the lock')
        with open(out, 'a') as f:
            rc = subprocess.run(['caffeinate', '-i', 'python3', 'scripts/timing_lock.py', 'run', '--wait', '3600', '--label', f'kv5-6-{name}', '--'] + cmd,
                                cwd=WT, stdout=f, stderr=subprocess.STDOUT).returncode
        say(f'{name}: rc={rc}')
        if rc != 75:
            break
        if dt.datetime.now() > deadline:
            say(f'{name}: refused, and past 19:30: stopping')
            sys.exit(75)
        time.sleep(600)
say('done')
