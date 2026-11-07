#!/usr/bin/env python3
"""Attribution for G2's MC3 cell at 4 slots (0.907 against f8632013 on the final build): context, not a gate. bench_w7_plain.py as the
cell runs it (the 1.5B, 4 clients, 6 turns of 128 greedy tokens, --fixed-nonce, -kv-sessions 4), from bf5e0a59's tree.
  A: main b727c55e (KV5-1 to KV5-5, no KV5-6) against the final build bf5e0a59, cells old new new old old new: is it KV5-6?
  B: the final build pinned at -ctx 16384 (each slot's share, so no raise) against the same build at its default (one conversation's cap
     raised to 32,768), cells pinned raised raised pinned: is it the raise?
  C: main b727c55e against 607f8420 (the table read through a slice taken once).
  D, E: C and B again on the final tree, from e2762eb0's tree: main 8b7e9941 against e2762eb0, then e2762eb0 pinned against raised.
Run under the exclusive timing lock, on AC."""
import os, subprocess, sys, time
D = '/Users/francistownsend-merino/goinfer-work/kv5-6'
SCRIPTS = f'{D}/new-bf5e0a59/scripts'
OUT = f'{D}/mc3-attrib'
os.makedirs(f'{OUT}/serve-logs', exist_ok=True)


def say(s):
    print(time.strftime('%H:%M:%S ') + s, flush=True)


def cell(name, serve, args, scripts=SCRIPTS):
    out = f'{OUT}/{name}.json'
    if os.path.exists(out):
        say(f'== {name}: done before, skipped')
        return
    say(f'== {name}')
    with open(f'{OUT}/{name}.log', 'w') as f:
        rc = subprocess.run(['python3', '-u', 'bench_w7_plain.py', '--clients', '4', '--engines', 'goinfer', '--backend', 'metal', '--fixed-nonce',
                             '--serve-args', args, '--server-log', f'{OUT}/serve-logs/{name}.log', out],
                            cwd=scripts, env=dict(os.environ, GOINFER_SERVE_CPU=serve), stdout=f, stderr=subprocess.STDOUT).returncode
    say(f'== {name} rc={rc}')
    if rc != 0:
        sys.exit(rc)


batt = subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True).stdout.splitlines()[0]
if 'AC Power' not in batt:
    sys.exit(f'not on AC power: {batt}')
say(batt)
steps = sys.argv[1:] or ['A', 'B']
if 'A' in steps:
    for i, arm in enumerate(['main', 'final', 'final', 'main', 'main', 'final'], 1):
        cell(f'A{i}-{arm}', f'{D}/serve-main-final' if arm == 'main' else f'{D}/serve-final', '-kv-sessions 4')
if 'B' in steps:
    for i, arm in enumerate(['pinned', 'raised', 'raised', 'pinned'], 1):
        cell(f'B{i}-{arm}', f'{D}/serve-final', '-kv-sessions 4 -ctx 16384' if arm == 'pinned' else '-kv-sessions 4')
if 'C' in steps:  # C: main b727c55e against 607f8420, the table read through a slice taken once
    for i, arm in enumerate(['main', 'cache', 'cache', 'main', 'main', 'cache'], 1):
        cell(f'C{i}-{arm}', f'{D}/serve-main-final' if arm == 'main' else f'{D}/serve-cache', '-kv-sessions 4')
F2 = f'{D}/new-e2762eb0/scripts'
if 'D' in steps:
    for i, arm in enumerate(['main', 'final', 'final', 'main', 'main', 'final'], 1):
        cell(f'D{i}-{arm}', f'{D}/serve-main-final2' if arm == 'main' else f'{D}/serve-final2', '-kv-sessions 4', F2)
if 'E' in steps:
    for i, arm in enumerate(['pinned', 'raised', 'raised', 'pinned'], 1):
        cell(f'E{i}-{arm}', f'{D}/serve-final2', '-kv-sessions 4 -ctx 16384' if arm == 'pinned' else '-kv-sessions 4', F2)
say('done')
