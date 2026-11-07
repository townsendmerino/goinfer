#!/usr/bin/env python3
"""Classify the census diff's lines: with every byte figure (free, kv) removed, do the main and branch lines still differ (a decision
moved) or only the figures (the price moved)? Also each kv delta's size."""
import collections, re
D = '/Users/francistownsend-merino/goinfer-work/kv5-6'
lines = open(f'{D}/census-diff.txt').read().splitlines()
minus = [l[1:] for l in lines if l.startswith('-') and not l.startswith('---')]
plus = [l[1:] for l in lines if l.startswith('+') and not l.startswith('+++')]
norm = lambda l: re.sub(r'free \d+', 'free N', re.sub(r'kv \d+', 'kv N', l))
mc, pc = collections.Counter(map(norm, minus)), collections.Counter(map(norm, plus))
print(f'changed lines: {len(minus)} removed, {len(plus)} added')
print(f'with byte figures removed, lines only on main: {sum((mc - pc).values())}, only on the branch: {sum((pc - mc).values())}')
for l, n in list((pc - mc).items())[:8]:
    print('  branch only:', n, l[:220])
for l, n in list((mc - pc).items())[:8]:
    print('  main only:  ', n, l[:220])
kv = lambda l: int(re.search(r'kv (\d+)', l).group(1)) if re.search(r'kv (\d+)', l) else None
deltas = collections.Counter()
for a, b in zip(minus, plus):
    if kv(a) is not None and kv(b) is not None:
        deltas[kv(b) - kv(a)] += 1
print('kv deltas (bytes: count):', sorted(deltas.items())[:20])
