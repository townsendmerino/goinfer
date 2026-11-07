#!/usr/bin/env python3
"""G2's MC3 cells (old new new old old new): each cell's aggregate tok/s and wall, the three old/new pairs and their median, and whether
every client's every turn has the same content_sha in every cell."""
import glob, json, os, statistics as S
D = '/Users/francistownsend-merino/goinfer-work/kv5-6/' + (os.environ.get('G2DIR') or 'g2-final')
cells = []
for i, arm in enumerate(['old', 'new', 'new', 'old', 'old', 'new'], 1):
    d = json.load(open(f'{D}/mc3s4-{i}-{arm}.json'))
    g = d['results']['goinfer']['4']
    shas = [[t.get('content_sha') for t in c] for c in g.get('per_client', [])]
    reused = sum(t.get('prefill_reused_tokens', 0) for c in g.get('per_client', []) for t in c)
    cells.append((arm, g['aggregate_tok_s'], g['wall_s'], shas, d['header'].get('goinfer_serve_mtime'), d['header'].get('uptime')))
    print(f"cell {i} {arm}: aggregate {g['aggregate_tok_s']} tok/s, wall {g['wall_s']} s, prefill reused {reused}, serve mtime {d['header'].get('goinfer_serve_mtime')}, "
          f"{d['header'].get('uptime')}")
pairs = []
for a, b in ((0, 1), (2, 3), (4, 5)):
    new = cells[a][1] if cells[a][0] == 'new' else cells[b][1]
    old = cells[b][1] if cells[a][0] == 'new' else cells[a][1]
    pairs.append(round(new / old, 3))
print('new/old pairs', pairs, 'median', S.median(pairs))
first = cells[0][3]
print('every turn content_sha equal across cells:', all(c[3] == first for c in cells), '; turns per client:', [len(x) for x in first])

