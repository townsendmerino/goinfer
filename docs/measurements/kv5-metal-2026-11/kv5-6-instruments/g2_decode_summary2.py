#!/usr/bin/env python3
"""KV5-6's G2 decode cells by day: new/old median per cell (every run of each arm), each A B B A block's ratio, swap-outs, swap-ins,
RSS and the load at each cell's start. Bands (KV5's plan): PASS at 0.98 or more, AMBIGUOUS from 0.97, FAIL under 0.97."""
import json, statistics as S, sys

D = '/Users/francistownsend-merino/goinfer-work/kv5-6/' + (__import__('os').environ.get('G2DIR') or 'g2-day')


def band(r):
    return 'PASS' if r >= 0.98 else ('AMBIGUOUS' if r >= 0.97 else 'FAIL')


for f in sys.argv[1:] or ('qwen15', 'deep15', 'gemma'):
    d = [c for c in json.load(open(f'{D}/{f}.json')) if c.get('engine')]
    for (m, dep) in sorted({(c['model'], c['depth']) for c in d}):
        cells = [c for c in d if c['model'] == m and c['depth'] == dep]
        new = [x for c in cells if c['engine'] == 'goinfer' for x in c['runs']]
        old = [x for c in cells if c['engine'] == 'goinfer_old' for x in c['runs']]
        if not new or not old:
            print(m, dep, 'an arm is missing')
            continue
        r = S.median(new) / S.median(old)
        blocks = []
        for b in (0, 1):
            bn = [x for c in cells if c['engine'] == 'goinfer' and c.get('block') == b for x in c['runs']]
            bo = [x for c in cells if c['engine'] == 'goinfer_old' and c.get('block') == b for x in c['runs']]
            if bn and bo:
                blocks.append(round(S.median(bn) / S.median(bo), 3))
        so = max(c['counts']['swap']['swapout_mb'] for c in cells)
        si = max(c['counts']['swap']['swapin_mb'] for c in cells)
        rss = {}
        for c in cells:
            rss.setdefault(c['engine'], []).append(c['counts'].get('rss_peak_kb') or 0)
        gap = (max(rss.get('goinfer', [0])) - max(rss.get('goinfer_old', [0]))) / 1024
        loads = [round(c['machine']['loadavg'][0], 2) for c in cells]
        errs = [c['error'] for c in cells if c['error']]
        print(f"{m:10s} {dep:5d}  new {S.median(new):6.1f}  old {S.median(old):6.1f} tok/s  new/old {r:.3f}  {band(r):9s} blocks {blocks}  "
              f"swap-outs {so} MB, swap-ins at most {si} MB  RSS new-old {gap:+.1f} MB  load {loads}{'  errors ' + str(errs) if errs else ''}")
