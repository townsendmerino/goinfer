#!/usr/bin/env python3
"""G-RF2-S never-swap (KV4 G4 as registered: cell by cell, median over the cell's runs): map procmem.jsonl's servers to bench_peer
records by launch order (checked: each server's binary must match its record's engine), then TTFT's 3 + 3 per model and MC3's six.
Run from this directory:  python3 g4_by_cell.py <procmem.jsonl>
"""
import json,sys,statistics
ps=sorted((json.loads(l) for l in open(sys.argv[1]) if l.strip()), key=lambda p:p['first'])
labels=[]
for f in ['decode-qwen','decode-deep','decode-gemma3']:
    for x in json.load(open(f+'.json')):
        if isinstance(x,dict) and x.get('engine') in ('goinfer','goinfer_old'):
            labels.append((f"{f} {x['model']}@{x['depth']}", 'new' if x['engine']=='goinfer' else 'old'))
n_bench=len(labels)
bench, rest = ps[:n_bench], ps[n_bench:]
bad=[i for i,(p,(c,a)) in enumerate(zip(bench,labels)) if p['name']!=f'serve-cuda-{a}']
print(f"{n_bench} bench servers mapped by order; arm mismatches: {bad}")
# the rest: TTFT (12) then MC3 (6), in time order; group TTFT by model by order (6 + 6), MC3 one cell
for i,p in enumerate(rest):
    c = 'ttft-1.5B' if i<6 else ('ttft-7B' if i<12 else 'mc3 1.5B x4')
    labels.append((c, p['name'].split('-')[-1]))
cells={}
for p,(c,a) in zip(ps,labels):
    cells.setdefault(c,{'old':[],'new':[]})[a].append(p['hwm_kb']/1024)
worst=0
for c,v in cells.items():
    d=statistics.median(v['new'])-statistics.median(v['old']); worst=max(worst,abs(d))
    print(f"{c:28s} old {len(v['old'])} new {len(v['new'])}  median new-old {d:+6.1f} MB {'OK' if abs(d)<=64 else 'OVER'}")
print(f"G4 as registered (cell by cell, median over the cell's runs): worst {worst:.1f} MB; swap peak max {max(p['swap_peak_kb'] for p in ps)/1024:.1f} MB ->", 'PASS' if worst<=64 and max(p['swap_peak_kb'] for p in ps)==0 else 'FAIL')
