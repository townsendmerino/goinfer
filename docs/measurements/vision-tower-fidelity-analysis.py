#!/usr/bin/env python3
"""Quantifies the R8 phase-A tower fidelity difference from the saved outputs of TestVisionTowerTiming
(cpu-ref = CPU int8 reference, exact = attn_img_batched arm, bm128 = fused arm; [4096,1152] f32 LE each).
Usage: vision-tower-fidelity-analysis.py <dir holding vision-{cpu-ref,resident-exact,resident-bm128}-{a,b}.bin>"""
import sys, numpy as np
L = sys.argv[1].rstrip('/') + '/'
ld = lambda n: np.fromfile(L + n, dtype='<f4').reshape(4096, 1152).astype(np.float64)
rowcos = lambda a, b: (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
for pat in 'ab':
    cpu, old, new = ld(f'vision-cpu-ref-{pat}.bin'), ld(f'vision-resident-exact-{pat}.bin'), ld(f'vision-resident-bm128-{pat}.bin')
    print(f'== pattern {pat}: relative L2  old-vs-cpu {rel(old,cpu):.4f}  new-vs-cpu {rel(new,cpu):.4f}  new-vs-old {rel(new,old):.4f}')
    for nm, a, b in [('old vs cpu', old, cpu), ('new vs cpu', new, cpu), ('new vs old', new, old)]:
        print(f'   per-row cosine {nm}: p1/p5/p25/med/p75 = ' + ' / '.join(f'{x:.3f}' for x in np.percentile(rowcos(a, b), [1, 5, 25, 50, 75])))
    co, cn = rowcos(old, cpu), rowcos(new, cpu)
    print(f'   new closer to cpu than old on {100*(cn>co).mean():.1f}% of rows; paired per-row cosine diff {(cn-co).mean():+.4f} +- {(cn-co).std()/np.sqrt(len(co)):.4f}')
    d, dd = np.abs(new - old), np.abs(old - cpu)
    top = np.argsort(-np.sqrt((d ** 2).mean(0)))[:5]; topo = np.argsort(-np.sqrt((dd ** 2).mean(0)))[:5]
    print(f'   output rms {np.sqrt((cpu**2).mean()):.2f}, abs max {np.abs(cpu).max():.1f}; max|new-old| {d.max():.1f}, max|old-cpu| {dd.max():.1f}; top-5 error dims new-old {top.tolist()} vs old-cpu {topo.tolist()} (overlap {len(set(top)&set(topo))})')
