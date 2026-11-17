import sys,glob,gzip
"""Replays each kv3r-seq-*.tsv.gz (the build's device allocations before the fit-check reading, with the card's free VRAM after each)
through candidate page-placement rules and prints, per rule: final error and worst per-step error against the card, MiB.
Usage: python3 -I kv3r-packing-sim.py <this directory>"""
MIB=1<<20; Q=2*MIB
def load(p): return [l.rstrip('\n').split('\t') for l in gzip.open(p,'rt')]
def seq(p):
    ev=load(p); start=int(ev[0][2])
    fit_i=[i for i,e in enumerate(ev) if e[0]=='S' and e[1]=='fit'][0]
    pre=[e for e in ev[:fit_i] if e[0]=='A' and e[3]=='1']
    return [int(e[2]) for e in pre],[start-int(e[5]) for e in pre]
def sim(sizes,policy,tails=True):
    used=0; pages=[]  # remaining bytes of partial pages, in creation order
    out=[]
    for n in sizes:
        if n>=Q:
            r=-(-n//Q)*Q; used+=r
            if tails and r-n>0: pages.append(r-n)
        else:
            cands=[i for i,x in enumerate(pages) if x>=n]
            if policy=='bump':
                # only the most recently created page with room is current
                cands=[len(pages)-1] if pages and pages[-1]>=n else []
            elif policy=='first': cands=cands[:1]
            elif policy=='best': cands=sorted(cands,key=lambda i:pages[i])[:1]
            elif policy=='last': cands=cands[-1:]
            if cands: pages[cands[0]]-=n
            else: used+=Q; pages.append(Q-n)
        out.append(used)
    return out
for pol,tails in [('bump',False),('bump',True),('first',True),('first',False),('best',True),('best',False),('last',False)]:
    row=[]
    for p in sorted(glob.glob(sys.argv[1]+'/kv3r-seq-*.tsv.gz')):
        s,o=seq(p); u=sim(s,pol,tails)
        mx=max(abs(a-(b-2*MIB)) for a,b in zip(u,o))  # obs has the +2MiB base
        row.append((p.split('kv3r-seq-')[1][:10],round((u[-1]-(o[-1]-2*MIB))/MIB,2),round(mx/MIB,2)))
    print(pol,tails,row)
