import sys,statistics as st
rows=[l.rstrip('\n').split('\t') for l in open(sys.argv[1]).read().splitlines()[1:]]
cells={}
for r in rows:
    try: v=float(r[12])
    except: continue
    cells.setdefault((r[1],r[2]),{})[(int(r[3]),int(r[4]))]=v
for (m,d),c in cells.items():
    pr=[];A=[];B=[]
    for b in range(1,6):
        if all((b,k) in c for k in (1,2,3,4)):
            pr+=[c[(b,2)]/c[(b,1)], c[(b,3)]/c[(b,4)]]; A+=[c[(b,1)],c[(b,4)]]; B+=[c[(b,2)],c[(b,3)]]
    if pr: print(f"{m} d{d}: pairs {len(pr)} median {st.median(pr):.4f} faster {sum(x>1 for x in pr)}/{len(pr)} range {min(pr):.3f}-{max(pr):.3f} A {st.mean(A):.3f} B {st.mean(B):.3f}")
