import csv,collections,gzip,sys,statistics
rows=list(csv.reader(gzip.open(sys.argv[1],"rt") if sys.argv[1].endswith(".gz") else open(sys.argv[1])))
hs=[i for i,r in enumerate(rows) if 'Kernel Name' in r]
if not hs: print("no data"); sys.exit()
h=rows[hs[0]]; kn=h.index('Kernel Name'); dc=h.index('gpu__time_duration.sum')
by=collections.defaultdict(lambda:[0,0.0])
for r in rows[hs[0]+2:]:
    if len(r)<=dc: continue
    try: v=float(r[dc].replace(',',''))/1000
    except: continue
    by[r[kn]][0]+=1; by[r[kn]][1]+=v
tot=sum(t for c,t in by.values()); n=sum(c for c,t in by.values())
lm=by.get('gemv_w8a8_fwd',[0,0])[0] or 1
print(f"{n} kernels, {tot/1000:.1f} ms total; lm-head launches (~tokens) = {lm}; per token: {n/lm:.0f} kernels, {tot/lm/1000:.2f} ms GPU")
for k,(c,t) in sorted(by.items(),key=lambda x:-x[1][1])[:12]: print(f"  {k:32s} n/tok={c/lm:6.1f}  us/tok={t/lm:8.1f}  {100*t/tot:5.1f}%  avg={t/c:7.1f}us")
