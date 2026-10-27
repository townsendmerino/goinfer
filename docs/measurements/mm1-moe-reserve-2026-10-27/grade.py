import re,bisect,collections,sys
def load(f):
    g={};raw={};cur=None
    for ln in open(f):
        if ln.startswith('## '): cur=ln[3:].strip(); g[cur]={}; raw[cur]=[]
        elif cur:
            raw[cur].append(ln)
            m=re.match(r'free (\d+): (\S+) ctx (\d+) slots (\d+)',ln)
            if m: g[cur][int(m.group(1))]=(m.group(2),int(m.group(3)))
            else:
                m=re.match(r'free (\d+): ctx (\d+) slots',ln)
                if m: g[cur][int(m.group(1))]=('metal',int(m.group(2)))
    return g,raw
a,ra=load(sys.argv[1]); b,rb=load(sys.argv[2])
moe=set()
for k,lines in ra.items():
    if k.startswith('D3 '):
        for l in lines:
            m=re.search(r'expertFull (\d+)',l)
            if m and int(m.group(1))>0: moe.add(k.split()[1]); break
fam=lambda k:k.split()[1] if len(k.split())>1 else ''
rank={'decline':0,'weight-paged':1,'expert-cached':2,'resident':3}
cls=lambda v:('floor' if v[1]<=4096 else 'above') if v[0]=='metal' else v[0]
def at(d,x):
    ks=sorted(d); i=bisect.bisect_right(ks,x)-1; return d[ks[max(i,0)]]
fails=[];changed=collections.Counter();moved=collections.Counter();mv=collections.Counter()
assert set(a)==set(b), 'group sets differ'
for k in a:
    if ra[k]==rb[k]: continue
    d=k.split()[0]; changed[d]+=1
    if d not in ('D3','D4'): fails.append(('changed outside D3/D4',k)); continue
    if fam(k) not in moe: fails.append(('non-MoE group changed',k)); continue
    xs=sorted(set(a[k])|set(b[k])); anym=False
    for x in xs:
        p,q=at(a[k],x),at(b[k],x)
        if cls(p)!=cls(q):
            anym=True; mv[f'{d} {cls(p)}->{cls(q)}']+=1
            if p[0]!='metal' and rank[q[0]]<rank[p[0]]: fails.append(('placement down',k,x,p,q))
            if p[0]=='metal' and cls(q)=='floor': fails.append(('metal fit to floor',k,x,p,q))
    for x in sorted(set(a[k])&set(b[k])):
        p,q=a[k][x],b[k][x]
        if cls(p)==cls(q) and p[0]!='decline' and q[1]<p[1]: fails.append(('context smaller',k,x,p,q))
    if anym: moved[d]+=1
print('changed groups',dict(changed),'with a placement/class move',dict(moved))
for m,n in sorted(mv.items()): print(' ',m,n)
print('failures',len(fails)); [print('  ',f) for f in fails[:10]]
d4=[k for k in a if k.startswith('D4') and ra[k]!=rb[k]]
print('D4 changed fixtures',len({fam(k) for k in d4}), 'lower values',collections.Counter(re.search(r'lower=(\d+)',k).group(1) for k in d4))
