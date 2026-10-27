import collections,re,bisect,sys,json
def groups(f):
    g={};raw={};cur=None
    for ln in open(f):
        if ln.startswith('## '):
            cur=ln[3:].strip(); g[cur]={}; raw[cur]=[]
        elif cur:
            raw[cur].append(ln)
            m=re.match(r'free (\d+): (\S+) ctx (\d+) slots (\d+)',ln)
            if m: g[cur][int(m.group(1))]=(m.group(2),int(m.group(3)),int(m.group(4)))
            else:
                m=re.match(r'free (\d+): ctx (\d+) slots (\d+)',ln)
                if m: g[cur][int(m.group(1))]=('metal',int(m.group(2)),int(m.group(3)))
    return g,raw
a,ra=groups(sys.argv[1]); b,rb=groups(sys.argv[2])
rank={'decline':0,'weight-paged':1,'expert-cached':2,'resident':3}
def cls(k,v):
    if v[0]!='metal': return v[0]
    fl=4096; cand=int(re.search(r'lower=(\d+)',k).group(1)) if 'lower=' in k else 0
    return 'floor' if v[1]<=fl else 'above-floor'
def at(d,x):
    ks=sorted(d); i=bisect.bisect_right(ks,x)-1; return d[ks[max(i,0)]]
changed=[k for k in a if ra[k]!=rb.get(k)]
rows=[];tot=collections.Counter();down=[]
for k in changed:
    mv=collections.Counter(); cx=collections.Counter()
    xs=sorted(set(a[k])|set(b[k]))
    for x in xs:   # placement (outcome class): change points are complete under the census's cap
        p,q=at(a[k],x),at(b[k],x)
        if cls(k,p)!=cls(k,q):
            mv[f'{cls(k,p)}->{cls(k,q)}']+=1
            if p[0]!='metal' and rank[q[0]]<rank[p[0]]: down.append((k,x,p,q))
            if p[0]=='metal' and cls(k,q)=='floor': down.append((k,x,p,q))
    for x in sorted(set(a[k])&set(b[k])):  # context: only where both record a reading
        p,q=a[k][x],b[k][x]
        if cls(k,p)==cls(k,q) and p[1]!=q[1]: cx['ctx '+('down' if q[1]<p[1] else 'up')+' at '+cls(k,p)]+=1
    tot.update(mv); tot.update(cx)
    rows.append((k,dict(mv),dict(cx)))
moved=[r for r in rows if r[1] or r[2]]
print('groups',len(a),'changed',len(changed),'with a placement/class or common-reading ctx move',len(moved))
print('by decision',collections.Counter(k.split()[0] for k,_,_ in moved),'all changed',collections.Counter(k.split()[0] for k in changed))
for m,n in tot.most_common(): print(' ',m,n)
print('downward class moves',len(down)); [print('  ',d) for d in down[:5]]
json.dump(rows,open(sys.argv[3],'w'))
