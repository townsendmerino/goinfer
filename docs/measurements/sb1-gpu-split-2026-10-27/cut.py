import collections,sys,json
DECL,XREF=sys.argv[1],sys.argv[2]
D={}; F={}
for l in open(DECL):
    n,f,dec,us=(l.rstrip("\n").split("\t")+[""])[:4]
    D[n]=(dec=="true", set(us.split())); F[n]=f
def ismeth(n): return "." in n and not n.startswith(("init@","_@"))
taint={n for n,(d,_) in D.items() if d}
while True:
    new=set()
    for n,(d,us) in D.items():
        if n not in taint and us & taint: new.add(n)
    cur=taint|new
    for n in list(cur):
        if ismeth(n):
            t,m=n.split(".",1)
            if m[:1].isupper() and t in D: new.add(t)
    for n in D:
        if ismeth(n) and n.split(".")[0] in cur: new.add(n)
    new-=taint
    if not new: break
    taint|=new
files=collections.defaultdict(set)
for n,f in F.items(): files[f].add(n)
stay={f for f,ns in files.items() if ns & taint}|{"doc.go"}
move=set(files)-stay
print("taint decls:",len(taint),"| stay files:",len(stay),"(tests",sum(f.endswith('_test.go') for f in stay),") move files:",len(move),"(tests",sum(f.endswith('_test.go') for f in move),")")
print("nontest stay:",sorted(f for f in stay if not f.endswith('_test.go')))
bad=collections.defaultdict(set)
for l in open(XREF):
    uf,what,df=l.rstrip("\n").split("\t")
    if uf in move and df in stay: bad[uf].add(what+" @"+df)
print("MOVE->STAY edges:",sum(len(v) for v in bad.values()))
for f,v in sorted(bad.items()): print(" ",f,sorted(v)[:8])
json.dump({"stay":sorted(stay),"move":sorted(move)},open("gpu-cut.json","w"))
