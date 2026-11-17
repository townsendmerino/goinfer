import re,sys
def norm(l):
    # strip the fields the packing term is meant to move
    l=re.sub(r'slack=\d+','slack=#',l)
    l=re.sub(r'(\w+)@\d+','\\1@#',l)
    return l
for g in ('sb2_cuda_census','sb2_cuda_mem_census'):
    o=open(f'{sys.argv[1]}/{g}.old.txt').read().splitlines(); n=open(f'{sys.argv[1]}/{g}.new.txt').read().splitlines()
    ch=[(a,b) for a,b in zip(o,n) if a!=b]
    rest=[(a,b) for a,b in ch if norm(a)!=norm(b)]
    print(g,'lines',len(o),len(n),'changed',len(ch),'changed beyond slack/kv@/prefill@ fields',len(rest))
    for a,b in rest[:5]: print(' OLD',a[:400],'\n NEW',b[:400])
