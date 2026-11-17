import re,sys,collections
def load(p):
    g=collections.defaultdict(list)
    for l in open(p):
        grp,line=l.rstrip('\n').split(' || ')
        m=re.match(r'free (\d+): (\S+) ctx (\d+) slots (\d+) dense (\d+) slack (\d+)',line)
        g[grp].append((int(m.group(1)),(m.group(2),int(m.group(3)),int(m.group(4))),int(m.group(6))))
    return g
old,new=load(sys.argv[1]+'/after.txt'),load(sys.argv[2]+'/after.txt')
def trans(pts):
    pts=sorted(pts); out=[]; prev=None
    for f,o,s in pts:
        if o!=prev: out.append((f,o)); prev=o
    return out
same=diff=0; shifts=collections.Counter(); bad=[]
slack={}
for g in old:
    to,tn=trans(old[g]),trans(new[g])
    so=old[g][0][2]; sn=new[g][0][2]; slack[g.split(' | ')[0]]=(so,sn)
    if [o for _,o in to]==[o for _,o in tn]:
        same+=1
        shifts[g.split(' | ')[0].split()[1]]+=0
        for (fo,_),(fn,_) in zip(to[1:],tn[1:]):
            shifts[(g.split()[1], round((fn-fo)/2**20,2))]+=1
    else:
        diff+=1; bad.append((g,to,tn))
print('groups',len(old),'same outcome sequence',same,'different',diff)
for k,v in sorted(((k,v) for k,v in shifts.items() if isinstance(k,tuple)),key=lambda x:str(x)): print(k,v)
for g,to,tn in bad[:6]: print('DIFF',g,'\n old',to,'\n new',tn)
for m,(a,b) in sorted(slack.items()): print(m,'slack old',a,'new',b,'delta MiB',(b-a)/2**20)
