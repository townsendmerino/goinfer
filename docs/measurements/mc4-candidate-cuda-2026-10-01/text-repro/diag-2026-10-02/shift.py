import json, sys
solo=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2]))
out=[]
for k in ('21','0','2','9'):
    a,b=solo[k]['lp'][:6],c[k]['lp'][:6]
    d=max(abs(x[1]-y[1]) for x,y in zip(a,b))
    out.append(f"{k}:reused={c[k]['reused']} shift={d:.4f}")
print("  "+"  ".join(out))
