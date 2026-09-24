import json,sys
G=sys.argv[1]
def rows(p):
    out=[]
    for t in json.load(open(p))["turns"]:
        for k in sorted(t["cells"]):
            for i,r in enumerate(t["cells"][k]):
                x=json.loads(json.dumps(r["raw"]))
                for c in (x.get("tool_calls") or []): c.pop("id",None)
                out.append(((t["n"],k,i),json.dumps(x,sort_keys=True),r["cls"]))
    return out
for m in sys.argv[2:]:
    a=rows(f"{G}/A-{m}-on.json"); b=rows(f"{G}/A-{m}-off.json")
    diff=[(ka,ca,cb) for (ka,xa,ca),(kb,xb,cb) in zip(a,b) if xa!=xb]
    print(m,f"{len(a)-len(diff)}/{len(a)} byte-identical", "| differing:",diff[:5])
