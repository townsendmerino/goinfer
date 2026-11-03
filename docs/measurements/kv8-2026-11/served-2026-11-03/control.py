#!/usr/bin/env python3
# The diagnosis of the served reading's FAIL: each client ALONE on one slot, so nothing about a second slot or another client is in it.
import importlib.util,sys,json,os,time,types
sp=importlib.util.spec_from_file_location('h','docs/measurements/kv8-2026-11/served_two_clients.py'); h=importlib.util.module_from_spec(sp); sp.loader.exec_module(h)
a=types.SimpleNamespace(serve=sys.argv[1],model=sys.argv[2])
out=sys.argv[3]; res=[]
for label,names in (("alone-A","A"),("alone-B","B")):
    s=h.Server(a,1,os.path.join(out,f"serve-control-{label}.log"))
    name=names; p=h.preamble({"A":"amber","B":"birch"}[name])
    for r in range(h.ROUNDS):
        text,ttft,u=h.turn(p)
        print(f"{label} round {r}: {u.get('prompt_tokens')} tokens, reused {u.get('prefill_reused_tokens')}, first token {ttft:.2f} s, reply {text!r}",flush=True)
        res.append({"cell":label,"round":r,"client":name,"reused":u.get('prefill_reused_tokens'),"ttft_s":ttft,"prompt_tokens":u.get('prompt_tokens'),"reply":text})
        p+=text+f"\nQuestion {r}: which shelf next?\nAnswer:"
    s.close(); time.sleep(3)
json.dump(res,open(os.path.join(out,"control.json"),"w"),indent=0)
