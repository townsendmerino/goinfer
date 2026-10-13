import json,sys,glob,os
for f in sorted(glob.glob(os.path.join(sys.argv[1],'req-*.json')), key=lambda p:int(p.split('-')[-1].split('.')[0])):
    r=json.load(open(f)); b=r['body']
    if not isinstance(b,dict): print(os.path.basename(f), r['method'], r['path'], '(no JSON body)'); continue
    roles=[m.get('role') for m in b.get('messages') or []]
    sysf=b.get('system'); tools=b.get('tools') or []
    print(f"{os.path.basename(f)} {r['method']} {r['path']} model={b.get('model')} max_tokens={b.get('max_tokens')} tools={len(tools)} "
          f"system={len(json.dumps(sysf)) if sysf else 0}ch tools={len(json.dumps(tools))}ch roles={roles} "
          f"keys={sorted(k for k in b if k not in ('messages','system','tools','model','max_tokens','stream'))}")
    print('   beta:', r['headers'].get('anthropic-beta'))
