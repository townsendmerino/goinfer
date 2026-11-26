#!/usr/bin/env python3
"""LS6 (aikit's typed WeightMat.Kind): add .String() where goinfer uses Kind() as a plain string.

Usage, from the module directory: python3 scripts/aikit_surface/kind_string_fix.py go vet -tags "<tags>" ./...
Run it once per module and tag set; a comparison with a string constant compiles unchanged and is not touched.
"""
import re,subprocess,sys,os
# Repeatedly: run the given go command; at each file:line whose error names linalg.Kind, add .String() after the line's
# .Kind() calls (or after `k` in a string concatenation). Stops when no such error remains.
done=set()
for it in range(60):
    p=subprocess.run(sys.argv[1:],capture_output=True,text=True)
    out=p.stderr+p.stdout
    sites=[]
    for m in re.finditer(r'^(?:vet: )?((?:\.\./)*[\w/.\-]+\.go):(\d+):(\d+): ([^\n]*linalg\.Kind[^\n]*)$', out, re.M):
        sites.append((os.path.normpath(m.group(1)),int(m.group(2)),m.group(4)))
    sites=[s for s in sites if (s[0],s[1]) not in done]
    if not sites: break
    for f,l,msg in sites:
        done.add((f,l))
        lines=open(f).read().split('\n'); ln=lines[l-1]
        n=re.sub(r'\.Kind\(\)(?!\.String\(\))', '.Kind().String()', ln)
        if n==ln and 'string type linalg.Kind' in msg:
            # A Kind held in a variable, used in a string concatenation: "..." + k + "...".
            n=re.sub(r'(?<=\+ )([A-Za-z_]\w*)(?= \+|\b(?=[,})\]]))(?![.(\w])', r'\1.String()', ln)
        if n==ln:
            print('UNHANDLED',f,l,ln.strip(),'|',msg); continue
        lines[l-1]=n; open(f,'w').write('\n'.join(lines)); print('fixed',f+':'+str(l))
print('remaining errors:', len(re.findall(r'linalg\.Kind', subprocess.run(sys.argv[1:],capture_output=True,text=True).stderr)))
