import json, struct, sys, os, hashlib, urllib.request, collections
local = os.path.expanduser('~/models/gemma-4-E4B-it/model.safetensors')
url = 'https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-unquantized/resolve/main/model.safetensors'
tok = os.environ['HF_TOKEN']
def rng(a, b):
    req = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + tok, 'Range': f'bytes={a}-{b}'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()
def lhead():
    with open(local, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return n, json.loads(f.read(n))
ln, lh = lhead()
rn = struct.unpack('<Q', rng(0, 7))[0]
rh = json.loads(rng(8, 8 + rn - 1))
lh.pop('__metadata__', None); rh.pop('__metadata__', None)
def top(n): return n.split('.')[0] if not n.startswith('model.') else '.'.join(n.split('.')[:2])
for name, h in (('local plain', lh), ('remote QAT-unquantized', rh)):
    c = collections.Counter(); b = collections.Counter()
    for k, v in h.items():
        c[top(k)] += 1; b[top(k)] += v['data_offsets'][1] - v['data_offsets'][0]
    print(name, 'tensors', len(h))
    for k in sorted(c): print('   %-34s %5d tensors %8.1f MB' % (k, c[k], b[k] / 1e6))
json.dump({'ln': ln, 'rn': rn, 'lh': lh, 'rh': rh}, open(sys.argv[1], 'w'))
