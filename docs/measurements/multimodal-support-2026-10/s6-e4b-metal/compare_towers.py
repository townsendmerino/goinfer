import json, sys, os, hashlib, urllib.request, time
d = json.load(open(sys.argv[1])); ln, rn, lh, rh = d['ln'], d['rn'], d['lh'], d['rh']
local = os.path.expanduser('~/models/gemma-4-E4B-it/model.safetensors')
url = 'https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-unquantized/resolve/main/model.safetensors'
tok = os.environ['HF_TOKEN']
pre = ('model.audio_tower', 'model.embed_audio', 'model.embed_vision', 'model.vision_tower')
names = sorted([k for k in rh if k.startswith(pre)], key=lambda k: rh[k]['data_offsets'][0])
missing = [k for k in names if k not in lh] + [k for k in lh if k.startswith(pre) and k not in rh]
meta_diff = [k for k in names if k in lh and (lh[k]['dtype'], lh[k]['shape']) != (rh[k]['dtype'], rh[k]['shape'])]
print('tower tensors', len(names), 'missing on one side', len(missing), 'dtype/shape differ', len(meta_diff))
# coalesce the remote ranges into chunks
chunks = []; cur = None
for k in names:
    a, b = rh[k]['data_offsets']
    if cur and a - cur[1] < (1 << 20) and b - cur[0] < (96 << 20): cur[1] = b; cur[2].append(k)
    else:
        cur = [a, b, [k]]; chunks.append(cur)
same = diff = 0; total = 0; t0 = time.time(); bad = []
lf = open(local, 'rb')
for ci, (a, b, ks) in enumerate(chunks):
    req = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + tok, 'Range': f'bytes={8 + rn + a}-{8 + rn + b - 1}'})
    with urllib.request.urlopen(req, timeout=300) as r: buf = r.read()
    assert len(buf) == b - a, (len(buf), b - a)
    total += len(buf)
    for k in ks:
        ra, rb = rh[k]['data_offsets']; rbytes = buf[ra - a: rb - a]
        if k not in lh: continue
        la, lb = lh[k]['data_offsets']; lf.seek(8 + ln + la); lbytes = lf.read(lb - la)
        if rbytes == lbytes: same += 1
        else: diff += 1; bad.append(k)
    if ci % 5 == 0 or ci == len(chunks) - 1:
        print(f'chunk {ci + 1}/{len(chunks)}: {total / 1e6:.0f} MB read, {same} tensors identical, {diff} different, {time.time() - t0:.0f} s', flush=True)
print('RESULT tower tensors byte-identical', same, 'different', diff, 'of', len(names))
by = {}
for k in bad: by['.'.join(k.split('.')[:2])] = by.get('.'.join(k.split('.')[:2]), 0) + 1
print('different by group', by); print('first different', bad[:5])
