"""Write a towers-only safetensors from google/gemma-4-E4B-it-qat-q4_0-unquantized: every tensor under
model.vision_tower, model.embed_vision, model.audio_tower, model.embed_audio, bytes fetched by range, names, dtypes and
shapes unchanged; plus the repo's small config files. A manifest of per-tensor sha256 goes beside it."""
import json, struct, sys, os, hashlib, urllib.request, time
repo = 'https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-unquantized/resolve/main/'
tok = os.environ['HF_TOKEN']; out = sys.argv[1]; os.makedirs(out, exist_ok=True)
def get(path, rng=None):
    h = {'Authorization': 'Bearer ' + tok}
    if rng: h['Range'] = 'bytes=%d-%d' % rng
    with urllib.request.urlopen(urllib.request.Request(repo + path, headers=h), timeout=300) as r: return r.read()
for small in ('config.json', 'processor_config.json', 'generation_config.json'):
    open(os.path.join(out, small), 'wb').write(get(small))
rn = struct.unpack('<Q', get('model.safetensors', (0, 7)))[0]
rh = json.loads(get('model.safetensors', (8, 8 + rn - 1))); meta = rh.pop('__metadata__', None)
pre = ('model.audio_tower', 'model.embed_audio', 'model.embed_vision', 'model.vision_tower')
names = sorted([k for k in rh if k.startswith(pre)], key=lambda k: rh[k]['data_offsets'][0])
hdr = {}; off = 0
for k in names:
    n = rh[k]['data_offsets'][1] - rh[k]['data_offsets'][0]
    hdr[k] = {'dtype': rh[k]['dtype'], 'shape': rh[k]['shape'], 'data_offsets': [off, off + n]}; off += n
if meta: hdr['__metadata__'] = meta
hb = json.dumps(hdr, separators=(',', ':')).encode(); hb += b' ' * ((8 - len(hb) % 8) % 8)
chunks = []; cur = None
for k in names:
    a, b = rh[k]['data_offsets']
    if cur and a == cur[1] and b - cur[0] < (96 << 20): cur[1] = b; cur[2].append(k)
    else: cur = [a, b, [k]]; chunks.append(cur)
man = {}; t0 = time.time(); total = 0
with open(os.path.join(out, 'model.safetensors.part'), 'wb') as f:
    f.write(struct.pack('<Q', len(hb))); f.write(hb)
    for ci, (a, b, ks) in enumerate(chunks):
        buf = get('model.safetensors', (8 + rn + a, 8 + rn + b - 1)); assert len(buf) == b - a
        for k in ks:
            ra, rb = rh[k]['data_offsets']; x = buf[ra - a: rb - a]; f.write(x); man[k] = hashlib.sha256(x).hexdigest(); total += len(x)
        print(f'chunk {ci + 1}/{len(chunks)}: {total / 1e6:.0f} MB, {time.time() - t0:.0f} s', flush=True)
os.rename(os.path.join(out, 'model.safetensors.part'), os.path.join(out, 'model.safetensors'))
json.dump({'source': 'google/gemma-4-E4B-it-qat-q4_0-unquantized model.safetensors', 'tensors': len(names), 'bytes': total, 'sha256': man},
          open(os.path.join(out, 'TOWERS-MANIFEST.json'), 'w'), indent=0)
print('wrote', len(names), 'tensors', total, 'bytes')
