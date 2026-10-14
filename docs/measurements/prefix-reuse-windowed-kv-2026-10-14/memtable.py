# Bytes a windowed CUDA layer holds per KV slot: 2 (K and V) x kvDim x 4 (windowed layers stay f32: cuda/kvpolicy.go) x min(ctx, window+slack)
# positions (cuda/resident.go:kvBytesForCap). Geometry from each checkpoint's config.json in ~/models; KV-shared layers hold nothing of their own.
import json, os
M = os.path.expanduser('~/models')
def text(d):
    c = json.load(open(os.path.join(M, d, 'config.json')))
    return c.get('text_config', c)
rows = []
for d in ['mellum2.1-thinking', 'gemma-2-2b-it', 'gemma-3-4b-it', 'gemma-4-E2B-unq', 'gemma-4-E4B-unq', 'gemma-4-26b-a4b-it', 'gemma-4-31B-it', 'command-r7b', 'gpt-oss-20b-hf', 'phi3-mini-4k']:
    c = text(d); mt = c['model_type']; L = c['num_hidden_layers']; W = c['sliding_window']
    nkv = c.get('num_key_value_heads') or (4 if mt == 'gemma3_text' else c['num_attention_heads'])
    hd = c.get('head_dim') or (256 if mt == 'gemma3_text' else c['hidden_size'] // c['num_attention_heads'])
    own = L - (c.get('num_kv_shared_layers') or 0)
    if c.get('layer_types'):
        nwin = sum(1 for t in c['layer_types'][:own] if 'sliding' in t)
    elif mt == 'gemma2':
        nwin = L // 2
    elif mt == 'gemma3_text':
        nwin = L - L // 6
    elif mt == 'cohere2':
        nwin = L - L // c['sliding_window_pattern']
    else:
        nwin = L
    pp = 2 * nkv * hd * 4
    rows.append((d, mt, W, nwin, L, nkv * hd, pp))
rows.append(('Mistral-7B-v0.1 (GGUF; geometry from its header)', 'mistral', 4096, 32, 32, 1024, 2 * 1024 * 4))
print('| checkpoint | family | window W | windowed layers (own KV) / all | kvDim | bytes per position per layer | per layer per slot, W+512 (before) | per layer per slot, W+1536 (after) | added, all windowed layers, per slot |')
print('|---|---|---|---|---|---|---|---|---|')
for d, mt, W, nwin, L, kvdim, pp in rows:
    a, b = pp * (W + 512), pp * (W + 1536)
    print('| %s | %s | %d | %d / %d | %d | %d | %.1f MiB | %.1f MiB | +%.0f MiB |' % (d, mt, W, nwin, L, kvdim, pp, a / 2**20, b / 2**20, nwin * (b - a) / 2**20))
