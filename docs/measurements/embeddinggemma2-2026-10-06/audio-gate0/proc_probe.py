import numpy as np
from transformers import AutoProcessor
p=AutoProcessor.from_pretrained('/Users/francistownsend-merino/models/embeddinggemma-2')
print(type(p).__name__, 'audio_seq_length', p.audio_seq_length, 'ms/token', p.audio_ms_per_token, p.audio_token, p.boa_token, p.eoa_token, p.audio_token_id)
rng=np.random.default_rng(0)
for N in [16000, 8000, 16000*15, 16000*40]:
    x=(rng.standard_normal(N)*0.1).astype(np.float32)
    o=p(audio=[x], return_tensors='np')
    ids=o['input_ids'][0]
    n_aud=(ids==258881).sum()
    print(N, 'keys', sorted(o.keys()), 'ids len', len(ids), 'head', ids[:3].tolist(), 'tail', ids[-3:].tolist(), 'n_audio', n_aud, 'feat', o['input_features'].shape, 'compute_num_tokens', p._compute_audio_num_tokens(np.zeros(N),16000))
x=(rng.standard_normal(16000)*0.1).astype(np.float32)
o=p(text=["hello <|audio|> world"], audio=[x], return_tensors='np'); ids=o['input_ids'][0]
print('text+audio ids head', ids[:4].tolist(), 'around', ids[-5:].tolist(), 'mm_token_type_ids' in o and np.unique(o['mm_token_type_ids']).tolist())
print(p.tokenizer.decode(ids[:3]), '|', p.tokenizer.decode(ids[-4:]))
