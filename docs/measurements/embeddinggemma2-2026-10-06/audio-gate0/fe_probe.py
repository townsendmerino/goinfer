import numpy as np, transformers
from transformers import AutoFeatureExtractor
print(transformers.__version__, transformers.__file__)
fe = AutoFeatureExtractor.from_pretrained('/Users/francistownsend-merino/models/embeddinggemma-2')
print(type(fe).__name__, fe.frame_length, fe.hop_length, fe.fft_length, fe.window.dtype, fe.window.shape, fe.mel_filters.dtype, fe.mel_filters.shape, fe.mel_floor, fe.padding_side)
w=fe.window.astype(np.float64); n=np.arange(320); print('window max err vs formula', np.abs(w-(0.5-0.5*np.cos(2*np.pi*n/320))).max(), w[0], w[160])
mf=fe.mel_filters; print('all-zero filters', np.where(mf.max(axis=0)==0)[0], 'nonzero per filter first/last', (mf>0).sum(0)[:4], (mf>0).sum(0)[-4:])
rng=np.random.default_rng(0)
for N in [16000, 16001, 1000, 321, 160, 100]:
    x=(rng.standard_normal(N)*0.1).astype(np.float32)
    o=fe([x], return_tensors='np')  # batched path, as processor does
    feats,mask=o['input_features'],o['input_features_mask']
    nmel=(N+160-321)//160+1
    print(N,'feat',feats.shape,feats.dtype,'mask',mask.shape,mask.dtype,'valid',mask.sum(),'formula nmel',nmel, 'zero rows on invalid', np.all(feats[0][~mask[0]]==0))
# non-batched path
x=(rng.standard_normal(16000)*0.1).astype(np.float32)
try:
    o=fe(x, return_tensors='np'); print('nonbatched', o['input_features'].shape, o['input_features_mask'].sum())
except Exception as e: print('nonbatched ERR', e)
o2=fe([x], return_tensors='np'); 
try: print('nonbatched==batched', np.array_equal(o['input_features'],o2['input_features']))
except Exception as e: print(e)
# f64 input same as f32?
o3=fe([x.astype(np.float64)], return_tensors='np'); print('f64 input == f32 input', np.array_equal(o3['input_features'],o2['input_features']))
# truncation
x=np.zeros(500000,np.float32); o=fe([x],return_tensors='np'); print('500000 samples ->', o['input_features'].shape, o['input_features_mask'].sum())
# batch of 2 differing lengths
a=(rng.standard_normal(16000)*0.1).astype(np.float32); b=a[:8000]
o=fe([a,b],return_tensors='np'); ob=fe([b],return_tensors='np')
print('batch2 shape',o['input_features'].shape, o['input_features_mask'].sum(1), 'b alone', ob['input_features'].shape, ob['input_features_mask'].sum(), 'b valid rows equal', np.array_equal(o['input_features'][1][:ob['input_features_mask'][0].sum()], ob['input_features'][0][:ob['input_features_mask'][0].sum()]))
# reproduce in pure numpy w/o FE to pin formula
x=a
L=len(x); Lp=((L+127)//128)*128
xp=np.zeros(Lp,np.float32); xp[:L]=x; m=np.zeros(Lp,np.int32); m[:L]=1
xp=np.concatenate([np.zeros(160,np.float32),xp]); m=np.concatenate([np.zeros(160,np.int32),m])
nf=(len(xp)-321)//160+1
fr=np.stack([xp[i*160:i*160+320] for i in range(nf)])
win=(0.5-0.5*np.cos(2*np.pi*np.arange(320)/320)).astype(np.float32)
spec=np.abs(np.fft.rfft(fr*win,n=512))
lm=np.log(spec@mf+0.001).astype(np.float32)
vm=m[np.arange(nf)*160+320].astype(bool)
lm=lm*vm[:,None]
print('manual repro exact', np.array_equal(lm, o2['input_features'][0]) if lm.shape==o2['input_features'][0].shape else (lm.shape,o2['input_features'][0].shape), np.abs(lm-o2['input_features'][0]).max())
