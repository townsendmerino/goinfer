import numpy as np
from transformers import AutoFeatureExtractor
fe = AutoFeatureExtractor.from_pretrained('/Users/francistownsend-merino/models/embeddinggemma-2')
mf=fe.mel_filters
rng=np.random.default_rng(1)
def manual(x, win_f32=True, prod_f32=True):
    L=len(x); Lp=((L+127)//128)*128
    xp=np.zeros(Lp,np.float32); xp[:L]=x; m=np.zeros(Lp,np.int32); m[:L]=1
    xp=np.concatenate([np.zeros(160,np.float32),xp]); m=np.concatenate([np.zeros(160,np.int32),m])
    nf=(len(xp)-321)//160+1
    fr=np.stack([xp[i*160:i*160+320] for i in range(nf)])
    win=(0.5-0.5*np.cos(2*np.pi*np.arange(320)/320))
    if win_f32: win=win.astype(np.float32)
    p = fr*win if prod_f32 else fr.astype(np.float64)*win.astype(np.float64)
    spec=np.abs(np.fft.rfft(p,n=512))
    lm=np.log(spec@mf+0.001).astype(np.float32)
    vm=m[np.arange(nf)*160+320].astype(bool)
    return lm*vm[:,None], vm
for N in [16000, 12345, 50000]:
    x=(rng.standard_normal(N)*0.1).astype(np.float32)
    o=fe([x],return_tensors='np'); ref=o['input_features'][0]
    for wf,pf in [(True,True),(False,False),(True,False)]:
        lm,vm=manual(x,wf,pf)
        print(N, 'win_f32',wf,'prod_f32',pf,'exact',np.array_equal(lm,ref),'maxdiff',np.abs(lm-ref).max(), 'mask eq', np.array_equal(vm,o['input_features_mask'][0]))
print('bin0 value on valid frame', ref[0,0], np.log(0.001))
# mel filter centers
from transformers.audio_utils import hertz_to_mel, mel_to_hertz
mels=np.linspace(hertz_to_mel(0.0),hertz_to_mel(8000.0),130); ff=mel_to_hertz(mels)
print('filter edge freqs first 4', ff[:4], 'last 3', ff[-3:])
print('fft freqs', np.linspace(0,8000,257)[:3])
