# Gate 0 desk spec: `gemma4_audio` tower → pure Go (EmbeddingGemma 2 first, Gemma 4 E2B/E4B second)

Date 2026-10-06. Read-only research (a research agent's desk read for Phase A of `docs/tasks/task-embeddinggemma2.md`; the probe scripts beside it are what it ran). The reference is **transformers 5.19.0** as installed in the
embeddinggemma2-2026-10-06 venv. The checkpoint's `config.json` says it was written by `5.18.0.dev0`.
Checkpoint: `~/models/embeddinggemma-2`, `REVISION` = `914f7f89142e33e77833254d9c9b90c3cef7303b`.
Nothing was read from `/Volumes/` or `/srv/models`.

Path abbreviations:
- `TF/` = `/Users/francistownsend-merino/goinfer-bench/embeddinggemma2-2026-10-06/venv/lib/python3.12/site-packages/transformers/`
- `MG4` = `TF/models/gemma4/modeling_gemma4.py`
- `FE4` = `TF/models/gemma4/feature_extraction_gemma4.py`
- `PG4` = `TF/models/gemma4/processing_gemma4.py`
- `MEG2` = `TF/models/embedding_gemma2/modeling_embedding_gemma2.py`
- `PEG2` = `TF/models/embedding_gemma2/processing_embedding_gemma2.py`
- `AU` = `TF/audio_utils.py`
- `SEQ` = `TF/feature_extraction_sequence_utils.py`
- `CKPT` = `~/models/embeddinggemma-2/`

**How this was checked.** Besides reading the code, I ran three kinds of probe (scripts in this directory):
1. Feature-extractor probes (`fe_probe*.py`). A plain NumPy re-derivation of the log-mel matches the HF extractor
   **bit for bit** (`np.array_equal` True at 16000, 12345 and 50000 samples).
2. Tiny random-weight tower probes (`model_probe*.py`): masks, the relative shift, padding, and the eager/sdpa split.
3. `spec_reimpl.py`. It is an independent f64 re-implementation of **this spec**: direct sliding-window attention,
   no blocking, run on only the valid frames. It loads the **real audio-tower tensors only** (the 752 `*audio*`
   tensors, about 305M params; the whole checkpoint file is 1.49 GB, not 10 GB). Against HF in f32 it matches to
   max |Δ| 5.6e-4 on the 1536-d tower output and 1.1e-4 on the 512-d embedder output (max |x| 5.7), for a 2.37 s
   chirp (59 tokens). It matches to 4e-5 / 8e-6 for a 0.37 s one (9 tokens). Nothing else was loaded: no text model,
   no full `from_pretrained`.

Anything marked **INFERRED** was not read directly.

---

## 0. Checkpoint facts (`CKPT/config.json`, safetensors header)

`audio_config` (`CKPT/config.json`):

| field | value |
|---|---|
| `model_type` | `gemma4_audio` |
| `hidden_size` | 1024 |
| `num_hidden_layers` | 12 |
| `num_attention_heads` | 8 (so head_dim = 128) |
| `hidden_act` | `silu` |
| `subsampling_conv_channels` | [128, 32] |
| `conv_kernel_size` | 5 |
| `residual_weight` | 0.5 |
| `attention_chunk_size` | 12 |
| `attention_context_left` | 13 |
| `attention_context_right` | 0 |
| `attention_logit_cap` | 50.0 |
| `attention_invalid_logits_value` | -1e9 |
| `gradient_clipping` | 1e10 |
| `use_clipped_linears` | true |
| `rms_norm_eps` | 1e-6 |
| `output_proj_dims` | 1536 |
| `dtype` | bfloat16 |

Config defaults (`TF/models/gemma4/configuration_gemma4.py:53-75`) agree with these. The docstring at :43 says
`attention_invalid_logits_value` defaults to `1e-9`, but the code at :70 is `-1.0e9`.

Top-level fields: `text_config.hidden_size` = **512**, `audio_token_id` = 258881, `boa_token_id` = 256000. The end
token is stored as **`eoa_token_index`** = 258883, not `eoa_token_id` (the field is declared at
`TF/models/embedding_gemma2/configuration_embedding_gemma2.py:174`).

**Safetensors header.** The header is 171,296 bytes, with metadata `{"format":"pt"}`. **All 1,376 tensors are BF16**,
and 752 of them have "audio" in their name:
- 12 layers × 62 tensors, plus 5 subsample tensors, plus 2 `output_proj` tensors, plus 1 `embed_audio` tensor.
- Layers 1–11 have exactly the same shapes as layer 0.

| tensor | dtype | shape |
|---|---|---|
| `audio_tower.subsample_conv_projection.layer0.conv.weight` | BF16 | [128, 1, 3, 3] |
| `audio_tower.subsample_conv_projection.layer0.norm.weight` | BF16 | [128] |
| `audio_tower.subsample_conv_projection.layer1.conv.weight` | BF16 | [32, 128, 3, 3] |
| `audio_tower.subsample_conv_projection.layer1.norm.weight` | BF16 | [32] |
| `audio_tower.subsample_conv_projection.input_proj_linear.weight` | BF16 | [1024, 1024] |
| `audio_tower.layers.L.feed_forward{1,2}.pre_layer_norm.weight` | BF16 | [1024] |
| `audio_tower.layers.L.feed_forward{1,2}.ffw_layer_1.linear.weight` | BF16 | [4096, 1024] |
| `audio_tower.layers.L.feed_forward{1,2}.ffw_layer_2.linear.weight` | BF16 | [1024, 4096] |
| `audio_tower.layers.L.feed_forward{1,2}.post_layer_norm.weight` | BF16 | [1024] |
| `audio_tower.layers.L.norm_pre_attn.weight` / `norm_post_attn.weight` / `norm_out.weight` | BF16 | [1024] |
| `audio_tower.layers.L.self_attn.{q,k,v}_proj.linear.weight` | BF16 | [1024, 1024] |
| `audio_tower.layers.L.self_attn.post.linear.weight` | BF16 | [1024, 1024] |
| `audio_tower.layers.L.self_attn.relative_k_proj.weight` (**no clip bounds, no `.linear.`**) | BF16 | [1024, 1024] |
| `audio_tower.layers.L.self_attn.per_dim_scale` | BF16 | [128] |
| `audio_tower.layers.L.lconv1d.pre_layer_norm.weight` | BF16 | [1024] |
| `audio_tower.layers.L.lconv1d.linear_start.linear.weight` | BF16 | [2048, 1024] |
| `audio_tower.layers.L.lconv1d.depthwise_conv1d.weight` | BF16 | [1024, 1, 5] |
| `audio_tower.layers.L.lconv1d.conv_norm.weight` | BF16 | [1024] |
| `audio_tower.layers.L.lconv1d.linear_end.linear.weight` | BF16 | [1024, 1024] |
| `audio_tower.layers.L.<clippable>.{input_min,input_max,output_min,output_max}` (10 clippables/layer) | BF16 | [] (scalar) |
| `audio_tower.output_proj.weight` | BF16 | [1536, 1024] |
| `audio_tower.output_proj.bias` | BF16 | [1536] |
| `embed_audio.embedding_projection.weight` | BF16 | [512, 1536] |

Each layer has 10 clippable linears, so 120 in the tower:
- `feed_forward1.ffw_layer_1`, `feed_forward1.ffw_layer_2`, `feed_forward2.ffw_layer_1`, `feed_forward2.ffw_layer_2`
- `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.post`
- `lconv1d.linear_start`, `lconv1d.linear_end`

There is **no `embed_audio.embedding_pre_projection_norm.weight`**. That is correct: the norm is `with_scale=False`
(§2.6). There is no `rel_pos_enc` tensor either, because it is a non-persistent buffer.

Sample values, decoded from BF16:
- Clip bounds are **asymmetric and finite**. Layer 0 `ffw_layer_1` input is [-12.875, 12.8125]. Layer 0 `self_attn.post`
  output is [-101.5, 100.5]. Across the tower, `input_max` ranges over [5.78, 32.25] and `output_max` over [6.19, 211.0].
- Layer 0 `q_proj`, `k_proj` and `v_proj` have identical bounds: [-20.375, 20.25] in, [-34.5, 34.25] out. Read each
  one separately anyway.
- `per_dim_scale` values are around -2 to -4, so softplus(per_dim_scale) ≈ 0.02–0.12.
- RMSNorm weights are around 0.8–5. These are **direct scales, not (1+w)**.
- `output_proj.bias` is not small (one entry is 8.25).

---

## 1. Feature extractor: waveform → log-mel (`Gemma4AudioFeatureExtractor`)

The class is at `FE4:49`. `CKPT/preprocessor_config.json` sets `feature_size` 128, `sampling_rate` 16000,
`frame_length` 320, `hop_length` 160, `fft_length` 512, `min_frequency` 0, `max_frequency` 8000, `mel_floor` 0.001,
`preemphasis` 0.0, `preemphasis_htk_flavor` true, `dither` 0.0, `input_scale_factor` 1.0, `per_bin_mean` and
`per_bin_stddev` null, `padding_side` right, `padding_value` 0.0, `fft_overdrive` false.

**Where the sizes come from.** `__init__` takes `frame_length_ms` and `hop_length_ms` (20 and 10 by default) and
recomputes the sizes as `int(round(sr*ms/1000))`, giving 320 and 160 (`FE4:130-131`). It also recomputes
`fft_length = 2**ceil(log2(320))` = 512 (`FE4:134-137`). The `frame_length`/`hop_length`/`fft_length` keys in the JSON
are therefore overwritten by values that happen to be equal. The probe confirms the instance has 320, 160 and 512.

The steps, in order (`__call__`, `FE4:226-294`; `_extract_spectrogram`, `FE4:169-224`):

0. **Input.** The input is mono float at **16 kHz**. Nothing resamples or checks it: `sampling_rate` passed to
   `__call__` is swallowed by `**kwargs` (`FE4:235`). It must be a **list** of 1-D arrays, which is what the processor
   passes (`TF/processing_utils.py:741`, `make_list_of_audio`).
   - A bare 1-D `np.ndarray` takes the non-batched branch (`FE4:270-271`), which produces shape (1,N). That raises
     `cannot select an axis to squeeze out` (measured).
   - `SequenceFeatureExtractor.pad` casts float64 to **float32** (`SEQ:218-219`). f64 and f32 inputs therefore give
     identical features (measured).
1. **Truncate and pad** (`SEQ:179-221`, called from `FE4:273-280`). The defaults are `padding="longest"`,
   `max_length=480000`, `truncation=True` and `pad_to_multiple_of=128` (`FE4:229-232`).
   - Truncation to 480,000 samples (30 s) happens first (`SEQ:296-335`; 480000 % 128 = 0).
   - The batch is then right-padded with 0.0 to the longest length, rounded **up to a multiple of 128 samples**
     (`SEQ:263-264`). A sample `attention_mask` of int32 1/0 is built (`SEQ:268-277`).
   - The padding adds extra mel frames, but they are all masked (step 7) and stripped later (§3).
2. **Dither and scale.** Both are off: `dither` is 0 (`FE4:173-174`) and `input_scale_factor` is 1.0 (`FE4:176-177`).
3. **Semicausal left pad.** `frame_length//2 = 160` zeros are prepended to the waveform, and 0s to the sample mask
   (`FE4:181-183`). There is **no right/center padding, no reflect padding**.
4. **Framing.** The waveform is unfolded with size **321** (`frame_length+1`) and step 160 (`FE4:185-188`).
   `num_frames = (L_padded + 160 - 321)//160 + 1` (`FE4:38`), where `L_padded` is the length after the 128-multiple
   padding.
   - With preemphasis 0, the frame is `frames_to_process[..., :-1]`: the first 320 samples (`FE4:197-198`). The 321st
     sample exists only for the preemphasis branches (`FE4:190-196`), which are unused here.
5. **Window.** Periodic Hann: `np.hanning(321)[:-1]`, i.e. `w[n] = 0.5 - 0.5*cos(2πn/320)` for n = 0..319
   (`AU:884,891,898`). It is cast to **float32** (`FE4:141`).
   - The product `frames*window` is computed in **float32** (f32 × f32, `FE4:202`).
   - **To be bit-exact this has to be an f32 multiply.** Doing it in f64 gives max |Δ| 1–4e-6 on the log-mel (measured).
6. **FFT and magnitude.** `np.fft.rfft(frames, n=512)` zero-pads the 320 samples on the right to 512 (`FE4:203`).
   NumPy computes the rfft in **float64/complex128**. `magnitude = |stft|` gives 257 bins (`FE4:205`). This is
   magnitude, **not power**: no squaring.
7. **Mel.** `mel = magnitude @ mel_filters`, where `mel_filters` is [257,128] float64 (`FE4:207`).
   `log_mel = ln(mel + 0.001)`: an additive floor, natural log, float64 (`FE4:208`). It is **not**
   `log(max(mel, floor))`. Per-bin mean and stddev are skipped because both are null (`FE4:210-214`). The result is
   cast to **float32** (`FE4:286`).
8. **Mel filterbank** (`FE4:149-157`, `AU:744-829`). The scale is HTK: `mel = 2595*log10(1+f/700)` and
   `f = 700*(10^(mel/2595)-1)` (`AU:572,608`). There is **no norm** (`norm=None`) and `triangularize_in_mel_space`
   is False.
   - `mel_freqs = linspace(mel(0), mel(8000), 130)`; `filter_freqs = hz(mel_freqs)` (`AU:809-810`).
   - `fft_freqs = linspace(0, 8000, 257)`, i.e. 31.25 Hz spacing (`AU:819`).
   - The triangle for filter m is `max(0, min((f - f_m)/(f_{m+1}-f_m), (f_{m+2}-f)/(f_{m+2}-f_{m+1})))`
     (`AU:647-666`).
   - **Filter 0 is all zeros** (measured: `where(max==0) → [0]`). Its band [0, 27.9] Hz contains only FFT bin 0
     (0 Hz), where the up-slope is 0. Mel bin 0 is therefore always exactly `ln(0.001)` = -6.9077554 on valid frames.
     The code comment at `FE4:144-146` blames the "uppermost" filter, which is wrong: it is the lowest one.
9. **Frame mask** (`FE4:219-223`). Frame i is valid iff the *padded-domain* sample at index `i*160 + 320` is real,
   i.e. original sample index `i*160 + 160` < N.
   - With N the real (pre-padding) length, `T_valid = (N + 160 - 321)//160 + 1 = (N - 161)//160 + 1`, and 0 if
     N ≤ 160. Equivalently, frame i is valid iff its whole 321-sample window, including the unused 321st sample, is
     real audio.
   - The **mask is a prefix of ones** because padding is on the right.
10. **Zero invalid frames.** `input_features = log_mel * mask[:, None]` (`FE4:289`). Invalid frames become **0.0**,
    not ln(0.001).

Output: `input_features` float32 [B, T_frames, 128] and `input_features_mask` bool [B, T_frames]. Measured examples:

| real samples | frames | valid frames |
|---|---|---|
| 16000 | 99 | 99 |
| 16001 | 100 | 100 |
| 8000 | 50 | 49 |
| 160 | 1 | 0 |
| 100 | 0 | 0 |
| 500000 | 2999 | 2999 (after truncation) |

**For the Go port.** Run framing on the real samples only and keep exactly `T_valid` frames. Their content does not
depend on the 128-sample padding, because a valid frame's window lies entirely inside real audio. Batch-mates also
do not affect it: a batch-of-2 row is bit-identical to running that input alone (measured).

---

## 2. The tower, op by op

The model is `Gemma4AudioModel` (`MG4:1891-1974`). `EmbeddingGemma2Model` builds it through
`AutoModel.from_config(config.audio_config)` (`MEG2:651`), so EmbeddingGemma 2 and Gemma 4 share **the same code**.

Dtype convention for the reference: run in f32 (`model_kwargs={"dtype": torch.float32}`, as
`scripts/pin_embeddinggemma2_real.py:65` does). The checkpoint is BF16. A BF16 run of the tower scored cosine
0.99933 against f32 (measured), so a golden must be pinned in f32.

Shapes below are for one sequence of `T` valid mel frames.

### 2.1 Subsampling conv stack (`Gemma4AudioSubSampleConvProjection`, `MG4:393-420`)

- **Input.** `input_features` [B,T,128] is unsqueezed to [B,1,T,128], i.e. NCHW with H = time and W = frequency
  (`MG4:414`).
- **Per layer** (`MG4:365-390`), layer0 has 1→128 channels and layer1 has 128→32:
  1. If a mask was given, multiply by it along **time**: `x * mask[:, None, :, None]` (`MG4:381-382`).
  2. `Conv2d(k=(3,3), stride=(2,2), padding=1, bias=False)` (`MG4:368-375`). This is a standard cross-correlation
     with symmetric zero-padding 1 on both time and frequency. The weight is [out, in, kt, kf].
  3. **LayerNorm** over the channel axis. It is a real LayerNorm (mean-subtracted, biased variance), not RMSNorm:
     `nn.LayerNorm(C, eps=rms_norm_eps=1e-6, elementwise_affine=True, bias=False)` (`MG4:376`). It is applied at
     every (t,f) after permuting to [B,T,F,C] (`MG4:385`). There is **weight but no bias**, and eps is **1e-6, not
     PyTorch's 1e-5 default**.
  4. **ReLU** (`MG4:377,385`).
  5. Downsample the mask: `mask = mask[:, ::2]` (`MG4:387-388`).
- **How the shapes shrink.** Each layer maps time `T → floor((T+2-3)/2)+1 = ceil(T/2)` and frequency
  `128 → 64 → 32`. After two layers: [B, 32 ch, ceil(ceil(T/2)/2), 32 freq].
  - `mask[::2]` keeps `ceil(T/2)` entries, so the mask length matches the conv output length.
  - On a prefix mask: valid subsampled frame t ⇔ input frame 2t valid ⇔ `t < ceil(T_valid/2)`.
- **Flatten.** `permute(0,2,3,1)` gives [B,T',F'=32,C=32], which is reshaped to [B,T',1024]. The flat index is
  **`f*32 + c`**, with channel fastest (`MG4:418-419`).
- **Projection.** `proj_input_dim` is hard-coded as `(channels[0]//4) * channels[1]` = 32×32 (`MG4:406`). That equals
  `F'*C` only because feature_size = 128 = channels[0] (INFERRED intent).
- `input_proj_linear`: Linear 1024→1024, **no bias, not clipped** (`MG4:407,420`).
- **The final layer's output is NOT re-masked.** Invalid frames carry garbage from here on. They cannot affect valid
  frames (§3), and they are stripped at the end.

### 2.2 Relative position table (`Gemma4AudioRelPositionalEncoding`, `MG4:226-254`)

- `context_size = chunk + left - 1 + right = 12+13-1+0 = 24` (`MG4:238-240`).
- `inv_ts[i] = exp(-i * ln(10000)/511)` for i = 0..511, using `num_timescales = 1024/2` (`MG4:241-245`).
- `position_ids = arange(12, -1, -1)` = [12, 11, …, 0], i.e. **13 rows in descending order** (`MG4:250`).
- `PE[r] = concat(sin(p_r*inv_ts), cos(p_r*inv_ts))`, with all 512 sines first and then all 512 cosines. It is
  **not interleaved** (`MG4:252-253`). The table is cast to the hidden dtype (`MG4:254`).
- Indexing by distance d = 12 - r, so `PE(d) = [sin(d·inv_ts) ‖ cos(d·inv_ts)]` for d = 0..12 (verified numerically).
- `inv_timescales` is a **non-persistent buffer** (`MG4:246`). `_init_weights` recomputes it (`MG4:1475-1481`), which
  guards against the internlm2 `from_pretrained` fast-init class of defect. The pin script should still assert it
  against the formula.

### 2.3 Conformer block × 12 (`Gemma4AudioLayer`, `MG4:533-581`)

`clip = min(1e10, finfo(dtype).max) = 1e10` (`MG4:557`, `MG4:440`, `MG4:523`). It is a `torch.clamp(±1e10)` and is
**an identity in practice**. It is not gradient-only: it runs in forward too.

All RMSNorms here are `Gemma4RMSNorm` (`MG4:205-223`):
- `y = x * pow(mean(x²) + eps, -0.5) * w`, computed in f32 and cast back.
- The weight is used **directly, not (1+w)**.
- eps is 1e-6. The FFN norms and the block norms are built without passing eps, so they use the class default 1e-6
  (`MG4:431-432,543-545`). The lconv norms pass `config.rms_norm_eps` (`MG4:507-508`). Both are 1e-6.

**ClippableLinear** (`MG4:176-202`):

```
y = clamp(linear(clamp(x, input_min, input_max)), output_min, output_max)
```

- The linear has no bias.
- The four bounds are per-module **scalars from the checkpoint** (`MG4:187-191`, loaded as persistent buffers). If
  they are missing, `_init_weights` sets them to ±inf (`MG4:1502-1506`), i.e. an identity, silently.

**Block order:**

1. **FFN1** (`Gemma4AudioFeedForward`, `MG4:423-455`):

   ```
   r = x
   y = clamp(x, ±1e10)
   y = RMSNorm_pre(y)
   y = clip_lin1(y)          [1024→4096]
   y = SiLU(y)
   y = clip_lin2(y)          [4096→1024]
   y = clamp(y, ±1e10)
   y = RMSNorm_post(y)
   x = r + 0.5*y             (residual_weight, MG4:436,452-453)
   ```

2. **Attention sub-block** (`MG4:559-573`):

   ```
   r = x
   y = RMSNorm(clamp(x), norm_pre_attn)
   y = Attn(y)
   y = RMSNorm(clamp(y), norm_post_attn)
   x = r + y
   ```

   The residual weight here is 1.0, not 0.5.
3. **LightConv1d** (`MG4:492-530`):

   ```
   r = x
   y = RMSNorm(x, pre_layer_norm)
   y = clip_linear_start(y)                            [1024→2048]
   y = GLU(y) = y[:, :1024] * sigmoid(y[:, 1024:])     (torch glu: first half × σ(second half), MG4:518)
   y = depthwise causal conv1d over time               (MG4:520)
   y = clamp(y, ±1e10)
   y = RMSNorm(y, conv_norm)
   y = SiLU(y)
   y = clip_linear_end(y)                              [1024→1024]
   x = r + y
   ```

   - **Depthwise causal conv** (`Gemma4AudioCausalConv1d`, `MG4:459-489`). It has groups = 1024, kernel 5, stride 1,
     no bias, and a left pad of `(5-1)*1+1-1 = 4` zeros with **no right pad** (`MG4:474-476,487`):
     `out[t,c] = Σ_{k=0..4} w[c,0,k] * y[t-4+k, c]`, where y at negative t is 0.
   - The conv weight is cross-correlation order, not flipped.
4. **FFN2.** Same as FFN1, with its own weights (`MG4:576`).
5. **Output norm.** `x = RMSNorm(clamp(x), norm_out)` (`MG4:578-579`). **The block output is the norm output.** There
   is no residual after it, so every block emits a unit-RMS × weight tensor.

### 2.4 Attention (`Gemma4AudioAttention`, `MG4:257-362`). Exact semantics.

H = 8, D = 128.

- **Projections.** `q, k, v = ClippableLinear(y)`, each 1024→1024. They are upcast to f32 and viewed as [T,H,D]
  (`MG4:322-324`).
- **Query scaling** (`MG4:326`, constants at `MG4:268`):

  ```
  q = q * q_scale * softplus(per_dim_scale)    per-dim, [D], shared across heads
  q_scale = D^-0.5 / ln 2 = 0.12751743
  ```

  - `softplus(0) = ln 2`, so the `/ln 2` cancels it at init (INFERRED design intent).
  - `per_dim_scale` is **the raw parameter, and softplus is applied in forward**. A port that uses the stored value as
    the scale gets negative scales.
- **Key scaling.** `k = k * k_scale`, where `k_scale = ln(1+e)/ln 2 = 1.89463612` (`MG4:269,327`).
- **Relative keys.** `relk = relative_k_proj(PE)` is a plain Linear 1024→1024 with no bias, no clip and **no
  k_scale**. It is viewed as [13,H,D] (`MG4:334-336`).
- **Logits.** For query position t and key position j, with distance d = t - j:

  ```
  logit[h,t,j] = Σ_d' q[t,h,d']·k[j,h,d']  +  Σ_d' q[t,h,d']·relk[d][h,d']
  ```

  - The first term is matrix_ac (`MG4:338-339`). The second is matrix_bd (`MG4:341-344`).
  - `q` here is the **scaled** query in both terms.
  - `_rel_shift` (`MG4:304-311`) maps block column j' to table row `c = j' - i` for 0 ≤ c ≤ 12, else 0. Row c holds
    position `12-c`, which equals d (verified: `out[i,j'] = bd[i, j'-i]` for 0 ≤ j'-i ≤ 12, else 0).
- **Softcap.** `logit = 50 * tanh(logit / 50)` (`MG4:346-349`).
- **Masking.** Masked entries are set to **-1e9** (`MG4:351-354`). `softmax` runs in f32 over the context
  (`MG4:356`), followed by `·v` (`MG4:357`).
  - **The softcap is applied BEFORE the mask fill**, so masked logits are exactly -1e9.
- **Output.** Crop to T (`MG4:358-359`), then `post` ClippableLinear 1024→1024 (`MG4:360`).

**Which keys a query sees.** This is the effective mask, as used by the default sdpa backend. Measured on the 4-D
mask: the allowed distances are exactly {0,…,11}.
- Key j is visible to valid query t iff **`t-11 ≤ j ≤ t` and j is valid**. That is 12 positions including self.
- Source: `sliding_window_mask_function((context_left-1, context_right)) = ((12, 0))` with `dist < 12`
  (`MG4:1875-1888,1958-1960`), ANDed with the padding mask by `create_bidirectional_mask` (`MG4:1954-1961`).
- Note that the blocked context (24 = 12 + 12 past) and the PE table (d up to 12) both reach **distance 12, but the
  mask never admits it**. The d = 12 PE row is dead.
- Future keys (d < 0) are masked. They also get bd = 0 from the shift.
- **This differs from Gemma3n's equivalent mask, which admits d ∈ {0..12}** (13 positions;
  `TF/models/gemma3n/modeling_gemma3n.py:350-362`, computed). Admitting d = 12 changes the real-weight output by
  max |Δ| 36 on the tower and 7.1 on the embedder (measured with `spec_reimpl_win13.py`). Match **0..11**, i.e. the
  installed HF, and record the transformers version in the golden. Which one matches Google's original is unknown
  (INFERRED: a possible upstream off-by-one).

**Blocking is an implementation detail.** The reference computes this in blocks: chunk = 12, and each block's context
is the 12 keys before the block start plus the block's own 12 (`MG4:286-302`, mask conversion `MG4:1915-1940`).
`spec_reimpl.py` uses the direct per-query sliding window above and matches HF to f32 noise. Every valid query has at
least its self key, so no valid row is fully masked.

### 2.5 Output projection

`output_proj`: Linear 1024→1536 **with bias** (`MG4:1911,1973`). The tower returns `last_hidden_state`
[B,T',1536] and `attention_mask` = the subsampled mask (`MG4:1974`).

### 2.6 Multimodal embedder → text space

`EmbeddingGemma2MultimodalEmbedder` (`MEG2:565-591`) is identical to `Gemma4MultimodalEmbedder` (`MG4:2045-2069`):

```
e = embedding_projection(RMSNorm_noscale(h))
```

- `RMSNorm_noscale` has `with_scale=False`, so there is **no weight**, and eps = `audio_config.rms_norm_eps` = 1e-6
  (`MEG2:576-579`).
- `embedding_projection` is a Linear `output_proj_dims` → `text_config.hidden_size` with no bias. That is 1536→**512**
  for EmbeddingGemma 2. For Gemma 4 E2B it is 1536→its text hidden size (INFERRED: 1536 for E2B, per
  `docs/multimodal.md:349`).
- Call path: `get_audio_features` = tower, then `embed_audio(last_hidden_state)` (`MEG2:859-880`, `MG4:2447-2468`).

---

## 3. Padded frames, and how many soft tokens

- **Valid outputs depend only on valid inputs.** All three cases were measured on a tiny random tower:
  - garbage in invalid mel frames changes the valid outputs by 0.0;
  - padded vs truncated-to-valid input: max |Δ| 1.2e-7;
  - `mask=None` vs an all-ones mask: identical.
- **Why.** The subsample layers zero the masked input frames before each conv (`MG4:381-382`), which reproduces the
  conv's zero padding. The attention masks invalid keys. The lconv is causal (left pad only) and padding is on the
  right. FFNs and norms are per-frame.
- **So the Go port can run on exactly the `T_valid` frames** and needs no attention mask beyond the sliding window
  (the reference does the same: `spec_reimpl.py`).
- **Token count.** `n_tokens = ceil(ceil(T_valid/2)/2)`, with `T_valid = (N-161)//160 + 1` (0 if N ≤ 160):
  - The processor computes it by simulating two stride-2 conv steps on the frame mask and summing
    (`PEG2:270-281`, the same as `PG4:192-203`).
  - The model strips padding tokens with `audio_features[audio_mask]` (`MEG2:827`) and asserts the count matches the
    placeholders (`MEG2:831-835`).
  - Examples: 1 s (16000) → 25; 0.5 s (8000) → 13; 2.37 s → 59; 15 s → 375; ≥30 s → 750 (truncated at 480,000
    samples). All measured.
- **`audio_ms_per_token` (40) and `audio_seq_length` (280) do not govern the real path.**
  - They are used only by `_compute_audio_num_tokens` (`PEG2:365-403`), the `_get_num_multimodal_tokens` helper for
    vLLM. That helper caps at 280, so for 15 s it returns 280 while the processor inserts 375 (measured). **Do not
    port that helper as the token count.**
  - The real cap is the 30 s truncation, which gives at most 750 tokens.
- **Short inputs.**
  - N ≤ 160 samples (≤ 10 ms): 0 valid frames, so 0 tokens.
  - N < 161 with nothing padded gives a [1,0,128] feature tensor, and the HF tower **crashes** on 0 frames ("Kernel
    size can't be greater than actual input", measured).
  - The Go port should reject audio with `T_valid == 0`, or emit no audio tokens for it.

---

## 4. Token layout and splice

- **Token ids** (`CKPT/tokenizer.json`):
  - `<bos>` = 2
  - `<|audio>` (boa) = 256000
  - `<|audio|>` (soft-token placeholder) = 258881
  - `<audio|>` (eoa) = 258883
  - The tokenizer config names them `audio_token`, `boa_token`, `eoa_token`.
- **Replacement string.** Each `<|audio|>` in the text becomes `boa + <|audio|>×n_tokens + eoa`, with no separators
  (`PEG2:281`; identical in `PG4:203`).
- **Audio-only input.** `EmbeddingGemma2Processor.prepare_inputs_layout` sets text to `"<|audio|>"` per audio
  (`PEG2:154-174`). The measured ids are `[2, 256000, 258881×n, 258883, 1]`: **BOS first and EOS (1) last**, both
  added by the tokenizer.
- **Text plus audio** (`"hello <|audio|> world"`): `[2, 23391, 236743, 256000, …, 258883, 1902, 1]`.
- **ST chat template.** `CKPT/chat_template.jinja` emits `<|audio|>` for each `{"type":"audio"}` content item, unless
  the text already contains a placeholder. It concatenates items with no separator.
- **Differences from Gemma4Processor:**
  - it accepts audio-only input (`PEG2:188-190`); Gemma4 requires text or images (`PG4:144-145`);
  - it fixes the operator-precedence bug in Gemma4's audio-token check (`PG4:147` vs `PEG2:192`);
  - it preserves per-sample audio counts when there is no text (`PEG2:127-132`).
  - The audio token expansion is identical.
- **Splice** (`MEG2:775-839`):
  - Placeholder ids are replaced by `pad_token_id` = 0 before the embedding lookup (`MEG2:780`).
  - That lookup scales by sqrt(512) (`MEG2:142-153,492`). Those rows are then overwritten by `masked_scatter` with the
    audio embedder output (`MEG2:837-839`).
  - **The audio rows are NOT scaled**, the same as the vision rows.
  - The splice happens **only if both `input_features` and `input_features_mask` are passed** (`MEG2:821`, `MG4:2366`).
    Without the mask, the audio is silently skipped and the placeholders stay as pad embeddings.
  - The projection-only PLE then runs on the spliced `inputs_embeds` (`MEG2:527`, `MEG2:172-200`), so audio rows feed
    PLE like vision rows do. The text path needs no new work.

---

## 5. Where a naive port silently goes wrong

1. **The clip bounds are load-bearing, not cosmetic.**
   - On a 2.37 s chirp, 107 of the 120 ClippableLinears clamped at least one element.
   - Dropping all clips gives tower-output cosine **0.567** (measured).
   - The bounds are asymmetric BF16 scalars. Missing ones default to ±inf (`MG4:1502-1506`), an identity, so a
     loader that skips scalar `[]` tensors still runs. Clamp both the input and the output.
2. **Never pin with `attn_implementation="eager"`.**
   - Under eager, `create_bidirectional_mask` returns a float additive mask: 0 where allowed, -3.4e38 where masked.
   - The tower treats it as a bool (`attention_mask.logical_not()`, `MG4:353`). It then fills the allowed positions
     with -1e9 and keeps the disallowed ones, which **inverts the mask**.
   - Measured: eager vs sdpa cosine **0.82** on the real tower; max |Δ| 0.51 on a tiny one.
   - `scripts/pin_embeddinggemma2_tiny.py:66` sets `_attn_implementation = "eager"`. **Do not copy that line into an
     audio pin.** Assert `audio_tower.config._attn_implementation == "sdpa"` and that the 4-D mask dtype is bool.
     `flex_attention` crashes.
3. **Off-by-one window.** Allowed d ∈ 0..11, not 0..12 (§2.4). A port written from Gemma3n, or from
   "context_left = 13", admits d = 12. **Below 13 tokens (about 0.5 s of audio) the window never binds**, so a short
   golden cannot see this. The 0.37 s case (9 tokens) is identity for the window; 2.37 s exposes it with max |Δ| 7.1.
4. **Flatten order after the subsampler.** It is `f*32 + c`. Both axes are 32, so a `c*32 + f` flatten is
   shape-legal and wrong.
5. **The subsampler norm is a LayerNorm with eps 1e-6 and no bias.** It is not RMSNorm and does not use eps 1e-5.
6. **The Gemma4RMSNorm weight is direct, not (1+w).** The values are about 1–5, which is the tell.
7. **The embedder norm has no weight.** Zero rows are legal for it: it simply is not there. Do not look for or
   synthesize a weight.
8. **`per_dim_scale` needs softplus in forward**, and `q_scale` includes `/ln 2`.
   - `k_scale` = ln(1+e)/ln 2 is applied to the keys **but not to the relative keys**.
   - All three are absent from the config, so getting one wrong gives a plausible-looking softmax.
9. **The relative PE layout.** It is [sin‖cos] concatenated, not interleaved. The table runs in descending position
   order (row r = distance 12-r). `inv_ts` uses the denominator 511, not 512. The d = 0 row is `[0…0, 1…1]`, so a PE
   that is wrong only at d > 0 still looks right on a 1-token input.
10. **The softcap is applied before the mask, and the mask value is -1e9.** Not -inf and not 1e-9 (the config
    docstring at `TF/models/gemma4/configuration_gemma4.py:43` says 1e-9).
11. **GLU half order.** It is `first * sigmoid(second)` (torch semantics).
12. **The causal conv pads on the left only**, with 4 zeros, and uses cross-correlation weight order. A centered pad
    (2+2) passes on stationary input.
13. **Block outputs are normed** (`norm_out`), and the FFN residual weight is 0.5 but the attention residual weight is
    1.0.
14. **Feature-extractor traps:**
    - the window/frame product must be f32;
    - FFT magnitude, not power;
    - `ln(x + 0.001)`, not `ln(max(x, 0.001))`;
    - HTK mel with no Slaney norm;
    - mel filter 0 is all zero, so bin 0 is constant (a torchaudio-style bank would differ);
    - left pad 160 only;
    - 321-sample unfold windows for the mask rule;
    - invalid frames are zeroed to 0.0;
    - nothing checks the 16 kHz sample rate, so the caller must resample.
15. **The token count is not 280 and not `ceil(ms/40)`.** It is `ceil(ceil(T_valid/2)/2)`.
16. **Checkpoint dtype.** Everything is BF16. Upcast to f32 for the reference (BF16 vs f32 cosine is 0.99933). A Go
    path that keeps BF16 activations will not meet a tight bar.
17. **The splice needs the mask.** If the reference pin is called without `input_features_mask`, the audio is
    silently skipped (`MEG2:821`).
18. **Non-persistent buffers.** `inv_timescales` and `softcap` are re-initialized in `_init_weights`
    (`MG4:1475-1484`). The pin script should assert `inv_timescales` matches the formula and `softcap == 50` after
    load. It should also assert one `per_dim_scale`, one `input_max` and one `output_max` equal the header values,
    which proves the checkpoint scalars loaded and did not fall back to the init.

---

## 6. What the reference pin script should capture

Load through ST in f32, the same way the vision pin does (`scripts/pin_embeddinggemma2_vision.py:50`). Assert sdpa.
Use at least three clips:
- one under 13 tokens (≤ 0.4 s): the edge case;
- one over 48 tokens (≥ 2 s), so the window binds and there are several 12-blocks;
- one whose length is not a multiple of 128 samples and not 160-aligned, which exercises the mask rule.

Use deterministic synthetic audio (seeded chirp plus noise) or a pinned WAV with its sha. Store the per-stage tensors
gzipped (`.json.gz`), per the repo's large-golden rule.

| stage | tensor | how to grab it |
|---|---|---|
| input | raw f32 samples (or sha + length) | — |
| features | `input_features` [1,T,128] and `input_features_mask` [1,T] | `processor(audio=[x], return_tensors="pt")` or `fe([x])`. Store only the valid rows plus `T_valid`. |
| tokens | `input_ids` (layout check) | processor output |
| subsample out | [T',1024] after `input_proj_linear` | `audio_tower.subsample_conv_projection` forward hook → `out[0]`; equals `hidden_states[0]` under `output_hidden_states=True` (measured) |
| subsample internals (optional) | layer0/layer1 outputs [C,T,F] | forward hooks on `subsample_conv_projection.layer0/1` → `out[0]` |
| per-block sub-stages (block 0, optionally all) | after FFN1, after attention-residual, after lconv1d, after FFN2 | forward hooks on `layers[L].feed_forward1`, `.lconv1d`, `.feed_forward2` (outputs). For the attention sub-block, hook `layers[L].norm_post_attn` output and add the FFN1 output. |
| attention internals (block 0) | `self_attn` output (after `post`) and the attention weights | forward hook on `layers[L].self_attn` → `out[0]` (out[1] = blocked weights [B,H,nb,12,24]) |
| each block output ×12 | [T',1024] | forward hook on each `audio_tower.layers[L]`. **Do not use `hidden_states[L+1]` for the last block**: `output_hidden_states` returns N+1 entries and **overwrites the last with `last_hidden_state`, i.e. post-`output_proj`, 1536-d** (measured; `TF/utils/output_capturing.py:243-244`). |
| tower out | `last_hidden_state` [T',1536] (valid rows) and `attention_mask` | return of `get_audio_features(...)` (`MEG2:859`) |
| embedder out | `pooler_output` [T',512] (valid rows) | same return (`.pooler_output`), or a hook on `embed_audio` |
| final | pooled + normalized 768-d embedding | ST `encode` with `{"type":"audio"}` input, as with vision |
| sanity | `inv_timescales`, `softcap`, the loaded clip scalars, the attention mask dtype | assert, do not store |

Store valid rows only: padded rows hold garbage by construction (§2.1).

Grade per stage. The f32-vs-f64 noise is about 6e-4 max abs on the 1536-d tower output; that is the f32 HF reference
against `spec_reimpl.py` in f64, which only gives the order of magnitude a bar needs. Grade stage by stage, so the
first divergent stage names the bug.

**What to reuse.** `spec_reimpl.py` (about 60 lines, f64) is a working executable form of §2. It can serve as the
port's line-by-line oracle, and it already proves that "run on valid frames only, with a direct sliding window" is
exact.
