#!/usr/bin/env python3
"""G-S14b2's fixture (docs/tasks/task-multimodal-support-2026-10.md, "G-S14b"): a tiny random-weight Qwen3-ASR (audio encoder + projector + a Qwen3 text decoder, for S14.3) built from
transformers' own classes, written in the REAL checkpoint's layout (everything under `thinker.`, the projector as proj1/proj2 inside the audio tower, config nested under thinker_config), and
the reference outputs of its audio tower and projector on three synthetic clips, with features made by transformers' own Qwen3ASRFeatureExtractor.
Run (repo root): ~/g4venv/bin/python -I scripts/pin_qwen3asr_tiny.py testdata/qwen3asr-tiny"""
import json, os, sys, zipfile
import numpy as np, torch, transformers
from safetensors.torch import save_file
from transformers import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration
from transformers.models.qwen3_asr.feature_extraction_qwen3_asr import Qwen3ASRFeatureExtractor

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
torch.manual_seed(7)
audio = dict(d_model=32, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=64, downsample_hidden_size=8, num_mel_bins=128, n_window=50, n_window_infer=200, output_dim=24)
text = dict(vocab_size=160, hidden_size=24, intermediate_size=48, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=12, max_position_embeddings=512, rms_norm_eps=1e-6,
            rope_theta=1000000.0, tie_word_embeddings=True, attention_bias=False,
            rope_scaling={"interleaved": True, "mrope_interleaved": True, "mrope_section": [2, 2, 2], "rope_type": "default", "type": "default"})
cfg = Qwen3ASRConfig(audio_config=audio, text_config=text, audio_token_id=150, pad_token_id=0, eos_token_id=[1, 2])
cfg._attn_implementation = "eager"
model = Qwen3ASRForConditionalGeneration(cfg).float().eval()
with torch.no_grad():  # random weights with a real spread, norms and biases non-trivial
    # Weights sized so the planted defects are SEEN: pre-activations of order 1-3 (the exact and tanh GELUs differ visibly there) and sharp attention (full attention over the clip differs
    # from windowed). A first draw with std 0.15 left tanh-GELU at cosine 1.000000 and full attention at 0.9991: a fixture too gentle to fail.
    for n, p in model.named_parameters():
        if "audio_tower" in n or "multi_modal_projector" in n:
            if p.dim() >= 2:
                fan_in = p.numel() // p.shape[0]
                gain = 5.0 if ("q_proj" in n or "k_proj" in n) else (3.0 if ("fc1" in n or "conv2d" in n or "linear_1" in n) else 1.5)
                p.normal_(0, gain / fan_in ** 0.5)
            elif "norm" in n and n.endswith("weight"): p.copy_(1 + 0.2 * torch.randn_like(p))
            else: p.normal_(0, 0.3)
        elif p.dim() >= 2: p.normal_(0, 0.15)
        elif "norm" in n and n.endswith("weight"): p.copy_(1 + 0.2 * torch.randn_like(p))
        else: p.normal_(0, 0.1)
sd = model.state_dict()
def rename(k):
    if k.startswith("model.audio_tower."): return "thinker." + k[len("model."):]
    if k.startswith("model.multi_modal_projector.linear_1."): return "thinker.audio_tower.proj1." + k.split(".")[-1]
    if k.startswith("model.multi_modal_projector.linear_2."): return "thinker.audio_tower.proj2." + k.split(".")[-1]
    if k.startswith("model.language_model."): return "thinker.model." + k[len("model.language_model."):]
    if k == "lm_head.weight": return "thinker.lm_head.weight"
    raise KeyError(k)
tensors = {rename(k): v.detach().contiguous().clone() for k, v in sd.items()}
save_file(tensors, f"{out}/model.safetensors")
full = cfg.to_dict()
thinker = {"model_type": "qwen3_asr_thinker", "architectures": ["Qwen3ASRThinkerForConditionalGeneration"], "audio_config": full["audio_config"], "text_config": full["text_config"],
           "audio_token_id": 150, "audio_start_token_id": 148, "audio_end_token_id": 149, "initializer_range": 0.02, "dtype": "float32"}
json.dump({"architectures": ["Qwen3ASRForConditionalGeneration"], "model_type": "qwen3_asr", "thinker_config": thinker, "transformers_version": transformers.__version__}, open(f"{out}/config.json", "w"), indent=1)

fe = Qwen3ASRFeatureExtractor()
def clip(n, f0):
    i = np.arange(n, dtype=np.float64); noise = ((np.arange(n, dtype=np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)).astype(np.float64) / 4294967296.0 - 0.5
    return (0.3 * np.sin(2 * np.pi * f0 * i / 16000) + 0.15 * np.sin(2 * np.pi * (f0 * 2.7) * i / 16000) * (1 + np.sin(i / 3000.0)) + 0.1 * noise).astype(np.float32)
cases = [("clip2s", clip(32000, 330.0)), ("clip5s3", clip(84800, 220.0)), ("clip7s", clip(112000, 150.0))]
meta = {"transformers": transformers.__version__, "torch": torch.__version__, "cases": []}
with zipfile.ZipFile(f"{out}/golden.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for name, x in cases:
        b = fe(x, sampling_rate=16000, return_tensors="pt", return_attention_mask=True, padding=True, truncation=False, n_window=50)
        feats, mask = b["input_features"], b["attention_mask"]
        with torch.no_grad():
            tower = model.model.audio_tower(input_features=feats, input_features_mask=mask).last_hidden_state
            proj = model.model.multi_modal_projector(tower)
        T, valid = feats.shape[-1], int(mask.sum())
        z.writestr(f"{name}.samples.f32", x.astype("<f4").tobytes()); z.writestr(f"{name}.feats.f32", feats[0].numpy().astype("<f4").tobytes())
        z.writestr(f"{name}.mask.u8", mask[0].numpy().astype("u1").tobytes())
        z.writestr(f"{name}.tower.f32", tower.numpy().astype("<f4").tobytes()); z.writestr(f"{name}.out.f32", proj.numpy().astype("<f4").tobytes())
        meta["cases"].append({"name": name, "samples": int(len(x)), "T": T, "valid": valid, "tokens": int(tower.shape[0]), "d_model": audio["d_model"], "out_dim": audio["output_dim"]})
        print(f"{name}: {len(x)} samples, T {T}, valid {valid}, tokens {tower.shape[0]}, tower rms {tower.pow(2).mean().sqrt():.3f}, out rms {proj.pow(2).mean().sqrt():.3f}")
    z.writestr("meta.json", json.dumps(meta, indent=1))
# NOTE: transformers 5.15.0 cannot read this nested layout (nor the real Qwen3-ASR checkpoint's: it takes the nested config for defaults and reports every weight MISSING, i.e. randomly initialised,
# with no error). The reference outputs above come from the in-memory model; the layout's names are the real checkpoint's, and the Go loader is exercised on the real one (G-S14b3).
