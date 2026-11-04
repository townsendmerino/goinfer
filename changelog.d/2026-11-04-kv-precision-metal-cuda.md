- **Metal and CUDA KV precision (`docs/kv-precision-matrix.md`).** On Metal, `--kv f32` is now delivered on every family (`--kv auto`
  stays `f16` on standard families and `f32` on `deepseek_v2` and `deepseek_v3` MLA, which store a single compressed latent vector per
  position and refuse explicit `f16` or `i8`), and `--kv i8` works on batched prefill and multi-sequence decode (`--moe-step`,
  `--recurrent-step`, `--gemma4-step`; Gemma 4 E-models still decline `i8` to `f16`). On CUDA, `--kv f16` is now delivered on request
  across all resident families (MoE, sliding-window, pairwise RoPE, m-RoPE, Gated-DeltaNet and Mamba hybrids, Gemma 4 E-models, and MLA),
  and `--kv auto` now selects `f16` for MoE (`qwen2_moe`, `qwen3_moe`, `qwen3_vl_moe`, `glm4_moe`, `mellum`, `mixtral`; `gpt-oss` stays
  `f32`) and m-RoPE (`qwen2_5_vl`, `qwen3_vl`), while sliding-window, pairwise RoPE, hybrids, and MLA stay `f32` under `auto`.
