- **Metal and CUDA KV precision (`docs/kv-precision-matrix.md`).** On Metal, `--kv f32` is delivered on every family, and `--kv i8`
  is delivered on batched prefill and multi-sequence decode (`--moe-step`, `--recurrent-step`, `--gemma4-step`), Gemma 4 E-models
  included. `--kv auto` stays `f16` on standard families and `f32` on `deepseek_v2` and `deepseek_v3` MLA, which store one compressed
  latent vector per position and refuse explicit `f16` or `i8`. On CUDA, `--kv f16` is delivered on request for the families the
  CUDA policy lists (`llama`, `mistral`, `qwen2`, `qwen3`, `phi3`, the MoE families `qwen2_moe`, `qwen3_moe`, `qwen3_vl_moe`, `mixtral`
  and `gpt-oss`, the sliding-window `gemma2` and `gemma3`, `cohere`, `cohere2`, `qwen2_5_vl`, `qwen3_vl`, the Gated-DeltaNet
  `qwen3_5`, `qwen3_5_moe` and `qwen3_next`, `gemma4` (including the E-models), and MLA `deepseek_v2`, `deepseek_v3` and `kimi_k2`).
  Other families refuse an explicit `f16` at load, among them Mamba-2 hybrids (`nemotron_h`), Olmo Hybrid, `glm4_moe` and `mellum`.
  Every CUDA `--kv i8` load refuses (no int8 KV on CUDA yet). `--kv auto` on CUDA selects `f16` for `llama`, `mistral`, `qwen2`,
  `qwen3`, `phi3`, `qwen2_5_vl`, `qwen3_vl`, `qwen2_moe`, `qwen3_moe`, `qwen3_vl_moe` and `mixtral`; sliding-window, pairwise RoPE,
  hybrid and MLA families stay `f32` under auto.
