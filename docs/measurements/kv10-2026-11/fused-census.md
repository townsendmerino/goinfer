# Fused CUDA prefill attention: which checkpoints get it (66 attention-decoder configs under /home/francis/models)

The rule (`backend/cuda/prefill_kernels.go:cudaResident.useAttnFused`): the fused kernel serves a layer only at head dim 64 or 128, with no attention softcap; every other layer runs the exact `attn_batched`.

## No fused prefill kernel

| model type | head dim | softcap | checkpoints here |
|---|---|---|---|
| `gemma` | 256 | no | codegemma-2b, gemma-2b-it |
| `gemma2` | 256 | yes | gemma-2-2b-it |
| `gemma3_text` | 256 | no | gemma-3-4b-it |
| `gemma4_text` | 256 | no | gemma-4-26b-a4b-it-vis560, gemma-4-26b-a4b-it, gemma-4-31B-it, gemma-4-E2B-unq, gemma-4-E4B-it, gemma-4-E4B-unq |
| `mistral` | 32 | no | tinymistral-248m |
| `phi3` | 96 | no | phi3-mini-4k |
| `qwen3_5_moe_text` | 256 | no | qwen3.6-35b-a3b-vision, qwen3.6-35b-a3b |
| `qwen3_5_text` | 256 | no | JEV-9B, clef-flash, qwen3.5-0.8b, qwen3.5-9b, qwen3.8-27b, qwen3_5-scaled4 |
| `spark2_5` | 256 | no | spark25-1.7b |

## Fused prefill kernel

| model type | head dim | softcap | checkpoints here |
|---|---|---|---|
| `cohere` | 128 | no | aya-expanse-8b |
| `cohere2` | 128 | no | command-r7b |
| `glm_ocr_text` | 128 | no | glm-ocr |
| `gpt_oss` | 64 | no | gpt-oss-20b-hf |
| `granite` | 64 | no | granite-4.2-3b |
| `internlm2` | 128 | no | internlm2-1_8b |
| `laguna` | 128 | no | laguna-xs2 |
| `llama` | 64 | no | tinyllama-1.1b-chat |
| `llama` | 128 | no | qwen3-1.7b-eagle3, voxtral-mini-3b-2507 |
| `mellum` | 128 | no | mellum2-unq, mellum2.1-thinking |
| `ministral3` | 128 | no | ministral3-3b-bf16 |
| `olmo3` | 128 | no | olmo3-7b-think |
| `olmo_hybrid` | 128 | no | olmo-hybrid-7b |
| `qwen2` | 64 | no | qwen05-st, qwen2.5-0.5b-instruct |
| `qwen2_5_vl` | 128 | no | qwen2.5-vl-3b, qwen25vl-3b-instruct |
| `qwen2_moe` | 128 | no | qwen15-moe-a27b-l4slice, qwen15-moe-a27b |
| `qwen3` | 64 | no | gptoss-20b-dflash |
| `qwen3` | 128 | no | dflash2-qwen38, dspark-qwen3-4b-f32, dspark-qwen3-4b, gemma4-26b-dflash, qwen3-0.6b-bf16, qwen3-0.6b-fp8, qwen3-1.7b-bf16, qwen3-4b-dflash-f32, qwen3-4b-dflash, qwen3-4b, qwen35-9b-dflash, qwen36-35b-dflash |
| `qwen3_moe` | 128 | no | qwen3moe-30b-a3b-bf16 |
| `qwen3_vl_moe_text` | 128 | no | qwen3-vl-30b-a3b-instruct |
| `qwen3_vl_text` | 128 | no | qwen3-vl-2b-instruct |
| `smollm3` | 128 | no | smollm3-3b |

## The rule does not read these

| model type | head dim | why | checkpoints here |
|---|---|---|---|
| `deepseek_v2` | 128 | MLA: latent attention, not the dense layer path the rule governs | deepseek-v2-lite |
| `deepseek_v3` | 128 | MLA: latent attention, not the dense layer path the rule governs | moonlight-16b |
| `granitemoehybrid` | 128 | Mamba-2 mixer: the prompt takes the sequential path, which has no batched attention (prefill.go: nonBatchableKind) | granite-hf |
| `lfm2` | 64 | not CUDA-resident (a resident build is declined, fit check 2026-11-16) | lfm25-2.6b, lfm25-vl-1.6b |
| `nemotron_h` | 96 | Mamba-2 mixer: the prompt takes the sequential path, which has no batched attention (prefill.go: nonBatchableKind) | nemotron-h-4b-instruct-128k |
| `nemotron_h` | 128 | Mamba-2 mixer: the prompt takes the sequential path, which has no batched attention (prefill.go: nonBatchableKind) | nemotron-h-4b-cfgfix |

