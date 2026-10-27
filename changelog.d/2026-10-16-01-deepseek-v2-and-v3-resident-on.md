- **DeepSeek-V2 and V3 resident on Metal.** Multi-head latent attention runs on the GPU in its absorbed form, with the latent
  rows as the KV cache. Gated on tiny models, a kernel check of both rope layouts against float64, and a two-layer slice of
  DeepSeek-V2-Lite against the CPU. Kimi K2 stays on the CPU: its 384-expert router is past Metal's 256.
