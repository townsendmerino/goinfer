- **CUDA MoE prefill parity on the real 26B is green again** (tests only). `TestPrefillMoE_real26B` compared the batched
  prefill, which is W4A8, with a sequential leg that decoded on float activations, the Gemma 4 MoE default. The two are
  different arithmetics by design, so every logit differed. The shared parity helper now selects the W4A8 decode itself.
  On the real Gemma-4-26B the batched prefill is bit-identical to the sequential W4A8 path (0 of 262,144 logits differ).
