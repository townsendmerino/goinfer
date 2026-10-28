# TestPrefillMoE_real26B: the test compared two arithmetics (2026-10-28)

nobara-mm01, assigned by Francis at 05:03 from SB12's recorded CUDA gate run (`docs/measurements/sb12-2026-10.md` §14).
nobara, RTX 2070 SUPER, NVIDIA driver 595.91.07, Go 1.27.3. Gemma-4-26B-A4B int4 `.giw` from `~/models`, loaded as serve
loads it (`-ctx 8192 -moe-cache-experts`). The runs held the exclusive timing lock. At the starts the load average was
2.3 to 4.6, with 57 GB available.

## The finding: not a prefill bug

`TestPrefillMoE_real26B` compares the batched MoE prefill with the sequential path at 48 positions, bit for bit. Its
sequential leg (`ForwardNoLogits`, then `Forward`) is the decode path. On a Gemma 4 MoE that path decodes on float
activations by default (`cuda/floatact.go`, S19), while prefill and the LM head stay W4A8. That split is by design. So the
test compared two different arithmetics, and a router argmax that flips under the difference sends every later row to
different experts.

Commit `f38f098aa` set `floatActOffForTest` on every W4A8 decode test when float-activation decode became the default. It
set it on `TestPrefillMoE_bitIdentical` but missed the heavy real-26B test, which shares the same helper.

| arm | logits differing | max abs diff | log |
|---|---:|---:|---|
| origin/main `0e69d19f8`, as is | 262,144 of 262,144 | 21.20784 | `main.log` |
| the same, with the W4A8 decode (`floatActOffForTest`) | 0 of 262,144 | 0 | `fix.log` |
| the fix (the flag set in the shared helper) | 0 of 262,144 | 0 | `fix2.log` |

The batched MoE prefill is bit-identical to the sequential W4A8 path on the real 26B. That includes the C′ expert cache
and K=V, the combination only this model exercises. With the fix, `gemma4-moe-scaled` also passes, 0 of 4,096 differ.

## The fix

`prefillMoEParity` sets `floatActOffForTest` itself, so no caller can leave it out. The float-activation decode keeps its
own gate, `TestGemma4MoE_floatParity`, which holds it to the CPU's float path.

## What users see

A served Gemma 4 MoE prefills its prompt on W4A8 and decodes on float activations, as S19 decided. A prompt's
logits therefore differ between the batched prefill and a token-by-token prefill, by design. That is the S19 trade, not a
regression; nothing here changes it.
