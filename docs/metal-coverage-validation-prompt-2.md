# Prompt for the Linux box — round 2 (the As-cap bug class + Qwen3-1.7B CUDA reference)

> Paste to the Linux-box Claude. Pull latest `main` first. Self-contained.

---

Round-2 follow-up on the Metal model-coverage work. **Metal is macOS-only, so I (the Mac) own
the Metal runs** — but the Qwen3-0.6B Metal run just caught a bug whose *bug class* may live in
the CUDA backend too, and there's a CUDA-runnable reference that would de-risk the whole thing.
Don't build/run `metal/` (empty on Linux).

## The bug I just found + fixed (Metal), and why it may bite CUDA

Validating **Qwen3-0.6B** on Metal surfaced a real bug: **Qwen3's `head_dim` is independent of
`hidden`** — 0.6B has `num_attention_heads·head_dim = 16·128 = 2048`, but `hidden_size = 1024`.
So `nH·hd ≠ H`. My Metal Stage-A GEMV staged the activation into a **hardcoded `threadgroup
short As[1536]`**, and the **o-proj**'s contraction dim `K = nH·hd = 2048` **overflowed** it →
corrupt output. It was invisible on qwen2.5 (there `nH·hd == H == 1536`). I fixed it with
per-dispatch dynamic threadgroup memory (size `As` to K each call). It would have *completely
broken* Qwen3-1.7B (H=2048), Mistral-7B (nH·hd=4096), and any model where `nH·hd ≠ H` or `H`
exceeds a hardcoded cap.

### Task A (highest value): audit the CUDA resident decoder for the SAME bug class
In `cuda/resident.go` / `cuda/backend.go` and the CUDA kernels, look for:
- any **hardcoded shared-memory / staging-buffer size** for activation staging (the CUDA analog
  of `As[1536]`), or a launch/config constant that assumes a max K;
- any place that uses **`H` (hidden) where it should use `nH·hd` (the q/attn width)** — e.g.
  o-proj input dim, attention ctx dim, `qDim`/`kvDim` computation. On qwen2.5 these are equal so
  a mix-up is silent; on Qwen3/any `nH·hd≠H` GQA model it corrupts.
- The CUDA `hostW`/`cudaResident` fields `qDim = nH·hd`, `kvDim`, `hidden` (resident.go) — confirm
  o-proj/down/attention use the right one for each matmul's K and for the activation buffers.

If the CUDA path has a fixed staging size or an `H`↔`nH·hd` conflation, **it will mis-run Qwen3
and Mistral-7B just like Metal did** — fix it (dynamic/size-to-K) and you can validate it directly
(Task B). If it's already dynamic and dim-correct, great — confirm and report.

### Task B: run Qwen3-1.7B on CUDA — a decoder-correctness reference
You can run CUDA; run **Qwen3-1.7B** through the CUDA resident path and report argmax parity vs the
CPU int8 reference + logit cosine (like the qwen2.5 gate). Two payoffs:
1. It exercises `nH·hd == H == 2048` AND QK-norm at a real size — proves the **decoder's Qwen3
   handling** (QK-norm application, head_dim, weights) is correct, independent of Metal.
2. If CUDA-Qwen3-1.7B parity is tight (~like qwen2.5), it confirms my Metal residual gap on
   Qwen3-**0.6B** (15/24, cos 0.93) is int4-vs-int8-on-a-tiny-Q8_0-model + adversarial teacher-
   forced ids, not a decoder or QK-norm bug. If CUDA parity is *also* loose, the issue is
   upstream (decoder/QK-norm) and we both chase it.

### Task C: the Phi-3 decoder bug you found (still open, all backends)
`phi3Architecture` (registry.go:~1043) never wires `cfg.SlidingWindow` (2047) into the arch — cf.
`mistralArchitecture:327`. goinfer runs Phi-3 full-attention on CPU + CUDA + Metal, diverging from
HF past 2047 tokens. Belongs in the decoder; you have Phi-3-mini to validate. Folds into your
admission-port/taxonomy task.

## What's mine (for reference)
I'll run **Mistral-7B-v0.1** (window=4096, NOT v0.3) on Metal next to prove the sliding-window
path, and re-run Qwen3 once you confirm the decoder side.

**Deliverable:** Task A (CUDA has the bug class? fixed?), Task B (Qwen3-1.7B CUDA parity numbers),
Task C status.
