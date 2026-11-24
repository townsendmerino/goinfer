# CUDA fused prefill attention at head dim 96 (Phi-3's depth term): registration (2026-11-23)

**Status: REGISTERED 2026-11-23, before any code or run.** Follows OP6-C (`../op6c-q4k-prefill-2026-11-23/README.md`). Gates in
order, each passing before the next; fixed N, no early stop. It changes output, so if it ships it lands opt-in, and the default
is Francis's call.

**Question.** Does the tensor-core fused prefill attention (`attn_fused`, L2), instantiated at head dim 96, cut Phi-3 mini's TTFT at
its default CUDA load (`--quant q4k`, f16 KV) at depth, without losing fidelity? After OP6-C the weights take about 10.1 s of a
3,900-token prefill and attention at head dim 96 about 18 s, on the exact kernels: `attnFusedFor` instantiates 64 and 128 only,
and a per-32 pass runs the exact kernels at every length.

**The change.** `attn_fused_hd96` (the kernel is a template on head dim; 96 is six 16-wide MMA steps), `attnFusedFor` admitting 96,
and per-32 passes allowed L2 above `fastPrefillFloor` behind an opt-in. L3 is untouched (it serves int4 only).

**Gates, in order.**
1. **Tier 0, correctness:** `attn_fused_hd96` against the float64 reference on the existing fused-attention tests' shapes
   (causal, windows 0 and 100, M across the block tile, f32 and f16 KV): cosine ≥ 0.9999, max abs ≤ 1e-2; a plant (one head
   dimension dropped from the P·V step) red; `TestAttnKernelDigests` unchanged for every existing kernel.
2. **Fidelity:** `TestPrefillGateVsReferenceCUDA`'s criteria (§3), exact arm against the L2 arm, both scored against the CPU f32
   reference: (a) hard flips ≤ exact's; (b) mean teacher-forced agreement ≥ exact − 1.0 pt and L2 ≥ exact on at least half the
   prompts; (c) mean KL ≤ 1.10 × exact's. Cell: Phi-3 mini q4k, held-out set B, K = 512 and 3,900 (the P3M references at 3,900
   exist from the hd-96 decode gate; 512 is generated first). **AMBIGUOUS → parked:** (b) within its last 0.2 pt, or (c) in
   1.05–1.10×.
3. **Speed:** TTFT of the exact pass (A) against the L2 pass (B) on one load, P = 512 and 3,900 from position 0, set A's first
   prompt, 10 pairs alternating A B / B A, the void rule per cell (load above 8.0, or available memory under the fit need plus 2 GB;
   at most 4 re-runs a P). **SHIP (opt-in):** median A ÷ B ≥ 1.3 at 3,900 with at least 9 of 10 pairs above 1, and ≥ 0.98 at 512.
   **KILL:** under 1.1 at 3,900. **Between:** parked.

**Stopping rule.** One run of each gate; a failed gate stops the sequence there.

**What each outcome means.** SHIP: the PR lands L2 at hd 96 behind an opt-in and states device bytes and the longest context that
fits; making it Phi-3's default is DECISION NEEDED to Francis. Gate 1 fails: fixed before gate 2 runs. Gate 2 FAIL: hd 96 stays
on the exact kernels; the record names the criterion. Gate 3 parks or kills: not merged; the record says which.

**Cost.** Gate 1 about 20 minutes; gate 2's 512 reference about 15 minutes of CPU (shared lock) and the gate about 15 minutes
exclusive; gate 3 about 10 minutes exclusive. By day; nothing on the card after 21:30 PST.
