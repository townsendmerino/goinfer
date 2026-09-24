# R6 flash-decode default-on — served A/B, 2026-09-23

**What was measured.** The shipped default (`GOINFER_CUDA_FLASH_DECODE` unset, lane ON at S=16) against
`GOINFER_CUDA_FLASH_DECODE=0` (exact attention), through `scripts/bench_splitkv.py --mode flash-default`:
a fresh `serve` per arm, arms interleaved, decode tok/s at a fixed context depth. Machine `nobara-pc`
(RTX 2070 SUPER, NVIDIA driver 595.91.07 — the current anchor), both arms on local-disk `~/models`
checkpoints, greedy, idle-gated (load < 0.6). The harness stamped commit `d88ef960` with a dirty tree:
the binary was built from the working tree carrying the flip, before it was committed.

| geometry | 128 keys | 2048 | 3900 |
|---|---|---|---|
| Qwen2.5-Coder-1.5B q4_k_m | 1.002 (252.97 / 252.43) | 1.296 (227.98 / 175.96) | 1.609 (215.42 / 133.85) |
| gemma3-1b | | | 1.022 (184.43 / 180.40) |
| phi3-mini | | | 1.001 (59.73 / 59.69) |

Cells are default tok/s / `=0` tok/s. One pair per cell; within-cell spreads were 0.00–1.57 tok/s, far
under every effect above 1.02.

**Reading.**
- Below the 2048-key floor the lane is not entered, so 128 keys must read ~1.000 — it does (1.002).
- The win grows with depth on the geometry the lane was gated on, as a key-split kernel should.
- phi3-mini (head_dim 96) is outside the lane's supported head dims and stays on the exact path, so
  1.001 is the expected no-op and confirms the automatic decline, not a speedup.
- gemma3-1b's 1.022 is a single pair at 1.18 tok/s spread on the `=0` arm; read it as "not slower",
  not as a measured +2%.

**Not established.** Fidelity: this measures speed only. The lane is not bit-identical, and its
fidelity evidence (held-out set B, D7 KL 0.9862×, S 1.0205×) covers Qwen2.5 dense only — no other
family, no chat-template prompts. D7 prompt 9 (lane KL 0.0143 vs exact 0.0021) is unexplained. One pair
per cell means no paired-difference statistic here.

Raw cells: `attn-decode-fa-default-2026-09-23-15b.json`, `…-other.json`; harness log
`attn-decode-fa-default-2026-09-23.log`.
