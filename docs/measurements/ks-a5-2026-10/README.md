# KS-A5 build: one-round row loads in the fused rmsnorm GEMVs, 2026-10-23

The build for `docs/tasks/task-kernel-survey-2026-10.md` KS-A5. Its first measurement
(`docs/measurements/kernel-survey-2026-10/README.md` § KS-A5) read `fused_rms_gu` at the 0.5B's shape (hidden 896, inter
4,864) at **55.2%** of DRAM peak. A K = 896 row is 112 words: three serialized load rounds of 64, 32 and 16 words a warp,
each waiting on the previous round's FMAs. The item waited for KS-A2 (#191), since the gain needs decode to be GPU-bound.

**The change** (`cuda/fused_qkv.cu`, `fused_qkv.ptx` regenerated; today's source first reproduced the committed PTX byte for byte):
- A lane's words in a row are `lane + 32m` for m ascending; the old 2×-unrolled loop and its tail visit them in that order,
  and the FMA chain runs in that order.
- A row of at most 256 words (K ≤ 2,048) loads all of its words and group scales in one round (`fq_load`), then runs the
  same chain (`fq_dot`). The output is the loop's, bit for bit.
- `fused_rms_gu` also issues the warp's first row before the norm prologue, and each next row while the current one is
  reduced.
- Longer rows keep the loop.
- Scope: `fused_rms_qkv` and `fused_rms_gu`. The rows-per-warp twins (`fused_qkv_rows.cu`, `fused_gu_rows.cu`) and aikit's
  `gemv_w4a8_fwd` are unchanged; the instrument records which kernel each model dispatches.

Machine: nobara-pc, RTX 2070 SUPER 8 GB, Ryzen 7 3700X, NVIDIA driver 595.91.07, Go 1.27.3. Checkpoints from `~/models` on NVMe.

## Pre-registration

Registered 2026-10-23 before any GPU run. Instrument: `instruments/ks_a5_rowload_test.go.txt`, copied into `cuda/` for the
run. The old arm is origin/main's committed `fused_qkv.ptx`, loaded beside the new one on the same resident and swapped into
`fQKV` / `fGU`. Held under the timing lock.
- **G1, bit-identity (tier 1):** full logits over 32 decode positions, old arm then new arm over the same positions,
  byte-equal. Models: Qwen2.5-Coder-0.5B (K 896), gemma3-1b (K 1,152) and Qwen2.5-Coder-1.5B (K 1,536), each loaded as
  `serve -backend cuda -quant int4` loads it. One differing logit stops the item.
  - Before merge also: the CUDA device suite, and `TestFusedQKVRowsBitIdentical` / `TestFusedGURowsBitIdentical`, which
    hold the twins equal to these kernels.
- **G2, speed (tier 1, in process):** the 0.5B, greedy decode at depth 128, 10 ABBA quads of 64 tokens. Readout: the per-quad
  ratio old over new, its median, and quads above 1.
  - **ship:** at least 9 of 10 above 1 (CLAUDE.md's small-gains rule: bit-identical, 9 of 10, no memory increase; the change
    allocates nothing);
  - **parked, to the owner:** 6–8 of 10;
  - **killed:** 5 or fewer.

  The survey's estimate was ~0.3 ms a token (~10%) on the 0.5B. It assumed the gate/up kernel goes from 200 GB/s to the
  ~360 GB/s of long rows, which is an estimate, not a bar.
- **Recorded, no bar:** gemma3-1b's and the 1.5B's dispatch (which kernel takes gate/up); registers and local bytes of the
  two kernels, old and new.
- **Load rule:** a quad started with the 1-minute load above 8.0 is void and re-run in order.
- **Cost:** three loads (~10–20 s each), 96 G1 forwards, 42 blocks of 64 tokens at ~2 ms: about 2 minutes.

## Readings

Run 2026-10-23 15:57 PDT at 6ababd5e3 (old arm: origin/main's committed `fused_qkv.ptx`), under the timing lock, the card
clear (402 MiB used) and the 1-minute load 4.81 at the start. Log: `~/goinfer-logs/ks-a5-2026-10/gate.log`.

| id | reading | verdict |
|---|---|---|
| G1 | logits byte-equal over 32 positions on all three: the 0.5B (151,936 a position), gemma3-1b (262,144) and the 1.5B (151,936) | **PASS** |
| G2 | 0.5B greedy token 1,909.3 → **2,000.5 µs** (523.7 → 499.9 tok/s); new speedup **median 0.9548×** (0.9507–0.9569), **0 of 10** quads above 1, 0 void; quad loads 4.53–4.75 | **KILLED** |

**Dispatch:**
- The 0.5B takes gate/up through the changed `fused_rms_gu` (rows-per-warp 8; 112 words a row).
- gemma3-1b (rows-per-warp 11, 144 words) and the 1.5B (14, 192 words) take it through `fused_rms_gu_rows`, which this build
  does not change.

So only the 0.5B's gate/up and every model's `fused_rms_qkv` (when it is dispatched) ran the new code.

**The build is slower, consistently: about 4.5%, 91 µs a token on the 0.5B.** Bit-identity held, so this is purely speed.
Not established: why. The registered no-bar readout of registers and local bytes was not in the instrument, and a probe
run afterwards to read them was not permitted in this session. The likely suspects:
- the next-row prefetch holding a second 16-register row set live across the reduction, which costs occupancy;
- the first row's loads held live across the norm prologue's barriers.

Each is a separate variant (one-round loads without the prefetch, or prefetch without the pre-prologue load), and each
would need its own pre-registration; none was tried. The code is in history at 6ababd5e3; `cuda/` on this branch is main's.
