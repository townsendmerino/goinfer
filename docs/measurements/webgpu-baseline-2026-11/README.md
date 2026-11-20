# WG0: where WebGPU stands against goinfer's own CUDA (nobara-pc, 2026-11-20)

**Read 2026-11-20 by the WG0 session, against [`registration.md`](registration.md)** (registered 11:15 PST at `9187d1cdc`,
amended 11:22 PST at `5df28899f`, both before any timed run). **G-WG0 met: all 9 cells valid**: 4 complete pairs each,
both arms on their resident decode path, the decode token gate ok, no void start and no swap void. The Mac half
(WebGPU against Metal) is not in this record; its commands are in the registration.

**Provenance:** goinfer `018cbe74c` (both binaries; sha256 `778850e2…` serve-cuda, `7e1c0a13…` serve-webgpu), aikit v1.76.0 ·
RTX 2070 SUPER, driver 595.112.07, WebGPU over Vulkan (wgpu-native, the NVIDIA adapter) · Ryzen 7 3700X, Linux
7.3.0-202.nobara.fc44 · qwen2.5-coder 0.5B / 1.5B and qwen2.5-7b-instruct, Q4_K_M GGUF from `~/models` on NVMe, loaded
`-quant int4` on both arms, ctx 4096 · greedy, seed 1 · one exclusive timing-lock session, **12:09–12:43 PST**, settle under
0.8 before each step, the checkpoint's page cache evicted before each step (decode) and each cell (TTFT) · desktop's
`kwin_wayland` held 19 MiB of the card · cells started at load 0.2–1.0, `MemAvailable` ≈ 60.7 GB, swap flat at 416 MB.
Raw: [`results/`](results/) (`provenance.txt`, `run.log`, the harness JSON and logs, `serve-logs/`, `fit/`, `grade.txt`).

## Results

Fraction = WebGPU ÷ CUDA per A B B A pair (TTFT: CUDA TTFT ÷ WebGPU TTFT, so both read as WebGPU's share of CUDA's speed);
median of 4 pairs with the min–max; the arms' medians beside it.

| cell | CUDA | WebGPU | WebGPU ÷ CUDA (4 pairs) | predicted |
|---|---|---|---|---|
| decode 0.5B @128 | 521.2 tok/s | 236.6 tok/s | **0.454** (0.452–0.456) | 0.45–0.60 ✓ |
| decode 1.5B @128 | 277.1 tok/s | 141.9 tok/s | **0.512** (0.511–0.513) | 0.50–0.65 ✓ |
| decode 7B @128 | 86.1 tok/s | 64.2 tok/s | **0.745** (0.745–0.746) | 0.65–0.80 ✓ |
| decode 0.5B @2048 | 447.2 tok/s | 58.0 tok/s | **0.130** (0.129–0.130) | ≥ 0.40 ✗ |
| decode 1.5B @2048 | 250.1 tok/s | 42.3 tok/s | **0.169** (0.169–0.169) | ≥ 0.46 ✗ |
| decode 7B @2048 | 79.2 tok/s | 30.9 tok/s | **0.389** (0.388–0.390) | ≥ 0.70 ✗ |
| TTFT 0.5B @1024 | 62.3 ms | 628.1 ms | **0.099** (0.098–0.101) | 0.15–0.50 ✗ (below) |
| TTFT 1.5B @1024 | 168.9 ms | 1433.5 ms | **0.118** (0.116–0.120) | 0.15–0.50 ✗ (below) |
| TTFT 7B @1024 | 760.5 ms | 4234.4 ms | **0.180** (0.179–0.180) | 0.15–0.50 ✓ |

TTFT prompts: 1,052 tokens by `usage` on all three (the 1,024-token filler plus the chat template and Qwen's default system
message). The cache check read repeat/fresh 0.33 / 0.19 / 0.10 on CUDA and 0.02 / 0.01 / 0.006 on WebGPU: both reuse a
repeated prompt and the unique-prefix requests miss it, the healthy reading.

**Against the prediction:**
- **Depth 128 held:** the fractions rose from §B8's 38 / 41 / 63% to 45 / 51 / 75%, inside the predicted bands (the 0.5B at
  its lower edge). Both arms moved: CUDA's own column also rose (§B8 332.7 / 220.8 / 73.1 tok/s), so the fraction moved less
  than WebGPU did. WebGPU against §B8's 127.6 / 90.1 / 46.1 tok/s reads 1.85× / 1.57× / 1.39×, but that is a cross-session
  comparison of two builds, indicative only (`noise-registry.md`, TE3).
- **Depth 2048 failed, by a wide margin.** WebGPU loses 75% / 70% / 52% of its depth-128 rate at 2048, where CUDA loses
  14% / 10% / 8%. The prediction leaned on G36's "the new kernel loses 12% from 128 to 1024" (the 1.5B at int8int8,
  2026-09); this run's int4 1.5B loses 70% by 2048. The key-split kernel is not disabled here: `attnKeysEligible` admits
  head sizes 64 and 128 on an f32 cache and `GOINFER_ATTN_KEYS` was unset. Where the token goes at 2048 is WG1's question
  (`TestDecode_dispatchProfile` at `GOINFER_DECODE_POS=2048`, KS-E5's nobara step), and it should come before any WG2 lever.
  Not established here: whether G36's 2026-09 shape still holds at int8int8, or whether something since has changed the
  long-context token on either quant.
- **TTFT** reads below the predicted band on the two small models (0.10, 0.12) and inside it on the 7B (0.18).

## Longest context that fits (descriptive, no gate; the amendment)

From each server's own banner (`results/ctxfit.tsv`, `results/fit/ctxfit-*.log`), `-quant int4`, untimed:

| model | CUDA default plan (KV f16) | CUDA, one conversation | WebGPU default plan (KV f32) | WebGPU, one conversation |
|---|---|---|---|---|
| 0.5B | a 65,536-position pool for 4 conversations, each guaranteed 16,384, one up to 32,768 | 32,768 (the model's window) | 4 × 16,384 | 16,384 (backend cap) |
| 1.5B | the same pool, 1.86 GiB of KV | 32,768 | 4 × 16,384 | 16,384 (backend cap) |
| 7B | a 55,040-position pool for 4, each guaranteed 13,760, one up to 32,768 | 32,768 | **1 × 16,384** (4 asked: slot 2's 1,792 MB of KV did not fit) | 16,384 (backend cap) |

What WebGPU's f32 cache costs on this 8 GB card: one conversation tops out at **half the model's window** on every size,
because WebGPU's cap is the ceiling its KV precision sets (serve's `-ctx` help: a larger `-ctx` is ignored) and f32 is the
only WebGPU KV. On the 7B it also leaves room for **one conversation where CUDA plans four**: 16,384 positions of f32 KV
are 1.75 GiB, the same bytes CUDA's f16 spends on 32,768. Whether one 7B conversation would fit past 16,384 on WebGPU is
not answerable from a banner while the cap holds.

## What the §B8 update would be (not made: Francis sees this first)

- The backend table's WebGPU column and its footnote ⁱ: replace "WebGPU runs at 38 / 41 / 63% of goinfer's own CUDA" with
  this run's **45 / 51 / 75% at depth 128** (WebGPU 236.6 / 141.9 / 64.2 tok/s beside CUDA 521.2 / 277.1 / 86.1, same build
  `018cbe74c`, same session), and add the depth-2048 row (**13 / 17 / 39%**) and TTFT at 1,052 tokens (**10 / 12 / 18%**),
  labelled cross-backend, never a peer cell. The CUDA cells here are this session's, not §B8's anchor; they move only the
  fraction.
- The old row moves to `docs/legacy-benchmarks.md` verbatim, and `docs/gpu-vendor-coverage.md`'s quote of 38 / 41 / 63%
  is retired with it (WG5).
- The "the gap narrows as the model grows, which is what a dispatch-overhead story predicts" sentence still holds at depth
  128; at 2048 the gap is far wider on every size, and the page should say so rather than quote the short-context figure
  alone.
