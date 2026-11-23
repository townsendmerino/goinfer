# WG0: where WebGPU stands against goinfer's own CUDA (nobara-pc, 2026-11-20)

**Read 2026-11-20 by the WG0 session, against [`registration.md`](registration.md)** (registered 11:15 PST at `9187d1cdc`,
amended 11:22 PST at `5df28899f`, both before any timed run). **G-WG0 met: all 9 cells valid**: 4 complete pairs each,
both arms on their resident decode path, the decode token gate ok, no void start and no swap void. The Mac half
(WebGPU against Metal) is the last section, read 2026-11-23.

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

## Decision on §B8 (Francis, 2026-11-20, option (a), relayed by the coordinator at 13:40 PST)

The update below is made in **WG5's docs PR, after the Mac half**: the 38 / 41 / 63% WebGPU figure is replaced by WG0's
45 / 51 / 75% at depth 128, the depth-2048 row (13 / 17 / 39%) and the TTFT row (10 / 12 / 18%) are added, and the old row
moves to `docs/legacy-benchmarks.md`. Nothing on `docs/benchmarks.md` changes before then.

## What the §B8 update is

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

## The Mac half: WebGPU against Metal (MacBook Pro M1 Pro, 2026-11-23)

**Read 2026-11-23 against [`registration.md`](registration.md) § "The Mac half", its commands as written, by
[`run-mac.sh`](run-mac.sh)** (`run-nobara.sh` with the macOS differences the registration names: no page-cache eviction, the
darwin idle gate, settle under load 2.0). **G-WG0 met on the Mac: all 6 cells valid**: 4 complete pairs each, both arms on
their resident decode path (`metal-resident (int4)`, `webgpu:metal-resident (int4)`), the token gate ok, no void start, swap flat.

**Provenance:** goinfer `018cbe74c` from a clean detached worktree, both binaries built there (`CGO_ENABLED=0 ./backend/metal/cmd/serve`,
sha256 `1827e32e…`; `CGO_ENABLED=1 -tags gpu ./backend/webgpu/cmd/serve`, sha256 `ebe60b75…`), aikit v1.76.0 · Apple M1 Pro
(14-core GPU), macOS 26.7.3 · qwen2.5-coder 0.5B and 1.5B Q4_K_M GGUF from `~/models` on the internal SSD (the 1.5B copied that
morning from nobara's `~/models`, byte count checked), `-quant int4` on both arms, ctx 4096 · greedy, seed 1 · one exclusive
timing-lock session, **08:18–09:01 PST**, Francis's apps open (the idle gate waited up to 208 s before a cell; decode cells started
at load 1.3–1.9) · swap 513 MB throughout, no growth. Raw: [`results-mac/`](results-mac/) (`provenance.txt`, `run.log`, `power.txt`,
the harness JSON and logs, `serve-logs/`, `fit/`, `ctxfit.tsv`, `grade.txt`).

| cell | Metal | WebGPU | WebGPU ÷ Metal (4 pairs) | per pair | power |
|---|---|---|---|---|---|
| decode 0.5B @128 | 209.8 tok/s | 61.2 tok/s | **0.292** (0.289–0.293) | 0.292 0.289 0.291 0.293 | battery |
| decode 0.5B @2048 | 174.8 tok/s | 27.8 tok/s | **0.159** (0.157–0.159) | 0.159 0.159 0.158 0.159 | battery |
| decode 1.5B @128 | 99.2 tok/s | 36.8 tok/s | **0.371** (0.370–0.377) | 0.370 0.371 0.377 0.370 | battery |
| decode 1.5B @2048 | 91.1 tok/s | 18.2 tok/s | **0.201** (0.195–0.206) | 0.205 0.204 0.195 0.197 | block 0's Metal cell battery, the rest AC |
| TTFT 0.5B @1024 | 289.1 ms | 2,739.9 ms | **0.106** (0.105–0.106) | 0.106 0.106 0.105 0.106 | AC |
| TTFT 1.5B @1024 | 1,041.7 ms | 7,811.0 ms | **0.133** (0.128–0.159) | 0.132 0.133 0.128 0.158 | block 0 AC, block 1 changes over, 2–3 battery |

TTFT prompts: 1,052 tokens by `usage` on every cell; repeat/fresh 0.07 / 0.04 on Metal and 0.01 / 0.005 on WebGPU, the healthy
reading. The registration wrote no prediction for the Mac, so none is graded.

**Power source, checked and kept.** The Mac ran on battery (low power mode off) until Francis plugged it in at 08:40:50 and on
battery again from 08:55:53 (`pmset -g log`, [`results-mac/power.txt`](results-mac/power.txt); each step's START and END lines in
`run.log` carry the source). Neither the registration nor `noise-registry.md` has a power rule, and the A B B A pairs interleave
inside each cell, so no cell is void and none was re-run. The two cells that changed source were read for a power effect and show
none: in decode 1.5B @2048 blocks 1–3 all ran on AC and still split 0.204 / 0.195 / 0.197, and in TTFT 1.5B the battery blocks
read 0.128 and 0.158 on either side of the AC block's 0.132. That TTFT cell is the noisiest here (block 3's Metal cell 1,210 ms
against 986–1,058); its median, 0.133, is quoted with that range.

**Beside nobara's half** (a different box, backend and session: two readings side by side, not a ratio of them): at depth 128 WebGPU
keeps less of the native backend on the Mac (29 / 37% of Metal) than on nobara (45 / 51% of CUDA); at 2048 the two are alike
(16 / 20% against 13 / 17%), as is TTFT at 1,052 tokens (11 / 13% against 10 / 12%). WebGPU loses 55% / 51% of its depth-128 rate
by 2048 on the Mac, Metal 17% / 8%: the same long-context shape as on nobara, so WG1's 2048 profile applies to both boxes.

**Longest context that fits** (`results-mac/ctxfit.tsv`, untimed, `-quant int4`): Metal's default plan is a 32,768-position pool
for 2 conversations (Metal's default of 2 slots), each guaranteed 16,384, one up to 32,768; with one conversation, 32,768 (the
model's window) on both sizes. WebGPU's default plan is 4 × 16,384 and one conversation 16,384, its backend cap, on both sizes.
At ctx 4096 Metal planned 2 conversations of the 4 asked and WebGPU 4. Neither size is memory-bound on this 16 GB Mac.

