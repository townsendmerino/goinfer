# WG0: where WebGPU stands against goinfer's own native backends (registered 2026-11-20)

**Status: REGISTERED 2026-11-20 11:15 PST, before any timed run.** Owner's go: Francis, 2026-11-20 ~11:00 PST (register now,
measure nobara this afternoon). Campaign: goinfer-notes `tasks/task-webgpu-2026-11.md`, row WG0. Results go in `README.md`
beside this file; a bar or cell that turns out wrong gets a dated amendment with its mechanism, not an edit.

**Why.** `docs/benchmarks.md` §B8's backend table (2026-08-26, goinfer `a161bd6`) is the only WebGPU-against-native figure
(WebGPU at 38 / 41 / 63% of goinfer's CUDA at 0.5B / 1.5B / 7B, depth 128). G35, G36, KS-E2 and KS-E3 landed after it.

## Cells (nobara-pc, RTX 2070 SUPER, driver 595.112.07, WebGPU over Vulkan through wgpu-native)

| cell | what | arms |
|---|---|---|
| decode @128 | qwen2.5-coder 0.5B, 1.5B and qwen2.5-7b-instruct, Q4_K_M GGUF, loaded `-quant int4`, ctx 4096 | CUDA resident vs WebGPU resident |
| decode @2048 | the same models and flags | the same |
| TTFT @1024 | the same models and flags, a 1,024-token prompt (`prompts.json` `0.5B:1024` / `1.5B:1024`; the 7B reuses `1.5B:1024`: one Qwen2.5 tokenizer, and `usage` records the count) | the same |

- **Same checkpoints and flags as §B8's harness**, which passed `-quant int4` to every backend (`bench_peer.py` at `a161bd6`).
  The task doc's "WebGPU (`int8int8`, resident)" does not match §B8 and cannot run the 7B: at int8int8 its 7.1 GB of weights
  do not fit resident and the load falls to `webgpu-staged` (fit check below). int4 is resident on both arms for all three.
  §B8's archive did not record a decode path, so whether its WebGPU column ran int4 resident is not known; this run records it.
- **One build:** both binaries from goinfer `018cbe74c` (origin/main, 2026-11-20), a clean detached worktree at
  `~/goinfer-bench/wg0-run/wt`; `serve-cuda-018cbe74` (sha256 `778850e2…`, `CGO_ENABLED=0 -tags cuda ./backend/cuda/cmd/serve`) and
  `serve-webgpu-018cbe74` (sha256 `7e1c0a13…`, `CGO_ENABLED=1 -tags gpu ./backend/webgpu/cmd/serve`). WebGPU has its own module,
  so no single binary carries both backends.
- **Decode instrument:** `scripts/bench_peer.py` at the pin, decode-only (inter-token rate from the first streamed token, tokens
  from `usage`, the proportional token gate), 8 completions × 64 tokens × 2 runs a cell, server restarted per cell. The two
  arms are its `goinfer` (serve-cuda) and `goinfer_old` (serve-webgpu, `BENCH_GOINFER_OLD_ARGS=-backend webgpu`, the later flag
  winning) engines, so `BENCH_ABBA=2` interleaves them: CUDA, WebGPU / WebGPU, CUDA, twice per cell = **4 pairs**.
  `BENCH_REQUIRE_BACKEND=1`. No peer arm: Ollama has no WebGPU build, so this is never a peer cell.
- **TTFT instrument:** [`wg0_ttft.py`](wg0_ttft.py) (`bench_peer_prefill.py` has no WebGPU arm and no A B B A): its request helpers,
  max_tokens 1, greedy, a unique `Session NNNN.` prefix per request (goinfer's cross-conversation KV reuse is prefix-keyed),
  its repeat/fresh cache check recorded, then 6 timed requests a cell, median. Same A B B A × 2 = 4 pairs, `-ctx 4096` on
  both arms (serve's own defaults differ: 32768 CUDA, 16384 WebGPU).
- **What is reported:** per cell, WebGPU as a fraction of CUDA: decode tok/s ratio, TTFT as CUDA TTFT ÷ WebGPU TTFT; the median
  of the 4 pair ratios with their min–max, and each arm's median. Grader: [`grade.py`](grade.py).

## Conditions

- Checkpoints from `~/models` on NVMe; the script refuses a path resolving under `/srv/models` or `/Volumes/` and checks the
  driver, the worktree's HEAD and cleanliness, and both binaries' hashes ([`run-nobara.sh`](run-nobara.sh)).
- The exclusive timing lock for the whole run (`timing_lock.py run --wait`), after nobara-aikit-timing's run.
- Page cache: [`evict_cache.py`](evict_cache.py) (the re-anchor record's, extended to take files) drops the three checkpoints'
  pages before each decode step (one model × depth; `bench_peer.py` has no per-cell hook, and a decode-only rate excludes the
  load) and before each TTFT cell. Then the load settles under 0.8 (decode steps) / 1.0 (TTFT cells), up to 10 minutes.
- Void rule (`noise-registry.md` §8, 2026-11-08): a cell starts only with load ≤ 8.0 and `MemAvailable` ≥ its fit need, and is
  VOID if system swap grows at all while it runs; a void re-runs in place and is listed, never averaged; three in a row stop
  the step. `bench_peer.py`'s own idle gate (load 1.0) applies on top.
- The desktop's `kwin_wayland` CUDA context is recorded (19 MiB at the fit check).

**Fit checks (2026-11-20 10:56–10:58 PST, these binaries, ctx 4096, `-quant int4`):** all six loads resident: CUDA
`cuda-resident (int4)`, 4 KV slots each; WebGPU `webgpu:vulkan-resident (int4)`, 4 × 4096 positions each. Not used:
WebGPU 7B int8int8 → `webgpu-staged` (resident build failed creating the W8A8 weight buffer); WebGPU 7B at serve's default
context (16384) holds 1 KV slot of 4. The run repeats every fit check, the WebGPU one with the harness's exact argv, and
exits 75 on a refusal or a wrong path.

## Prediction (written before any run)

From G35 + G36 (1.38× at 128 tokens, 2.50× at 1024, on the 1.5B at int8int8), KS-E2 (1.13×) and KS-E3 (1.01–1.09×), against
CUDA's own gains since §B8 (09-25: +3 / +15 / +11% at 128, and flash-decode default-on since 09-23):
- **Depth 128:** WebGPU's fraction has risen, more than "modestly": 0.5B 0.45–0.60 (§B8 0.38), 1.5B 0.50–0.65 (0.41),
  7B 0.65–0.80 (0.63; the 7B is the most bandwidth-bound, so dispatch wins move it least). If a fraction is within 0.03 of
  §B8's, that is reported as "not moved", meaning those wins sat at depths or quants §B8 did not measure.
- **Depth 2048:** no §B8 baseline, so no "moved more" claim is possible; predicted fraction at 2048 ≥ the same model's
  fraction at 128 − 0.05 (G36 left WebGPU losing 12% from 128 to 1024; CUDA's flash-decode holds its own curve).
- **TTFT @1024:** WebGPU 0.15–0.50 of CUDA's speed (CUDA's prefill uses tensor cores; WebGPU's is a W8A8 GEMM).

**Gate G-WG0:** every cell valid: no void path, the lock held, all 4 pairs complete, each arm on its resident decode path,
token gate ok. There is no pass or fail on the fraction: WG0 is the baseline the campaign is judged against.

## Template (`noise-registry.md` §8)

Tier: day. Testing tier 2: the figure replaces a published one (§B8), which tiers 0 and 1 cannot. Interleaving A B B A, one
session, one box. Reused readings: none (no arm of this shape has a reading at this build). Never-swap: n/a (no memory gate).
Noise: goinfer's own cross-session drift 3.6% (TE3); within-session pairs here. power.py: no bar, so no N; 4 pairs per cell,
fixed. Stopping rule: fixed N, no early stop. Ambiguous band: none (no decision rides on the number; WG5 decides the page text
with Francis). **Cost:** decode 6 steps × 8 cells × ~30–40 s, TTFT 3 × 8 cells × ~45 s, plus settles: **about 60–75 minutes**,
started when the lock frees (expected ~12:15 PST), finish ~13:30 PST. Over 2 h of machine time stops for Francis's call.

## The Mac half (not run by this session): WebGPU against Metal, the 0.5B and 1.5B

By day under the exclusive lock with Francis's apps closed, or queued for night per the overlay. From a clean detached
worktree at the same `018cbe74c` (`git worktree add --detach ~/goinfer-bench/wg0-mac/wt 018cbe74c`, `bash scripts/make_work.sh`):

```sh
B=~/goinfer-bench/wg0-mac/bin; R=~/goinfer-bench/wg0-mac/results; mkdir -p $B $R
CGO_ENABLED=0 go build -o $B/serve-metal-018cbe74 ./backend/metal/cmd/serve
CGO_ENABLED=1 go build -tags gpu -o $B/serve-webgpu-018cbe74 ./backend/webgpu/cmd/serve
# fit checks, the server's own banner (want `metal-resident (int4)` / `webgpu:metal-resident (int4)`), for 0.5B and 1.5B:
$B/serve-metal-018cbe74 -model bench=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf -backend metal -ctx 4096 -quant int4 -require-backend
$B/serve-webgpu-018cbe74 -model bench=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf -backend metal -ctx 4096 -quant int4 -require-backend -backend webgpu
# decode, one step per model x depth (0.5B, 1.5B x 128, 2048), each under the exclusive lock:
python3 scripts/timing_lock.py run --wait 3600 --label wg0-mac -- env GOINFER_SERVE_METAL=$B/serve-metal-018cbe74 \
  GOINFER_SERVE_METAL_OLD=$B/serve-webgpu-018cbe74 BENCH_GOINFER_OLD_ARGS="-backend webgpu" BENCH_ENGINES=goinfer,goinfer_old \
  BENCH_BACKENDS=metal BENCH_DEPTH_BACKEND=metal BENCH_PHASES=B BENCH_ABBA=2 BENCH_RUNS=2 BENCH_REQUIRE_BACKEND=1 \
  BENCH_MODELS=1.5B BENCH_DEPTHS=128 python3 -u scripts/bench_peer.py $R/decode-1.5B-128.json
# TTFT @1024 (no page-cache eviction on macOS: the script says so in each cell's line):
python3 scripts/timing_lock.py run --wait 3600 --label wg0-mac-ttft -- env WG0_REPO=$PWD WG0_SETTLE_LOAD=2.0 python3 -u \
  <this directory>/wg0_ttft.py $R/ttft.json --arm metal=$B/serve-metal-018cbe74:metal --arm webgpu=$B/serve-webgpu-018cbe74:webgpu \
  --models 0.5B,1.5B --quads 2 --n 6 --ctx 4096
python3 <this directory>/grade.py $R
```

The Mac's own idle gate (`BENCH_IDLE_GATE=instant`, the darwin default) and the void rule apply. The Mac rows are a separate
registration's to grade only if this text turns out not to fit the Mac (an amendment here, dated, before its run).
