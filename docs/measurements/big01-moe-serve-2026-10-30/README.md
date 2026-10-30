# BIG1 — a 35B MoE served on nobara's 8 GB card, experts host-streamed (2026-10-30)

Francis, 2026-10-30 10:35: the 7-9B models that fit nobara's card pass about 1 in 10 of tack's tiny coding tasks, so
try a bigger MoE today, Qwen3.6-35B-A3B first and gpt-oss-20b second, experts streamed from host RAM
(`-moe-cache-experts`). This record is the serve check that hands a command line to tack's E0.

## Why tack's E0 of 2026-10-18 was void (from the code and that run's own logs)

tack's C3 run (tack-runs `e0/c3-2026-10-18`, goinfer f20f0d93) loaded `qwen3.6-35b-a3b-q4_k_m.gguf` with
`-backend cuda -moe-cache-experts -thinking off -ctx 32768`. Its server log shows:

- The load did **not** re-quantize the `.gguf` into the heap. On Linux a `.gguf` routes to its sidecar
  (`decoder/sidecar.go:DefaultToSidecar`, `sidecarRoute`), and the v17 sidecar
  `qwen3.6-35b-a3b-q4_k_m.int4.e4h.cuda.v17.giw` (19.6 GB, built 12:51 that day) is what loaded: "reading its 19.6 GB
  once before the load".
- The anonymous term is the C′ expert cache's host copy. `cuda/pack_stream.go:cudaResident.packStackPinned` allocates
  the whole int4 expert stack as fresh `cuMemAllocHost` memory (`NewMappedHostBufferOn`, plan line `LinePinnedExperts`)
  and packs every expert into it from the mapped `.giw`. For this model that is 40 layers × 256 experts × 3 × 2048×512
  weights = 32.2 G weights: 16.1 GB of nibbles + 2.0 GB of f16 group scales ≈ 18.1 GB pinned, which is the fit line's
  "weights 18.2 GB". Pinned pages are anonymous and locked: they cannot be paged out, so the kernel finds them by
  reclaiming something else.
- The file-backed term is the `.giw` mapping itself: the pre-read and the pack touch all 19.6 GB, and those pages count
  in RSS while resident. So RSS ≈ 18 GB pinned + up to 19.6 GB file-backed; the 33.1 GB peak tack recorded is that sum
  with part of the file already reclaimed.
- The box was already 9.82 GB into system swap at the load ("swap guard: baseline 9.82 GB (system swap)"). Pinning
  18 GB on top pushed 0.90 GB of the server's own anonymous pages out ("TRIPPED — server swap grew 0.90 GB over
  baseline"), past the per-process guard's 512 MB, and every request after that got 503.

So the question for today is whether the same load stays out of swap on a box that is not already paging, and if not,
what to change. No transcode is needed: the v17 sidecar is current (`decoder/serialize.go:giwVersion` = 17).

## Registration (written before the run)

- **Tier 1**, one load, one pass, by day under the timing lock. Expected wall: load about 1.5 min, probe about 5 min.
- **Instrument:** `memwatch.py` samples the server's `/proc/<pid>/status` (VmRSS, RssAnon, RssFile, VmPin, VmLck, VmSwap,
  VmHWM) and system swap and MemAvailable each second from the server's start to its exit; `probe.py` sends, in order,
  a forced tool call (`tool_choice: "required"`, a seven-tool coding-agent list), a 256-token decode, a cold ~4k and a
  cold ~12k prompt, an append-only follow-up to the 12k conversation, then a harness turn pair at ~12k (a cold forced
  call over a 12k context, then the tool result appended and the next forced call).
- **Clean serve** = the model loads cuda-resident, the server's VmSwap stays under 64 MB and system swap stays within
  +512 MB of its value at the server's start for the whole run, and the forced call returns a call to a listed tool with
  arguments that parse as JSON.
- **Stopping rule:** memwatch sends SIGTERM at the first sample over either swap bound; the run is then void and the
  record says so. Otherwise one pass, no repeats.
- **Void at the start:** 1-minute load average above 8.0, or MemAvailable below the fit need plus 2 GB.
- The timings are single observations of one build, for tack's planning, not a peer or A/B claim.

## Results

Two loads on nobara (RTX 2070 SUPER 8 GB, driver 595.112.07, 62.7 GiB RAM), both under the exclusive timing lock, model
`~/models/qwen3.6-35b-a3b-q4_k_m.gguf` (served from its v17 sidecar), flags
`-backend cuda -require-backend -moe-cache-experts -thinking off -ctx 32768`, greedy (temperature 0). `kwin_wayland`
held its 16 MiB CUDA context throughout. Neither run was void at its start: load 0.56 and 2.43, 59 GB available, swap
0.94 and 0.91 GB used (zram).

- **Run 1**, origin/main 6369e578 (`goinfer-serve-cuda-6369e578`, sha256 ddd2279bda11…), 10:55-11:08 PDT.
- **Run 2**, this branch 00584c35, which adds the shared expert to the expert-major prefill (below), 11:13-11:19 PDT.

**Memory: clean in both runs.** Server VmSwap 0.00 GB at every sample and system swap flat (peak 1.02 and 0.94 GB).
Peak VmHWM 38.8 GB: RssShmem 19.09 GB (the pinned expert copy, 17.8 GB at load, plus the prefix snapshots, which
pin host memory while held), RssFile 19.07 GB (the `.giw` mapping), RssAnon 0.69 GB. The pinned copy shows in
`RssShmem`, not `RssAnon`: the driver maps `cuMemAllocHost` memory as shared pages. The load took 22.5 s and 21.3 s
(the `.giw` was in the page cache); the C′ cache capped at 58 slots per layer (4.2 GB of VRAM).

| step | prompt tokens | run 1 (main) | run 2 (fix) |
|---|---|---|---|
| forced tool call (`tool_choice: required`, 7 tools), time to the call | 529 | 12.3 s | 5.6 s |
| decode, 256 tokens | 40 | 43.6 tok/s | 43.5 tok/s |
| cold first token, ~4k | 3,649 | 72.2 s | 33.5 s |
| cold first token, ~12k | 13,228 | 289.9 s | 149.8 s |
| append-only follow-up to the 12k conversation (12,800 reused) | 13,297 | 13.8 s | 6.9 s |
| harness turn, cold forced call over 12k | 11,704 | 269.3 s | 129.9 s |
| harness turn, tool result appended, next forced call (11,730 reused) | 12,066 | 9.4 s | 7.1 s |

Every forced call came back as a listed tool with arguments that parse (`read_file {"path": "cuda/pack_stream.go"}`,
then `run_command {...}` after the tool result), the same calls in both runs; the free-text answers are coherent.
Prefix reuse works on an append-only follow-up: the reused count is a multiple of 512, and the remainder (500 and 336
tokens) is what the follow-up pays for.

**Why cold prefill was ~46 tok/s.** `cuda/moe_expert_major.go:cudaResident.prefillMoEExpertMajorEligible` refused
any layer with a shared expert, and every Qwen3.6 MoE layer has one, so prefill ran the per-row loop in
`cuda/prefill.go:cudaResident.prefillCore`: one decode-shaped FFN per prompt row, each with its own routing readback
and its own expert misses through 58 slots. The fix runs the shared expert after the rank-order fold, once per row,
through `cuda/resident.go:cudaResident.sharedExpert` (the decode path's own shared block, moved out unchanged), so
each row sees the same kernels in the same order. Gate: `TestMoEExpertMajorCUDA_residualIdentical` over
`qwen3next-tiny` (sigmoid-gated shared expert) and `glm-tiny` (ungated): red before the fix (expert-major never ran),
green after on all 2,560 residual elements of 40 rows against both the per-row path and the fully resident build; a
plant adding the shared expert before the fold goes red on 1,107 and 710 of 2,560 elements, every row. Logs:
`~/goinfer-bench/big01-2026-10-30/` on nobara (`redgreen.log`, `plant.log`, `qwen-1055/`, `qwen-1113/`).

**What is left in cold prefill (not measured further).** Run 2 is about 90 tok/s. The server's own counters for the
cold 12k harness turn: 6,005 expert-cache hits and 190,479 misses (3.1%). Each 512-row chunk touches nearly every
expert of every layer, and 58 slots hold under a quarter of them. The routing and shared-expert stages still launch
per row.

## gpt-oss-20b (one load, 11:23-11:36 PDT)

Branch build 00584c35 (the fix above does not touch gpt-oss: its own route kernel keeps it off the expert-major path),
`~/models/gpt-oss-20b-MXFP4.gguf` from its v17 sidecar, `-backend cuda -require-backend -moe-cache-experts -ctx 32768`
(gpt-oss always reasons; `probe.py` counts the first reasoning token as the first token). Not void at its start: load
3.25, 58 GB available, swap 0.91 GB.

- **Memory: clean.** Server VmSwap 0.00 GB throughout, system swap flat (peak 0.93 GB). Peak VmHWM 23.5 GB: RssShmem
  12.05 GB (pinned experts and snapshots), RssFile 11.44 GB, RssAnon 0.53 GB. The C′ cache capped at 12 slots per
  layer (4.1 GB); its hit rate was 76-82% on every turn.
- Load 14.2 s; decode 32.6 tok/s (256 tokens); the forced call is a post-check, not a grammar (`forced_tool_call:
  "post-check"`), and came back as `run_command {"command":"ls -R"}` in 9.1 s at 330 tokens.
- Cold first token 89.5 s at 3,522 tokens and 311.4 s at 12,524 (about 40 tok/s, per-row prefill: gpt-oss's route
  kernel is excluded from expert-major); an append-only follow-up reused 12,525 tokens and answered in 1.2 s.
- **The harness turn's cold forced call at ~12k returned no call and no usage** after 312 s, although the server
  prefilled it (its log has the KV snapshot of 10,880 rows). The probe did not keep the raw stream, so the cause is not
  recorded; the follow-up turn did not run. tack's E0 of 2026-10-18 on 0de92cd9 had long prompts that streamed no
  tokens; this may be the same defect, unconfirmed.

gpt-oss-20b is not handed to tack: its cold prefill is slower than Qwen3.6's and its long-context forced call failed once.
