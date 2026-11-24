# WG3 families: results

Registered in [registration.md](registration.md). nobara-pc, RTX 2070 SUPER, driver 595.112.07, WebGPU over Vulkan;
the compositor's context present throughout (`kwin_wayland`, 19 MiB). Every arm runs `-backend webgpu -quant int4 -ctx
4096`, greedy (temperature 0, seed 1), from `~/models`.

## Disclosed deviation: a poller on nobara (all GLM-OCR cells; Francis, 2026-11-24, verdicts kept)

From 2026-11-23 11:22:52 to 2026-11-24 07:45 PST, a leftover waiter on mbp2 (this campaign's session) ran
`ssh nobara 'grep -q "decode rc=" ~/goinfer-bench/tmp/mbp2-wg3/speed4.log'` once a minute: an sshd session, a zsh login
and one grep each, during every nobara timed cell in that span. Its span covers all of GLM-OCR's cells below
(11:52-12:13). Francis's call (2026-11-24, via the coordinator): keep the verdicts and record the poller here, no re-run.
The margins it sits against: decode 8.65-8.82× per pair against the 1.05 bar, TTFT 1.012-1.019 per pair against 0.8.
Ministral 3's first cell (2026-11-24 06:35) is void as a whole for the same poller and re-ran in full (below).

## GLM-OCR (WG3-pair, #1247) — 2026-11-23 11:52-12:13 PST

Checkpoint `~/models/glm-ocr` (text decoder). Arms: base `3a826524b`, head `8e4f95aba` (serve sha256 `5d77b58f…`,
`dc418e4d…`); fit: base `webgpu-staged (int4)` (webgpu does not implement `pairwise-mrope`, `pairwise-rope`), head
`webgpu:vulkan-resident (int4)`. Calibrated prompts: 129 tokens (decode), 1,031 (TTFT). No void cell; decode errors 0.

| block | decode head tok/s | base tok/s | head ÷ base | TTFT base ms | head ms | base ÷ head |
|---|---|---|---|---|---|---|
| 0 | 270.7 | 31.3 | 8.65 | 5,150 | 5,061 | 1.018 |
| 1 | 272.6 | 30.9 | 8.82 | 5,128 | 5,065 | 1.013 |
| 2 | 272.2 | 31.1 | 8.75 | 5,167 | 5,071 | 1.019 |
| 3 | 271.3 | 31.2 | 8.70 | 5,135 | 5,073 | 1.012 |

**Verdict, by the registration:** decode a **win** (median 8.72×, all four pairs > 1) and TTFT **no regression** (median
1.015 ≥ 0.8). #1247 was armed on it and merged 2026-11-24.

## Ministral 3 (WG3-temp, #1249) — 2026-11-24 09:31-10:27 PST (the re-run)

Checkpoint `ministral3-3b-bf16`. Arms: base `52f8adb90`, head `5b15f156f` (serve sha256 `9eb29d43…`, `02b5bfd1…`); fit:
base `webgpu-staged (int4)` (webgpu does not implement `attn-temp`), head `webgpu:vulkan-resident (int4)`.

**The first cell (06:35-07:57) is void as a whole.** The poller above ran through it, and so did a per-minute chain monitor
from this session and the coordinator's reads at 06:47, 06:51 and 07:35. Its results are kept on nobara as
`ministral3-void-0635`, never averaged, and the cell re-ran in full with no contact until 09:45. **Disclosed in the re-run:**
this session's scheduled read at 09:45 and five ssh `cat`s of result files right after it (about 09:45-09:47) fell inside
the decode cell (09:34:52-09:49:40); every TTFT cell ran after them, and no TTFT cell is void.

| block | decode head tok/s | base tok/s | head ÷ base | TTFT base ms | head ms | base ÷ head |
|---|---|---|---|---|---|---|
| 0 | 107.9 | 9.4 | 11.48 | 23,143 | 11,209 | 2.065 |
| 1 | 107.0 | 9.4 | 11.38 | 23,077 | 11,245 | 2.052 |
| 2 | 107.0 | 9.4 | 11.38 | 23,101 | 11,207 | 2.061 |
| 3 | 107.1 | 10.5 | 10.20 | 23,075 | 11,209 | 2.059 |

**Verdict, by the registration:** decode a **win** (median 11.38×, all four pairs > 1) and TTFT **no regression** (median
2.060). The resident's token-by-token prompt is faster to first token than the CPU's batched prefill here.

## Command-R7B (WG3-coh, #1255) — 2026-11-24 07:57-09:29 PST

Checkpoint `command-r7b`. Arms: base `52f8adb90`, head `c744c0f1b` (serve sha256 `81d274a6…`, `63910b80…`); fit: base
`webgpu-staged (int4)` (webgpu does not implement `layer-norm`, `pairwise-rope`, `parallel-block`), head
`webgpu:vulkan-resident (int4)`. No contact from this session during the cell; no void cell,
decode errors 0.

| block | decode head tok/s | base tok/s | head ÷ base | TTFT base ms | head ms | base ÷ head |
|---|---|---|---|---|---|---|
| 0 | 57.0 | 7.2 | 7.92 | 47,321 | 18,660 | 2.536 |
| 1 | 57.1 | 7.2 | 7.93 | 47,194 | 18,669 | 2.528 |
| 2 | 57.1 | 7.2 | 7.93 | 47,281 | 18,726 | 2.525 |
| 3 | 57.0 | 7.2 | 7.92 | 47,173 | 18,660 | 2.528 |

**Verdict, by the registration:** decode a **win** (median 7.92×) and TTFT **no regression** (median 2.528). #1255 is
not armed on it: its fidelity question (worst position 0.860 at prompt position 13) is open, recorded in
#1255's measurement record (`docs/measurements/wg3-coh-2026-11/` on its branch until it merges).

## Harness review (goinfer #1254, merged before its review posted), checked against the code 2026-11-23

1. `-require-backend` was stripped from both arms' TTFT servers, the head arm included. **Holds**, fixed: only the base
   binary (`WG3_BASE_SERVE`) runs without it. Every TTFT cell also records its server's decode path, so a head cell that
   fell back to the CPU would show; WG3-phi's four head cells (its own harness, the same strip) read
   `webgpu:vulkan-resident (int8int8)` each, so none is void.
2. Decode mode never assigned the calibrated prompts. **Refuted**: `scripts/bench_peer.py` reads `BENCH_PROMPTS` itself
   at import (`_PROMPTS`), and `bench.py` imports it after the run script exports the copy's path.
3. The load settle proceeded on timeout, and ran once. **Holds**, fixed: it settles before each cell and stops (exit 75)
   if the load does not fall under 0.8 in 15 minutes. bench_peer's own idle gate and WG0's per-cell void rule applied
   downstream in either case.
4. The nested timing lock, and `$?` through `tee`. The lock is **refuted**: `timing_lock.acquire` inherits a parent
   `timing_lock.py run`'s lock (`inherited=True`). The rc line printed the right code (the shell expands `$?` before the
   pipe runs), but the script went on to the TTFT cell and exited 0 after a failed decode: **holds**, fixed (each cell's
   rc is kept and the script exits with the first failure).
5. `kwin_wayland` unrecorded, a weak base fit check, a hang on a crashed server. The first is **refuted**: on nobara the
   compositor holds a CUDA context, and the compute-apps query lists it (WG3-phi's provenance:
   `/usr/bin/kwin_wayland, 19 MiB`). The other two **hold**, fixed: the base arm's decode path must be `cpu…` or
   `webgpu-staged…`, and `fit()` stops waiting when its server exits.
