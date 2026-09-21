# R2 `attention_fa` — speed band, pre-registered 2026-09-21

Pre-registered per R2's own brief (`docs/tasks/red-october.md`, "Record: … before the first
cell"), before any cell of this measurement runs. The band itself was registered in the brief on
2026-09-18 and is restated here unchanged; this file adds the exact protocol and what will and
will not be counted.

## What is being decided

Whether `GOINFER_METAL_ATTN_FA=1` (`attention_fa` + `attention_fa_combine`, dense-GQA hd=128 decode
attention gridded by kvHead × split, engaging only at `curNKeys ≥ attnFADepthFloor` = 1536) may
become a candidate for default-on. Precondition already met: gate (1) (isolation, 2026-09-19) and
gate (3) (teacher-forced fidelity vs the CPU f32 reference at K=3900, PASSES —
`r2-attn-fa-rootcause-2026-09-21.md`). This band is the remaining condition.

## Band (verbatim from the brief)

**Depth bench, 1.5B, same protocol, min-of-batches: ships at ≥60 tok/s at 4000 AND ≥58 at 2048
AND no regression beyond 3% at 128; parked at 48–60 at 4000; killed below 48.** Decision rule:
"the band above, on the depth bench. A shallow regression beyond 3% at 128 parks the kernel
regardless of the depth win, because the 1.5B at 128 is the cell the promotion gate reads first."

Standing (the brief, 2026-09-13 depth bench, W4A8 1.5B): 72.3 / 67.9 / 51.4 / 40.4 tok/s at
128 / 512 / 2048 / 4000. Re-measured today for R1 (single run, shipped arm): 69.9 / 66.7 / 48.4 /
37.6. Peer (Ollama, `completed/metal-verdict.md` row 4): 85.2 / 79.1 / ~80 / 77.5.

## Instrument of record

`metal/depth_bench_test.go` `TestZZ_metalDepthBench` (`GOINFER_METAL_DEPTH_BENCH=1`): one resident,
`qwen2.5-coder-1.5b` int4, KV warmed incrementally, steady greedy decode via `ForwardArgmax`
(synchronous `Begin`/`End` command buffer per token, `encodeTrunkInto` → the same
`encodeAttentionResidualWith` dispatch site that gates `attention_fa`), best-of-5 batches at each
of {128, 512, 2048, 4000}, 4000 clamped inside the resident KV. **Three runs per arm, interleaved
(off, on, off, on, off, on), each run its own process** — one resident at a time on this 16 GB
machine, the same serialization every real-checkpoint test today has used. The arm is selected by
the process environment (`GOINFER_METAL_ATTN_FA` is read once at `BuildResident`), so it cannot be
toggled inside one run; interleaving processes is the drift control available. Per depth, the
number scored is the **median of the three runs' best-of-batches**; the spread is reported.

Structural note, stated before measuring: at depths 128 and 512 the two arms dispatch the
**identical** shipped kernel (`canUseAttnFA` declines below 1536), so any difference there is
run-to-run noise plus the one extra `SetU32` pair `setPos` now performs per token (the 2026-09-20
race fix). The "≤3% at 128" criterion is therefore expected to be met by construction; it is
measured anyway because it is registered.

## Served cross-check (reported, not deciding)

`scripts/bench_peer.py`, `BENCH_MODELS=1.5B BENCH_BACKENDS=metal BENCH_DEPTH_BACKEND=metal
BENCH_DEPTHS=3900 BENCH_ENGINES=goinfer,ollama` (Phase A gives depth 128; Phase B's calibrated
3900-token prompt is the "~4000" every standing Metal depth figure already uses), greedy, script
defaults (2 runs × 8 completions × 64 tokens), `BENCH_MAX_LOADAVG=2.0` as the harness registers —
once with the env var unset, once with `GOINFER_METAL_ATTN_FA=1` (same two-invocation limitation
as above; Ollama runs in both as the drift control). This measures the real serving loop
(`execLoop`, pipelined command buffers) rather than the synchronous depth bench; it is reported
beside the band, and a disagreement between the two instruments is itself a finding.

## Not measured, said in advance

The brief's "cross-check the depth term's GPU-busy time with the `metal/tax_test.go` timestamps":
`tax_test.go` is a nop-kernel binding-tax microbenchmark, not a GPU-busy timer for a real forward,
so that cross-check is not available as written. The resident does record command-buffer GPU
timestamps (`gpuStart`/`gpuEnd` in `forwardLogits`) but the depth bench's `ForwardArgmax` path
does not surface them; adding that is a separate change, not made for this run.

## Decision

Read straight off the band: median-of-three depth-bench tok/s at 4000 and 2048, and the 128 cell's
ratio to the shipped arm's own 128 cell. Ambiguous → parked, per the band's own 48–60 zone. No
re-baselining of the floor if the shipped arm's numbers come in below the 2026-09-13 standing;
the band is absolute tok/s, as registered.
