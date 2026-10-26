# KS-H6 build: what the card loses after the KV pools (a reading, 2026-10-25)

For the KS-H6 build registration (`docs/tasks/task-kernel-survey-2026-10.md`, "KS-H6 build: gates"), part 1 of the margin: "the build's
own allocations after the KV pools". A reading, not a gate. `TestKSH6C_postPoolBreakdown` (`cuda/ksh6c_breakdown_test.go`), run by
`run-breakdown.sh`.

- Setup: nobara, RTX 2070 SUPER 8 GB, driver 595.91.07, int4, f16 KV.
- Models: Qwen3-8B Q4_K_M and Qwen2.5-7B-Instruct Q4_K_M, both at `-ctx 16384`, and Qwen2.5-Coder-1.5B unpinned (16,384).
- Each model loads in a fresh process, from `~/models`.
- Free VRAM is read through the primary context at every allocation the hook sees.
- Each large drop is attributed to the call site of the allocation before it and the one after it.

## Readings

| run | log | allocations after the pools | requested | card's drop to the end of the load |
|---|---|---|---|---|
| as serve loads (startup self-test on), rev f352d2120 | `logs/breakdown-2026-10-25-2218.log` | 70 | 0.3 MiB | 130.0 MiB, all of it in one step of 132.00 MiB |
| self-test off (`KSH6C_NO_SELFTEST=1`), rev debbe3434 | `logs/breakdown-2026-10-25-2222.log` | 26 | 0.8–0.9 MiB | 0.0–2.0 MiB |

The three models read the same on both rows.

## What takes the 132 MiB

**It is not the build.** With the self-test on, the step falls after `BuildResident` has returned. On one side of it is an
allocation of `allocRecurrentCheckpoint`, which a dense Qwen never reaches; on the other, the first allocation of a `prefillCore`
pass.
- Both belong to the decoder's startup self-test (`gpuProbeAfterResident`, `decoder/selftest_gpu.go`). Once per process, after the
  first resident is attached, it builds its four tiny fixtures as CUDA residents on the same context.
- One of them, `qwen35-tiny` (DeltaNet and a MoE FFN), launches `moe_route`. That first launch reserves the context's local-memory
  backing store, which is shared and sized by the largest kernel launched. It is 132 MiB on this card (138,412,032 B,
  `TestMoERouteFirstLaunchReservation`), and the context keeps it for the life of the process.
- The fixtures' own allocations (the other 44 small buffers on the first row) are freed when they close.

**The build's own part is about 1 MiB.** Twenty-six buffers totalling 0.8–0.9 MiB, all under the 2 MiB quantum. The card drops by one
2 MiB page on Qwen3-8B and by nothing on the other two (the driver's sub-allocator had room).

**So under the old flat margin**, the first model a serve process loads was fitted against free VRAM that the self-test then took
132 MiB of. The 384 MiB margin covered that without naming it. The figure H6-R recorded as "post-pool allocations" is this pool.
