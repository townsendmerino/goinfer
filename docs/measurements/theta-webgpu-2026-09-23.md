# WebGPU Theta — measured, wired, and confirmed to fix a real regression

**P22** (`docs/queue-performance.md`): "WebGPU Theta is unmeasured, and falls through to the 0.5
default — explicitly now, rather than by accident." CPU (0.5), CUDA (0.251) and Metal (0.96) all
got real probes (`docs/measurements/theta-per-backend-2026-09-01.md`,
`theta-cuda-ab-2026-09-01.md`); WebGPU never had. This measures it, wires it, and confirms the wire
fixes a real, measured regression — matching Metal's own precedent, which found the SAME defect
class (a domain-excluded measured value silently replaced by the worst available default).

## Provenance

nobara-pc, RTX 2070 SUPER (WebGPU via Vulkan), commit `1f2010ff` at measurement time. Models:
qwen2.5-coder 0.5B and 1.5B instruct, from `~/models` on local NVMe. Probe:
`gpu/theta_probe_test.go` (new — mirrors `decoder/theta_probe_test.go`,
`cuda/theta_probe_test.go`, `metal/theta_probe_test.go`, same definition: `T(n)` = wall time of one
verify pass over `n` tokens at fixed depth; `Theta = (least-squares slope of T(n)) / T(1)`).

## Measured

| model | depth | T(1) | slope | Theta |
|---|---|---|---|---|
| 0.5B | 128 | 5766 µs | 5643.4 µs/node | **0.979** |
| 0.5B | 512 | 7986 µs | 8167.3 µs/node | **1.023** |
| 1.5B | 128 | 9775 µs | 9563.7 µs/node | **0.978** |
| 1.5B | 512 | 12804 µs | 13164.7 µs/node | **1.028** |

Reproduced on a second run (0.5B @128: Theta 1.062) — consistent, tight range across both runs.
`T(n)/T(1)` is linear to n=16 in every cell (e.g. 0.5B @512: 1.00 → 2.01 → 2.98 → 3.98 → 6.05 →
8.11 → 12.20 → 16.32) — indistinguishable from a plain per-token loop, exactly the shape Metal's
own probe found (1.006-1.048, linear to n=16) before Metal's fix.

**This is despite `gpu/decoderunner.go`'s `runBatch` genuinely recording every row into ONE
command buffer and ONE `Submit`/`Poll`** (checked directly, not inferred from Theta alone): the
single-submit structure removes Go-side dispatch/sync overhead between rows, but each runner still
issues its own full set of per-layer GPU dispatches — no fusion of compute across rows — so the
GPU-side wall-clock still scales ~linearly with n. A structurally-batched submission is not what
"batched" means for the `VerifyPathReporter` contract if it doesn't move Theta; the interface is
about measured marginal cost, and code shape without a measurement predicted the wrong answer here
too, matching this repo's own recurring lesson about not trusting a "batched"-looking doc comment
over an actual measurement.

## Wired

`gpu.residentDecoder` now implements `decoder.VerifyPathReporter` (`gpu/residency.go`), returning
`(false, ...)` unconditionally — no configuration measured a different shape, and there is no
per-model branch in `ForwardN`/`runBatch` that would behave otherwise. This routes
`decoder.verifyTheta()` to the existing `sequentialVerifyTheta = 1.02` constant (already shared
with Metal's own non-batched case) rather than `thetaFor("webgpu")`'s unmeasured 0.5 default — no
new per-backend constant needed, since the measured range (0.978–1.028) already falls inside what
that shared constant represents.

## Confirmed with a real A/B, not asserted from the probe alone

`gpu/theta_ab_test.go` (mirrors `metal/theta_ab_test.go`): three interleaved arms, real hardware,
qwen2.5-coder-0.5B, int8int8 (int4 is out of scope — M-09, `decoder/spec_verify_guard.go`, refuses
ALL speculative decoding on a staged webgpu-int4 model outright; a correctness guard, not routed
around), 256-token prompt from real repo source, 48 generated tokens, median of 3, do-nothing arm
included:

| arm | median | vs off | vs Theta=0.5 |
|---|---:|---:|---:|
| off (no speculation) | 321.9 ms (296.2 ms repeat) | — | — |
| spec, Theta=0.5 (was) | 391.2 ms | **1.22× SLOWER than not speculating at all** | — |
| spec, Theta wired (1.02) | 304.4 ms | 0.95× (within noise of off) | **1.28× faster** |

**The old default was actively harmful, not merely suboptimal**: under 0.5 the controller drafted
several nodes deep believing each one was half-price, and on real hardware that cost 22% versus
just not speculating. The fix recovers it — wired Theta declines to draft almost entirely (Theta ≈
1.0 makes `Depth()` return 0 for essentially every acceptance rate, per `decoder/spec_adaptive.go`'s
own domain logic — no new logic needed here, only the missing plumbing to reach it) and lands
within noise of plain generation, as it should for a backend whose verify genuinely costs a full
step per node.

## Gates

`gofmt`/`go vet`/`staticcheck` clean under both `-tags gpu` and `-tags "gpu goinfer_testhooks"`.
Full portable `gpu` suite: 97 pass / 0 fail. `decoder`'s own Theta suite
(`TestTheta_geOneIsLegalAndDisablesDrafting`, `TestVerifyTheta_*`, `TestThetaFor_measuredValues`,
etc.) is backend-agnostic and unaffected — this change supplies a new true input to logic those
tests already cover generically, so no new decoder-level test was needed for correctness (declining
to draft changes scheduling only, never the verified token stream).

## Not established

A wider sweep (more depths, more models, larger MaxDraft) was not run — the A/B's own do-nothing
arm and the probe's four-configuration consistency are what license trusting this, matching the
bar `theta-per-backend-2026-09-01.md` set for Metal's own fix, not a claim that 1.02 is exact for
every WebGPU configuration.
