# The cuda-device-suite night job's red `cuda/kernels`: a card-free read (2026-10-30, nobara-sb01)

The coordinator asked for a read of nobara's 18:00 `cuda-device-suite` night job. It ran at `e99594541` (origin/main at 18:01)
and came back with 896 passed and 55 failed, all in `cuda/kernels` (`verdict-2026-10-30.log`). Issue #229 has been collecting the
same red automatically since 10-27.

**Not the driver, and not today's commits.** The package has been red every night since 10-26, on driver 595.91.07 as well as on
tonight's 595.112.07 (`first-failures.txt`, from each night's `go-test.jsonl`):

| night | rev | failed | first failure | first `CUDA_ERROR_ILLEGAL_ADDRESS` |
|---|---|---|---|---|
| 10-23 | 5275808c8 | none | — | — |
| 10-24, 10-25 | d9a1cb4dd, e3ef27691 | 2 in `cuda` (`TestKEqVCopy_…`) | — | — |
| 10-26 | 3afc2d90a | 3 in `cuda/kernels` | `TestAllocGranularity` | — |
| 10-27 | 11fda06e9 | 45 + 2 | `TestAllocGranularity` | `TestAttnKernelDigests` (`prefill_batched_kv16/attn_batched`) |
| 10-28 | d7185c1ce | 47 | the same | `TestAttnKernelDigests` (`decode_splitkv/splitkv_vsum`) |
| 10-29 | aeec6b0c1 | 51 | the same | `TestAttnKernelDigests` (`decode_splitkv_kv16/splitkv_vsum`) |
| 10-30 | e99594541 | 53 (55 with subtests) | the same | the same |

**When each began: the SB1 test moves.**
- `TestAllocGranularity` first fails on the first night whose revision contains 05464762c (10-26 16:11, "move 25 kernel-level test
  files out of package cuda"). 10-25's revision does not contain it.
- The `ILLEGAL_ADDRESS` first appears on the first night that contains d014819e3 (10-26 22:10, "4 digest tests follow"), which moved
  `attn_digest_test.go` into `cuda/kernels`.

Both tests passed in package `cuda`. `TestAttnKernelDigests` passed on today's driver when run by itself (nobara-ci01, 11:35,
verifying #518).

**What the failures look like.**
- **`TestAllocGranularity`** runs before anything faults. It reads the driver charging a 5 MiB allocation as 4.00 MiB, 6 MiB as 5.94
  and 9 MiB as 7.75: less than requested, so free memory moved for another reason during the measurement. Tests that run before it in
  the new package (the act-group kernel tests) allocate and free, and an allocation-delta measurement is order-sensitive.
- **`CUDA_ERROR_ILLEGAL_ADDRESS` is sticky:** after it, every CUDA call in the process fails, `cuModuleLoadData` included. That is why
  most of the 55 are fallout; only 8 tests passed after it.
- **The first fault moves** between three kernels night to night, and the digests themselves are unchanged. That is the pattern of
  an out-of-bounds access whose effect depends on memory layout, and so on which tests ran before it in the same process. A fault is
  also reported at the next synchronize, so the faulting launch can be earlier than the kernel named.

**What reading cannot settle.** Reading `splitkv_vsum` and `KV_SUM_RUNS` (`cuda/kvpage.cuh`) finds no row or table read past `nKeys`.
The rig sizes each KV buffer to exactly `nKeys × kvDim`, with an identity block table (`attn_digest_rig_test.go`). nobara has no
CUDA toolkit, so no `compute-sanitizer`. The faulting launch has to be found on the card, by test order (registered in the task doc,
"The red cuda/kernels package").

## G-KR1, the card bisection (2026-10-31 06:08–06:14 PDT, nobara-sb01): not test order; two test processes on one card

Seven runs of the `cuda/kernels` test binary at f18a97c21 (origin/main of the morning), under the timing lock held exclusively. Driver
595.112.07, the card at 532 MiB (the desktop compositor's 20 MiB context only), load average under 1.7 at every start. Logs:
`g-kr1-2026-10-31/`.

| run | what | result |
|---|---|---|
| 1 | `TestAttnKernelDigests` alone | pass |
| 2 | `TestAllocGranularity` alone | pass (6.00 / 6.00 / 10.00 MiB) |
| 3 | the 11 tests before the digests, then the digests, in package order | all pass |
| 4 | run 3's set, with package `cuda`'s test binary running beside it | `TestAllocGranularity` fails (6.06 MiB for a 6 MiB request); the digests fault, `prefill_batched_kv16/attn_batched: CUDA_ERROR_ILLEGAL_ADDRESS` |
| 5 | run 4 on the fixed rig | the digests pass (36 tests of package `cuda` ran beside them); `TestAllocGranularity` still fails |
| 6 | run 5 again | the digests pass |
| 7 | the whole fixed package, alone | 64 pass, 8 skip, 0 fail |

**The registered rule did not anticipate the outcome.** Neither of its cases held: the digests passed alone and passed after the
whole prefix (runs 1 and 3). The variable was not test order but another process. The night job runs `go test ./...` in `cuda/`, and
`go test` runs package binaries in parallel. Before SB1, these tests ran in package `cuda`'s process. Now `cuda/kernels` is its own
binary and shares the card with package `cuda`, `gpudev`, `qwencuda` and `visioncuda`. On 10-30 the first fault came 7 s into the run,
while `cuda`'s `TestB35Snapshot_restoreEqualsCold` and `gpudev`'s `TestCUDA_asyncLaunchThenSync` were running (`go-test.jsonl`).
Run 4 reproduced both failures by recreating that pairing alone.

**The out-of-bounds access: the digest rig's uploads.**
- `adRig`'s `f32`, `i32` and `kv` copied with a pageable `cuMemcpyHtoD`, which returns once the bytes are staged, not once they
  land.
- The rig launches on a non-blocking stream, which is not ordered after that copy.
- A KV kernel could therefore read its block table (`kvt`, through `i32`) before the identity table landed. It got whatever the
  fresh allocation held, often an earlier case's floats read as block indices, and read out of bounds.
- Alone, the copy finishes in time. With another process's copies on the card, it does not. That is why the faulting kernel moved
  from night to night.
- The package's own `gcKVIdentityTab` already synchronizes the context after its copy, for this reason. The rig did not.

**`TestAllocGranularity`** reads the device's free memory, which another process's allocations move. It is not a driver change:
alone it reads 6 / 6 / 10 MiB exactly, as before.

**The fix:**
- Every rig upload goes through `adUpload`, which synchronizes the context after the copy.
- `TestAttnDigestRig_uploadsSynchronize`, an untagged source scan that CI runs, holds the rig to that. It was red on the old rig
  (`redfirst-scan-on-old-rig.log`) before the fix made it green.
- The device suite runs its packages with `-p 1`, as `cmd/gate`'s gpu suite already does.

The night suite's wall time rises from package `cuda`'s 12.3 min on 10-30 to about that plus the other packages' ~1 min. The
`TestKEqVCopy_…` failures in package `cuda` (10-24, 10-25) are a different matter, not touched here.
