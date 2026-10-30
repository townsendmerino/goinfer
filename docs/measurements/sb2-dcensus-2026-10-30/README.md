# SB2 memory reads, step 3, PR 3: the derived census (2026-10-30, nobara-sb01)

Gates registered before the code: `docs/tasks/task-subsystems-2026-10.md`, "SB2 memory reads, step 3, PR 3 — the derived census".
CPU only. No device run: the derive's error against the card is G-DM2's (`../sb2-derived-2026-10-29/README.md`).

## What the census now says

`TestBackendMemoryCensus_cuda` (`cuda/sb2_census_test.go`), golden `cuda/testdata/sb2_cuda_mem_census.amd64.txt.gz`: 558 lines,
279 planned cells × the two nobara profiles. Each line's head is unchanged from step 2, byte for byte (the profile's figures, the
planned context, the margin, the weights verdict). Its `kv: step 3` tail is replaced by `derived(G-DM2 band -62/+58 MiB):`,
then each reading the build takes, as derived (`kv@`, `ckpt@`, `grantPre@`, `slots@`, `grant@`, `prefill@`), then the decisions.

| decision | over the 558 lines |
|---|---|
| KV fit | none refuse; context 4096 on 550, 16384 on 8 (qwen3next-tiny); one slot each |
| recurrent checkpoint | kept on 102, none on 456 (no recurrent state) |
| windowed slack, expert-cache slots | not reached: layer B-cuda's option cells hold no windowed-KV or expert-cache load |
| prefill width | 2048 on all |
| near-threshold mark `~` | none |

So the census covers three of the five post-allocation decisions. Adding windowed and expert-cache cells would be a new cell set
for layer B-cuda, not part of this step.

## Gates (`gates.log`)

- **G-DC1: passes.** Built twice in one process, equal. It matches its rewritten golden and covers all 279 planned cells × 2
  profiles. On every line the derive's pre-allocation decisions equal `planMemory`'s over step 2's figures: no `G-DC1` error.
- **G-DC2: passes.** With the derived KV reading skewed by −8 GiB, all 558 lines refuse at `kv`. The second half of the check,
  that a line refused before the KV fit stays unchanged, is vacuous here: no clean line refuses before it.
- **G-DC3: passes.** `TestNearMarks` covers each decision changed at −64 MiB alone, at +64 MiB alone and at neither, and a
  refusal appearing at one skew.
- Step 2's own plant (`TestBackendMemoryCensus_cuda_plant`) still passes: all 558 lines refuse at the weights, 8 against the
  post-pool figure.

## Found while writing it: the derive marked readings the build does not take (red first, `readflags-red.log`)

Derive mode marked the checkpoint and grant sites as read on every load. The build reads the checkpoint's figure only with
recurrent state to keep, and a grant's only on a windowed load (`cudaResident.kvGrantRollback`). That broke `memReadings`'
contract ("false means no reading"), and the census would have printed readings that never happen. The figures were still
derived; only the flags were wrong, and no decision read them. G-DM1's `TestDeriveMemory` now asserts the flags: llama-tiny reads
only the width, and qwen3next-tiny reads the checkpoint and no grant. Red before the fix, green after. The census also prints
`cacheSlots` only where the cache is sized.

## Checks

gofmt clean. `go vet` under no tag, `cuda`, and `cuda goinfer_testhooks`. staticcheck (CI's pinned build) under the same three:
nothing in the changed files. Under `-tags cuda` alone it lists 20 pre-existing U1000s, test-hook-only callers, which also shows
it analysed. `internal/layout` green, and no new test file in package `cuda`. `TestPTX_matchesSourcesAndBindings` (the source scan
over cuda's non-test files) passes. `touched.py` was not run: it would have run cuda's device tests while nobara-ci01's
3.6-hour card run was next in line.
