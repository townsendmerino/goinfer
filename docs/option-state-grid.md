# Option × path and state × lifecycle grids

**Generated** from `decoder/optiongrid.go` and `decoder/cachestate.go` by `TestOptionStateGrid_fresh`; do not edit.
Regenerate: `go test ./decoder -run OptionStateGrid -update`. Design and history:
[`tasks/task-option-path-admission-2026-10.md`](tasks/task-option-path-admission-2026-10.md) §4.

## Load options × execution paths

**tested** names a test that drives the option through the path; **declined** names the function where the path refuses it;
**untested** runs today with no test that drives it (47 cells; `TestOptionGrid_ratchet` lets that number only fall);
**n/a**: the option does not reach the path (reasons below the table).

| option | CPU decode | CPU batched prefill | CPU batched decode | GPU resident decode | GPU resident prefill | speculative verify | session reuse and snapshot |
|---|---|---|---|---|---|---|---|
| `ActQuantGroup` | tested: `TestActQuantGroup_perModel` | tested: `TestOptionPath_cpuBatchedPrefill` | untested | tested: `TestActGroup_phi3ResidentMatchesCPU` | untested | tested: `TestOptionPath_specVerify` | n/a |
| `Backend` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `CPUBatchDecode` | n/a | n/a | tested: `TestEnableCPUBatch_policy` | n/a | n/a | tested: `TestOptionPath_specVerify` | untested |
| `EmbedInt4` | untested | tested: `TestOptionPath_cpuBatchedPrefill` | untested | untested | untested | tested: `TestOptionPath_specVerify` | n/a |
| `ExactPrefill` | n/a | tested: `TestOptionPath_cpuBatchedPrefill` | n/a | n/a | untested | tested: `TestOptionPath_specVerify` | n/a |
| `KVPrecision` | n/a | n/a | n/a | untested | declined at `PrefillPath` (`TestPrefill_declinesInt8KV`) | untested | untested |
| `KVQuant` | untested | tested: `TestOptionPath_cpuBatchedPrefill` | untested | n/a | n/a | tested: `TestOptionPath_specVerify` | tested: `TestKVI8_snapshotRoundtrip` |
| `Knobs` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `MoECacheExperts` | n/a | n/a | n/a | untested | untested | untested | untested |
| `MoECacheSlots` | n/a | n/a | n/a | untested | untested | untested | untested |
| `MoEPager` | untested | untested | untested | n/a | n/a | untested | n/a |
| `Quant` | tested: `TestDecodeParityInt4` | tested: `TestOptionPath_cpuBatchedPrefill` | untested | untested | untested | declined at `SpecDecodeConflict` (`TestSpecDecodeConflict_refusesStagedWebGPUInt4`) | untested |
| `ResidentContext` | n/a | n/a | n/a | untested | untested | untested | untested |
| `ResidentKVSlots` | n/a | n/a | n/a | untested | untested | untested | untested |
| `ResidentPrefillChunk` | n/a | n/a | n/a | n/a | untested | untested | n/a |
| `StreamWeights` | untested | untested | untested | n/a | n/a | untested | n/a |
| `WeightCacheBytes` | untested | untested | untested | n/a | n/a | untested | n/a |

**n/a reasons.**

- `ActQuantGroup`: an activation-quantization setting; the session's cache does not depend on it
- `Backend`: selects which paths run; it is the row, not an input to one
- `CPUBatchDecode`: selects whether decode joins a batch; a single generation is unaffected
- `CPUBatchDecode`: a decode setting
- `CPUBatchDecode`: chooses the CPU's MC3c batching; a resident batches through its own MC3
- `EmbedInt4`: a weight format; the session's cache does not depend on it
- `ExactPrefill`: selects how a prompt is ingested; decode is one token
- `ExactPrefill`: selects prefill numerics; a session's reuse rules do not depend on them
- `KVPrecision`: "Ignored off the residency path"
- `KVQuant`: the CPU KV cache's precision ("selects the CPU KV cache storage precision"); a resident uses KVPrecision
- `Knobs`: routes per-model operator knobs by name to the code that reads each (knobs.go); each knob is an environment read listed in testdata/env_reads.txt and owned there
- `MoECacheExperts`: "CUDA and Metal residency; the CPU's expert paging is StreamWeights"
- `MoECacheSlots`: "Only meaningful with MoECacheExperts", a residency option
- `MoEPager`: the CPU expert pager's backing mode
- `MoEPager`: pages weights, not the session's cache
- `ResidentContext`: "Ignored off the residency path"
- `ResidentKVSlots`: asks a GPU-resident backend for independent KV caches
- `ResidentPrefillChunk`: chunks a resident's batched prefill under MC3
- `ResidentPrefillChunk`: chunks prefill; decode is one token
- `ResidentPrefillChunk`: schedules a prefill; the session's cache is the same either way (chunk-invariant)
- `StreamWeights`: CPU expert paging of an mmap-backed .giw ("the CPU's expert paging is StreamWeights")
- `StreamWeights`: pages weights, not the session's cache
- `WeightCacheBytes`: the budget of StreamWeights, a CPU paging option
- `WeightCacheBytes`: budgets weights, not the session's cache

**Load-only options** (read while loading, by no path afterwards):

- `AcceptSlowMoE`: an acknowledgement checked when a paged-MoE load is planned; a no-op afterwards
- `BackendAuto`: tells a backend at load that "auto" chose it, so it may decline a precision it would requantize
- `DisableFit`: selects load-time fit-by-default behaviour (resolveCtxCapFit, guardFit)
- `ExtraResidentBytes`: prices a companion allocation when the resident is planned
- `ExtraResidentKVPerPosition`: prices a companion allocation's K/V when the resident context is chosen
- `LoRA`: merged into the base weights at load ("merged into the base at load"), so every path runs the merged weights; compute-time adapters are Session state, in the cache-state grid
- `LoadAbort`: checked between layers during a GGUF weight build; nil afterwards in effect
- `ResidentKVSlotsDefault`: tells the backend at load that ResidentKVSlots is the caller's default, not a chosen count
- `noSelfTest`: the resident self-test's own fixture loads; read by the self-test at load

## Kinds of cache state × lifecycle paths

Every `KVCache` field belongs to one of these kinds, or is declared geometry, a counter or scratch (`TestKVCache_everyFieldHasAState`).
`Snapshot` and a partial `TruncateTo` read this table.

| kind | partial rewind | full reset | snapshot | session prefix reuse | speculative rollback |
|---|---|---|---|---|---|
| positional KV | exact | cleared | persisted | exact | exact |
| sliding-window ring | exact unless wrapped | cleared | persisted | exact unless wrapped | refused |
| int8 KV + scales | exact | cleared | persisted | exact | exact |
| MLA latent | exact | cleared | refused | exact | exact |
| DeltaNet state | inexact → cold | cleared | refused | inexact → cold | refused |
| short-conv window | inexact → cold | cleared | refused | inexact → cold | refused |
| Mamba-2 state | inexact → cold | cleared | refused | inexact → cold | refused |
| KDA state | inexact → cold | cleared | refused | inexact → cold | refused |
| image blocks | inexact → cold | cleared | refused | inexact → cold | transient |
| m-RoPE positions | inexact → cold | cleared | refused | inexact → cold | transient |
| deepstack rows | transient | transient | transient | transient | transient |
| capture rows | transient | transient | transient | transient | transient |
| tree-verify mask | transient | transient | transient | transient | transient |
| LoRA adapter binding | kept | kept | persisted | cold on mismatch | kept |

**Reasons.**

- sliding-window ring × partial rewind: ring.truncate returns false for a wrapped ring rewound by more than one position (C1); TruncateTo passes it on
- sliding-window ring × snapshot: the snapshot writes each ring's physical rows and count (C2)
- sliding-window ring × session prefix reuse: rewindForReuse goes cold on an inexact TruncateTo
- sliding-window ring × speculative rollback: specRollbackSafe refuses every SlidingWindow > 0 model: rollback sites do not consume an inexact rewind (audit C-04)
- MLA latent × partial rewind: TruncateTo re-slices the per-layer latent by latentDim
- MLA latent × snapshot: the snapshot format does not carry the latent store
- DeltaNet state × partial rewind: mutated in place per token; no per-position history to rewind to
- DeltaNet state × full reset: resetRecurrent (audit C-01, C-03)
- DeltaNet state × snapshot: the snapshot format does not carry recurrent state
- DeltaNet state × session prefix reuse: follows the rewind; rewindForReuse goes cold
- DeltaNet state × speculative rollback: specRollbackSafe refuses every model with recurrent state
- short-conv window × partial rewind: mutated in place per token; no per-position history to rewind to
- short-conv window × full reset: resetRecurrent (audit C-01, C-03)
- short-conv window × snapshot: the snapshot format does not carry recurrent state
- short-conv window × session prefix reuse: follows the rewind; rewindForReuse goes cold
- short-conv window × speculative rollback: specRollbackSafe refuses every model with recurrent state
- Mamba-2 state × partial rewind: mutated in place per token; no per-position history to rewind to
- Mamba-2 state × full reset: resetRecurrent (audit C-01, C-03)
- Mamba-2 state × snapshot: the snapshot format does not carry recurrent state
- Mamba-2 state × session prefix reuse: follows the rewind; rewindForReuse goes cold
- Mamba-2 state × speculative rollback: specRollbackSafe refuses every model with recurrent state
- KDA state × partial rewind: mutated in place per token; no per-position history to rewind to
- KDA state × full reset: resetRecurrent (audit C-01, C-03)
- KDA state × snapshot: the snapshot format does not carry recurrent state
- KDA state × session prefix reuse: follows the rewind; rewindForReuse goes cold
- KDA state × speculative rollback: specRollbackSafe refuses every model with recurrent state
- image blocks × partial rewind: the blocks are position ranges with no per-position rewind; a rewind below one would let new text attend bidirectionally
- image blocks × full reset: resetMultimodal (audit M-25)
- image blocks × snapshot: the snapshot format does not carry image blocks
- image blocks × session prefix reuse: follows the rewind
- image blocks × speculative rollback: the speculative loops take text prompts; nothing they run puts an image block in their cache
- m-RoPE positions × partial rewind: positions and the decode delta are set by the image prefill and have no per-position rewind
- m-RoPE positions × full reset: resetMultimodal (audit M-25)
- m-RoPE positions × snapshot: the snapshot format does not carry m-RoPE positions
- m-RoPE positions × session prefix reuse: follows the rewind
- m-RoPE positions × speculative rollback: the speculative loops take text prompts; nothing they run sets m-RoPE positions in their cache
- deepstack rows × partial rewind: GenerateQwenVL sets the rows for the image prefill and clears them right after (generate_vl.go)
- deepstack rows × full reset: cleared after the prefill that set them
- deepstack rows × snapshot: cleared after the prefill that set them
- deepstack rows × session prefix reuse: cleared after the prefill that set them
- deepstack rows × speculative rollback: cleared after the prefill that set them
- capture rows × partial rewind: captured rows are overwritten by every forward before anything reads them
- capture rows × full reset: overwritten by every forward
- capture rows × snapshot: overwritten by every forward
- capture rows × session prefix reuse: overwritten by every forward
- capture rows × speculative rollback: overwritten by every forward
- tree-verify mask × partial rewind: nothing in production sets the tree fields since EAGLE tree drafting was removed (be9aeea8, 2026-09-24); only tests do, on caches they own
- tree-verify mask × full reset: unset in production (be9aeea8)
- tree-verify mask × snapshot: unset in production (be9aeea8)
- tree-verify mask × session prefix reuse: unset in production (be9aeea8)
- tree-verify mask × speculative rollback: unset in production (be9aeea8)
- LoRA adapter binding × partial rewind: the adapter is the stream's binding, not sequence state: positions are rewound under the same projections
- LoRA adapter binding × full reset: a reset session keeps its adapter; serve's per-adapter session LRUs rely on it
- LoRA adapter binding × snapshot: format v3 records the adapter name; LoadSession rebinds it, or refuses when the model has not loaded it
- LoRA adapter binding × session prefix reuse: rewindForReuse goes cold when the bound adapter is not the one the cached prefix was built under
- LoRA adapter binding × speculative rollback: a rollback stays under the same projections
