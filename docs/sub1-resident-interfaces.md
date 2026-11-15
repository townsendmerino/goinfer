<!-- citations-at: 12bf57d5b -->
# SUB1 inventory: the old Resident* interfaces (48 declared in decoder; 5 are the contract + provider, 43 below)

Generated 2026-11-14 from origin/main 12bf57d5b by an AST scan (identifier uses, comments excluded) over every module, cross-checked with grep -w.
"bridge" = decoder/resident_subsystems.go (the legacy* wrappers that make a test fake satisfy the contracts) and each backend's subsystems.go. "pin" = a backend `var _ decoder.X = (*resident)(nil)`.

| interface | file | replaced by (contract: methods) | users outside the bridge | verdict |
|---|---|---|---|---|
| ResidentAdapter | decoder/residency.go | Runtime: SetAdapter | 3 backend pin(s); decoder prod: decoder/model.go:1306(field) | delete (tier B: decoder production names it, call site moves to the contract) |
| ResidentBatchStepper | decoder/residency.go | Decode: BatchStepRange, StepBatch | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentBlockDrafter | decoder/blockspec.go | — | 1 backend pin(s); backend prod: backend/cuda/drafter.go:714(param/result); decoder prod: decoder/blockspec.go:61(param/result), decoder/blockspec.go:203(field), decoder/testhooks.go:763(param/result); 10 decoder test site(s) in 6 file(s); 2 backend test site(s) in 1 file(s) | STAYS: a value the Runtime contract returns (ResidentRuntimeSubsystem.AttachBlockDrafter): its DraftBlock/FuseContext methods are the drafter handle, not a capability probe |
| ResidentCapsReporter | decoder/residency.go | Runtime: ResidentCaps | 3 backend pin(s); 14 decoder test site(s) in 2 file(s) | delete (tier C: tests name it, they move to the contract) |
| ResidentCost | decoder/cost_resident.go | Runtime: CostSpan | 1 backend pin(s); decoder prod: decoder/cost_resident.go:15(param/result), decoder/cost_resident.go:36(param/result), decoder/model.go:1314(field) | delete (tier B: decoder production names it, call site moves to the contract) |
| ResidentDrafterHost | decoder/blockspec.go | Runtime: AttachBlockDrafter, PrefillLastNArgmax, SetBatchedCapture, BatchedCapture | 1 backend pin(s); decoder prod: decoder/blockdrafter.go:137(iface-embed), decoder/blockspec.go:202(field), decoder/blockspec.go:279(param/result); 4 decoder test site(s) in 4 file(s) | delete (tier B+C: decoder production and tests name it) |
| ResidentExpertCache | decoder/residency.go | Runtime: ExpertCacheStats, ResizeExpertCache | none | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentForward | decoder/residency.go | — | 4 backend pin(s); backend prod: backend/cuda/backend.go:139(param/result), backend/cuda/backend.go:155(param/result), backend/cuda/backend.go:161(param/result), backend/cuda/backend.go:173(param/result), backend/cuda/backend.go:183(param/result), backend/metal/backend.go:78(param/result), backend/webgpu/residency.go:235(param/result), backend/webgpu/residency.go:254(param/result); decoder prod: decoder/mc3_batch.go:65(field), decoder/model.go:53(field), decoder/residency.go:487(param/result), decoder/residency.go:575(param/result), decoder/residency.go:1505(param/result), decoder/resident_reuse.go:564(param/result), decoder/selftest_gpu.go:264(param/result), decoder/step.go:161(assert), decoder/step.go:262(assert), decoder/testhooks.go:78(param/result); 57 decoder test site(s) in 36 file(s); 14 backend test site(s) in 11 file(s) | STAYS: the base of every resident (Model.resident, Forward/ForwardN/UploadKV/TruncateTo/Reset/Close are in no subsystem); 73 test uses |
| ResidentGreedy | decoder/residency.go | Decode: ForwardArgmax | 1 backend pin(s); 1 decoder test site(s) in 1 file(s) | delete (tier C: tests name it, they move to the contract) |
| ResidentGreedyChain | decoder/residency.go | Decode: GreedyChainAvailable, GreedyChainStart, GreedyChainNext, GreedyChainStop | 1 backend pin(s); decoder prod: decoder/residency.go:194(iface-embed) | delete (tier B: decoder production names it, call site moves to the contract) |
| ResidentHiddenLast | decoder/residency.go | Prefill: HiddenLast | none | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentHybridMRoPEPrefill | decoder/residency.go | Prefill: HybridMRoPEPrefill | 3 backend test site(s) in 3 file(s) | delete (tier C: tests name it, they move to the contract) |
| ResidentImageBlocksPrefill | decoder/residency.go | Prefill: PrefillImageBlocksLast | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentImagePrefill | decoder/residency.go | Prefill: PrefillImageLast | 1 backend pin(s); decoder prod: decoder/testhooks.go:115(param/result), decoder/testhooks.go:120(param/result) | delete (tier B: decoder production names it, call site moves to the contract) |
| ResidentKVCopier | decoder/resident_reuse.go | KV: CopyKV | 3 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentKVDemoter | decoder/resident_reuse.go | KV: SetKVHostBudget, PromoteKV, CopyDemotedKV, ReserveKVHost | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentKVPoolInfo | decoder/resident_reuse.go | KV: KVPool | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentKVSlotActivity | decoder/resident_reuse.go | KV: SetKVSlotBusy | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentKVSlotter | decoder/residency.go | KV: KVSlots, UseKVSlot | 2 backend pin(s); 2 decoder test site(s) in 1 file(s) | delete (tier C: tests name it, they move to the contract) |
| ResidentKVSnapshot | decoder/kvsnap.go | — | backend prod: backend/cuda/kvsnap.go:189(param/result), backend/cuda/kvsnap.go:287(param/result); decoder prod: decoder/kvsnap.go:53(param/result), decoder/kvsnap.go:56(param/result), decoder/kvsnap.go:131(field); 3 decoder test site(s) in 2 file(s); 5 backend test site(s) in 3 file(s) | STAYS: a handle the KV contract returns (SnapshotKV): Rows/Bytes/Serves/Release are the handle, not a capability probe |
| ResidentKVSnapshotter | decoder/kvsnap.go | KV: SnapshotKV, RestoreKV, KVSnapshotBytes, KVSnapshotRollback, KVSnapshotRecurrent | 1 backend pin(s); decoder prod: decoder/kvsnap.go:411(param/result); 1 decoder test site(s) in 1 file(s) | delete (tier B+C: decoder production and tests name it) |
| ResidentKVUploadPass | decoder/resident_reuse.go | KV: BeginKVUpload, EndKVUpload | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentLayerView | decoder/backend_view.go | — | decoder prod: decoder/backend_view.go:72(param/result), decoder/backend_view.go:308(param/result) | STAYS: not a capability: BackendModelView.Layer returns it (MZ4 owns it); 24 methods, no contract method matches |
| ResidentMRoPE | decoder/residency.go | Decode: ForwardMRoPE | 3 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentMRoPEDeepstackPrefill | decoder/residency.go | Prefill: PrefillMRoPEDeepstackLast | none | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentMRoPEPrefill | decoder/residency.go | Prefill: PrefillMRoPELast | 1 backend pin(s); decoder prod: decoder/testhooks.go:127(param/result), decoder/testhooks.go:133(param/result) | delete (tier B: decoder production names it, call site moves to the contract) |
| ResidentPrefillBlocks | decoder/residency.go | Prefill: PrefillBlockRows, PrefillScratchBytes | 1 backend pin(s); decoder prod: decoder/residency.go:504(param/result) | delete (tier B: decoder production names it, call site moves to the contract) |
| ResidentPrefillChunks | decoder/prefill_chunk_scratch.go | Prefill: PrefillChunkRows, PrefillScratchBytes | none | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentPrefillKV | decoder/residency.go | Decode: ForwardNoLogits | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentPrefillMarked | decoder/kvsnap.go | Prefill: PrefillMarkAlign, PrefillLastMarked | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentPrefillRows | decoder/prefill_rows.go | Prefill: PrefillLastRows | 2 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentPrefillSpanScratch | decoder/prefill_chunk_scratch.go | Prefill: PrefillSpanScratchBytes | none | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentRaggedStepper | decoder/residency.go | Decode: StepRagged | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentRecurrentCheckpoint | decoder/recurrent_checkpoint.go | KV: RecurrentCheckpointBytes, SaveRecurrentCheckpoint, RestoreRecurrentCheckpoint, RecurrentCheckpointHeld | 2 backend pin(s); decoder prod: decoder/recurrent_checkpoint.go:37(param/result); 2 decoder test site(s) in 2 file(s) | delete (tier B+C: decoder production and tests name it) |
| ResidentRecurrentSpec | decoder/blockspec.go | Runtime: PrepareRecurrentSpec, RecurrentSpecRows, VerifyLastNArgmax, FoldVerified, FlushFold, DropFold | 1 backend pin(s); decoder prod: decoder/blockspec.go:306(param/result), decoder/forwardn.go:113(param/result); 1 decoder test site(s) in 1 file(s) | delete (tier B+C: decoder production and tests name it) |
| ResidentResidualAll | decoder/residency.go | Prefill: ResidualAll | 2 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentReusableKV | decoder/resident_reuse.go | KV: ReusableKV | 1 backend pin(s); 1 decoder test site(s) in 1 file(s) | delete (tier C: tests name it, they move to the contract) |
| ResidentReusableKVSlot | decoder/resident_reuse.go | KV: ReusableKVSlot | 2 backend pin(s); decoder prod: decoder/resident_reuse.go:589(param/result) | delete (tier B: decoder production names it, call site moves to the contract) |
| ResidentSample | decoder/residency.go | Decode: SampleAvailable, ForwardSample | 4 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentSampleChain | decoder/residency.go | Decode: SampleChainAvailable, SampleChainStart, SampleChainNext | 1 backend pin(s); decoder prod: decoder/model.go:1332(field) | delete (tier B: decoder production names it, call site moves to the contract) |
| ResidentSeedArgmax | decoder/blockspec.go | Runtime: PrefillSeedArgmax | none | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentSpanPrefill | decoder/residency.go | Prefill: PrefillSpansLast | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |
| ResidentTopK | decoder/residency.go | Decode: TopKAvailable, ForwardTopK | 1 backend pin(s) | delete (tier A: no user beyond the bridge and the backend pins) |

## Tiers

- **A** (22): ResidentBatchStepper, ResidentExpertCache, ResidentHiddenLast, ResidentImageBlocksPrefill, ResidentKVCopier, ResidentKVDemoter, ResidentKVPoolInfo, ResidentKVSlotActivity, ResidentKVUploadPass, ResidentMRoPE, ResidentMRoPEDeepstackPrefill, ResidentPrefillChunks, ResidentPrefillKV, ResidentPrefillMarked, ResidentPrefillRows, ResidentPrefillSpanScratch, ResidentRaggedStepper, ResidentResidualAll, ResidentSample, ResidentSeedArgmax, ResidentSpanPrefill, ResidentTopK
- **B** (12): ResidentAdapter, ResidentCost, ResidentDrafterHost, ResidentGreedyChain, ResidentImagePrefill, ResidentKVSnapshotter, ResidentMRoPEPrefill, ResidentPrefillBlocks, ResidentRecurrentCheckpoint, ResidentRecurrentSpec, ResidentReusableKVSlot, ResidentSampleChain
- **C** (5): ResidentCapsReporter, ResidentGreedy, ResidentHybridMRoPEPrefill, ResidentKVSlotter, ResidentReusableKV
- **stays** (4): ResidentBlockDrafter, ResidentForward, ResidentKVSnapshot, ResidentLayerView

## What "delete" means here

Every method of every deletable interface is already a method of a contract (the table), so no contract changes. A deleted interface is removed from the exported surface: the legacy* wrappers in `decoder/resident_subsystems.go` keep their structural probe for test fakes as an unexported interface of the same method set (a fake that implements `ForwardArgmax` still satisfies the Decode contract through the bridge); the backends' pins are deleted, since each backend already pins the four contracts in `subsystems.go`; tests that name an old interface assert on the contract through `SubsystemsOf`. No method body changes.
Not planned here: removing the legacy* wrappers (every test fake would have to implement a contract or the provider), MZ5's leaf move, `ResidentLayerView` (MZ4).

## Progress

- **PR1 (tier A, 22):** the backends' pins are gone (commit 1) and the 22 are unexported in decoder as `legacy<Name>` probes inside the bridge (commit 2). The old exported names no longer exist; the comments in the backends point at the contract that carries the method.
