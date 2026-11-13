# What the backends take from `decoder`, and what could move to a leaf

Row **SUB1** of `docs/tasks/task-forward-and-memory-2026-11.md` (the question that closes it: which `decoder` names could a leaf package own so a backend stops importing the 55,000-line loader).

**Provenance.** An AST scan at `311afd331`: every selector `decoder.X` in the non-test `.go` files under `backend/cuda`, `backend/metal` and `backend/webgpu` (subdirectories included; comments and tests excluded) is a use. Distinct symbols: cuda 90, metal 102, webgpu 37; 149 across the three. Each symbol's declaration was read from the non-test files of `decoder/`.

**Method.** "Movable" is a syntactic reading, not a trial move: the symbol's declaration, plus every method declared on it, plus every package-level name those mention, transitively, reaches none of `Model`, `Weights`, `LayerWeights`, `Architecture`, `Options`, `Cache`, `Session` or `Generation`. A movable symbol would stay in `decoder` as an alias (`type X = leaf.X`), as `decoder/fit` did, so no caller changes until it chooses to. The scan does not check imports a moved declaration would need (a leaf may import `internal/runtime/*`), cycles with the other leaves, or the tests that name a symbol. Those are the questions for the row that moves one. 136 of 149 read as movable and 13 do not.

**What this says.** The blocker is mainly one type. `Model` is used 94 times in production code across the three backends (cuda 28, metal 58, webgpu 8) and cannot move; the rest of the backends' surface is contracts, sentinels, constants and small data types that could. `decoder/backend_view.go:BackendModelView` is the existing narrowing of `Model` for a builder. Whether a leaf of the movable contracts, with `BackendModelView` as the way in, is the design is for the owner to decide with RF5 and SB1; this note only reports what the scan found.

| symbol | kind | cuda | metal | webgpu | movable | why |
|---|---|---|---|---|---|---|
| `Backend` | interface | 2 | 2 | 1 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `BlockDrafterWeights` | interface | 3 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `BuildDemand` | struct | 2 | 2 | 0 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `BuildQuery` | struct | 1 | 1 | 0 | no | carries `Options`, which is the load-time configuration struct |
| `CUDAKVF16AutoDecline` | func | 1 | 0 | 0 | no | takes `Options` |
| `CUDAKVF16Decline` | func | 1 | 0 | 0 | no | takes `Options` |
| `CUDAKVI8Decline` | func | 1 | 0 | 0 | no | takes `Options` |
| `DeclineResident` | func | 1 | 6 | 7 | yes | a function whose body and signature reach no `Model`, `Weights` or `Options` |
| `DecodeVerifyDiverger` | interface | 1 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `DrafterGeometry` | struct | 1 | 0 | 0 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `DrafterRows` | struct | 3 | 0 | 0 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `ErrKVPoolFull` | var | 4 | 4 | 0 | yes | a sentinel value |
| `ErrLoadAborted` | var | 5 | 0 | 0 | yes | a sentinel value |
| `ExactAttentionScoper` | interface | 1 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ExpertCacheStats` | struct | 4 | 0 | 0 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `FeatAttnLogitSoftcap` | const | 0 | 1 | 0 | yes | a constant |
| `FeatDeltaNet` | const | 0 | 1 | 0 | yes | a constant |
| `FeatEmbedScale` | const | 0 | 1 | 0 | yes | a constant |
| `FeatFinalLogitSoftcap` | const | 0 | 1 | 0 | yes | a constant |
| `FeatGatedGELU` | const | 0 | 1 | 0 | yes | a constant |
| `FeatLayerNorm` | const | 0 | 1 | 0 | yes | a constant |
| `FeatLogitScale` | const | 0 | 1 | 0 | yes | a constant |
| `FeatMoE` | const | 0 | 1 | 0 | yes | a constant |
| `FeatMoEGatedShared` | const | 0 | 1 | 0 | yes | a constant |
| `FeatNoPE` | const | 0 | 1 | 0 | yes | a constant |
| `FeatNonGatedMLP` | const | 0 | 1 | 0 | yes | a constant |
| `FeatNonGatedReLU2` | const | 0 | 1 | 0 | yes | a constant |
| `FeatPairwiseMRoPE` | const | 0 | 1 | 0 | yes | a constant |
| `FeatPairwiseRoPE` | const | 0 | 1 | 0 | yes | a constant |
| `FeatParallelBlock` | const | 0 | 1 | 0 | yes | a constant |
| `FeatPartialRotary` | const | 0 | 1 | 0 | yes | a constant |
| `FeatPerLayerRoPE` | const | 0 | 1 | 0 | yes | a constant |
| `FeatPostOnlyNorm` | const | 0 | 1 | 0 | yes | a constant |
| `FeatQKNorm` | const | 0 | 1 | 0 | yes | a constant |
| `FeatQKNormWhole` | const | 0 | 1 | 0 | yes | a constant |
| `FeatRMSAddOne` | const | 0 | 1 | 0 | yes | a constant |
| `FeatRopeMscale` | const | 0 | 1 | 0 | yes | a constant |
| `FeatSSM` | const | 0 | 1 | 0 | yes | a constant |
| `FeatSandwichNorm` | const | 0 | 1 | 0 | yes | a constant |
| `FeatSlidingWindow` | const | 0 | 1 | 0 | yes | a constant |
| `FreeBytesFor` | func | 1 | 1 | 0 | yes | a function whose body and signature reach no `Model`, `Weights` or `Options` |
| `Gemma4MoEResidentBundle` | struct | 0 | 3 | 0 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `HostRAMAvailableBytes` | func | 0 | 2 | 1 | yes | a function whose body and signature reach no `Model`, `Weights` or `Options` |
| `HostRAMBytes` | func | 0 | 1 | 1 | yes | a function whose body and signature reach no `Model`, `Weights` or `Options` |
| `Int4F32` | func | 0 | 3 | 3 | yes | a function whose body and signature reach no `Model`, `Weights` or `Options` |
| `KQuantReader` | struct | 1 | 6 | 0 | no | reads tensors out of `Weights` |
| `KVBlockRows` | const | 10 | 27 | 0 | yes | a constant |
| `KVCopyDecline` | struct | 16 | 7 | 5 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `KVF16Policy` | struct | 1 | 0 | 0 | no | holds a func over `Options` |
| `KVHostBudget` | interface | 3 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `KVPoolDetail` | struct | 3 | 4 | 0 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `KVWindowNextBase` | func | 1 | 0 | 0 | yes | a function whose body and signature reach no `Model`, `Weights` or `Options` |
| `KVWindowReusable` | func | 1 | 0 | 0 | yes | a function whose body and signature reach no `Model`, `Weights` or `Options` |
| `KVWindowStepBelowBase` | const | 1 | 0 | 0 | yes | a constant |
| `KVWindowStepOverCap` | const | 1 | 0 | 0 | yes | a constant |
| `LayerWeights` | struct | 0 | 2 | 0 | no | a field type of `Weights`, which holds every family's tensors |
| `MetalCtxCeiling` | const | 0 | 1 | 0 | yes | a constant |
| `MetalCtxDefault` | const | 0 | 1 | 0 | yes | a constant |
| `MetalDefaultKVSlots` | const | 0 | 1 | 0 | yes | a constant |
| `MetalFit` | struct | 0 | 5 | 0 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `MetalFitInput` | struct | 0 | 1 | 0 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `MetalFitWhy` | func | 0 | 1 | 0 | yes | a function whose body and signature reach no `Model`, `Weights` or `Options` |
| `MixerAttention` | const | 2 | 2 | 0 | yes | a constant |
| `MixerDeltaNet` | const | 16 | 1 | 0 | yes | a constant |
| `MixerMLA` | const | 11 | 2 | 0 | yes | a constant |
| `MixerMamba` | const | 15 | 1 | 0 | yes | a constant |
| `MixerNone` | const | 15 | 3 | 0 | yes | a constant |
| `Model` | struct | 28 | 58 | 8 | no | the loaded model itself: ~200 methods and the weights, KV, config and loader state; every backend builder takes it as input. Narrowing it is `BackendModelView`'s job, not a move |
| `NoopResidentDecode` | struct | 1 | 1 | 1 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `NoopResidentKV` | struct | 1 | 1 | 1 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `NoopResidentPrefill` | struct | 1 | 1 | 1 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `NoopResidentRuntime` | struct | 1 | 1 | 1 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `NotePrefillFallback` | func | 0 | 1 | 0 | yes | a function whose body and signature reach no `Model`, `Weights` or `Options` |
| `PrefillPartial` | struct | 1 | 0 | 0 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `PrefillPathReporter` | interface | 0 | 1 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `Prefiller` | interface | 0 | 1 | 1 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `QuantBackend4` | interface | 0 | 0 | 1 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `QuantBatchBackend4` | interface | 0 | 0 | 1 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `RegisterBackend` | func | 1 | 1 | 1 | yes | a function whose body and signature reach no `Model`, `Weights` or `Options` |
| `RegisterBuildDemand` | func | 1 | 1 | 0 | no | registers a `BuildDemandFunc`, which takes a `BuildQuery` (over `Options`) |
| `RegisterHardwareInfo` | func | 1 | 1 | 1 | yes | a function whose body and signature reach no `Model`, `Weights` or `Options` |
| `RegisterKVF16Policy` | func | 1 | 0 | 0 | no | registers a `KVF16Policy` (over `Options`) |
| `RegisterMemoryProbe` | func | 1 | 1 | 0 | yes | a function whose body and signature reach no `Model`, `Weights` or `Options` |
| `RegisterPlanSlack` | func | 1 | 0 | 0 | no | registers a `PlanSlackFunc`, which takes `*Model` |
| `ReleasableDrafterWeights` | interface | 1 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidencyBackend` | interface | 1 | 1 | 0 | no | its methods take `*Model` |
| `ResidencyPricer` | interface | 0 | 1 | 0 | no | its methods take `*Model` |
| `ResidentAdapter` | interface | 1 | 1 | 1 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentAdapterLayer` | alias adapter.ResidentAdapterLayer | 5 | 5 | 3 | yes | already defined in a leaf (`adapter.ResidentAdapterLayer`); the decoder name is a forwarding alias |
| `ResidentAdapterProj` | alias adapter.ResidentAdapterProj | 2 | 2 | 0 | yes | already defined in a leaf (`adapter.ResidentAdapterProj`); the decoder name is a forwarding alias |
| `ResidentBackendFeatures` | func | 0 | 0 | 1 | yes | a function whose body and signature reach no `Model`, `Weights` or `Options` |
| `ResidentBackendMoECap` | func | 2 | 2 | 1 | yes | a function whose body and signature reach no `Model`, `Weights` or `Options` |
| `ResidentBatchDraw` | struct | 2 | 1 | 0 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `ResidentBatchOut` | struct | 5 | 4 | 0 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `ResidentBatchSeq` | struct | 2 | 2 | 0 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `ResidentBatchStepper` | interface | 1 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentBlockDrafter` | interface | 3 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentCaps` | struct | 3 | 3 | 3 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `ResidentCapsReporter` | interface | 1 | 1 | 1 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentCost` | interface | 1 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentDecodeSubsystem` | interface | 1 | 1 | 1 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentDrafterHost` | interface | 1 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentFeature` | type | 0 | 4 | 0 | yes | a named type with no `Model`, `Weights` or `Options` in its closure |
| `ResidentForward` | interface | 6 | 2 | 3 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentGreedy` | interface | 1 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentGreedyChain` | interface | 0 | 1 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentImageBlocksPrefill` | interface | 0 | 1 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentImagePrefill` | interface | 0 | 1 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentKVCopier` | interface | 1 | 1 | 1 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentKVDemoter` | interface | 1 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentKVPoolInfo` | interface | 0 | 1 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentKVSlotActivity` | interface | 0 | 1 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentKVSlotter` | interface | 1 | 0 | 1 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentKVSnapshot` | interface | 4 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentKVSnapshotter` | interface | 1 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentKVSubsystem` | interface | 1 | 1 | 1 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentKVUploadPass` | interface | 0 | 1 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentMRoPE` | interface | 1 | 1 | 1 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentMRoPEPrefill` | interface | 0 | 0 | 1 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentMixer` | type | 3 | 0 | 0 | yes | a named type with no `Model`, `Weights` or `Options` in its closure |
| `ResidentPrefillBlocks` | interface | 0 | 1 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentPrefillKV` | interface | 0 | 0 | 1 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentPrefillMarked` | interface | 1 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentPrefillRows` | interface | 1 | 1 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentPrefillSubsystem` | interface | 1 | 1 | 1 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentPrice` | struct | 0 | 3 | 0 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `ResidentRaggedOut` | struct | 4 | 0 | 0 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `ResidentRaggedSeq` | struct | 3 | 0 | 0 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `ResidentRaggedStepper` | interface | 1 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentRecurrentCheckpoint` | interface | 1 | 1 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentRecurrentSpec` | interface | 1 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentResidualAll` | interface | 1 | 1 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentReusableKV` | interface | 0 | 1 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentReusableKVSlot` | interface | 1 | 1 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentRuntimeSubsystem` | interface | 1 | 1 | 1 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentSample` | interface | 1 | 2 | 1 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentSampleChain` | interface | 0 | 1 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentSpanPrefill` | interface | 1 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentSubsystemProvider` | interface | 1 | 1 | 1 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `ResidentSubsystems` | struct | 2 | 2 | 2 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `ResidentTopK` | interface | 1 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `StreamedDrafterWeights` | interface | 1 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `StreamingDrafterHost` | interface | 1 | 0 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `TopKRow` | struct | 15 | 0 | 0 | yes | plain data; no `Model`, `Weights` or `Options` in its closure |
| `VerifyCostReporter` | interface | 0 | 1 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `VerifyPathReporter` | interface | 1 | 1 | 0 | yes | an interface over plain values and other movable contracts; no `Model`, `Weights` or `Options` in its closure |
| `WebGPUCtxCeiling` | func | 0 | 0 | 1 | yes | a function whose body and signature reach no `Model`, `Weights` or `Options` |
| `WeightsMemFraction` | const | 0 | 1 | 2 | yes | a constant |
| `WindowedKVSlack` | const | 1 | 0 | 0 | yes | a constant |
