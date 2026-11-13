package decoder

import "fmt"

// layoutBackend is the backend whose weight layout a load builds: the one its sidecar stand-in is for, else the one it runs on.
func (o Options) layoutBackend() string {
	if o.SidecarFor != "" {
		return o.SidecarFor
	}
	return o.Backend
}

// Options configures Load.
type Options struct {
	noSelfTest bool   // set only by the resident self-test's own fixture loads (selftest_gpu.go); unexported so the type stays comparable and the API unchanged
	Backend    string // "cpu" (default), "webgpu", "cuda", "metal", or "auto" (AutoBackend); a name not compiled in falls back to cpu
	Quant      string // "" (f32), "int8" (weight-only per-row), "int8int8" (full int8×int8 W8A8), "int4" (group-wise), or "bf16" / "f16" (16-bit weights widened per tile: bit-identical to f32 over a bf16 / f16 checkpoint, half the memory, CPU only)
	LoRA       string // optional PEFT adapter dir (adapter_config.json + adapter_model.safetensors), merged into the base at load. Safetensors base only.
	// DirectLoad builds a .gguf or a safetensors directory's weights in the heap. Off (the default) on darwin and linux, Load
	// opens the source's sidecar .giw instead, transcoding it once beside the source when it is missing or stale, so the
	// weights are a mapping of that file rather than an anonymous copy; serve and chat do the same. A .gguf whose sidecar
	// cannot be written (a read-only directory, a full disk) is refused, naming this field. A q4k .gguf takes its sidecar too,
	// except under StreamWeights. A test binary loads directly, unless the test turns the default on.
	DirectLoad bool
	// KVPrecision selects the GPU residency KV cache precision: "" / "auto" (the default: f16 on CUDA for a family the f16 kernels cover and
	// the gates cleared, otherwise what the backend always did: f32, or f16 on Metal), "f32" (bit-exact, on every backend), "f16" (lossy, twice
	// the context) or "i8" (lossy, four times). Ignored off the residency path. See task-gpu-f16-kv.md / task-gpu-kv-i8.md and
	// task-cuda-f16-kv-2026-10.md.
	KVPrecision string
	// MoECacheExperts streams routed MoE experts host→VRAM per token instead of holding the whole
	// expert stack resident — the path to running a model whose experts exceed VRAM with every
	// expert still executing on the GPU. Off by default; bit-identical to fully-resident when on
	// (cuda.TestGemma4MoE_cacheExpertsBitExact_*; on Metal metal.TestMoECacheExperts_bitExactMetal, against
	// a resident load that prefills the same way: a paged Metal model's prompt takes a prefill
	// bit-identical to sequential, so beside the default batched f16 prefill a prompt over its floor
	// differs exactly as ExactPrefill does). The experts must be int4: Metal declines int8 experts by
	// name. CUDA and Metal residency; the CPU's expert paging is StreamWeights.
	MoECacheExperts bool
	// MoEExpertsUnpinned makes a CUDA expert cache (MoECacheExperts) upload its slots from pageable pages of a file holding the packed experts,
	// written once beside the model and unlinked while mapped, instead of a pinned host copy: the copy's host memory (about 12 GiB on Gemma 4 26B)
	// is not held, at a measured decode cost of 0.82x at short context and 0.75x at long (docs/measurements/mem10-2026-11/ab-registration.md).
	// Off by default; the fit guard names it when the pinned copy is what does not fit.
	MoEExpertsUnpinned bool
	// FlashDecodeSoftcap lets a model with an attention softcap (Gemma 2) take CUDA's flash-decode lane, in its capped variant, for decode and
	// verify rows past the lane's key floor. Off by default: such a model decodes on the exact split-KV lane, and the flash-decode lane's
	// outputs differ from it, with no fidelity gate read yet (docs/measurements/op1o2-fa-softcap-2026-11/README.md). No effect on an uncapped
	// model, which takes the lane by default, or off CUDA.
	FlashDecodeSoftcap bool
	// MoECacheSlots is the per-layer expert-slot count for MoECacheExperts (0 = ask for all and
	// auto-cap to measured free VRAM). More slots ⇒ higher LRU hit rate ⇒ fewer per-token DMAs,
	// at VRAM cost. Only meaningful with MoECacheExperts.
	MoECacheSlots int
	// KVQuant selects the CPU KV cache storage precision: "" / "f32" (default, bit-exact, the reference), "f16" (each stored row
	// rounded to binary16, half the bytes; the generic loop, Architecture.kvF16OK, else f32) or "i8" (per-(position,
	// KV-head) symmetric int8, 4x smaller; declined where Architecture.kvInt8OK says no). The lossy two are opt-in. See
	// docs/completed/task-cpu-kv-f16-2026-10.md and task-cpu-kv-quant.md.
	KVQuant string
	// StreamWeights enables on-demand weight residency (idea #2): for an mmap-backed
	// .giw MoE model, expert weights are paged out of the mapping under a RAM budget
	// (WeightCacheBytes) instead of all held resident. Bit-exact (read-only re-fault);
	// trades RAM for cold-miss fault latency. No-op for non-MoE / non-.giw models.
	StreamWeights bool
	// WeightCacheBytes is the resident-bytes budget for streamed weights (0 = auto,
	// ~half of available RAM), and the budget a mapped MoE .giw's working-set prediction prices the page cache at.
	WeightCacheBytes int64
	// AcceptSlowMoE is the explicit acknowledgement a MoE .giw load that decodes on the CPU (mapped, or paged under
	// StreamWeights) needs when its predicted working-set rate falls below moeSlowTokPerSecThreshold (moeworkingset.go):
	// without it the load is refused. A no-op everywhere else.
	AcceptSlowMoE bool
	// RequireBackend makes a decline an error: when Backend names a backend other than the CPU and the load builds no
	// resident runner on it (the backend is not linked, no device answered, or it declined the model), Load closes the model
	// and returns an error that errors.Is matches against ErrBackendDeclined, carrying the decline's reason, instead of
	// running on the CPU. A CPU load is unaffected.
	RequireBackend bool
	// MoEPager is the CPU expert pager's backing mode for a .giw-paged MoE model: "mmap" (advice-based,
	// zero-copy) or "pool" (owned buffers + pread; a firm cap on every platform). "" = the platform
	// default (MoEPagerDefault: pool on darwin, mmap elsewhere), unless GOINFER_MOE_PREAD_CPU overrides.
	MoEPager string
	// Knobs sets per-model operator knobs by environment-variable name (knobs.go's list, e.g.
	// "GOINFER_FUSED_ATTENTION": "0"), overriding the process environment for THIS model only. Unset names
	// take the environment's value, read once at Load. Unknown names are ignored. A pointer, so Options
	// stays comparable (a hard-tier API property, docs/api-tiers.md): Knobs: &decoder.Knobs{...}.
	Knobs *Knobs
	// EmbedInt4 relaxes the int8 pin on the token-embedding/LM-head table in int4 mode, storing it at int4 too: this halves the
	// single largest resident tensor on a big-vocab small model, at a lossy cost (mostly on rare tokens). Off in a zero Options,
	// which keeps the bit-exact int8 pin; the CLIs turn it on by default (internal/loadflags). GGUF load path only.
	EmbedInt4 bool
	// ResidentContext requests a GPU-resident KV capacity in positions. 0 (default) keeps the
	// backend's built-in default, so nobody who did not ask allocates deep-KV VRAM. When set, the
	// backend caps it at the model's own context window — the effective cap is
	// min(model context window, this) — and fails at LOAD if the KV that implies does not fit
	// beside the weights, rather than OOM-ing mid-decode. Ignored off the residency path.
	ResidentContext int
	// DisableFit is --fit=off: it restores every "fit by default" behavior to its previous default. Currently: CUDA's unpinned
	// resident context stays the flat historical constant instead of asking Plan for more when there is room
	// (cuda/resident.go's resolveCtxCapFit), and a dense .gguf that will not fit resident RAM stays a plain refusal instead of
	// getting an automatic -stream-weights retry (guardFit in fitguard.go, read by internal/modelload's Load, not by Load itself).
	// An explicitly pinned request (ResidentContext, MoECacheSlots, StreamWeights, ...) is never affected in either direction:
	// fit-by-default only acts on the unpinned case. See docs/tasks/task-fit-to-hardware.md.
	DisableFit bool
	// ExtraResidentBytes prices a companion allocation that will claim VRAM on the same device after this model's own residency
	// is built (a --drafter's weights; internal/serveapp's loadDecoder computes it via decoder.DrafterResidentBytesEstimate before
	// calling Load). See Model.ExtraResidentBytes. 0 (the default) means nothing else is attaching.
	ExtraResidentBytes int64
	// ExtraResidentKVPerPosition is ExtraResidentBytes' ctx-scaling twin: a companion allocation's device K/V scales with the
	// resident context THIS model ends up choosing, which is unknowable when ExtraResidentBytes is priced (in loadDecoder). See
	// Model.ExtraResidentKVPerPosition and decoder.DrafterKVBytesPerPosition. 0 (the default) means no such term.
	ExtraResidentKVPerPosition int64
	// Drafter reports that a block drafter will be attached to this model after the load (serve's --drafter). The drafter keeps f32
	// K/V and attends with the target's kernels, so on CUDA auto resolves the KV to f32, and an explicit f16 refuses (ResolveKV).
	Drafter bool
	// ExactPrefill forces bit-exact prompt ingestion on every backend that has a faster, non-exact default: CUDA's tensor-core
	// batched prefill (GOINFER_CUDA_FAST_PREFILL), Metal's f16-MMA batched prefill (GOINFER_METAL_FAST_PREFILL) and CPU's
	// f32-attention fast path (GOINFER_CPU_FAST_ATTENTION), all default ON above their own thresholds. false (the default)
	// leaves whatever env state the process already has untouched, so a caller managing these knobs itself (serve's
	// --cpu-exact-prefill and --cpu-fast-attention, more granular than this bool, travel through Options.Knobs) is not
	// overridden. This is the library-level chokepoint docs/completed/task-prefill-gap.md describes; --exact-prefill sets it.
	ExactPrefill bool
	// ResidentKVSlots asks a GPU-resident backend for this many independent KV caches ("slots"), so several interleaved
	// conversations each keep their own prefix resident instead of evicting one another's (docs/tasks/parked/task-concurrency-2026-09.md).
	// Still one generation at a time unless MC3 is enabled: a slot is bound per generation. 0 or 1 = one slot. A backend clamps it
	// to what its fit guard allows and says so; a backend that does not implement ResidentKVSlotter, and every family with
	// recurrent state, keep one slot. serve sets it from -kv-sessions.
	ResidentKVSlots int
	// ResidentKVSlotsDefault says ResidentKVSlots is the caller's default, not a count the operator chose (serve sets it when
	// -kv-sessions was not given). A backend may then lower it to its own default: Metal keeps 2 slots, since every slot's KV is
	// resident from the first token on unified memory.
	ResidentKVSlotsDefault bool
	// ResidentRecurrentStepDefault and ResidentGemma4StepDefault say the step option beside each is the caller's default, not a choice: a
	// backend that grants one KV slot leaves that step out, since its scratch then batches nothing.
	ResidentRecurrentStepDefault, ResidentGemma4StepDefault bool
	// ResidentWindowedKV asks the CUDA resident to keep only a window's worth of K/V (plus WindowedKVSlack positions of headroom) for each
	// sliding-window layer instead of the whole context, which lets a windowed model such as Mellum2.1 fit a long context fully resident
	// beside its weights (docs/completed/task-cuda-windowed-kv-2026-10.md). Off by default: it changes how much memory a load takes. A
	// backend or family that cannot honour it keeps the full cache and says why (ResidentWindowedKVDecline); the decode is bit-identical
	// either way. Only CUDA reads it today.
	ResidentWindowedKV bool
	// ResidentMoEStep lets a generic MoE model (the routed FFN, its shared expert included) run on the Metal resident's
	// multi-sequence step: concurrent generations decode in one batched step, and a prompt shorter than the batched pass's
	// floor prefills on it, each bit-identical to the single-token forward (docs/completed/task-forward-architecture-2026-10.md
	// §4.9, OP4m-d3). It costs the step's scratch, a few MiB. Off leaves every MoE generation on single-token decode. Only
	// Metal reads it. Chat and serve set it from --moe-step, on by default.
	ResidentMoEStep bool
	// ResidentRecurrentStep lets a Qwen3.5-family model (Gated-DeltaNet and gated attention) or a Nemotron-H model (Mamba-2)
	// run on the Metal resident's multi-sequence step: concurrent generations, each on its own KV slot with its own recurrent
	// state, decode in one batched step, and a prompt shorter than the batched pass's floor prefills on it, each bit-identical
	// to the single-token forward (docs/completed/task-forward-architecture-2026-10.md §4.9, OP4m-d4b, d4c and d6b). Concurrency needs several slots
	// (ResidentKVSlots, asked for explicitly). It costs the step's scratch. Only Metal reads it.
	ResidentRecurrentStep bool
	// ResidentGemma4Step lets a Gemma 4 dense model (the E-models with their per-layer embeddings included) run on the Metal
	// resident's multi-sequence step: concurrent generations decode in one batched step, each row's layer on decode's own
	// encoder with the front of its dense FFN batched, bit-identical to the single-token forward
	// (docs/completed/task-forward-architecture-2026-10.md §4.9, OP4m-d5). It costs the step's scratch. Only Metal reads it.
	ResidentGemma4Step bool
	// ResidentPrefillChunk, under MC3 (EnableResidentConcurrency), prefills a long prompt suffix in chunks of this many tokens
	// while other generations are decoding, one decode step between chunks, instead of in one pass that stalls them all for the
	// whole prompt (docs/tasks/parked/task-concurrency-2026-09.md). 0 = off: whole prefill. Sound only where the resident's batched
	// prefill is chunk-invariant (Metal's is: TestMC5_prefillChunkInvariance). serve defaults it to 512, the graded value
	// (docs/measurements/chunked-prefill-2026-09-27.md).
	ResidentPrefillChunk int
	// KVShareOff stops a request taking KV rows another conversation computed: a resident request reuses only a
	// continuation of a slot's own sequence (KVScope.Off for every request; docs/tasks/task-kv-architecture-2026-10.md
	// §5.15). serve sets it for -kv-share off and when it answers the network without an API key.
	KVShareOff bool
	// KVHostBytes is the host memory budget for prefix snapshots (docs/completed/task-kv7-prefix-snapshot-2026-10.md): after a cold
	// first prefill of a long prompt a GPU-resident backend that implements ResidentKVSnapshotter copies the prompt's KV to pinned
	// host memory, and a later request that starts with the same ids restores it instead of prefilling it. 0 turns it off, and
	// nothing is held while no snapshot is. Pinned memory is allocated when a snapshot is taken and freed when it is dropped.
	// serve sets it from -kv-host-bytes.
	KVHostBytes int64
	// KVDemote keeps a released idle conversation's rows in host memory under the KVHostBytes budget, for a resident that implements
	// ResidentKVDemoter, and brings them back when it returns (docs/measurements/kv7-demote-2026-11/README.md). The copy waits on the
	// request taking the blocks, so it is off until a timed reading says otherwise. serve sets it when -kv-host-bytes names a budget. KVShareOff
	// turns it off, as it does the snapshots.
	KVDemote bool
	// CPUBatchDecode chooses whether concurrent CPU generations of this model join their decode tokens into one batched
	// forward (MC3c step 2, docs/tasks/parked/task-concurrency-2026-09.md; Model.EnableCPUBatch): CPUBatchAuto (0, the
	// default) batches an eligible model with at least 2 GiB of dense weights; CPUBatchOn batches every
	// eligible model; CPUBatchOff keeps step 1's independent workers. Every reply is bit-identical either way.
	CPUBatchDecode int
	// ActQuantGroup selects per-group activation quantization for the int8-activation projections (int4 = W4A8, int8int8 =
	// W8A8, int4mix): 0 (the default) scales each activation vector by one max/127, 32 gives every 32 inputs their own scale. A
	// family with massive activation outliers (Phi-3) loses nearly the whole vector under one scale; per-32 keeps an outlier's
	// damage inside its group (docs/tasks/task-actquant-pergroup-2026-09.md). Per model: two models in one process may differ.
	// Honoured on the CPU and by CUDA residency; other resident backends decline to the CPU path when it is set.
	ActQuantGroup int

	// LoadAbort, if non-nil, is checked between layers during a direct GGUF weight build, during a .giw's CRC pass and layer walk, and by a
	// backend's resident build (Model.LoadAbort): the load-time consumer of the swap tripwire. Closing it aborts the load with an error satisfying errors.Is(err, ErrLoadAborted). Load does not know
	// why it closed (it does not own decoder.SwapWatch), so a caller arming a watch is expected to wrap the returned error with
	// its own reason/pricing detail. A nil channel (the zero value) blocks forever in a select, so it is a genuine no-op.
	//
	// Limitation: the safetensors direct-build path and StreamTranscodeGGUF's transcode do not check it, and a backend's resident build
	// does only if it polls Model.LoadAbort (docs/code-notes/decoder.md#Options.LoadAbort).
	LoadAbort <-chan struct{}

	// BackendAuto says Backend was chosen by "auto" rather than named. A backend auto chose declines a model it would
	// run only at another precision (Metal re-quantizes int8 weights to int4), where a named one runs it as asked.
	// Load sets it when Backend is "auto"; a caller that resolved auto itself (the CLIs, for their own checks by name)
	// sets it beside the name.
	BackendAuto bool

	// SidecarFor names the backend whose sidecar .giw this load stands in for, "" for none: a planner (goinfer-chat fit) opens a source
	// whose sidecar the real load would build first, to price that load without building it. The weights are built in the layout the
	// sidecar holds (on metal, canonical int4 with no row4 repack: wantsRow4Fallback), and the resident host copy is priced as the
	// bundle maps it, every weight payload (mmapAliasedBytes). A load from a .giw ignores it. Per model: Model.SidecarFor.
	SidecarFor string

	// planResidency is set only by FitQuery's open of a .giw: withResidency builds nothing, so FitQuery can ask the backend what
	// its build would decide (Model.plannedDecodePath).
	planResidency bool
}

// Validate checks the stringly-typed knobs against their allowed values, so an
// invalid enum (e.g. -kv-quant=int8 instead of i8) is a clear load-time error
// rather than a silent fall-through to the default. Callers building Options from
// untrusted input (cmd/serve's per-model flags) should call it before Load; Load
// itself does not, to keep existing direct callers unchanged. Path/arch-dependent
// rules (StreamWeights needs a .giw; KVQuant i8 excludes MoE) stay runtime checks.
func (o Options) Validate() error {
	if _, err := parseQuant(o.Quant); err != nil {
		return err
	}
	switch o.Backend {
	case "", "cpu", "webgpu", "cuda", "metal", "auto":
		// Accepting the NAME is not a claim that the backend is built in: an unregistered
		// one falls back to CPU with a note (NewBackend). Rejecting it here instead meant
		// `serve --backend cuda|metal` failed at flag-validation even when the module WAS
		// compiled in (-tags cuda / -tags metal), because Validate runs before registration
		// is ever consulted.
	default:
		return errorOfKind(ErrUnknownBackend, "decoder: invalid backend %q (cpu | webgpu | cuda | metal | auto)", o.Backend)
	}
	switch o.ActQuantGroup {
	case 0, 32:
	default:
		return fmt.Errorf("decoder: invalid ActQuantGroup %d (0 | 32)", o.ActQuantGroup)
	}
	switch o.KVPrecision {
	case "", "auto", "f32", "f16", "i8":
	default:
		return fmt.Errorf("decoder: invalid -kv %q (auto | f32 | f16 | i8)", o.KVPrecision)
	}
	switch o.KVQuant {
	case "", "f32", "f16", "i8":
	default:
		return fmt.Errorf("decoder: invalid -kv-quant %q (f32 | f16 | i8)", o.KVQuant)
	}
	return nil
}
