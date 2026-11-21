package resident

import (
	"context"

	"github.com/townsendmerino/goinfer/internal/cost"
	"github.com/townsendmerino/goinfer/internal/runtime/adapter"
)

// SubsystemProvider is implemented by GPU resident runners (cuda, metal, webgpu) to expose their capabilities
// through four cohesive subsystem interfaces instead of ad-hoc per-method type assertions. Backends pin their subsystem
// wrappers to DecodeSubsystem, PrefillSubsystem, KVSubsystem and RuntimeSubsystem at
// compile time, so signature drift fails the build instead of silently disarming an optional fast path at runtime.
//
// A resident that is not a provider has no optional capability: SubsystemsOf returns the no-op subsystems for it.
type SubsystemProvider interface {
	ResidentSubsystems() Subsystems
}

// Subsystems groups a resident runner's capabilities by domain: single-token and multi-sequence decode, batched
// prompt prefill, KV slot and snapshot management, and runtime control / speculation / telemetry.
type Subsystems struct {
	Decode  DecodeSubsystem
	Prefill PrefillSubsystem
	KV      KVSubsystem
	Runtime RuntimeSubsystem
}

// DecodeSubsystem covers single-token decode fast paths (argmax, top-K, device Gumbel sample, m-RoPE, KV-only),
// device-chained greedy and sampled decode, and multi-sequence batched / ragged decode steps.
type DecodeSubsystem interface {
	// GreedyAvailable is an optional capability on a Forward: compute the token's greedy
	// argmax on-device and hand back just the id, skipping the full-logits readback. The decode loop
	// uses it ONLY when Sampler.ArgmaxEquivalent() OR Sampler.GreedyEquivalent() (the `top_k=1` shape)
	// holds and no LogitProcessor is set, so the emitted tokens are identical to the logits path.
	// Backends may skip implementing it.
	GreedyAvailable() bool
	ForwardArgmax(embedding []float32, pos int) (int, error)

	// TopKAvailable is an optional Forward extension: run a token's forward and return only the K
	// best logits (plus Z when wantZ), instead of the whole vocab-wide row. The decode loop uses it only
	// for a filtered sampler (top_k / top_p / min_p at temperature > 0) with no bias, penalties, logprobs
	// or logit processor — anything that needs, or rewrites, the full row keeps the full path.
	// TopKAvailable is false for a backend whose logits are transformed on the host after readback
	// (final-logit softcap, logit scale): ranking the raw device row would not be ranking what the
	// sampler sees.
	TopKAvailable() bool
	ForwardTopK(embedding []float32, pos, k int, temperature float64, wantZ bool) (TopKRow, error)

	// SampleAvailable is an optional Forward extension: run a token's forward and DRAW the next token
	// on-device by Gumbel-max (sampler_gumbel.go), returning just the id — no logits readback and no host
	// normalisation. It serves the temperature-only configuration (temperature > 0, no top-k / top-p / min-p), the
	// server's default sampling shape. (seed, draw) are the sampler's own (Sampler.NextDraw), so whichever path
	// serves a step the stream is a function of (seed, step, logits) only. SampleAvailable is false for a backend
	// whose logits are transformed on the host after readback (final-logit softcap, logit scale).
	SampleAvailable() bool
	ForwardSample(embedding []float32, pos int, temperature float64, seed, draw uint64) (int, error)

	// MRoPEDecodeAvailable is an OPTIONAL Forward extension: a resident backend whose rotation kernel takes a
	// rope-angle position (ropePos) separate from the KV-cache/attention position (pos). Qwen2.5-VL decode past
	// an image block needs it: decoder/rope.go's CPU reference rotates by pos+mropeDelta while storage and
	// attention still use plain pos. A new capability rather than a Forward signature change, since only Qwen-VL
	// decode needs the split. ForwardMRoPE(emb, pos, pos) must equal Forward(emb, pos) exactly. Without it
	// GenerateQwenVL's resident path declines to the CPU/staged decode.
	MRoPEDecodeAvailable() bool
	ForwardMRoPE(embedding []float32, pos, ropePos int) (logits []float32, err error)

	// NoLogitsAvailable is an OPTIONAL Forward extension: run a token's forward to build ONLY its
	// resident KV, skipping the final norm + LM head matmul + full-logits readback (+ any host-side
	// softcap). generateInto uses it for prompt[:-1] — every prefill token except the last needs only its
	// K/V in the cache; only the LAST prompt token's logits seed decode. The layer chain (hence the KV
	// written) is identical to Forward, so decode is byte-identical; the LM head on a big-vocab model is
	// a large share of a forward (262144×hidden MACs + a ~1 MB D2H + softcap), so skipping it on every
	// prompt token but the last cuts prefill/TTFT materially. Backends without it keep full-logits prefill.
	NoLogitsAvailable() bool
	ForwardNoLogits(embedding []float32, pos int) error

	// GreedyChainAvailable is an optional capability on a Forward (docs/audit-metal-2026-09-30.md): greedy
	// decode with the token chained on the device. The resident picks each token's argmax on-device and gathers its
	// embedding as the next forward's input itself, so it can queue the next forward before the host has seen the token;
	// the host reads 4 bytes per token. GreedyChainStart opens a chain whose first input is token id at position pos;
	// each GreedyChainNext returns the argmax of the forward at the chain's next position (pos, pos+1, ...), which is also
	// that forward's successor's input; GreedyChainStop closes it, waiting out any forward queued past the last token
	// taken. Between Start and Stop the caller makes no other call on the resident. The decode loop uses it under the
	// same conditions as GreedyAvailable, and only where its own embedding lookup is a plain table row (embedIsTableRow),
	// with no batched concurrency, adapter or processor; the tokens are those of the full-logits path.
	GreedyChainAvailable() bool
	GreedyChainStart(id, pos int) error
	GreedyChainNext() (int, error)
	GreedyChainStop()

	// SampleChainAvailable is GreedyChainAvailable's temperature-only sibling: the chained forward ends with the
	// resident's own Gumbel-max draw (SampleAvailable's ForwardSample) instead of the argmax. SampleChainStart opens it with
	// the draw of the forward at pos; the resident draws each later forward with the next one, which is how
	// Sampler.NextDraw advances. Each SampleChainNext passes the caller's own draw for the forward it returns, and the
	// resident refuses one that is not the draw it used. GreedyChainStop closes it. The decode loop uses it where it would
	// call ForwardSample, with no batched concurrency, adapter or processor and a plain table-row embedding; the tokens are ForwardSample's.
	SampleChainAvailable() bool
	SampleChainStart(id, pos int, temperature float64, seed, draw uint64) error
	SampleChainNext(seed, draw uint64) (int, error)

	// BatchStepperAvailable is an OPTIONAL KVSlotterAvailable extension: one
	// decode token for several sequences at once, each on its own KV slot at its own position, in one step. It reads and
	// writes only the named slots — never the bound one's binding — and each returned row must equal what that sequence's
	// own call would return on its slot, bit for bit (the batched decode's identity gate): Forward's logits, or, for a sequence
	// that carries a Draw, SampleAvailable.ForwardSample's id. BatchStepRange is the batch sizes the step serves better than
	// per-sequence calls; hi == 0 means the resident cannot batch at all.
	BatchStepperAvailable() bool
	BatchStepRange() (lo, hi int)
	StepBatch(seqs []BatchSeq) ([]BatchOut, error)

	// RaggedStepperAvailable is an optional Forward extension: one step over several sequences, each with one row or more
	// at consecutive positions from its Pos on its own KV slot, every row in the kernel class its own shape runs today (a row
	// alone is a decode token, a sequence's rows a verify). Row j of a sequence equals that sequence's own Forward at Pos+j bit
	// for bit, or ForwardSample's id for a last row that carries the Draw. The slots are distinct; the bound slot is restored.
	RaggedStepperAvailable() bool
	StepRagged(seqs []RaggedSeq) ([]RaggedOut, error)
}

// PrefillSubsystem covers batched prompt ingestion, row-chunked and boundary-marked prefill, block and chunk
// scratch pricing, hidden / residual extraction, and multimodal / m-RoPE / span prefill.
type PrefillSubsystem interface {
	PrefillAvailable() bool
	PrefillLast(ctx context.Context, embeddings [][]float32, startPos int) (logits []float32, err error)

	// PrefillRowsAvailable is the optional capability of a Prefiller that reads a suffix's embeddings a range of rows at a time, so the
	// host holds a chunk of them instead of the whole suffix. PrefillLastRows is
	// PrefillLast for n rows at startPos: rows(lo, hi) returns rows [lo, hi) of the suffix, valid only until the next call, and the
	// resident may ask for a range more than once (a declined route re-reads it).
	PrefillRowsAvailable() bool
	PrefillLastRows(ctx context.Context, n, startPos int, rows func(lo, hi int) [][]float32) ([]float32, error)

	// PrefillMarkedAvailable is the optional capability of a resident to tell its caller when a batched prefill has just written a given position,
	// between two of its passes: the one moment a windowed layer's ring holds exactly what a shorter prompt would leave there, so a prefix snapshot
	// can be taken in the middle of a cold prefill.
	// PrefillMarkAlign is the width of the passes of a batched prefill, in positions: a mark can fire only at startPos plus a multiple of it.
	PrefillMarkedAvailable() bool
	PrefillMarkAlign() int
	// PrefillLastMarked is PrefillLast that calls at(mark) on the caller's goroutine after the pass that ends exactly at position mark, with no
	// pass running. A mark that is not at a pass boundary of this prefill (a single pass, a start that is not aligned to it, the passes cut
	// differently) calls nothing. at must not fail the prefill; a snapshot it cannot take is the caller's to count.
	PrefillLastMarked(ctx context.Context, embeddings [][]float32, startPos, mark int, at func(pos int)) ([]float32, error)

	PrefillPathInfo() (batched bool, reason string, ok bool)
	PrefillTailExact() bool

	// PrefillBlocksAvailable is the optional capability of a Prefiller whose pass allocates scratch for every row it is
	// handed, so a long prompt's scratch grows with the prompt. PrefillBlockRows is the most rows one pass should take,
	// or 0 when the resident does not want its prefill split; PrefillScratchBytes is what a pass of that many rows
	// allocates beyond the resident's standing buffers. residentPrefillSeed takes a longer suffix in passes of
	// PrefillBlockRows, and AdmitPrefillMemoryShare prices a resident request at one pass instead of the whole prompt.
	// Splitting changes no bit only for a resident whose continuation equals the whole pass, so both callers also require
	// PrefillTailExact (prefillBlockRows).
	PrefillBlocksAvailable() bool
	PrefillBlockRows() int
	// PrefillChunksAvailable is the optional capability of a Prefiller that splits a long prompt into passes itself: the decoder hands it
	// the whole suffix in one PrefillLast call and the resident cuts it. PrefillChunkRows is the most rows one of its passes takes, and
	// PrefillScratchBytes what a pass of that many rows allocates beyond the resident's standing buffers. Admission prices a request at
	// that one pass instead of the whole-prompt figure (residentPrefillScratch). It is not PrefillBlocksAvailable: that interface makes the
	// decoder split the prompt and needs PrefillTailExact, which a resident whose passes differ in kernel class by length cannot claim.
	PrefillChunksAvailable() bool
	PrefillChunkRows() int
	PrefillScratchBytes(rows int) int64
	// PrefillSpanScratchBytes is the optional capability of a chunking resident whose pass allocates more when its rows attend
	// many keys: PrefillSpanScratchBytes is at least what any pass of rows rows of a prompt whose last row attends keys keys
	// allocates beyond PrefillScratchBytes(rows). Admission adds it at the request's prompt length.
	PrefillSpanScratchBytes(rows, keys int) int64

	// HiddenLastAvailable is an OPTIONAL Forward extension: ingest a whole sequence at positions
	// startPos..startPos+len-1 and return the LAST position's hidden state AFTER the model's final norm — the resident
	// twin of Prefiller, stopping before the LM head. Used by HiddenLast so an embedding request on a GPU box does
	// not run the whole text decoder on the CPU. Bit-identical to the CPU path is the bar (embed.go): a backend whose
	// batched forward is not bit-identical to its own sequential one must implement this as a per-token forward that
	// stops before the head, or not at all, never reuse a declining Prefiller.
	HiddenLastAvailable() bool
	HiddenLast(ctx context.Context, embeddings [][]float32, startPos int) (hidden []float32, err error)

	// ResidualAllAvailable is an OPTIONAL Forward extension: ingest a whole sequence at positions
	// startPos..startPos+len-1 and return the residual stream of EVERY row after the last layer and BEFORE the final
	// norm, [len][hidden] f32, each row a fresh slice the caller owns: the resident twin of Model.PromptHiddenAll.
	// It stops before the norm because the decoder applies it on
	// the host in f32, as its CPU path does; HiddenLastAvailable cannot serve this (post-norm, last row only, and on CUDA
	// already through the int8 activation quantization). A backend whose batched forward is not what its decode computes
	// must take the exact path; a decline falls back to the CPU, a cancellation returns.
	ResidualAllAvailable() bool
	ResidualAll(ctx context.Context, embeddings [][]float32, startPos int) (residuals [][]float32, err error)

	// ImagePrefillAvailable is an OPTIONAL Prefiller extension: bidirectional attention over a contiguous
	// [imgStart,imgEnd) sub-range (Gemma 3's image-block mask, SetImageBlocks/attendHi in decoder/kvcache.go), the
	// resident twin of prefillLogitsVL's CPU forward. embeddings are the ALREADY-SPLICED per-position vectors (text
	// embeddings with the projected vision features overwritten at [imgStart,imgEnd), no embed scale). The whole prompt,
	// image block included, must fit in ONE weight-stationary pass; a longer prompt or any other decline returns an
	// error rather than chunking, because a bidirectional block split across a chunk boundary is unverified. Without it
	// GenerateVL takes the CPU-prefill-then-UploadKV bridge (docs/multimodal.md).
	ImagePrefillAvailable() bool
	PrefillImageLast(ctx context.Context, embeddings [][]float32, startPos, imgStart, imgEnd int) (logits []float32, err error)

	// ImageBlocksPrefillAvailable is ImagePrefillAvailable for a prompt with several images: blocks are each image's [start, end), in order and disjoint, each its
	// own bidirectional block. OPTIONAL: a backend without it takes a multi-image turn through the CPU-prefill + UploadKV
	// bridge, which handles any number of blocks.
	ImageBlocksPrefillAvailable() bool
	PrefillImageBlocksLast(ctx context.Context, embeddings [][]float32, startPos int, blocks [][2]int) (logits []float32, err error)

	// SpanPrefillAvailable is an OPTIONAL Forward extension: the batched prefill of a Gemma 4 bidirectional
	// ("vision") prompt. spans are each image's or audio clip's [start, end), absolute, in order and disjoint, and a row
	// attends as gemma4BatchRange says: inside a span it sees the whole span, and a sliding layer's window reaches back to
	// the span's start; elsewhere it is causal. The backend may cut passes, but never inside a span. Any error other than a
	// cancel is a decline, and GenerateGemma4VLSpans takes the CPU prefill and upload instead.
	SpanPrefillAvailable() bool
	PrefillSpansLast(ctx context.Context, embeddings [][]float32, startPos int, spans [][2]int) (logits []float32, err error)

	HybridMRoPEPrefill() bool

	// MRoPEPrefillAvailable is an OPTIONAL Prefiller extension: batched prefill under Qwen2.5-VL's m-RoPE 3D rotary
	// positions, the resident twin of prefillLogitsQwenVL's CPU forward. Unlike MRoPEDecodeAvailable (decode's single ropePos
	// split) it needs every row's own position up front: mropePos is decoder/rope.go's mropePositions output, one (t,h,w)
	// triple per ABSOLUTE position over the WHOLE prompt, which GenerateQwenVL already computes, so a backend must not
	// re-derive it. embeddings are the ALREADY-SPLICED vectors (ImagePrefillAvailable's convention). Unlike it,
	// implementations MAY chunk: Qwen's image tokens attend causally and each row's rotation is independent. Without it
	// GenerateQwenVL falls back to the CPU-prefill+UploadKV bridge.
	MRoPEPrefillAvailable() bool
	PrefillMRoPELast(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int) (logits []float32, err error)

	// MRoPEDeepstackPrefillAvailable is an OPTIONAL extension of MRoPEPrefillAvailable for Qwen3-VL: the same batched m-RoPE
	// prefill, which also adds DeepStack set l to the image rows [imgStart, imgStart+imgLen) after decoder layer l, as the CPU
	// prefill does (addDeepstack; HF's hidden_states[visual_pos_masks] += deepstack_visual_embeds[l]). deep holds one set per
	// injected layer, imgLen rows of the hidden size each. Without it GenerateQwenVLDeepstack keeps a turn that has DeepStack
	// sets on the CPU prefill.
	MRoPEDeepstackPrefillAvailable() bool
	PrefillMRoPEDeepstackLast(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int, deep [][]float32, imgStart, imgLen int) (logits []float32, err error)
}

// KVSubsystem covers multi-slot KV binding, shared block-pool telemetry, slot activity, cross-slot KV copy,
// windowed reuse queries, atomic upload passes, host prefix snapshots, host KV demotion, and recurrent checkpoints.
type KVSubsystem interface {
	// KVSlotterAvailable is an OPTIONAL Forward extension: a resident
	// that holds several independent KV caches ("slots") and binds exactly one at a time. Every Forward method —
	// Forward, ForwardN, UploadKV, TruncateTo, Reset — reads and writes the BOUND slot only, so a slot switch is the whole
	// interface: the decoder keeps each slot's reuse bookkeeping and binds the slot a generation will use before it
	// prefills. KVSlots is the count actually allocated (>= 1, after the backend's fit guard); UseKVSlot(i) binds slot i
	// and must leave nothing encoded against the previous one. A resident without it has one slot.
	KVSlotterAvailable() bool
	KVSlots() int
	UseKVSlot(i int) error

	// SetKVSlotBusy is the optional capability of a resident whose slots share one block pool to be told which slots a running
	// generation holds (the batched decode's busy slots), so that it releases the blocks of an idle slot to make room, never a running one's. The bound
	// slot counts as running whether or not it is marked.
	SetKVSlotBusy(slot int, busy bool)
	// KVPool is the optional capability of a resident whose slots share one block pool to say what it holds.
	KVPool() KVPoolDetail

	// ReusableKV is implemented by a resident whose KV for some layers holds only a recent window of positions (Options.ResidentWindowedKV).
	// ReusableKV(prefix) is how many of the first prefix positions of the bound slot's sequence it can serve to a forward that continues at
	// position prefix: prefix itself, or 0 when the window that forward attends to has already been dropped, in which case the caller prefills
	// the prompt from the start. A positive answer other than prefix is never returned.
	ReusableKV(prefix int) int
	// ReusableKVSlotAvailable is ReusableKV's question for any slot, bound or not: of the first prefix positions
	// of slot's sequence, how many it can serve to a forward that continues at position prefix (prefix or 0).
	ReusableKVSlotAvailable() bool
	ReusableKVSlot(slot, prefix int) int

	// KVUploadPassAvailable is the optional capability of a resident to treat the UploadKV calls between BeginKVUpload and EndKVUpload as one write of the
	// bound slot: it holds the rows only once every layer's have landed, and failed says one did not.
	KVUploadPassAvailable() bool
	BeginKVUpload() error
	EndKVUpload(failed bool) error

	// KVCopierAvailable is the optional capability of a resident with several KV slots (KVSlotterAvailable) to copy rows
	// between them. CopyKV copies rows [start, end) of every layer from slot src to slot dst, src != dst, start and end
	// multiples of kvmgr.BlockRows; dst then continues at position end, and src is unchanged. A range the backend cannot
	// serve (a window that no longer holds rows dst will attend to, a layout it does not copy) returns a *KVCopyDecline
	// before anything is written. The caller treats any error as having written nothing it may trust from start on.
	KVCopierAvailable() bool
	CopyKV(src, dst, start, end int) error

	// KVSnapshotterAvailable is the optional capability of a resident to hold a slot's KV rows in host memory and put them back. Like
	// KVCopierAvailable it works on today's slot storage: a windowed layer's rows are those its window and the rollback need, and the
	// restore sets the slot's ring base as a copy does.
	// SnapshotKV copies rows [0, end) of slot into pinned host memory, end a whole number of blocks; for a windowed layer, the rows
	// [end-(window-1)-rollback, end) its slot holds. A *KVCopyDecline, before anything is allocated, refuses what it cannot hold.
	KVSnapshotterAvailable() bool
	SnapshotKV(slot, end int) (KVSnapshot, error)
	// RestoreKV writes the first end rows of snap into slot, so that the slot continues at position end. A *KVCopyDecline, before
	// anything is written, refuses what it cannot serve.
	RestoreKV(snap KVSnapshot, slot, end int) error
	// KVSnapshotBytes is the host memory SnapshotKV(_, end) would hold, 0 when it would decline for a reason that needs no slot.
	KVSnapshotBytes(end int) int64
	// KVSnapshotRollback is how many positions before a snapshot's end a windowed layer still serves a restore; 0 with no windowed layer.
	KVSnapshotRollback() int
	// KVSnapshotRecurrent reports whether a snapshot carries a recurrent model's state with its rows: a snapshot to end holds the state
	// as it stood after position end-1, so it serves a restore to exactly end and to no shorter prefix. False when it carries none.
	KVSnapshotRecurrent() bool

	// KVDemoterAvailable is the optional capability of a resident to keep a released slot's rows in host memory (the slot still answers
	// ReusableKVSlotAvailable for them) and put them back when a request reuses it (docs/measurements/kv7-demote-2026-11/README.md).
	// SetKVHostBudget gives it the budget it charges them to; without one it keeps none.
	KVDemoterAvailable() bool
	SetKVHostBudget(b KVHostBudget)
	// PromoteKV puts back the blocks holding slot's first rows of its demoted copy and returns them, 0 when the slot holds its rows. The copy is
	// then gone, and the slot answers for those rows only. An error leaves the slot released and the copy in place.
	PromoteKV(slot, rows int) (blocks int, err error)
	// CopyDemotedKV writes rows [start, end) of src's demoted copy into dst, which holds the rows below start, and returns the blocks written. A
	// released dst holds none and takes them all. src stays released with its copy, for its own conversation's return. held is false when src is
	// not released, which is CopyKV's case. A *KVCopyDecline leaves dst as it was.
	CopyDemotedKV(src, dst, start, end int) (blocks int, held bool, err error)
	// ReserveKVHost charges n bytes of the budget for a prefix snapshot, dropping demoted copies and then held snapshots to make room, and dropping
	// nothing when that could not make it fit.
	ReserveKVHost(n int64) bool

	// RecurrentCheckpointBytes is the optional capability of a resident with recurrent state to keep one copy of it, every
	// recurrent layer's state and conv window, beside the live state. RecurrentCheckpointBytes is the copy's size, 0 when the resident
	// keeps none (its load did not admit it). SaveRecurrentCheckpoint copies the live state into it and RestoreRecurrentCheckpoint
	// copies it back; neither touches the attention KV. An error leaves the live state unknown to the caller.
	//
	// RecurrentCheckpointHeld reports whether slot holds a checkpoint now. A checkpoint can be taken from a slot, by another slot's save
	// or by a reset, while the slot's record still names its position, so a load of several slots scores only a checkpoint its slot holds:
	// a restore that fails has already cost that slot its sequence. A resident that cannot tell returns false, and scores none on such a load.
	RecurrentCheckpointBytes() int64
	SaveRecurrentCheckpoint() error
	RestoreRecurrentCheckpoint() error
	RecurrentCheckpointHeld(slot int) bool
}

// RuntimeSubsystem covers load-time capabilities, device cost spans, compute-time LoRA adapters, MoE expert
// cache management, speculative verify path / cost / divergence guards, batched argmax verify, block-drafting host, and
// recurrent speculative fold.
type RuntimeSubsystem interface {
	// Caps is the optional capability of a resident to report its Caps.
	ResidentCaps() Caps

	// CostAvailable is a resident that counts device costs (internal/cost) over a span of its work: CostSpan runs work with c
	// counting into phase p, adds the span's device-level counts and wall time, and counts the tokens work reports it covered.
	// Optional. A generation that asks for its cost on a resident without it, or on one shared with other generations (the batched decode), gets
	// no phase from the resident.
	CostAvailable() bool
	CostSpan(c *cost.Counter, p cost.Phase, work func() (tokens int, err error)) error

	// AdapterAvailable is an OPTIONAL Forward extension: a backend that can apply a compute-time LoRA
	// delta on the resident decode path. SetAdapter(nil) clears a previously-bound adapter. generateInto holds the
	// same resBusy exclusive-access window a Forward/ForwardN call requires for the WHOLE bind→forward→clear
	// sequence, so an implementation needs no locking and two adapters sharing one base model's resident runner
	// never race. A backend without it is simply not type-asserted: generateInto declines to the CPU/staged path,
	// never running an adapter session's tokens through the base weights.
	AdapterAvailable() bool
	SetAdapter(layers []adapter.ResidentAdapterLayer) error

	// ExpertCacheAvailable is an OPTIONAL Forward extension, implemented by a resident that pages MoE experts through a
	// device cache. ExpertCacheStats' ok is false when this build keeps no cache; its counters are written by the generation in
	// flight, so a caller reads them between generations. ResizeExpertCache sizes the cache to what free device memory holds
	// with reserve bytes left over, up to the slots the load asked for, and returns the slots it now has; the cache restarts
	// empty. It is called between generations, and an error leaves the cache as it was.
	ExpertCacheAvailable() bool
	ExpertCacheStats() (s ExpertCacheStats, ok bool)
	ResizeExpertCache(reserve int64) (slots int, err error)

	VerifyPathInfo() (batched bool, reason string, ok bool)
	VerifyCost() []float64
	DecodeVerifyDivergence() error
	EnterExactAttention() (leave func())

	// SeedArgmaxAvailable is an OPTIONAL narrowing of PrefillLastNArgmax for the prompt seed, the one place the loop asks for M rows of argmax
	// and reads exactly one of them. Without it the seed materialises the batched logits (M x vocab: gigabytes on a large-vocab target at
	// prompt lengths over ~1k) and runs the head over every row to obtain one token id. The capture the seed needs comes from the layer loop,
	// not the head, so heading one row loses nothing.
	//
	// Optional rather than part of DrafterHost, following GreedyAvailable: a backend without it keeps working
	// through the wide path.
	// PrefillSeedArgmax runs the same batched forward and returns only the LAST row's argmax.
	// The batched capture seam must be armed and filled exactly as PrefillLastNArgmax fills it.
	SeedArgmaxAvailable() bool
	PrefillSeedArgmax(embeddings [][]float32, startPos int) (int, error)

	LastNArgmaxAvailable() bool
	PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error)

	DrafterHostAvailable() bool
	CanStreamDrafterRows() bool
	AttachBlockDrafter(w BlockDrafterWeights) (BlockDrafter, error)
	SetBatchedCapture(taps []int) error
	BatchedCapture() [][]float32

	// RecurrentSpecAvailable is the OPTIONAL capability that lets a resident with recurrent state verify speculatively. Without it, BlockSpec refuses every recurrent model,
	// because PrefillLastNArgmax advances the state by every row it verifies, rejected or not.
	// PrepareRecurrentSpec allocates what verifies of up to rows rows need, before a drafter attaches; a later call for no more
	// rows is a no-op.
	// It returns why when this load cannot take the path (a recurrent layer it does not serve, no room for the rows), and
	// RecurrentSpecRows then stays 0.
	RecurrentSpecAvailable() bool
	PrepareRecurrentSpec(rows int) error
	// RecurrentSpecRows is the widest verify the path holds: 0 when this load cannot take it (a recurrent layer it does not
	// serve, or no room for its rows).
	RecurrentSpecRows() int
	// VerifyLastNArgmax is PrefillLastNArgmax that writes none of its own rows into any recurrent layer's committed state or conv
	// window. It first folds the rows the previous verify had marked (FoldVerified). Each row starts from the row above it, so the
	// rows are a chain.
	VerifyLastNArgmax(embeddings [][]float32, startPos int) ([]int, error)
	// FoldVerified marks the last verify's first n rows committed. They fold into the state, leaving it as a prefill of those
	// rows would, at the next VerifyLastNArgmax or FlushFold, whichever comes first.
	FoldVerified(n int) error
	// FlushFold folds the marked rows now. Called before anything other than a verify reads or advances the state.
	FlushFold() error
	// DropFold forgets the marked rows without folding them: an exit that commits nothing leaves no rows for a later turn to fold.
	DropFold()
}
