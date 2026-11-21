package resident

// Forward is one token's GPU forward: embedding[hidden] in, logits[vocab]
// out, with the model + KV resident on the device. Implemented by the gpu package.
type Forward interface {
	// Forward runs token-at-pos: returns logits for the embedding at absolute
	// position pos (the runner appends this position's K/V to its resident cache).
	Forward(embedding []float32, pos int) (logits []float32, err error)
	// ForwardN runs K tokens at consecutive positions startPos..startPos+K-1, appending K KV positions and returning K
	// logit rows — the batched verify for speculative decoding. Causal: row i attends to positions [0, startPos+i].
	// Bit-identical to K sequential Forward calls (TestResidentForwardN_parity), so no implementation may change numerics
	// to batch. nil/empty embeddings ⇒ no-op.
	//
	// "Batched" is an amortization opportunity, not a guarantee: each backend reports at load which path it took
	// (VerifyPathReporter). Callers must not assume "K rows in" means "one command buffer out".
	ForwardN(embeddings [][]float32, startPos int) (logits [][]float32, err error)
	// UploadKV writes a layer's post-RoPE K and raw V (each [n*kvDim]) into the resident GPU
	// caches at absolute positions base..base+n-1 — the prefill bridge. base is normally 0
	// (a fresh prefill), but a CPU-side sliding-window (ring) layer's live K/V can start at a
	// nonzero absolute position once its ring has wrapped (KVCache.LayerKV's own base return);
	// UploadKV must land those rows at their real position, not row 0.
	UploadKV(layer, base int, keys, vals []float32) error
	// TruncateTo drops resident KV positions ≥ pos — the rollback after a partial
	// speculative accept. The resident cache is positional and Forward sets
	// nKeys=pos+1, so stale entries past pos are simply never read and get overwritten
	// next round; this is therefore a no-op (the caller tracks pos), kept for the
	// interface symmetry with the CPU KVCache.
	TruncateTo(pos int)
	// Reset clears resident KV for a fresh generation (positions overwritten).
	Reset()
	// Close releases the resident GPU buffers.
	Close() error
}

// BatchSeq is one sequence of a BatchStepperAvailable step: its KV slot, the position it decodes, its input
// embedding (embedResident's), and, when its token is drawn on-device, that draw.
type BatchSeq struct {
	Slot, Pos int
	Emb       []float32
	Draw      *BatchDraw // nil: return the row's logits
}

// BatchDraw is a temperature-only draw a batched step makes on-device for its row: ForwardSample's
// arguments, from the sequence's own Sampler.NextDraw. Greedy marks a greedy generation's row
// (docs/audit-metal-2026-09-30.md), sent only to a resident that reports Caps.BatchGreedyDraw: the id must be the
// argmax of the logits the row would have returned (after any softcap or logit scale), first maximum on a tie, exactly
// as the decoder's sampler would pick it from them.
type BatchDraw struct {
	Temperature float64
	Seed, Draw  uint64
	Greedy      bool
}

// BatchOut is one row of a StepBatch: the drawn id (>= 0) for a sequence that carried a Draw, else its logits.
type BatchOut struct {
	Logits []float32
	ID     int
}

// RaggedSeq is one sequence of a RaggedStepperAvailable step: its KV slot, the position of its first row, its rows'
// embeddings (embedResident's), and, when its last row's token is drawn on-device, that draw. Prefill marks the rows as one
// chunk of a prompt's prefill instead of decode or verify rows; such a sequence carries no Draw.
type RaggedSeq struct {
	Slot, Pos int
	Embs      [][]float32
	Draw      *BatchDraw     // nil: every row's logits
	Prefill   *RaggedPrefill // nil: decode or verify rows
}

// RaggedPrefill makes a RaggedSeq one pass of a prompt's PrefillLast: its rows take the kernel class that pass
// takes (the fast kernels where the prompt's length allows them), so its K/V and its last row's logits equal that pass's bit
// for bit. PromptLen is the whole prompt's length, which decides the class; Last is the prompt's final pass, the only one
// whose last row's logits are returned. The caller cuts the prompt into passes as PrefillLast does.
type RaggedPrefill struct {
	PromptLen int
	Last      bool
}

// RaggedOut is one sequence of a StepRagged: each row's logits, with the last nil when the sequence carried a Draw,
// and the drawn id (>= 0), or -1. A prefill sequence's rows are nil but for a Last pass's final row.
type RaggedOut struct {
	Logits [][]float32
	ID     int
}

// TopKRow is one decode step's logits row reduced on-device to its K best entries. IDs/Logits are ordered (logit DESCENDING, token id ASCENDING) — the tie
// order internal/sampling's topKByLogit defines, which feeds the sampler's cumulative draw — so Logits[0] is the
// row's exact maximum and IDs[0] its argmax. Z is the full-vocabulary softmax denominator
// Σ exp((l−max)/T) at the temperature the row was requested with; 0 unless it was asked for. Full
// reads the whole row for the token where K was not enough, and is valid only until the next forward.
type TopKRow struct {
	IDs    []int32
	Logits []float32
	Z      float64
	Full   func() ([]float32, error)
}

// ExpertCacheStats describes a resident's device cache of MoE experts: Slots is the per-layer slot count it was granted,
// and Hits and Misses sum every MoE layer's lookups since the resident was built. A miss is one expert copied host to device.
type ExpertCacheStats struct {
	Slots        int
	Hits, Misses uint64
}

// PrefillTailExact is an OPTIONAL Prefiller extension: true when the backend's PrefillLast over a SHORT suffix that
// continues a prefix it prefilled computes, bit for bit, what one PrefillLast over the whole prompt computes for those
// positions, and its routing between its prefill lanes depends on the whole prompt's length, not the suffix's. Then a
// reused prompt (resident_reuse.go) can hand the batched path a suffix of any length and get the cold run's bits.
//
// Without it residentPrefillSeed offers only suffixes of 8 tokens or more: a 1-token suffix would run its last
// position through decode, which is not bit-identical to the batched pass the cold run used.
// Found on Metal: docs/code-notes/decoder.md#PrefillTailExact.
type PrefillTailExact interface {
	PrefillTailExact() bool
}

// Caps is what a resident reports about itself (Caps), read at each use. The zero value is what a
// resident that reports nothing gives, and each field's zero means "none".
type Caps struct {
	// ContextCap is the KV capacity in positions. A write past it is an out-of-bounds device write: the backends refuse one
	// mid-generation, and the decoder clamps a turn to it up front (clampToResidentCap) so the turn stops cleanly at the cap.
	ContextCap int
	// HasFastPrefill reports that the batched prefill switches to non-exact kernels (CUDA's attn_fused and gemm_w4a8_mma) from
	// FastPrefillFloor prompt tokens, a floor of 0 when the fast kernels are off. Rows a short request computed on the exact kernels
	// differ from the same rows on the fast ones, so prefix reuse declines a reuse that would mix them
	// (docs/measurements/mc4-candidate-cuda-2026-10-01.md, ROOT CAUSE). The class rule keys on HasFastPrefill, not on the floor.
	HasFastPrefill   bool
	FastPrefillFloor int
	// HasKernelFloor reports that the batched prefill runs one kernel class below a whole-prompt length of KernelFloor and another,
	// not bit-identical, from it (Metal: decode rows, then f16 MMA), a floor of 0 when one class runs at every length. Chunked
	// prefill keeps every chunk on the class the whole prompt takes. A narrower contract than HasFastPrefill's, which prefix reuse
	// also reads.
	HasKernelFloor bool
	KernelFloor    int
	// TailMinRows is the fewest rows a prefill's last pass may have to run the kernels of the pass before it (CUDA: the fused
	// attention's floor); every loop that cuts a prompt into passes takes the larger of it and prefillTailMin.
	TailMinRows int
	// Quant is the precision the weights run at, when the decoder cannot infer it from the backend and the load quant (Metal runs
	// some int8 models natively and re-quantizes others to int4); "" means residentQuantLabel applies.
	Quant string
	// ContextNote is a few words on how the backend arrived at ContextCap, for a load banner; "" when the caller's own -ctx set it.
	ContextNote string
	// BatchGreedyDraw reports that StepBatch returns a Greedy draw row's argmax id (BatchDraw.Greedy) instead of its
	// logits, so a greedy generation's batched token costs no logits readback.
	BatchGreedyDraw bool
}
