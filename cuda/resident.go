//go:build cuda

package cuda

import "github.com/townsendmerino/goinfer/decoder"

// cudaResident satisfies decoder.ResidentForward.
var _ decoder.ResidentForward = (*cudaResident)(nil)

// cudaResident is the resident decode runner (Layer B) — a scaffold. The per-token
// flow is laid out below; every launch hits the not-wired stub driver until the box
// fills megakernel.cu + a gocudrv-backed driver. Buffers, the JIT'd module, and the
// three fused super-kernel handles are populated once in BuildResident.
type cudaResident struct {
	drv driver

	// dims (from m.Dims()): hidden, layers, query/kv heads, head dim, ffn width, vocab.
	hidden, nLayers, nH, nKV, hd, inter, vocab int
	kvDim                                      int // nKV*hd

	mod        module // JIT'd megakernelPTX
	k1, k2, k3 fn     // the 3 fused super-kernels (spec §5.2): pre-attn | attn | ffn

	// Device buffers, allocated once (weights per layer, KV caches, scratch, the input
	// embedding xd, logits). Elided in the skeleton.
}

// Forward runs one token at absolute position pos and returns logits[vocab].
//
// Flow (the box implements the launches):
//  1. H2D the embedding into xd; set per-token uniforms (rope pos, KV-store base
//     pos*kvDim, attn nKeys=pos+1) — the only per-token writes.
//  2. For each layer L: Launch k1(L) → k2(L) → k3(L). The launch boundaries provide
//     the grid-wide sync between data-dependent stages (spec §5.2), so no
//     cooperative launch is needed for the first cut.
//  3. Final RMSNorm+quant → LM-head GEMV → logits.
//  4. D2H logits[vocab]; Synchronize (CUDA-event timed).
func (r *cudaResident) Forward(embedding []float32, pos int) ([]float32, error) {
	return nil, errCUDANotWired
}

// ForwardN runs K tokens at consecutive positions in one submit — the batched verify
// for speculative decoding. Causal: row i attends [0, startPos+i]. Amortizes the
// per-call channel-hop over K. Must be bit-identical to K sequential Forward calls.
func (r *cudaResident) ForwardN(embeddings [][]float32, startPos int) ([][]float32, error) {
	return nil, errCUDANotWired
}

// UploadKV writes a layer's post-RoPE K and raw V into the resident caches — the
// prefill bridge (same packed layout the megakernel reads: [pos*kvDim + head*hd + d]).
func (r *cudaResident) UploadKV(layer int, keys, vals []float32) error {
	return errCUDANotWired
}

// TruncateTo is a no-op: the resident KV is positional and Forward sets nKeys=pos+1,
// so entries past pos are never read and get overwritten (matches the WebGPU path).
func (r *cudaResident) TruncateTo(pos int) {}

// Reset clears resident KV for a fresh generation (positions overwritten).
func (r *cudaResident) Reset() {}

// Close releases the resident device buffers.
func (r *cudaResident) Close() error { return nil }
