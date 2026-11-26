//go:build darwin

package metal

import (
	"fmt"
	"math"
	"slices"
	"sync"

	"github.com/townsendmerino/goinfer/decoder"
)

// metalResident adapts *resident (whose Forward takes a token id and returns logits, no error)
// to decoder.ResidentForward (Forward takes a precomputed embedding and returns logits+error).
type metalResident struct {
	r      *resident
	hidden int
	exact  bool // the model was loaded with Options.ExactPrefill: fast prefill off for THIS resident
	// poisonPrefillLogitsForTest makes PrefillLast write a NaN into its logits before the finite check, so a test can
	// show the check declines. False in production; only tests set it.
	poisonPrefillLogitsForTest bool
	// ctxNote is how the build arrived at its context (metalContextNote), for the load banner.
	ctxNote string
	// quant is the precision ResidentQuant reports: the model's own quant when it runs int8 weights natively (r.w8),
	// "" otherwise, which leaves the decoder's label (an int8 model on the int4 path is re-quantized, and says so).
	quant string
	// need is the memory guard's need for this model (residentNeedBytes), which prefillBlockBudget subtracts from the guard's
	// budget; nil (a test's struct literal) leaves the block at prefillBlockScratch.
	need func() int64

	// fast is the batched-prefill decision, made once per resident on first use, so a change to the process environment mid-serve
	// cannot switch a loaded model's kernels. Lazy (not set at build) so a test's struct-literal resident gets the same default.
	fastOnce sync.Once
	fast     bool
}

// ContextNote is ResidentCaps' ContextNote: how the load arrived at the context cap.
func (a *metalResident) ContextNote() string { return a.ctxNote }

// ctxCap is this resident's resolved KV capacity: a.r.ctxCap, or metalCtxCapDefault for a zero-value &metalResident{} (r == nil),
// which TestMetalResidentCheckCap builds to test checkCap and ContextCap as pure logic with no Metal device.
func (a *metalResident) ctxCap() int {
	if a.r == nil || a.r.ctxCap == 0 {
		return metalCtxCapDefault
	}
	return a.r.ctxCap
}

// checkCap refuses positions [pos, pos+n) outside the resident KV [0, ctxCap()). Every layer's cache is sized ctxCap()*kvDim, so a
// write past it is an out-of-bounds device write that on unified memory silently corrupts adjacent MTLBuffers (other models'
// resident weights). The decode loop increments pos unbounded, so this is the guard: the loop surfaces the error and the caller
// can fall back to the staged path.
func (a *metalResident) checkCap(pos, n int) error {
	c := a.ctxCap()
	if pos < 0 || pos+n > c {
		return fmt.Errorf("metal: KV position %d(+%d) exceeds resident context cap %d — use the staged path for longer contexts", pos, n, c)
	}
	return nil
}

// checkPass is checkCap for a batched pass of n rows: a bounded window group takes at most decoder.MetalKVWindowPass rows in one, so a
// longer pass (an image turn, say) declines and the decoder runs it sequentially.
func (a *metalResident) checkPass(pos, n int) error {
	if e := a.checkCap(pos, n); e != nil {
		return e
	}
	if g := a.r.kvWin; g != nil && g.bounded && n > decoder.MetalKVWindowPass {
		return fmt.Errorf("metal: a pass of %d rows exceeds the %d a bounded window group takes: run it in blocks or sequentially", n, decoder.MetalKVWindowPass)
	}
	return nil
}

// ContextCap is the resident KV capacity in positions (ResidentCaps' ContextCap), so generateInto clamps maxTokens to the cap
// up front instead of erroring mid-decode.
func (a *metalResident) ContextCap() int { return a.ctxCap() }

// ResidentCaps is decoder.ResidentRuntimeSubsystem's. The batched prefill always reports a kernel-class floor, 0 when one class runs
// at every length.
func (a *metalResident) ResidentCaps() decoder.ResidentCaps {
	return decoder.ResidentCaps{
		ContextCap:      a.ContextCap(),
		HasKernelFloor:  true,
		KernelFloor:     a.PrefillKernelFloor(),
		Quant:           a.ResidentQuant(),
		ContextNote:     a.ContextNote(),
		BatchGreedyDraw: a.BatchGreedyDraw(),
	}
}

// ForwardMRoPE is decoder.ResidentDecodeSubsystem: like Forward, but the rotation angle (ropePos) and the
// KV-cache/attention position (pos) are supplied separately — Qwen2.5-VL decode past an image
// block needs them to differ. Forward itself is ForwardMRoPE(pos, pos).
func (a *metalResident) ForwardMRoPE(embedding []float32, pos, ropePos int) ([]float32, error) {
	if len(embedding) != a.embLen() {
		return nil, fmt.Errorf("metal: embedding len %d != %d (hidden %d + PLE inputs)", len(embedding), a.embLen(), a.hidden)
	}
	if e := a.checkCap(pos, 1); e != nil {
		return nil, e
	}
	a.r.kvQuiesce(a.r.kvSlot, pos+1)
	if e := a.r.kvWrite(a.r.kvSlot, pos, pos+1); e != nil { // its block, before the forward is encoded (setPos's kvWriteMust then holds)
		return nil, e
	}
	if pos == 0 {
		// A fresh sequence re-zeroes any recurrent state (DeltaNet, Mamba-2) left by a prior Generate on this resident (no-op for other
		// families), once its write is mapped: a refused one leaves the slot as it was.
		a.Reset()
	}
	logits := a.r.ForwardEmbMRoPEPipe(embedding, pos, ropePos)
	if err := a.r.takeExecErr(); err != nil {
		return nil, err // a command buffer aborted: surface it, never return stale logits
	}
	return logits, nil
}

// Forward runs one token given its embedding[H] at absolute position pos, returning logits[V].
// The returned slice is reused across calls (the decode loop consumes it before the next call).
func (a *metalResident) Forward(embedding []float32, pos int) ([]float32, error) {
	return a.ForwardMRoPE(embedding, pos, pos)
}

// SampleAvailable and ForwardSample (decoder.ResidentDecodeSubsystem, gumbel_sample.go) delegate to *resident. They must exist on
// *metalResident because decoder holds one, and Go does not promote a named field's methods.
func (a *metalResident) SampleAvailable() bool { return a.r != nil && a.r.SampleAvailable() }

func (a *metalResident) ForwardSample(embedding []float32, pos int, temperature float64, seed, draw uint64) (int, error) {
	if len(embedding) != a.embLen() {
		return 0, fmt.Errorf("metal: embedding len %d != %d (hidden %d + PLE inputs)", len(embedding), a.embLen(), a.hidden)
	}
	if e := a.checkCap(pos, 1); e != nil {
		return 0, e
	}
	a.r.kvQuiesce(a.r.kvSlot, pos+1)
	if e := a.r.kvWrite(a.r.kvSlot, pos, pos+1); e != nil {
		return 0, e
	}
	return a.r.ForwardSample(embedding, pos, temperature, seed, draw)
}

// embLen is every embedding row's length: hidden, or for a Gemma 4 E-model hidden + L·P PLE inputs, the
// decoder's EmbedLen. Derived from the resident, so a test-built metalResident gets it right too.
func (a *metalResident) embLen() int {
	if a.r == nil {
		return a.hidden
	}
	return a.r.H + a.r.nL*a.r.pleP
}

// ForwardNoLogits (decoder.ResidentDecodeSubsystem) runs the token's forward to build only its resident K/V: no final-norm LM-head
// dispatch, no logits readback, no softcap. residentPrefillSeed calls it for every prompt token but the last; the layers, hence
// the K/V written at pos, are identical to Forward, so decode from the last prompt token is byte-identical. It runs through the
// encode-ahead executor (ForwardEmbNoLogitsPipe, a noHead bit on execJob). A paged MoE has no paged branch in the executor's
// trunk encoder, so it runs the synchronous paged forward with its final norm and head skipped (pagedNoHead).
func (a *metalResident) ForwardNoLogits(embedding []float32, pos int) error {
	if len(embedding) != a.embLen() {
		return fmt.Errorf("metal: embedding len %d != %d (hidden %d + PLE inputs)", len(embedding), a.embLen(), a.hidden)
	}
	if e := a.checkCap(pos, 1); e != nil {
		return e
	}
	a.r.kvQuiesce(a.r.kvSlot, pos+1)
	if e := a.r.kvWrite(a.r.kvSlot, pos, pos+1); e != nil {
		return e
	}
	if pos == 0 {
		a.Reset() // a fresh sequence, once its write is mapped, as Forward does
	}
	a.r.ForwardEmbNoLogitsPipe(embedding, pos)
	return a.r.takeExecErr() // a command buffer aborted: surface it
}

// ResidentQuant (decoder.ResidentCaps.Quant) is the precision this resident runs its weights at, when the decoder cannot
// infer it: an int8 model on the native path (r.w8) runs at its own quant, where one on the int4 path is re-quantized. ""
// leaves the decoder's label.
func (a *metalResident) ResidentQuant() string { return a.quant }

// ForwardN runs a batch of embeddings at consecutive positions (prefill/verify).
// It sequences all N token forward steps inside a SINGLE Metal command buffer
// and single compute encoder (one commit, one wait), returning all N logits vectors.
// For paged MoE (which requires mid-layer host interaction), it falls back to the
// sequential loop.
func (a *metalResident) ForwardN(embeddings [][]float32, startPos int) ([][]float32, error) {
	// Fail-fast before any write: checking the whole batch up front refuses an over-cap run
	// without partial KV writes.
	if e := a.checkPass(startPos, len(embeddings)); e != nil {
		return nil, e
	}
	if len(embeddings) == 0 {
		return nil, nil
	}
	// Fall back to sequential loop for paged MoE where per-layer host interaction is required, and for a Gemma 4
	// E-model, whose PLE inputs ForwardBatch's layer-major encode has no per-row buffer for.
	if (a.r.g4moe != nil && a.r.g4moe.paged) || (a.r.moe != nil && a.r.moe.paged) || a.r.pleP > 0 {
		out := make([][]float32, len(embeddings))
		for i, emb := range embeddings {
			l, err := a.Forward(emb, startPos+i)
			if err != nil {
				return nil, err
			}
			out[i] = slices.Clone(l)
		}
		return out, nil
	}
	if startPos == 0 {
		a.Reset() // fresh sequence — same DeltaNet-state reset Forward(pos==0) does
	}
	out, err := a.r.ForwardBatch(embeddings, startPos)
	if err != nil {
		return nil, err
	}
	if err := a.r.takeExecErr(); err != nil {
		return nil, err // a command buffer aborted: surface it
	}
	return out, nil
}

// VerifyPath (decoder.VerifyPathReporter) reports whether this resident's ForwardN executes
// a single-command-buffer batched pass or falls back to a sequential loop.
func (a *metalResident) VerifyPath() (bool, string) {
	if (a.r.g4moe != nil && a.r.g4moe.paged) || (a.r.moe != nil && a.r.moe.paged) {
		return false, "sequential — paged MoE requires per-layer host staging"
	}
	return true, "batched layer-major single-command-buffer"
}

// SetAdapter implements decoder.ResidentRuntimeSubsystem —
// generateInto calls this to bind/clear a compute-time LoRA adapter for an admitted session.
func (a *metalResident) SetAdapter(layers []decoder.ResidentAdapterLayer) error {
	return a.r.SetAdapter(layers)
}

// UploadKV writes a layer's post-RoPE K and raw V into the resident caches at absolute
// position base..base+n-1, where n = len(keys)/kvDim. Used by decoder.residentUploadPrefill
// to push a CPU-computed prefill's KV into the resident GPU cache, letting GenerateVL and
// GenerateQwenVL run text decode on resident GPU rather than dropping to CPU.
//
// The contract: one call is one layer's part of a pass that uploads, for every layer that owns K/V, every row that layer reads
// ([base, base+n), its own live range; residentUploadPrefill's loop). A block table serves every layer, so the call maps the bound
// slot's blocks for all of them at once (kvOwn): a block another slot shares is copied for the rows below base, and the rest of it is
// the pass's to write. A layer whose rows the pass leaves out inside that range would read stale rows there.
func (a *metalResident) UploadKV(layer, base int, keys, vals []float32) error {
	if a.r == nil {
		return fmt.Errorf("metal: resident is nil")
	}
	if layer < 0 || layer >= len(a.r.layers) {
		return fmt.Errorf("metal: UploadKV layer %d out of range", layer)
	}
	if a.r.kc[layer] == (Buffer{}) {
		return nil // recurrent DeltaNet layer with no attention KV cache
	}
	if a.r.layers[layer].kvShared {
		// its kc/vc alias the source layer's cache; writing here would overwrite the source's rows.
		return fmt.Errorf("metal: UploadKV layer %d owns no KV (it reads layer %d's)", layer, a.r.layers[layer].kvSrc)
	}
	g := a.r.layers[layer].geom
	if g == nil {
		return fmt.Errorf("metal: layer %d has no attention geometry", layer)
	}
	kvDim := g.kvDim
	if kvDim == 0 {
		return fmt.Errorf("metal: layer %d has zero kvDim", layer)
	}
	if len(keys) != len(vals) || len(keys)%kvDim != 0 {
		return fmt.Errorf("metal: UploadKV len(keys)=%d len(vals)=%d not aligned to kvDim=%d", len(keys), len(vals), kvDim)
	}
	n := len(keys) / kvDim
	if e := a.checkCap(base, n); e != nil {
		return e
	}
	a.r.stopExec() // it writes K/V from the host beside the executor, whose pipelined forward may still be in flight
	if err := a.r.kvUploadMap(layer, base, base+n); err != nil {
		return err
	}
	// Each position goes to its row through the bound slot's table (kvHostRow), the pool's contents indexed from its start. A bounded
	// window group holds the slot from its base (kvWriteUpload): the rows below it have no block, so they are not written.
	lowest := 0
	if gr := a.r.kvGroupOf(layer); gr.win > 0 && gr.bounded && !kvRetainOffForTest {
		lowest = gr.kvBase[a.r.kvSlot]
	}
	if a.r.kvI8 {
		nKV := g.nKV
		hd := g.hd
		kc, vc := a.r.kc[layer].Int8s(), a.r.vc[layer].Int8s()
		ks, vs := a.r.ks[layer].Floats(), a.r.vs[layer].Floats()
		for p := max(0, lowest-base); p < n; p++ {
			row := a.r.kvHostRow(layer, base+p)
			for h := range nKV {
				kHead := keys[p*kvDim+h*hd : p*kvDim+(h+1)*hd]
				vHead := vals[p*kvDim+h*hd : p*kvDim+(h+1)*hd]
				var amaxK, amaxV float32
				for _, v := range kHead {
					if a := float32(math.Abs(float64(v))); a > amaxK {
						amaxK = a
					}
				}
				for _, v := range vHead {
					if a := float32(math.Abs(float64(v))); a > amaxV {
						amaxV = a
					}
				}
				scK := amaxK / 127.0
				if scK == 0 {
					scK = 1.0
				}
				scV := amaxV / 127.0
				if scV == 0 {
					scV = 1.0
				}
				ks[row*nKV+h] = scK
				vs[row*nKV+h] = scV
				invK := 1.0 / scK
				invV := 1.0 / scV
				for d := range hd {
					kc[row*kvDim+h*hd+d] = int8(math.Round(float64(kHead[d] * invK)))
					vc[row*kvDim+h*hd+d] = int8(math.Round(float64(vHead[d] * invV)))
				}
			}
		}
	} else {
		// A run of positions inside one block is consecutive rows, so each block's run is one copy.
		for p := max(0, lowest-base); p < n; {
			run := min(n-p, decoder.KVBlockRows-(base+p)%decoder.KVBlockRows)
			at, src := a.r.kvHostRow(layer, base+p)*kvDim, p*kvDim
			if a.r.kvF32 {
				copy(a.r.kc[layer].Floats()[at:at+run*kvDim], keys[src:src+run*kvDim])
				copy(a.r.vc[layer].Floats()[at:at+run*kvDim], vals[src:src+run*kvDim])
			} else {
				parallelF32ToF16(a.r.kc[layer].U16s()[at:at+run*kvDim], keys[src:src+run*kvDim])
				parallelF32ToF16(a.r.vc[layer].U16s()[at:at+run*kvDim], vals[src:src+run*kvDim])
			}
			p += run
		}
	}
	return nil
}

// KVSlots and UseKVSlot implement decoder.ResidentKVSubsystem: the slots metalKVSlots
// allocated, and binding one.
func (a *metalResident) KVSlots() int          { return a.r.kvSlotCount() }
func (a *metalResident) UseKVSlot(i int) error { return a.r.useKVSlot(i) }

// TruncateTo is a no-op: KV positions are overwritten on write, and attention reads only keys[0..pos], so stale positions past
// the current one are never observed.
func (a *metalResident) TruncateTo(pos int) {}

// Reset zeroes every Gated-DeltaNet layer's causal-conv ring and recurrent matrix state (no-op for every other family; a.r.dnet is
// nil). Unlike KV positions this state compounds, so a fresh sequence on the same resident must reset it or it keeps decaying the
// previous sequence's state (resetDeltaNet).
func (a *metalResident) Reset() {
	if a.r.ckptSlot == a.r.kvSlot { // a new sequence on the slot: what the checkpoint holds is no request's
		a.r.ckptSlot = -1
	}
	a.r.resetRecurrent()
}

// GreedyChainAvailable (decoder.ResidentDecodeSubsystem): the greedy chain is exact on this resident
// (greedyChainWhyNot, greedy_chain.go) and its gather table is on the device, or fits there now (chainEmbedTable).
func (a *metalResident) GreedyChainAvailable() bool {
	if a.r.greedyChainWhyNot() != "" {
		return false
	}
	_, _, why := a.r.chainEmbedTable()
	return why == ""
}

// GreedyChainStart opens the chain at token id, position pos.
func (a *metalResident) GreedyChainStart(id, pos int) error {
	if e := a.checkCap(pos, 1); e != nil {
		return e
	}
	return a.r.chainStart(id, pos)
}

// GreedyChainNext returns the next greedy token.
func (a *metalResident) GreedyChainNext() (int, error) { return a.r.chainNext() }

// GreedyChainStop closes the chain.
func (a *metalResident) GreedyChainStop() { a.r.stopChain() }

// SampleChainAvailable (decoder.ResidentDecodeSubsystem): the chain is available and so is the device draw.
func (a *metalResident) SampleChainAvailable() bool {
	return a.r.SampleAvailable() && a.GreedyChainAvailable()
}

// SampleChainStart opens a sampled chain at token id, position pos, drawing that forward with (seed, draw).
func (a *metalResident) SampleChainStart(id, pos int, temperature float64, seed, draw uint64) error {
	if e := a.checkCap(pos, 1); e != nil {
		return e
	}
	return a.r.chainStartSampled(id, pos, temperature, seed, draw)
}

// SampleChainNext returns the next sampled token; (seed, draw) is the caller's draw for it.
func (a *metalResident) SampleChainNext(seed, draw uint64) (int, error) {
	return a.r.chainNextSampled(seed, draw)
}

// Close stops the pipelined executor (waiting for it) and frees every MTLBuffer this resident
// allocated. Metal buffers are unified/system memory and purego has no ARC, so without this a
// multi-model serve (or /admin/models/unload) leaks the whole model per load.
func (a *metalResident) Close() error { return a.r.Close() }
