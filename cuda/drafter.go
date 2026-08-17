//go:build cuda

package cuda

import (
	"fmt"

	"github.com/townsendmerino/aikit/linalg"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// residentDrafter is a block drafter's weights on the device, attached to an existing resident
// target (P10 / docs/spec/08).
//
// IT SHARES THE TARGET'S DEVICE, STREAM AND KERNELS rather than building its own context. The
// n-gram/EAGLE precedent (decoder/speculative.go) runs its draft as a separate *Model with its
// own context, which is right there because the draft IS a model. A block drafter is not: it has
// no embedding and no LM head (it borrows the target's), its context comes from the target's
// hidden states rather than from tokens, and it is FIVE layers against the target's thirty-six.
// Sharing means the drafter's block forward and the target's verify are already ordered on one
// stream, with no cross-context synchronisation between them — which matters because the two
// alternate every round.
//
// Built against decoder.BlockDrafterWeights, so DFlash and DSpark both work with no type switch.
type residentDrafter struct {
	r   *cudaResident // the target this is attached to; owns the device, stream and kernels
	geo decoder.DrafterGeometry

	fc         cudaWQ // [hidden, nTaps*hidden] — the fusion projection
	hiddenNorm Buffer // RMSNorm weight applied to the fused context
	finalNorm  Buffer
	layers     []drafterLayer
	invF       Buffer // RoPE inverse frequencies, shared by every layer

	// Per-layer context K/V, built by ExtendContext and read by the block's attention. The
	// drafter keeps its OWN caches: the target's hold token K/V at token positions, while these
	// hold K/V projected from the FUSED tap hidden states, which is a different tensor at the
	// same indices.
	kc, vc []Buffer
	kvCap  int // positions each cache can hold
	ctxLen int // positions currently valid
	rhalf  int

	// Scratch for the context fusion and the per-layer projections, sized to the widest ctx seen.
	ctxCap  int
	ctxIn   Buffer // [ctxCap, fcCols] f32 tap rows as handed in
	ctxQ    Buffer // quantized ctxIn
	ctxSc   Buffer
	ctxFuse Buffer // [ctxCap, hidden] fused + normed
	ctxFQ   Buffer // quantized ctxFuse (input to k/v projections)
	ctxFSc  Buffer
	ctxQ2   Buffer // discarded q scratch: rope_kv_batched ropes q and k together, and the
	//                 context has no q — projecting one and throwing it away is cheaper than
	//                 a second kernel, and keeps this on the SAME code path as the block.
	ctxKB  Buffer
	ctxVB  Buffer
	extIn  Buffer // ExtendContext's own upload buffer
	extCap int    // rows the ExtendContext scratch covers
}

// drafterLayer is one trunk layer on the device. Same projections as a Qwen3 layer, which is why
// every kernel here is the target's own — the ONLY thing that differs is the attention mask.
type drafterLayer struct {
	q, k, v, o     cudaWQ
	gate, up, down cudaWQ
	qNorm, kNorm   Buffer
	inputNorm      Buffer
	postAttnNorm   Buffer
}

// AttachDrafter uploads a block drafter's weights to the device this resident already owns.
//
// The geometry is CHECKED against the target rather than trusted: a drafter reads the target's
// residual stream, so a hidden-dim mismatch is not a resizing problem, it is the wrong pairing —
// and the failure mode is a drafter that runs and drafts noise. docs/spec/08 records that exact
// shape of bug costing a full measurement round.
func (r *cudaResident) AttachDrafter(w decoder.BlockDrafterWeights) (*residentDrafter, error) {
	geo := w.DrafterGeometry()
	if geo.Hidden != r.hidden {
		return nil, fmt.Errorf("cuda drafter: hidden %d != target hidden %d — wrong pairing", geo.Hidden, r.hidden)
	}
	if geo.Layers == 0 {
		return nil, fmt.Errorf("cuda drafter: zero layers")
	}
	d := &residentDrafter{r: r, geo: geo}
	err := r.do(func() error {
		fcw, e := packWeight(w.DrafterFC())
		if e != nil {
			return fmt.Errorf("cuda drafter: pack fc: %w", e)
		}
		d.fc = r.upW(fcw)
		d.hiddenNorm = r.up32(w.DrafterHiddenNorm())
		d.finalNorm = r.up32(w.DrafterFinalNorm())
		inv := make([]float32, len(geo.InvFreq))
		for i, v := range geo.InvFreq {
			inv[i] = float32(v)
		}
		d.invF = r.up32(inv)
		d.rhalf = geo.HeadDim / 2 // full rotary: the drafter rotates every pair
		d.layers = make([]drafterLayer, geo.Layers)
		for i := range d.layers {
			lw := w.DrafterLayer(i)
			for _, m := range []struct {
				dst *cudaWQ
				src *linalg.WeightMat
				nm  string
			}{
				{&d.layers[i].q, lw.Q, "q"}, {&d.layers[i].k, lw.K, "k"},
				{&d.layers[i].v, lw.V, "v"}, {&d.layers[i].o, lw.O, "o"},
				{&d.layers[i].gate, lw.Gate, "gate"}, {&d.layers[i].up, lw.Up, "up"},
				{&d.layers[i].down, lw.Down, "down"},
			} {
				h, e := packWeight(m.src)
				if e != nil {
					return fmt.Errorf("cuda drafter: layer %d %s: %w", i, m.nm, e)
				}
				*m.dst = r.upW(h)
			}
			d.layers[i].qNorm = r.up32(lw.QNorm)
			d.layers[i].kNorm = r.up32(lw.KNorm)
			d.layers[i].inputNorm = r.up32(lw.InputNorm)
			d.layers[i].postAttnNorm = r.up32(lw.PostAttnNorm)
		}
		return r.stream.Sync()
	})
	if err != nil {
		return nil, err
	}
	return d, nil
}

// FuseContext projects the target's concatenated tap hidden states down to the trunk's width and
// norms them — the drafter's `fc` + `hiddenNorm`, batched over all rows in one pass.
//
// This is the one projection with no counterpart in a normal decoder layer, which is why it is
// the first thing built: if packWeight/upW mishandle the drafter's weights, it shows up here
// rather than eighteen kernels later.
//
// The rows are quantized to int8 on the way in, exactly as the target's own activations are, so
// the result is NOT bit-identical to the CPU f32 path — it is the same int8 arithmetic the
// resident target runs everywhere else, and the parity gate compares by cosine accordingly.
func (d *residentDrafter) FuseContext(rows [][]float32) ([][]float32, error) {
	n := len(rows)
	if n == 0 {
		return nil, fmt.Errorf("cuda drafter: empty context")
	}
	k := d.fc.K
	for i, row := range rows {
		if len(row) != k {
			return nil, fmt.Errorf("cuda drafter: context row %d is %d wide, want %d", i, len(row), k)
		}
	}
	hidden := d.geo.Hidden
	out := make([][]float32, n)
	err := d.r.do(func() error {
		if n > d.ctxCap {
			d.ctxIn = d.r.af(n * k)
			d.ctxQ = d.r.ai(n * (k / 4))
			d.ctxSc = d.r.af(n)
			d.ctxFuse = d.r.af(n * hidden)
			d.ctxCap = n
		}
		flat := make([]float32, 0, n*k)
		for _, row := range rows {
			flat = append(flat, row...)
		}
		if e := gpu.Upload(d.ctxIn, flat); e != nil {
			return e
		}
		// quantize each row, then ONE batched GEMV over all of them: the fc weights are read
		// once for the whole context instead of once per row.
		if e := d.r.launch(d.r.bQuant, LaunchConfig{GridX: uint32(n), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
			Arg(d.ctxIn), gpu.ArgValue(int32(k)), Arg(d.ctxQ), Arg(d.ctxSc), gpu.ArgValue(int32(n))); e != nil {
			return e
		}
		if e := d.r.bGemvB(d.fc, d.ctxQ, d.ctxSc, ArgNull(), d.ctxFuse, n, 0); e != nil {
			return e
		}
		// NOT r.bNormF32B: that hardcodes the TARGET's eps and add-one flag. The drafter has
		// its own normEps, and its RMSNorm is plain (addOne=0) per blockTrunk's CPU path —
		// applying the target's would be a silent, plausible-looking wrong normalization.
		if e := d.r.launch(d.r.bNormF32, LaunchConfig{GridX: 1, GridY: uint32(n), GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
			Arg(d.ctxFuse), Arg(d.hiddenNorm), gpu.ArgValue(int32(hidden)),
			gpu.ArgValue(float32(d.geo.NormEps)), gpu.ArgValue(int32(0)),
			gpu.ArgValue(int32(n))); e != nil {
			return e
		}
		if e := d.r.stream.Sync(); e != nil {
			return e
		}
		host := make([]float32, n*hidden)
		if e := gpu.Download(d.ctxFuse, host); e != nil {
			return e
		}
		for i := 0; i < n; i++ {
			out[i] = append([]float32(nil), host[i*hidden:(i+1)*hidden]...)
		}
		return d.r.launchErr
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ExtendContext projects the fused context rows into every layer's K/V cache, at positions
// [ctxLen, ctxLen+len(fused)), and advances ctxLen.
//
// TWO THINGS HERE ARE NOT WHAT A DECODER LAYER DOES, and both are load-bearing:
//
//	NO INPUT NORM. The context's K/V come from the fused rows RAW. blockTrunk.layer says it
//	outright — "input_layernorm normalizes the BLOCK only... the reference passes target_hidden
//	straight into k_proj/v_proj while only hidden_states goes through the norm. Norming both
//	would be the natural-looking port and would be wrong."
//
//	NO Q. The context supplies keys and values only; queries come from the block. rope_kv_batched
//	rotates q and k together, so a q scratch is projected and discarded — cheaper than a second
//	kernel, and it keeps the context on the SAME code path as the block, which is what stops the
//	two drifting apart.
//
// INCREMENTAL BY CONSTRUCTION: it appends at ctxLen rather than rebuilding. The CPU measurement
// (TestDFlashDraftScaling) put the rebuild path at 2.4x the incremental one at a 1024-token
// context, widening with length — a full drafter-prefill of the whole context on every block.
func (d *residentDrafter) ExtendContext(fused [][]float32) error {
	n := len(fused)
	if n == 0 {
		return nil
	}
	geo := d.geo
	hidden, hd, nKV, nH := geo.Hidden, geo.HeadDim, geo.NumKVHeads, geo.NumHeads
	kvDim, qDim := nKV*hd, nH*hd
	need := d.ctxLen + n
	return d.r.do(func() error {
		if need > d.kvCap {
			cap := need + 512 // headroom so a long generation does not realloc every round
			d.kc = make([]Buffer, geo.Layers)
			d.vc = make([]Buffer, geo.Layers)
			for l := range d.kc {
				d.kc[l] = d.r.af(cap * kvDim)
				d.vc[l] = d.r.af(cap * kvDim)
			}
			if d.ctxLen > 0 {
				return fmt.Errorf("cuda drafter: context grew past capacity mid-sequence (%d > %d); "+
					"reallocating would drop the K/V already written", need, d.kvCap)
			}
			d.kvCap = cap
		}
		// Sized from THIS call's row count, not from whatever FuseContext last allocated.
		// Coupling the two would make ExtendContext depend on a prior FuseContext having run
		// with at least as many rows — an ordering rule between two exported methods that
		// nothing states and that fails as a confusing capacity error.
		if n > d.extCap {
			d.extIn = d.r.af(n * hidden)
			d.ctxFQ = d.r.ai(n * (hidden / 4))
			d.ctxFSc = d.r.af(n)
			d.ctxQ2 = d.r.af(n * qDim)
			d.ctxKB = d.r.af(n * kvDim)
			d.ctxVB = d.r.af(n * kvDim)
			d.extCap = n
		}
		flat := make([]float32, 0, n*hidden)
		for _, row := range fused {
			flat = append(flat, row...)
		}
		if e := gpu.Upload(d.extIn, flat); e != nil {
			return e
		}
		// quantize the fused rows ONCE — every layer's k/v projection reads the same activations.
		if e := d.r.launch(d.r.bQuant, LaunchConfig{GridX: uint32(n), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
			Arg(d.extIn), gpu.ArgValue(int32(hidden)), Arg(d.ctxFQ), Arg(d.ctxFSc),
			gpu.ArgValue(int32(n))); e != nil {
			return e
		}
		ropeN := nH*d.rhalf + nKV*d.rhalf + nKV*(hd-2*d.rhalf)
		for l := range d.layers {
			L := &d.layers[l]
			if e := d.r.bGemvB(L.q, d.ctxFQ, d.ctxFSc, ArgNull(), d.ctxQ2, n, 0); e != nil {
				return e
			}
			if e := d.r.bGemvB(L.k, d.ctxFQ, d.ctxFSc, ArgNull(), d.ctxKB, n, 0); e != nil {
				return e
			}
			if e := d.r.bGemvB(L.v, d.ctxFQ, d.ctxFSc, ArgNull(), d.ctxVB, n, 0); e != nil {
				return e
			}
			// per-head Q/K RMSNorm before rope, with the DRAFTER's eps and a plain (addOne=0)
			// norm — not the target's, for the same reason the fusion norm is launched directly.
			if L.kNorm.Len() > 0 {
				if e := d.r.launch(d.r.bQKN, LaunchConfig{GridX: uint32(nH + nKV), GridY: uint32(n), GridZ: 1,
					BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8},
					Arg(d.ctxQ2), Arg(d.ctxKB), Arg(L.qNorm), Arg(L.kNorm),
					gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(hd)),
					gpu.ArgValue(float32(geo.NormEps)), gpu.ArgValue(int32(0)),
					gpu.ArgValue(int32(n))); e != nil {
					return e
				}
			}
			// rope at absolute positions ctxLen+i, and store K/V into this layer's cache.
			if e := d.r.launch(d.r.bRopeKV, LaunchConfig{GridX: uint32((ropeN + 255) / 256), GridY: uint32(n), GridZ: 1,
				BlockX: 256, BlockY: 1, BlockZ: 1},
				Arg(d.ctxQ2), Arg(d.ctxKB), Arg(d.ctxVB), Arg(d.invF), Arg(d.kc[l]), Arg(d.vc[l]),
				gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(hd)),
				gpu.ArgValue(int32(d.ctxLen)), gpu.ArgValue(int32(d.rhalf)), gpu.ArgValue(int32(n))); e != nil {
				return e
			}
		}
		if e := d.r.stream.Sync(); e != nil {
			return e
		}
		d.ctxLen = need
		return d.r.launchErr
	})
}

// ContextLen is how many positions the drafter's K/V caches currently hold.
func (d *residentDrafter) ContextLen() int { return d.ctxLen }
