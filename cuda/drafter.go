//go:build cuda

package cuda

import (
	"fmt"

	"github.com/townsendmerino/aikit/linalg"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
	"math"
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

	blk       blockScratch
	headIn    Buffer
	headQ     Buffer
	headSc    Buffer
	headOut   Buffer
	headCap   int
	attnBlock Pipeline // attn_block_full — bound HERE, where the consumer now exists
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
	// N-08: r.upW / r.up32 do not return errors — they record into r.setupErr, which the BUILD
	// path already consumed by the time the drafter is constructed. So every upload failure
	// below landed in a field nothing reads again, and the drafter was returned successfully
	// with zeroed weights: a drafter that proposes garbage, which lossless verify then rejects,
	// so it costs acceptance rather than correctness and no gate can see it.
	//
	// Snapshot and compare rather than clear: another goroutine's build is not in flight here
	// (this runs inside r.do), but leaving an unrelated earlier error in place is not this
	// function's business either.
	setupBefore := r.setupErr
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
		bmod, e2 := r.dev.CompileLibrary(attnBlockPTX)
		if e2 != nil {
			return e2
		}
		if d.attnBlock, e2 = r.dev.NewComputePipeline(bmod, "attn_block_full"); e2 != nil {
			return e2
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
	if r.setupErr != nil && r.setupErr != setupBefore {
		return nil, fmt.Errorf("cuda drafter: weight upload failed: %w", r.setupErr)
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
			// REFUSE BEFORE ALLOCATING, NOT AFTER. The check sat below the allocation, so a
			// mid-sequence overflow had ALREADY replaced d.kc/d.vc with fresh empty buffers before
			// returning its error — the K/V it warns about dropping was dropped by the line above
			// the warning (audit-2026-09-02 M-15).
			if d.ctxLen > 0 {
				return fmt.Errorf("cuda drafter: context grew past capacity mid-sequence (%d > %d); "+
					"reallocating would drop the K/V already written", need, d.kvCap)
			}
			// SIZED TO THE TARGET'S CONTEXT, not need+512. The drafter is a 5-layer trunk, so its
			// whole K/V is layers*ctxCap*kvDim*2 floats — small. At need+512 the capacity froze at
			// len(prompt)+512 on the first call and never grew, so any greedy generation past
			// ~500 tokens failed mid-stream, prompt-length independent, and blockspec returned that
			// as the generation's terminal error. Every committed block-spec test stops at <= 96
			// tokens, so none of them could reach it.
			capRows := need + 512
			if d.r.ctxCap > capRows {
				capRows = d.r.ctxCap
			}
			d.kc = make([]Buffer, geo.Layers)
			d.vc = make([]Buffer, geo.Layers)
			for l := range d.kc {
				d.kc[l] = d.r.af(capRows * kvDim)
				d.vc[l] = d.r.af(capRows * kvDim)
			}
			d.kvCap = capRows
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
				gpu.ArgValue(int32(d.ctxLen)), gpu.ArgValue(int32(d.rhalf)), gpu.ArgValue(int32(n)),
				// Drafters build from a geometry (no decoder.Model), which carries no YaRN
				// attention_factor — so 1.0, not a value fetched from nowhere. A YaRN drafter
				// would have to thread RopeMscaleLayer through geo first; this is where it lands.
				gpu.ArgValue(float32(1))); e != nil {
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

// blockScratch is the per-block working set, sized to the widest block seen.
type blockScratch struct {
	cap                int
	x                  Buffer // [M, hidden] the residual stream
	aq, mq, cq, dq     Buffer // quantized activations for the four GEMV groups
	aSc, mSc, cSc, dSc Buffer
	q, k, v            Buffer
	cctx               Buffer // attention output, [M, qDim]
	g, u, dScr         Buffer
}

// checkDrafterShmem is DraftBlock's shared-memory guard, extracted so it is testable without a
// live device (V-05, docs/review-2026-09-04.md): the drafter's attnBlock launch sizes its dynamic
// shared memory the same way decode's does, (nKeys+128)*4 bytes, with no split-KV fallback here
// either. Every drafter layer attends the WHOLE context plus the whole block (no per-layer
// window, unlike the target's prefill), so one check covers all layers — unlike
// checkPrefillShmem's per-layer loop. M-15's kvCap = ctxCap means a --drafter run can reach this
// boundary exactly where the target's own batched prefill does.
func checkDrafterShmem(ctxLen, M int) error {
	nKeys := ctxLen + M
	if splitKVRequired(nKeys) {
		return fmt.Errorf("cuda drafter: attention at %d attended keys needs %d B of shared "+
			"memory, past this device's %d B limit — the drafter has no split-KV path, so this "+
			"context length needs a shorter --drafter block or a lower -ctx", nKeys,
			attnShmemBytes(nKeys), singleBlockAttnShmemLimit)
	}
	return nil
}

// DraftBlock runs the trunk over one block and returns its output rows.
//
// The block occupies absolute positions [ctxLen, ctxLen+M), directly after the context
// ExtendContext wrote — so the attention sees ctx‖block exactly, with attn_block_full's uniform
// nKeys giving every row all of it (the drafter's block is bidirectional; see attn_block.cu).
//
// Every kernel here is the target's own except that one. The drafter is five layers of the same
// Qwen3 shape, which is the whole reason this is assembly rather than new numerics.
//
// THE NORM CONSTANTS ARE THE DRAFTER'S, NOT THE TARGET'S, everywhere. bRmsB/bNormF32B/the
// prefill qk-norm all read r.eps and r.addOneArg(), and reusing them here would apply the
// target's normalization to the drafter's weights — silent, plausible, and wrong. Each is
// launched directly with the drafter's own eps and a plain (addOne=0) norm.
func (d *residentDrafter) DraftBlock(blockIn [][]float32) ([][]float32, error) {
	M := len(blockIn)
	if M == 0 {
		return nil, fmt.Errorf("cuda drafter: empty block")
	}
	geo := d.geo
	hidden, hd, nH, nKV, inter := geo.Hidden, geo.HeadDim, geo.NumHeads, geo.NumKVHeads, geo.Intermediate
	qDim, kvDim := nH*hd, nKV*hd
	for i, row := range blockIn {
		if len(row) != hidden {
			return nil, fmt.Errorf("cuda drafter: block row %d is %d wide, want %d", i, len(row), hidden)
		}
	}
	if d.ctxLen == 0 {
		return nil, fmt.Errorf("cuda drafter: no context — call ExtendContext first")
	}
	if d.ctxLen+M > d.kvCap {
		return nil, fmt.Errorf("cuda drafter: block at %d..%d exceeds K/V capacity %d", d.ctxLen, d.ctxLen+M, d.kvCap)
	}
	if e := checkDrafterShmem(d.ctxLen, M); e != nil {
		return nil, e
	}
	out := make([][]float32, M)
	err := d.r.do(func() error {
		s := &d.blk
		if M > s.cap {
			s.x = d.r.af(M * hidden)
			s.aq, s.aSc = d.r.ai(M*(hidden/4)), d.r.af(M)
			s.mq, s.mSc = d.r.ai(M*(hidden/4)), d.r.af(M)
			s.cq, s.cSc = d.r.ai(M*(qDim/4)), d.r.af(M)
			s.dq, s.dSc = d.r.ai(M*(inter/4)), d.r.af(M)
			s.q, s.k, s.v = d.r.af(M*qDim), d.r.af(M*kvDim), d.r.af(M*kvDim)
			s.cctx = d.r.af(M * qDim)
			s.g, s.u, s.dScr = d.r.af(M*inter), d.r.af(M*inter), d.r.af(M*inter)
			s.cap = M
		}
		flat := make([]float32, 0, M*hidden)
		for _, row := range blockIn {
			flat = append(flat, row...)
		}
		if e := gpu.Upload(s.x, flat); e != nil {
			return e
		}
		eps := float32(geo.NormEps)
		ropeN := nH*d.rhalf + nKV*d.rhalf + nKV*(hd-2*d.rhalf)
		nKeys := d.ctxLen + M
		for l := range d.layers {
			L := &d.layers[l]
			// input_layernorm: the BLOCK only (the context was projected raw).
			if e := d.r.launch(d.r.bRms, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1,
				BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((256 + hidden) * 4)},
				Arg(s.x), Arg(L.inputNorm), gpu.ArgValue(int32(hidden)), gpu.ArgValue(eps),
				gpu.ArgValue(int32(0)), Arg(s.aq), Arg(s.aSc)); e != nil {
				return e
			}
			if e := d.r.bGemvB(L.q, s.aq, s.aSc, ArgNull(), s.q, M, 0); e != nil {
				return e
			}
			if e := d.r.bGemvB(L.k, s.aq, s.aSc, ArgNull(), s.k, M, 0); e != nil {
				return e
			}
			if e := d.r.bGemvB(L.v, s.aq, s.aSc, ArgNull(), s.v, M, 0); e != nil {
				return e
			}
			if L.kNorm.Len() > 0 {
				if e := d.r.launch(d.r.bQKN, LaunchConfig{GridX: uint32(nH + nKV), GridY: uint32(M), GridZ: 1,
					BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8},
					Arg(s.q), Arg(s.k), Arg(L.qNorm), Arg(L.kNorm),
					gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(hd)),
					gpu.ArgValue(eps), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(M))); e != nil {
					return e
				}
			}
			// rope at the block's absolute positions, and write its K/V after the context's.
			if e := d.r.launch(d.r.bRopeKV, LaunchConfig{GridX: uint32((ropeN + 255) / 256), GridY: uint32(M), GridZ: 1,
				BlockX: 256, BlockY: 1, BlockZ: 1},
				Arg(s.q), Arg(s.k), Arg(s.v), Arg(d.invF), Arg(d.kc[l]), Arg(d.vc[l]),
				gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(hd)),
				gpu.ArgValue(int32(d.ctxLen)), gpu.ArgValue(int32(d.rhalf)), gpu.ArgValue(int32(M)),
				gpu.ArgValue(float32(1))); e != nil { // no YaRN on a drafter geometry — see the note above
				return e
			}
			// THE one kernel that is not the target's: uniform nKeys, so every block row
			// attends the whole context AND the whole block.
			if e := d.r.launch(d.attnBlock, LaunchConfig{GridX: uint32(nH), GridY: uint32(M), GridZ: 1,
				BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nKeys + 128) * 4)},
				Arg(s.q), Arg(d.kc[l]), Arg(d.vc[l]),
				gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(hd)),
				// N-11: the DRAFTER's scale, not the target's. d.r.attnScale is the target
				// model's, and this file's own rule three dozen lines up says the norm
				// constants are the drafter's everywhere — this launch was the exception.
				// Equal today only because DFlash drafters share head_dim 128 with Qwen3
				// targets at the default scale, and lossless verify makes any mismatch
				// perf-only: the drafter proposes worse tokens, verify rejects them,
				// acceptance falls, and every correctness gate stays green.
				gpu.ArgValue(int32(d.ctxLen)), gpu.ArgValue(d.attnScale()),
				gpu.ArgValue(int32(0)), gpu.ArgValue(int32(M)), Arg(s.cctx)); e != nil {
				return e
			}
			if e := d.r.launch(d.r.bQuant, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1,
				BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
				Arg(s.cctx), gpu.ArgValue(int32(qDim)), Arg(s.cq), Arg(s.cSc), gpu.ArgValue(int32(M))); e != nil {
				return e
			}
			// o-proj accumulates straight into the residual (accum=1) — no sandwich norm here.
			if e := d.r.bGemvB(L.o, s.cq, s.cSc, ArgNull(), s.x, M, 1); e != nil {
				return e
			}
			// post_attention_layernorm → SwiGLU MLP → residual.
			if e := d.r.launch(d.r.bRms, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1,
				BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((256 + hidden) * 4)},
				Arg(s.x), Arg(L.postAttnNorm), gpu.ArgValue(int32(hidden)), gpu.ArgValue(eps),
				gpu.ArgValue(int32(0)), Arg(s.mq), Arg(s.mSc)); e != nil {
				return e
			}
			if e := d.r.bGemvB(L.gate, s.mq, s.mSc, ArgNull(), s.g, M, 0); e != nil {
				return e
			}
			if e := d.r.bGemvB(L.up, s.mq, s.mSc, ArgNull(), s.u, M, 0); e != nil {
				return e
			}
			// act=1 is SiLU (ACT_SILU in prefill_batched.cu). Passed literally rather than
			// r.act: that is the TARGET's activation, and a Gemma target would hand a GELU-tanh
			// to a SwiGLU drafter — the FeatGatedGELU class of bug, silently.
			if e := d.r.launch(d.r.bSw, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1,
				BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
				Arg(s.g), Arg(s.u), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(0)),
				gpu.ArgValue(int32(inter)), gpu.ArgValue(int32(1)),
				Arg(s.dq), Arg(s.dSc), Arg(s.dScr), gpu.ArgValue(int32(M))); e != nil {
				return e
			}
			if e := d.r.bGemvB(L.down, s.dq, s.dSc, ArgNull(), s.x, M, 1); e != nil {
				return e
			}
		}
		// final RMSNorm over the block rows, in place.
		if e := d.r.launch(d.r.bNormF32, LaunchConfig{GridX: 1, GridY: uint32(M), GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
			Arg(s.x), Arg(d.finalNorm), gpu.ArgValue(int32(hidden)), gpu.ArgValue(eps),
			gpu.ArgValue(int32(0)), gpu.ArgValue(int32(M))); e != nil {
			return e
		}
		if e := d.r.stream.Sync(); e != nil {
			return e
		}
		host := make([]float32, M*hidden)
		if e := gpu.Download(s.x, host); e != nil {
			return e
		}
		for i := 0; i < M; i++ {
			out[i] = append([]float32(nil), host[i*hidden:(i+1)*hidden]...)
		}
		return d.r.launchErr
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// SetBatchedCapture arms the batched hidden-state seam on the target: the next PrefillLastN /
// PrefillLastNArgmax records the residual for ALL its rows at each named layer.
//
// The per-token seam (SetHiddenCapture) costs a sync and a download per tap PER TOKEN — measured
// at 0.465 ms/token for five taps, ~2.3 ms per round at four accepted. This pays one download
// per tap for the whole block, because the batched forward already has every row's residual in
// one buffer.
func (r *cudaResident) SetBatchedCapture(taps []int) error {
	if len(taps) == 0 {
		r.capBTaps, r.capBOut = nil, nil
		return nil
	}
	prev := -1
	for _, t := range taps {
		if t <= prev {
			return fmt.Errorf("cuda: batched capture taps must be ascending, got %v", taps)
		}
		if t < 0 || t >= r.nLayers {
			return fmt.Errorf("cuda: batched capture tap %d out of range [0,%d)", t, r.nLayers)
		}
		prev = t
	}
	r.capBTaps = append([]int(nil), taps...)
	r.capBOut = make([][]float32, len(taps))
	return nil
}

// BatchedCapture returns the rows recorded by the last batched forward, as [tap][M*hidden].
func (r *cudaResident) BatchedCapture() [][]float32 { return r.capBOut }

// DraftTokens turns trunk output rows into token ids using the TARGET's LM head.
//
// A block drafter ships no head of its own — it borrows the target's, which is why the pairing
// is fixed and why the head cost already sits inside the verify's budget rather than the draft's.
// The trunk's final norm has already been applied, so this is head + argmax and nothing else.
//
// Batched for the same reason the verify's head is: one weight read of the head's ~389 M
// parameters for all M rows instead of one per row.
func (d *residentDrafter) DraftTokens(trunk [][]float32) ([]int, error) {
	M := len(trunk)
	if M == 0 {
		return nil, nil
	}
	hidden := d.geo.Hidden
	r := d.r
	ids := make([]int, M)
	err := r.do(func() error {
		if M > d.headCap {
			d.headIn = r.af(M * hidden)
			d.headQ, d.headSc = r.ai(M*(hidden/4)), r.af(M)
			d.headOut = r.af(M * r.vocab)
			d.headCap = M
		}
		flat := make([]float32, 0, M*hidden)
		for _, row := range trunk {
			if len(row) != hidden {
				return fmt.Errorf("cuda drafter: trunk row is %d wide, want %d", len(row), hidden)
			}
			flat = append(flat, row...)
		}
		if e := gpu.Upload(d.headIn, flat); e != nil {
			return e
		}
		if e := r.launch(r.bQuant, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1,
			BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
			Arg(d.headIn), gpu.ArgValue(int32(hidden)), Arg(d.headQ), Arg(d.headSc),
			gpu.ArgValue(int32(M))); e != nil {
			return e
		}
		if e := r.bGemvB(r.lmW, d.headQ, d.headSc, ArgNull(), d.headOut, M, 0); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		host := make([]float32, M*r.vocab)
		if e := gpu.Download(d.headOut, host); e != nil {
			return e
		}
		for m := 0; m < M; m++ {
			row := host[m*r.vocab : (m+1)*r.vocab]
			bi, bv := 0, row[0]
			for i, v := range row {
				if v > bv {
					bi, bv = i, v
				}
			}
			ids[m] = bi
		}
		return r.launchErr
	})
	if err != nil {
		return nil, err
	}
	return ids, nil
}

// TruncateContext drops drafter context positions at index >= n — the rollback after a partial
// accept. The K/V beyond n stay in the buffers and are simply overwritten by the next
// ExtendContext, exactly as the target's positional resident cache handles its own rollback.
func (d *residentDrafter) TruncateContext(n int) {
	if n < 0 {
		n = 0
	}
	if n < d.ctxLen {
		d.ctxLen = n
	}
}

// AttachBlockDrafter satisfies decoder.ResidentDrafterHost. The concrete AttachDrafter returns
// *residentDrafter; this returns it through the interface so `decoder` can drive the loop
// without importing this package.
func (r *cudaResident) AttachBlockDrafter(w decoder.BlockDrafterWeights) (decoder.ResidentBlockDrafter, error) {
	return r.AttachDrafter(w)
}

// Compile-time proof that the CUDA resident is a block-drafting host and its drafter satisfies
// the trunk interface. A backend that grows one of these and not the other fails here.
var (
	_ decoder.ResidentDrafterHost  = (*cudaResident)(nil)
	_ decoder.ResidentBlockDrafter = (*residentDrafter)(nil)
)

// attnScale is the drafter's own 1/sqrt(head_dim), matching decoder/dflash.go's CPU reference.
// Not the target's r.attnScale — see N-11 at the call site.
func (d *residentDrafter) attnScale() float32 {
	return float32(1 / math.Sqrt(float64(d.geo.HeadDim)))
}
