package cpuengine

import (
	"fmt"
	"time"

	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/inkling2"
	"github.com/townsendmerino/goinfer/internal/runtime/normact"
	"github.com/townsendmerino/goinfer/internal/runtime/rope"
)

// WalkToken runs prog's layers for one position over h ([hidden], the residual stream, updated in place) and returns h. Each
// operation reads its parameters from prog; the walker supplies the stream's buffers, the cache, and the hooks that are not
// operations (the layer pager, LoRA, the captures, activeHooks().DecodeTiming).
//
// pleID is the token id behind h for the embedding stage's PLE inputs (Gemma 4); an image row passes the pad id.
func (m *Engine) WalkToken(prog *layerProgram, h []float32, pleID int, cache *Cache) ([]float32, error) {
	arch := m.W.Arch
	scr := cache.Scr
	if m.LayerPager != nil {
		defer m.LayerPager.FinishLayers()
	}
	pos := cache.Pos() // this position, stable across layers in one forward
	var t attnTimer
	ink := inkling2.NewToken(arch) // Inkling 2's per-token attention-residual blocks; nil on every other family
	var g4 *gemma4Token            // Gemma 4's per-token state: its PLE inputs are computed from the embedding, before layer 0
	if arch.Gemma4 != nil {
		g4 = m.newGemma4Token(prog, h, pleID, pos, &scr.G4)
		if activeHooks().G4TraceHidden != nil {
			activeHooks().G4TraceHidden(-1, h)
		}
	}
	for l, ops := range prog.Layers {
		if activeHooks().SSMStopEnabled && l > activeHooks().SSMStopLayer {
			break // the resident-SSM bring-up's layer sweep (SetSSMStopLayerForTest)
		}
		if cache.LayerInputs != nil && cache.LayerInputs[l] != nil {
			copy(h, cache.LayerInputs[l])
		}
		if m.LayerPager != nil {
			m.LayerPager.EnterLayer(l)
		}
		lw := &m.W.Layers[l]
		var ld *LoRALayerDelta
		if len(cache.LoRALayers) > l {
			ld = &cache.LoRALayers[l]
		}
		nH, hd := arch.HeadsAt(l), arch.HeadDim
		q, k, v := scr.q[:nH*hd], scr.k, scr.v
		ctx := scr.ctx[:nH*hd]
		lastFFN := false // which sublayer the next single residual add belongs to, for the sublayer capture
		var planted []float32
		for i := range ops {
			if activeHooks().SeamPlant == PlantCaptureEarly && i == len(ops)-1 {
				m.captureLayer(cache, l, h)
			}
			o := &ops[i]
			if g4 == nil && needsGemma4State(o) {
				return nil, errNoGemma4State(l, o)
			}
			t0 := CostStart(cache)
			switch o.Kind {
			case opNorm:
				src, dst := m.reg(o.In, h, scr), m.reg(o.Out, h, scr)
				if activeHooks().SeamPlant == PlantSubPrePostNorm && cache.SubCapture && o.Norm.Slot == slotPostAttn {
					planted = append(planted[:0], src...)
				}
				w, b := lw.NormWeights(o.Norm)
				ApplyNorm(o.Norm, dst, src, w, b, arch.HiddenDim)
			case opProjQKV:
				t.start()
				projQKVToken(m.Be, scr, lw, m.reg(o.In, h, scr), q, k, v, ld, o.Proj)
				t.qkv()
			case opQKNorm:
				applyQKNorm(o.QKNorm, q, k, lw.QNorm, lw.KNorm, nH, arch.NumKVHeads, hd)
			case opRope:
				applyRope(&o.Rope, q, k, nH, arch.NumKVHeads, hd, pos, cache.MRopePos, cache.MRopeDelta)
			case opAttnTemp:
				scale := o.Temp.Scale(pos)
				for j := range q {
					q[j] *= scale
				}
			case opAttention:
				switch o.Attn.Order {
				case attnSinksF64:
					AttendSinksToken(o.Attn, q, k, v, ctx, cache, arch, lw, l, pos)
				case attnQuery:
					cache.Append(l, k, v)
					AttendQueryP(o.Attn, q, ctx, scr.ScoresBuf(cache.StoredRows(l, arch.NumKVHeads*hd)), cache, l, pos, arch)
				default:
					ap := o.Attn
					ap.Sinks = lw.AttnSinks // gpt-oss's learned sinks; nil for every other family
					if err := attendToken(ap, q, k, v, ctx, cache, scr, arch, l, pos); err != nil {
						return nil, err
					}
				}
			case opOutGate:
				gates := scr.gateBuf(lw.GProj.Rows())
				MatmulInto(scr.WS, m.Be, &lw.GProj, m.reg(o.In, h, scr), gates, 1)
				if o.Gate.Sigmoid {
					applySigmoidGateRow(gates, ctx, lw.GProj.Rows() == nH, nH, hd)
				} else {
					applyGateRow(gates, ctx, lw.GProj.Rows() == nH, nH, hd)
				}
			case opProjO:
				out := m.reg(o.Out, h, scr)
				t.oStart()
				MatmulInto(scr.WS, m.Be, &lw.OProj, ctx, out, 1)
				t.o()
				if ld != nil {
					ApplyLoRA(ld.O, ctx, out, scr)
				}
				if o.Proj.Bias {
					AddBias(out, lw.OBias)
				}
				t.attn()
				if cache.SubCapture { // ctx is the pre-o-proj context
					cache.SubCtx[l] = append(cache.SubCtx[l][:0], ctx...)
				}
				lastFFN = false
			case opFFN:
				var t0 time.Time
				if activeHooks().DecodeTiming {
					t0 = time.Now()
				}
				out := m.reg(o.Out, h, scr)
				if o.FFN.Form == ffnGemma4Dense {
					if err := m.gemma4DenseFFN(o.FFN.Act, m.reg(o.In, h, scr), out, lw, arch.FFNAt(l), g4); err != nil {
						return nil, err
					}
				} else if err := m.ffnToken(&o.FFN, m.reg(o.In, h, scr), out, lw, scr, ld); err != nil {
					return nil, err
				}
				if activeHooks().DecodeTiming {
					activeHooks().DecodeTimingStats.add(func() *int64 {
						if st := activeHooks().DecodeTimingStats; st != nil {
							return st.MLP
						}
						return nil
					}(), int64(time.Since(t0)))
				}
				if cache.SubCapture { // before any post-MLP norm
					cache.SubMLPPre[l] = append(cache.SubMLPPre[l][:0], out...)
				}
				lastFFN = true
			case opResidual:
				residualToken(o, h, scr, cache, l, lastFFN, planted)
			case opGemma4Attn:
				m.gemma4AttnToken(&o.G4, m.reg(o.In, h, scr), m.reg(o.Out, h, scr), lw, cache, l, pos, g4)
				lastFFN = false
			case opGemma4MoE:
				copy(h, gemma4MoEFFNWith(m.Be, arch.HiddenDim, o.Norms.Eps, o.Norms.AddOne, h, lw.Gemma4MoE, m.Pager, &g4.moe))
			case opPLE:
				m.gemma4PLEBranch(h, l, g4, lw, &o.Norms, scr.sub)
				if cache.SubPLE != nil && !(activeHooks().SeamPlant == PlantSkipPLECapture && l == 1) { // the contribution just added
					cache.SubPLE[l] = append(cache.SubPLE[l][:0], scr.sub...)
				}
			case opLayerScale:
				if lw.LayerScalar != 0 {
					for i := range h {
						h[i] *= lw.LayerScalar
					}
				}
			case opMLA:
				copy(m.reg(o.Out, h, scr), m.mlaAttentionWith(&o.MLA, m.reg(o.In, h, scr), lw, arch, cache, l, pos))
				lastFFN = false
			case opDeltaNet:
				dp := *arch.Qwen35
				dp.ONormEps, dp.NegEigval = o.Delta.ONormEps, o.Delta.NegEigval
				copy(m.reg(o.Out, h, scr), GatedDeltaNetStep(m.Be, m.reg(o.In, h, scr), lw.Delta, dp, arch.HiddenDim, o.Delta.Eps, cache.Delta[l]))
				lastFFN = false
			case opGatedAttn:
				if o.Gated.Attn.Order != attnQuery {
					return nil, fmt.Errorf("decoder: layer %d: gated attention runs AttendQuery only, not order %d", l, o.Gated.Attn.Order)
				}
				copy(m.reg(o.Out, h, scr), m.qwen35AttentionWith(&o.Gated, m.reg(o.In, h, scr), lw, arch, cache, l, pos))
				lastFFN = false
			case opMamba2:
				mp := arch.Mamba2Shapes(o.Mamba.NormGroups)
				copy(m.reg(o.Out, h, scr), mamba2Step(m.reg(o.In, h, scr), lw.Mamba, mp, o.Mamba.Eps, cache.Mamba[l]))
				lastFFN = false
			case opShortConv:
				copy(m.reg(o.Out, h, scr), shortConvStep(m.reg(o.In, h, scr), lw.ShortConv, *arch.LFM2, arch.HiddenDim, cache.Conv[l]))
				lastFFN = false
			case opKDA:
				kp := *arch.KDA
				kp.LowerBound = o.KDA.LowerBound
				copy(m.reg(o.Out, h, scr), kdaMixerStep(m.Be, m.reg(o.In, h, scr), lw.KDA, kp, arch.HiddenDim, o.KDA.Eps, cache.KDA[l]))
				lastFFN = false
			case opInkling2:
				if ink == nil {
					return nil, fmt.Errorf("decoder: layer %d: an Inkling 2 visit on a model without Inkling 2 geometry", l)
				}
				inkling2.Visit(inkling2Host{m.Be, cache}, arch, lw, o, cache.Ink[l], ink, l, pos, h)
			default:
				return nil, fmt.Errorf("decoder: layer %d: operation %v is not implemented on the CPU's per-token walker", l, o.Kind)
			}
			if cache.Cost != nil {
				m.CountOp(cache, o, lw, l, t0)
			}
		}
		if activeHooks().G4TraceHidden != nil && arch.Gemma4 != nil {
			activeHooks().G4TraceHidden(l, h)
		}
		if activeHooks().SeamPlant != PlantCaptureEarly {
			m.captureLayer(cache, l, h)
		}
	}
	inkling2.Final(arch, ink, h)
	if cache.ManualPos {
		cache.Advance() // a layer that stores no K/V row (a latent, a recurrent state) does not step the position
	}
	return h, nil
}

// captureLayer is the hidden-state seam the block drafters read: the residual AFTER layer l is complete (HF's
// output_hidden_states shifted by one: its index 0 is the embedding, ours is layer 0's output), copied, since later layers
// mutate h.
func (m *Engine) captureLayer(cache *Cache, l int, h []float32) {
	for i, cl := range cache.CaptureLayers {
		if cl == l {
			cache.Captured[i] = append(cache.Captured[i][:0], h...)
		}
	}
}

// FinalNorm applies the head stage's norm to one row in place: the norm logitsFromHidden applies before the head, for the seams
// that stop before it (the hidden states).
func (m *Engine) FinalNorm(x []float32) {
	if activeHooks().SeamPlant == PlantHiddenNoNorm {
		return
	}
	hp := &m.Program().Head
	b := m.W.FinalNormBias
	if !hp.Norm.Bias {
		b = nil
	}
	ApplyNorm(hp.Norm, x, x, m.W.FinalNorm, b, m.W.Arch.HiddenDim)
}

// reg resolves an operand to the stream's buffer.
func (m *Engine) reg(r opReg, h []float32, scr *DecodeScratch) []float32 {
	switch r {
	case regX:
		return scr.norm
	case regA:
		return scr.sub
	case regB:
		return scr.sub2
	}
	return h
}

// ApplyNorm computes one row's normalisation from src into dst (they may be the same buffer).
func ApplyNorm(p normParams, dst, src, w, b []float32, dim int) {
	switch p.Form {
	case normCopy:
		copy(dst, src)
	case normLayer:
		normact.LayerInto(dst, src, w, b, 1, dim, p.Eps)
	default:
		normact.RMSInto(dst, src, w, 1, dim, p.Eps, p.AddOne)
	}
}

// projQKVToken projects x to q, k and v for one position, with LoRA's deltas and the bias.
func projQKVToken(be Backend, scr *DecodeScratch, lw *darch.LayerWeights, x, q, k, v []float32, ld *LoRALayerDelta, p projParams) {
	if IsW8A8(&lw.QProj) && IsW8A8(&lw.KProj) && IsW8A8(&lw.VProj) {
		// One batched dispatch: the activation is quantized once and the weights are read in place.
		scr.qkvOps[0] = linalg.BatchOp{W: &lw.QProj, Dst: q}
		scr.qkvOps[1] = linalg.BatchOp{W: &lw.KProj, Dst: k}
		scr.qkvOps[2] = linalg.BatchOp{W: &lw.VProj, Dst: v}
		MatmulW8A8Batch(be, scr.WS, x, 1, lw.QProj.Cols(), scr.qkvOps[:], lw.QProj.ActQuantGroup())
	} else if activeHooks().W4A8BatchEnabled && IsW4A8(&lw.QProj) && IsW4A8(&lw.KProj) && IsW4A8(&lw.VProj) {
		var group int
		scr.qkvOpsW4[0], group = WMW4A8Op(&lw.QProj, q)
		scr.qkvOpsW4[1], _ = WMW4A8Op(&lw.KProj, k)
		scr.qkvOpsW4[2], _ = WMW4A8Op(&lw.VProj, v)
		scr.WS.SetThreshold(Int4ParThreshold)
		MatmulW4A8Batch(be, scr.WS, x, 1, lw.QProj.Cols(), group, scr.qkvOpsW4[:], lw.QProj.ActQuantGroup())
	} else {
		scr.hq.Prepare(be, &lw.QProj, x, 1) // x is quantized once for the three
		MatmulIntoPre(scr.WS, be, &lw.QProj, &scr.hq, x, q, 1)
		MatmulIntoPre(scr.WS, be, &lw.KProj, &scr.hq, x, k, 1)
		MatmulIntoPre(scr.WS, be, &lw.VProj, &scr.hq, x, v, 1)
	}
	if ld != nil { // where a merge would have folded the delta into the weight
		ApplyLoRA(ld.Q, x, q, scr)
		ApplyLoRA(ld.K, x, k, scr)
		ApplyLoRA(ld.V, x, v, scr)
	}
	if p.Bias {
		AddBias(q, lw.QBias)
		AddBias(k, lw.KBias)
		AddBias(v, lw.VBias)
	}
}

// applyQKNorm normalises q and k: weighted RMSNorm per head, or over each whole vector when p.Whole; or Llama 4's
// parameter-free RMS per head.
func applyQKNorm(p qkNormParams, q, k, qw, kw []float32, nH, nKV, hd int) {
	if p.Form == qkL2 {
		l2NormHeads(q, nH, hd, p.Eps)
		l2NormHeads(k, nKV, hd, p.Eps)
		return
	}
	if p.Whole {
		normact.RMS(q, qw, 1, nH*hd, p.Eps, p.AddOne)
		normact.RMS(k, kw, 1, nKV*hd, p.Eps, p.AddOne)
		return
	}
	normact.RMS(q, qw, nH, hd, p.Eps, p.AddOne)
	normact.RMS(k, kw, nKV, hd, p.Eps, p.AddOne)
}

// applyRope rotates one position's q and k.
func applyRope(r *ropeParams, q, k []float32, nH, nKV, hd, pos int, mropePos [][3]int, mropeDelta int) {
	if r.Kernel == ropeMLAKernel {
		MLARope(q, nH, hd, pos, r.InvFreq, r.Mscale, r.Interleave)
		MLARope(k, nKV, hd, pos, r.InvFreq, r.Mscale, r.Interleave)
		return
	}
	rope.At(q, nH, hd, pos, r.InvFreq, r.Mscale, r.MRopeSection, mropePos, mropeDelta, r.Interleave, r.MRopeInterleaved)
	rope.At(k, nKV, hd, pos, r.InvFreq, r.Mscale, r.MRopeSection, mropePos, mropeDelta, r.Interleave, r.MRopeInterleaved)
}

// attendToken stores this position's k and v in layer l's cache and writes the attention context of q over the attended keys
// into ctx. The cache's layout picks the read (an f32 ring read in place, another ring through scratch, a widened global layer, an
// f32 global layer); p picks which keys are attended and how they are scored.
func attendToken(p attentionParams, q, k, v, ctx []float32, cache *Cache, scr *DecodeScratch, arch *darch.Architecture, l, pos int) error {
	if p.Order != attnBatchedAcc64 {
		return fmt.Errorf("decoder: layer %d: attention order %d is not implemented on the CPU's per-token walker", l, p.Order)
	}
	nH, hd := arch.HeadsAt(l), arch.HeadDim
	kvDim := arch.NumKVHeads * hd
	if activeHooks().KVRoundF16 { // the f16 exactness gate's reference: the f32 path with each stored row rounded to f16 by hand
		roundF16InPlace(k)
		roundF16InPlace(v)
	}
	const acc64 = true // decode is bit-identical to the batched forward only with float64 accumulation
	r := cache.Rings[l]
	switch {
	case r != nil && RingDirectDecode && r.Quant == KVQF32:
		// The slot written takes position pos-W, the one row just outside the window, so writing before reading is safe.
		r.Write(pos, k, v)
		base := max(pos-r.W+1, 0)
		nKeys := pos - base + 1
		wk, wv := r.Window(base, nKeys)
		pool := scr.HeadWorkerPool(nH, 1, nKeys, hd, false, acc64)
		AttendBatchedHeadsP(q, ctx, wk, wv, base, cache, l, pos, 1, p, arch, acc64, pool)
		if !cache.ManualPos && l == cache.NumLayers-1 {
			cache.PosN++ // the write does not step pos; Append's last-layer advance
		}
	case r != nil:
		rows := pos - cache.WindowStart(pos, false) + 1
		lk, lv := scr.LocalBufs(rows * kvDim)
		base, nKeys := cache.BatchReadLocal(l, pos, 1, k, v, lk, lv)
		pool := scr.HeadWorkerPool(nH, 1, nKeys, hd, false, acc64)
		AttendBatchedHeadsP(q, ctx, lk[:nKeys*kvDim], lv[:nKeys*kvDim], base, cache, l, pos, 1, p, arch, acc64, pool)
		cache.CommitBatch(l, pos, 1, k, v)
		if !cache.ManualPos && l == cache.NumLayers-1 {
			cache.PosN++ // commitBatch does not step pos; Append's last-layer advance
		}
	case cache.Quant == KVQF16:
		// The f16 cache's rows, read in place by the f16 acc64 kernels: no widening of the history into scratch.
		cache.Append(l, k, v)
		nKeys, rs := cache.StoredRows(l, kvDim), cache.RowStride(kvDim)
		pool := scr.HeadWorkerPool(nH, 1, nKeys, hd, false, acc64)
		AttendBatchedHeadsKV(q, ctx, nil, nil, cache.KeysH[l][:nKeys*rs], cache.ValsH[l][:nKeys*rs], rs, 0, cache, l, pos, 1, p, arch, acc64, pool)
	case cache.Quant.widened():
		cache.Append(l, k, v)
		lk, lv := scr.LocalBufs(cache.StoredRows(l, kvDim) * kvDim)
		nKeys := cache.DequantGlobalLayer(l, kvDim, lk, lv)
		pool := scr.HeadWorkerPool(nH, 1, nKeys, hd, false, acc64)
		AttendBatchedHeadsP(q, ctx, lk[:nKeys*kvDim], lv[:nKeys*kvDim], 0, cache, l, pos, 1, p, arch, acc64, pool)
	default:
		cache.Append(l, k, v)
		nKeys := cache.StoredRows(l, kvDim)
		pool := scr.HeadWorkerPool(nH, 1, nKeys, hd, false, acc64)
		keys, vals, rs := cache.LayerRows(l, kvDim)
		AttendBatchedHeadsKV(q, ctx, keys, vals, nil, nil, rs, 0, cache, l, pos, 1, p, arch, acc64, pool)
	}
	return nil
}

// ffnToken writes the feed-forward output of x into out for one position.
func (m *Engine) ffnToken(p *ffnParams, x, out []float32, lw *darch.LayerWeights, scr *DecodeScratch, ld *LoRALayerDelta) error {
	arch := m.W.Arch
	switch p.Form {
	case ffnGemma4Dense:
		return ErrNotImplemented // the walker runs Gemma 4's dense FFN itself (it shares the token's quantized input)
	case ffnLlama4MoE:
		m.llama4MoEWith(x, out, lw, &p.MoE, arch.HiddenDim)
		return nil
	case ffnGptOss:
		g, err := m.GptOssMoEWith(x, lw, p, arch.HiddenDim, scr)
		if err != nil {
			return err
		}
		copy(out, g)
		return nil
	case ffnMoE:
		g, err := moeMLP(x, lw, &p.MoE, p.Act, arch.HiddenDim, m.Be, scr, m.Pager)
		if err != nil {
			return err
		}
		copy(out, g)
		return nil
	case ffnNemotronMoE:
		copy(out, m.nemotronMoEWith(x, lw, &p.MoE, arch.HiddenDim))
		return nil
	case ffnNonGated:
		var mid *[]float32
		if scr != nil {
			mid = &scr.ffnMid
		}
		return nonGatedMLPWith(x, out, lw, p.Act, lw.UpProj.Rows(), arch.HiddenDim, m.Be, mid)
	}
	return GatedMLPAct(x, out, lw, p.Act, m.Be, scr, ld)
}

// residualToken is opResidual on one position: the sublayer's contribution in scr.sub (and scr.sub2 for a parallel block) added
// into h, scaled when the op says so, after the sublayer capture records it. lastFFN says which sublayer a single add belongs to;
// planted is a test seam's stand-in for the attention contribution.
func residualToken(o *op, h []float32, scr *DecodeScratch, cache *Cache, l int, lastFFN bool, planted []float32) {
	if o.Res.Two {
		if cache.SubCapture {
			cache.SubAttn[l] = append(cache.SubAttn[l][:0], scr.sub...)
			cache.SubMLP[l] = append(cache.SubMLP[l][:0], scr.sub2...)
		}
		AddResidual2(h, scr.sub, scr.sub2)
		return
	}
	if cache.SubCapture { // the contribution about to reach the residual
		switch {
		case lastFFN:
			cache.SubMLP[l] = append(cache.SubMLP[l][:0], scr.sub...)
			if activeHooks().SeamPlant == PlantMLPPreAfterNorm && l == 1 {
				cache.SubMLPPre[l] = append(cache.SubMLPPre[l][:0], scr.sub...)
			}
		case planted != nil:
			cache.SubAttn[l] = append(cache.SubAttn[l][:0], planted...)
		default:
			cache.SubAttn[l] = append(cache.SubAttn[l][:0], scr.sub...)
		}
	}
	if o.Res.Scale != 0 {
		AddScaled(h, scr.sub, float32(o.Res.Scale))
	} else {
		AddResidual(h, scr.sub)
	}
}

// attnTimer accumulates activeHooks().DecodeTiming's attention counters across the operations of one attention sublayer.
type attnTimer struct{ t0, t1 time.Time }

func (t *attnTimer) start() {
	if activeHooks().DecodeTiming {
		t.t0 = time.Now()
	}
}

func (t *attnTimer) qkv() {
	if activeHooks().DecodeTiming {
		activeHooks().DecodeTimingStats.add(func() *int64 {
			if st := activeHooks().DecodeTimingStats; st != nil {
				return st.QKV
			}
			return nil
		}(), int64(time.Since(t.t0)))
	}
}

func (t *attnTimer) oStart() {
	if activeHooks().DecodeTiming {
		t.t1 = time.Now()
	}
}

func (t *attnTimer) o() {
	if activeHooks().DecodeTiming {
		activeHooks().DecodeTimingStats.add(func() *int64 {
			if st := activeHooks().DecodeTimingStats; st != nil {
				return st.O
			}
			return nil
		}(), int64(time.Since(t.t1)))
	}
}

func (t *attnTimer) attn() {
	if activeHooks().DecodeTiming {
		d := int64(time.Since(t.t0))
		activeHooks().DecodeTimingStats.add(func() *int64 {
			if st := activeHooks().DecodeTimingStats; st != nil {
				return st.AttnAll
			}
			return nil
		}(), d)
		activeHooks().DecodeTimingStats.add(func() *int64 {
			if st := activeHooks().DecodeTimingStats; st != nil {
				return st.Attn
			}
			return nil
		}(), d)
	}
}

// needsGemma4State reports whether o reads Gemma 4's per-token or per-pass state (PLE inputs, attention and FFN scratch).
func needsGemma4State(o *op) bool {
	switch o.Kind {
	case opGemma4Attn, opGemma4MoE, opPLE:
		return true
	case opFFN:
		return o.FFN.Form == ffnGemma4Dense
	}
	return false
}

// errNoGemma4State is a walker's refusal of an operation needsGemma4State names on a model without that state. The walkers
// return it rather than dereference the nil state, because a recovered access violation in a walker can corrupt the Go heap.
func errNoGemma4State(l int, o *op) error {
	if o.Kind == opFFN {
		return fmt.Errorf("decoder: layer %d: Gemma 4's dense FFN form on a model without Gemma 4 state", l)
	}
	return fmt.Errorf("decoder: layer %d: Gemma 4 operation %v on a model without Gemma 4 state", l, o.Kind)
}
