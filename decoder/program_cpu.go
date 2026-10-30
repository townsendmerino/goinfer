package decoder

import (
	"fmt"
	"math"
	"sync/atomic"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// seamPlantForTest plants one defect in a hidden-state seam, so the seams' gates (decoder/seams_test.go) can show they go red.
// Zero in production.
var seamPlantForTest seamPlant

type seamPlant uint8

const (
	plantNone            seamPlant = iota
	plantHiddenNoNorm              // HiddenLast's sequential path skips the head's norm (finalNorm)
	plantCaptureEarly              // ForwardCapture copies the residual before the layer's last operation
	plantSubPrePostNorm            // ForwardSubCapture records the attention contribution before its post-attention norm
	plantCostDoubleCount           // the cost counters record each attention output projection twice (SB11's plant)
)

// program returns m's program. A Model built without modelFromOptions (a test that assembles one) gets a fresh one per call.
func (m *Model) program() *layerProgram {
	if m.prog != nil {
		return m.prog
	}
	return buildProgram(m.w.arch, m.w.hasExperts)
}

// hasExperts reports whether layer l carries MoE experts (Gemma 4's 26B carries them in its own block).
func (w *Weights) hasExperts(l int) bool {
	return l < len(w.Layers) && (w.Layers[l].Experts != nil || w.Layers[l].gemma4moe != nil)
}

// walkToken runs prog's layers for one position over h ([hidden], the residual stream, updated in place) and returns h. Each
// operation reads its parameters from prog; the walker supplies the stream's buffers, the cache, and the hooks that are not
// operations (the layer pager, LoRA, the captures, decodeTiming).
//
// pleID is the token id behind h for the embedding stage's PLE inputs (Gemma 4); an image row passes the pad id.
func (m *Model) walkToken(prog *layerProgram, h []float32, pleID int, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	scr := cache.scr
	if m.layerPager != nil {
		defer m.layerPager.finishLayers()
	}
	pos := cache.Pos() // this position, stable across layers in one forward
	var t attnTimer
	var g4 *gemma4Token // Gemma 4's per-token state: its PLE inputs are computed from the embedding, before layer 0
	if arch.gemma4 != nil {
		g4 = m.newGemma4Token(prog, h, pleID, pos)
		if g4traceHidden != nil {
			g4traceHidden(-1, h)
		}
	}
	for l, ops := range prog.Layers {
		if ssmStopLayer >= 0 && l > ssmStopLayer {
			break // the resident-SSM bring-up's layer sweep (SetSSMStopLayerForTest)
		}
		if m.layerPager != nil {
			m.layerPager.enterLayer(l)
		}
		lw := &m.w.Layers[l]
		var ld *loraLayerDelta
		if cache.lora != nil {
			ld = &cache.lora.layers[l]
		}
		nH, hd := arch.headsAt(l), arch.HeadDim
		q, k, v := scr.q[:nH*hd], scr.k, scr.v
		ctx := scr.ctx[:nH*hd]
		lastFFN := false // which sublayer the next single residual add belongs to, for the sublayer capture
		var planted []float32
		for i := range ops {
			if seamPlantForTest == plantCaptureEarly && i == len(ops)-1 {
				m.captureLayer(cache, l, h)
			}
			o := &ops[i]
			t0 := costStart(cache)
			switch o.Kind {
			case opNorm:
				src, dst := m.reg(o.In, h, scr), m.reg(o.Out, h, scr)
				if seamPlantForTest == plantSubPrePostNorm && cache.subCapture && o.Norm.Slot == slotPostAttn {
					planted = append(planted[:0], src...)
				}
				w, b := lw.normWeights(o.Norm)
				applyNorm(o.Norm, dst, src, w, b, arch.HiddenDim)
			case opProjQKV:
				t.start()
				projQKVToken(m.be, scr, lw, m.reg(o.In, h, scr), q, k, v, ld, o.Proj)
				t.qkv()
			case opQKNorm:
				applyQKNorm(o.QKNorm, q, k, lw.QNorm, lw.KNorm, nH, arch.NumKVHeads, hd)
			case opRope:
				applyRope(&o.Rope, q, k, nH, arch.NumKVHeads, hd, pos, cache.mropePos, cache.mropeDelta)
			case opAttnTemp:
				scale := o.Temp.Scale(pos)
				for j := range q {
					q[j] *= scale
				}
			case opAttention:
				switch o.Attn.Order {
				case attnSinksF64:
					attendSinksToken(o.Attn, q, k, v, ctx, cache, arch, lw, l, pos)
				case attnQuery:
					cache.Append(l, k, v)
					attendQueryP(o.Attn, q, ctx, scr.scoresBuf(cache.storedRows(l, arch.NumKVHeads*hd)), cache, l, pos, arch)
				default:
					ap := o.Attn
					ap.Sinks = lw.AttnSinks // gpt-oss's learned sinks; nil for every other family
					if err := attendToken(ap, q, k, v, ctx, cache, scr, arch, l, pos); err != nil {
						return nil, err
					}
				}
			case opOutGate:
				gates := scr.gateBuf(lw.GProj.Rows())
				matmulInto(scr.ws, m.be, &lw.GProj, m.reg(o.In, h, scr), gates, 1)
				if o.Gate.Sigmoid {
					applySigmoidGateRow(gates, ctx, lw.GProj.Rows() == nH, nH, hd)
				} else {
					applyGateRow(gates, ctx, lw.GProj.Rows() == nH, nH, hd)
				}
			case opProjO:
				out := m.reg(o.Out, h, scr)
				t.oStart()
				matmulInto(scr.ws, m.be, &lw.OProj, ctx, out, 1)
				t.o()
				if ld != nil {
					applyLoRA(ld.o, ctx, out, scr)
				}
				if o.Proj.Bias {
					addBias(out, lw.OBias)
				}
				t.attn()
				if cache.subCapture { // ctx is the pre-o-proj context
					cache.subCtx[l] = append(cache.subCtx[l][:0], ctx...)
				}
				lastFFN = false
			case opFFN:
				var t0 time.Time
				if decodeTiming {
					t0 = time.Now()
				}
				out := m.reg(o.Out, h, scr)
				if o.FFN.Form == ffnGemma4Dense {
					// A model without Gemma 4's per-token state gets an error, never a nil dereference: a recovered access
					// violation here corrupted the Go heap on some Windows hosts (docs/queue-correctness.md G50).
					if g4 == nil {
						return nil, fmt.Errorf("decoder: layer %d: Gemma 4's dense FFN form on a model without Gemma 4 state", l)
					}
					if err := m.gemma4DenseFFN(o.FFN.Act, m.reg(o.In, h, scr), out, lw, arch.ffnAt(l), &g4.hq); err != nil {
						return nil, err
					}
				} else if err := m.ffnToken(&o.FFN, m.reg(o.In, h, scr), out, lw, scr, ld); err != nil {
					return nil, err
				}
				if decodeTiming {
					atomic.AddInt64(&dtMLP, int64(time.Since(t0)))
				}
				if cache.subCapture { // before any post-MLP norm
					cache.subMLPpre[l] = append(cache.subMLPpre[l][:0], out...)
				}
				lastFFN = true
			case opResidual:
				if o.Res.Two {
					if cache.subCapture {
						cache.subAttn[l] = append(cache.subAttn[l][:0], scr.sub...)
						cache.subMLP[l] = append(cache.subMLP[l][:0], scr.sub2...)
					}
					addResidual2(h, scr.sub, scr.sub2)
					break
				}
				if cache.subCapture { // the contribution about to reach the residual
					switch {
					case lastFFN:
						cache.subMLP[l] = append(cache.subMLP[l][:0], scr.sub...)
					case planted != nil:
						cache.subAttn[l] = append(cache.subAttn[l][:0], planted...)
					default:
						cache.subAttn[l] = append(cache.subAttn[l][:0], scr.sub...)
					}
				}
				if o.Res.Scale != 0 {
					addScaled(h, scr.sub, float32(o.Res.Scale))
				} else {
					addResidual(h, scr.sub)
				}
			case opGemma4Attn:
				m.gemma4AttnToken(&o.G4, m.reg(o.In, h, scr), m.reg(o.Out, h, scr), lw, cache, l, pos, g4)
				lastFFN = false
			case opGemma4MoE:
				copy(h, gemma4MoEFFNWith(m.be, arch.HiddenDim, o.Norms.Eps, o.Norms.AddOne, h, lw.gemma4moe, m.pager))
			case opPLE:
				m.gemma4PLEBranch(h, g4.ple(l), lw, &o.Norms, scr.sub)
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
				dp := *arch.qwen35
				dp.ONormEps, dp.NegEigval = o.Delta.ONormEps, o.Delta.NegEigval
				copy(m.reg(o.Out, h, scr), gatedDeltaNetStep(m.be, m.reg(o.In, h, scr), lw.delta, dp, arch.HiddenDim, o.Delta.Eps, cache.delta[l]))
				lastFFN = false
			case opGatedAttn:
				if o.Gated.Attn.Order != attnQuery {
					return nil, fmt.Errorf("decoder: layer %d: gated attention runs attendQuery only, not order %d", l, o.Gated.Attn.Order)
				}
				copy(m.reg(o.Out, h, scr), m.qwen35AttentionWith(&o.Gated, m.reg(o.In, h, scr), lw, arch, cache, l, pos))
				lastFFN = false
			case opMamba2:
				mp := arch.mamba2Shapes(o.Mamba.NormGroups)
				copy(m.reg(o.Out, h, scr), mamba2Step(m.reg(o.In, h, scr), lw.mamba, mp, o.Mamba.Eps, cache.mamba[l]))
				lastFFN = false
			case opShortConv:
				copy(m.reg(o.Out, h, scr), shortConvStep(m.reg(o.In, h, scr), lw.shortConv, *arch.lfm2, arch.HiddenDim, cache.conv[l]))
				lastFFN = false
			case opKDA:
				kp := *arch.kda
				kp.LowerBound = o.KDA.LowerBound
				copy(m.reg(o.Out, h, scr), kdaMixerStep(m.be, m.reg(o.In, h, scr), lw.kda, kp, arch.HiddenDim, o.KDA.Eps, cache.kda[l]))
				lastFFN = false
			default:
				return nil, fmt.Errorf("decoder: layer %d: operation %v is not implemented on the CPU's per-token walker", l, o.Kind)
			}
			if cache.cost != nil {
				m.countOp(cache, o, lw, l, t0)
			}
		}
		if g4traceHidden != nil && arch.gemma4 != nil {
			g4traceHidden(l, h)
		}
		if seamPlantForTest != plantCaptureEarly {
			m.captureLayer(cache, l, h)
		}
	}
	if cache.manualPos {
		cache.Advance() // a layer that stores no K/V row (a latent, a recurrent state) does not step the position
	}
	return h, nil
}

// captureLayer is the hidden-state seam the block drafters read: the residual AFTER layer l is complete (HF's
// output_hidden_states shifted by one: its index 0 is the embedding, ours is layer 0's output), copied, since later layers
// mutate h.
func (m *Model) captureLayer(cache *KVCache, l int, h []float32) {
	for i, cl := range cache.captureLayers {
		if cl == l {
			cache.captured[i] = append(cache.captured[i][:0], h...)
		}
	}
}

// finalNorm applies the head stage's norm to one row in place: the norm logitsFromHidden applies before the head, for the seams
// that stop before it (the hidden states).
func (m *Model) finalNorm(x []float32) {
	if seamPlantForTest == plantHiddenNoNorm {
		return
	}
	hp := &m.program().Head
	b := m.w.FinalNormBias
	if !hp.Norm.Bias {
		b = nil
	}
	applyNorm(hp.Norm, x, x, m.w.FinalNorm, b, m.w.arch.HiddenDim)
}

// reg resolves an operand to the stream's buffer.
func (m *Model) reg(r opReg, h []float32, scr *decodeScratch) []float32 {
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

// normWeights returns the weight and bias an opNorm applies: the bias only when the operation says so.
func (lw *LayerWeights) normWeights(p normParams) (w, b []float32) {
	switch p.Slot {
	case slotPreAttn:
		w, b = lw.PreAttnNorm, lw.PreAttnNormBias
	case slotPreMLP:
		w, b = lw.PreMLPNorm, lw.PreMLPNormBias
	case slotPostAttn:
		w = lw.PostAttnNorm
	case slotPostMLP:
		w = lw.PostMLPNorm
	}
	if !p.Bias {
		b = nil
	}
	return w, b
}

// applyNorm computes one row's normalisation from src into dst (they may be the same buffer).
func applyNorm(p normParams, dst, src, w, b []float32, dim int) {
	switch p.Form {
	case normCopy:
		copy(dst, src)
	case normLayer:
		layerNormInto(dst, src, w, b, 1, dim, p.Eps)
	default:
		rmsNormInto(dst, src, w, 1, dim, p.Eps, p.AddOne)
	}
}

// projQKVToken projects x to q, k and v for one position, with LoRA's deltas and the bias.
func projQKVToken(be Backend, scr *decodeScratch, lw *LayerWeights, x, q, k, v []float32, ld *loraLayerDelta, p projParams) {
	if isW8A8(&lw.QProj) && isW8A8(&lw.KProj) && isW8A8(&lw.VProj) {
		// One batched dispatch: the activation is quantized once and the weights are read in place.
		scr.qkvOps[0] = linalg.W8A8Op{BQ: wmInt8(&lw.QProj), Scales: wmScales(&lw.QProj), Dst: q, N: lw.QProj.Rows()}
		scr.qkvOps[1] = linalg.W8A8Op{BQ: wmInt8(&lw.KProj), Scales: wmScales(&lw.KProj), Dst: k, N: lw.KProj.Rows()}
		scr.qkvOps[2] = linalg.W8A8Op{BQ: wmInt8(&lw.VProj), Scales: wmScales(&lw.VProj), Dst: v, N: lw.VProj.Rows()}
		matmulW8A8Batch(be, scr.ws, x, 1, lw.QProj.Cols(), scr.qkvOps[:], lw.QProj.ActQuantGroup())
	} else if w4a8BatchEnabled && isW4A8(&lw.QProj) && isW4A8(&lw.KProj) && isW4A8(&lw.VProj) {
		var group int
		scr.qkvOpsW4[0], group = wmW4A8Op(&lw.QProj, q)
		scr.qkvOpsW4[1], _ = wmW4A8Op(&lw.KProj, k)
		scr.qkvOpsW4[2], _ = wmW4A8Op(&lw.VProj, v)
		scr.ws.SetThreshold(int4ParThreshold)
		matmulW4A8Batch(be, scr.ws, x, 1, lw.QProj.Cols(), group, scr.qkvOpsW4[:], lw.QProj.ActQuantGroup())
	} else {
		scr.hq.prepare(be, &lw.QProj, x, 1) // x is quantized once for the three
		matmulIntoPre(scr.ws, be, &lw.QProj, &scr.hq, x, q, 1)
		matmulIntoPre(scr.ws, be, &lw.KProj, &scr.hq, x, k, 1)
		matmulIntoPre(scr.ws, be, &lw.VProj, &scr.hq, x, v, 1)
	}
	if ld != nil { // where a merge would have folded the delta into the weight
		applyLoRA(ld.q, x, q, scr)
		applyLoRA(ld.k, x, k, scr)
		applyLoRA(ld.v, x, v, scr)
	}
	if p.Bias {
		addBias(q, lw.QBias)
		addBias(k, lw.KBias)
		addBias(v, lw.VBias)
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
		rmsNorm(q, qw, 1, nH*hd, p.Eps, p.AddOne)
		rmsNorm(k, kw, 1, nKV*hd, p.Eps, p.AddOne)
		return
	}
	rmsNorm(q, qw, nH, hd, p.Eps, p.AddOne)
	rmsNorm(k, kw, nKV, hd, p.Eps, p.AddOne)
}

// applyRope rotates one position's q and k.
func applyRope(r *ropeParams, q, k []float32, nH, nKV, hd, pos int, mropePos [][3]int, mropeDelta int) {
	if r.Kernel == ropeMLAKernel {
		mlaRope(q, nH, hd, pos, r.InvFreq, r.Mscale, r.Interleave)
		mlaRope(k, nKV, hd, pos, r.InvFreq, r.Mscale, r.Interleave)
		return
	}
	ropeAt(q, nH, hd, pos, r.InvFreq, r.Mscale, r.MRopeSection, mropePos, mropeDelta, r.Interleave, r.MRopeInterleaved)
	ropeAt(k, nKV, hd, pos, r.InvFreq, r.Mscale, r.MRopeSection, mropePos, mropeDelta, r.Interleave, r.MRopeInterleaved)
}

// Scale is the query's temperature at pos.
func (p *attnTempParams) Scale(pos int) float32 {
	return float32(1 + p.Beta*math.Log1p(math.Floor(float64(pos+p.PosOffset)/p.OrigMaxPos)))
}

// attendToken stores this position's k and v in layer l's cache and writes the attention context of q over the attended keys
// into ctx. The cache's layout picks the read (an f32 ring read in place, another ring through scratch, a widened global layer, an
// f32 global layer); p picks which keys are attended and how they are scored.
func attendToken(p attentionParams, q, k, v, ctx []float32, cache *KVCache, scr *decodeScratch, arch *Architecture, l, pos int) error {
	if p.Order != attnBatchedAcc64 {
		return fmt.Errorf("decoder: layer %d: attention order %d is not implemented on the CPU's per-token walker", l, p.Order)
	}
	nH, hd := arch.headsAt(l), arch.HeadDim
	kvDim := arch.NumKVHeads * hd
	if kvRoundF16ForTest { // F1's reference: the f32 path with each stored row rounded to f16 by hand
		roundF16InPlace(k)
		roundF16InPlace(v)
	}
	const acc64 = true // decode is bit-identical to the batched forward only with float64 accumulation
	r := cache.rings[l]
	switch {
	case r != nil && ringDirectDecode && r.quant == kvqF32:
		// The slot written takes position pos-W, the one row just outside the window, so writing before reading is safe.
		r.write(pos, k, v)
		base := max(pos-r.w+1, 0)
		nKeys := pos - base + 1
		wk, wv := r.window(base, nKeys)
		pool := scr.headWorkerPool(nH, 1, nKeys, hd, false, acc64)
		attendBatchedHeadsP(q, ctx, wk, wv, base, cache, l, pos, 1, p, arch, acc64, pool)
		if !cache.manualPos && l == cache.numLayers-1 {
			cache.pos++ // the write does not step pos; Append's last-layer advance
		}
	case r != nil:
		rows := pos - cache.WindowStart(pos, false) + 1
		lk, lv := scr.localBufs(rows * kvDim)
		base, nKeys := cache.batchReadLocal(l, pos, 1, k, v, lk, lv)
		pool := scr.headWorkerPool(nH, 1, nKeys, hd, false, acc64)
		attendBatchedHeadsP(q, ctx, lk[:nKeys*kvDim], lv[:nKeys*kvDim], base, cache, l, pos, 1, p, arch, acc64, pool)
		cache.commitBatch(l, pos, 1, k, v)
		if !cache.manualPos && l == cache.numLayers-1 {
			cache.pos++ // commitBatch does not step pos; Append's last-layer advance
		}
	case cache.quant.widened():
		cache.Append(l, k, v)
		lk, lv := scr.localBufs(cache.storedRows(l, kvDim) * kvDim)
		nKeys := cache.dequantGlobalLayer(l, kvDim, lk, lv)
		pool := scr.headWorkerPool(nH, 1, nKeys, hd, false, acc64)
		attendBatchedHeadsP(q, ctx, lk[:nKeys*kvDim], lv[:nKeys*kvDim], 0, cache, l, pos, 1, p, arch, acc64, pool)
	default:
		cache.Append(l, k, v)
		nKeys := cache.storedRows(l, kvDim)
		pool := scr.headWorkerPool(nH, 1, nKeys, hd, false, acc64)
		attendBatchedHeadsP(q, ctx, cache.Keys(l), cache.Vals(l), 0, cache, l, pos, 1, p, arch, acc64, pool)
	}
	return nil
}

// ffnToken writes the feed-forward output of x into out for one position.
func (m *Model) ffnToken(p *ffnParams, x, out []float32, lw *LayerWeights, scr *decodeScratch, ld *loraLayerDelta) error {
	arch := m.w.arch
	switch p.Form {
	case ffnGemma4Dense:
		return errNotImplemented // the walker runs Gemma 4's dense FFN itself (it shares the token's quantized input)
	case ffnLlama4MoE:
		m.llama4MoEWith(x, out, lw, &p.MoE, arch.HiddenDim)
		return nil
	case ffnGptOss:
		g, err := m.gptOssMoEWith(x, lw, p, arch.HiddenDim)
		if err != nil {
			return err
		}
		copy(out, g)
		return nil
	case ffnMoE:
		g, err := moeMLP(x, lw, &p.MoE, p.Act, arch.HiddenDim, m.be, scr, m.pager)
		if err != nil {
			return err
		}
		copy(out, g)
		return nil
	case ffnNemotronMoE:
		copy(out, m.nemotronMoEWith(x, lw, &p.MoE, arch.HiddenDim))
		return nil
	case ffnNonGated:
		g, err := nonGatedMLPWith(x, lw, p.Act, lw.UpProj.Rows(), arch.HiddenDim, m.be)
		if err != nil {
			return err
		}
		copy(out, g)
		return nil
	}
	return gatedMLPAct(x, out, lw, p.Act, m.be, scr, ld)
}

// attnTimer accumulates decodeTiming's attention counters across the operations of one attention sublayer.
type attnTimer struct{ t0, t1 time.Time }

func (t *attnTimer) start() {
	if decodeTiming {
		t.t0 = time.Now()
	}
}

func (t *attnTimer) qkv() {
	if decodeTiming {
		atomic.AddInt64(&dtQKV, int64(time.Since(t.t0)))
	}
}

func (t *attnTimer) oStart() {
	if decodeTiming {
		t.t1 = time.Now()
	}
}

func (t *attnTimer) o() {
	if decodeTiming {
		atomic.AddInt64(&dtO, int64(time.Since(t.t1)))
	}
}

func (t *attnTimer) attn() {
	if decodeTiming {
		d := int64(time.Since(t.t0))
		atomic.AddInt64(&dtAttnAll, d)
		atomic.AddInt64(&dtAttn, d)
	}
}
