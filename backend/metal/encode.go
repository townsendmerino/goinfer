//go:build darwin

package metal

import (
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// w4f16Lane holds the W4F16 decode lane's pipelines, enable flag, and half-typed activation scratch buffers.
type w4f16Lane struct {
	pRmsF16                         Pipeline // rmsnorm_f16_act: f32 residual in, half activation out, no quant
	pF32ToF16                       Pipeline // bare convert, for o-proj's input
	pSAf16, pSAf16Bias, pSAf16Resid Pipeline
	decodeLaneW4F16                 bool
	axF16, mxF16, cxF16             Buffer // half-typed activation buffers (QKV-in, gate/up-in, o-proj-in)
}

// q4kState holds the q4k decode lane's enable flag, pipelines, simdgroup row counts, and f32 activation buffers.
type q4kState struct {
	q4kLane                                                    bool
	pQ4K, pQ4KResid, pQ4KBias, pW8F32, pW8F32Resid, pW8F32Bias Pipeline
	pQ5K, pQ6K                                                 [3]Pipeline // native Q5_K and Q6_K GEMVs: plain, residual, bias
	q4kR, w8F32R                                               int
	pRmsF32Out, pSwF32                                         Pipeline
	axF32, swF32                                               Buffer
	pQ4KRows, pW8F32Rows                                       [batchMaxSeqs + 1]Pipeline
	pQ5KRows, pQ6KRows                                         [batchMaxSeqs + 1]Pipeline
	pRmsF32OutRows, pSwF32Rows                                 Pipeline
}

// attnFAState holds the split-KV attention_fa decode lane's pipelines, flags, partial scratch, uniforms, and test overrides.
type attnFAState struct {
	pAttnFA, pAttnFACombine         Pipeline
	pAttnFARows, pAttnFACombineRows Pipeline
	attnFAPartialLen                int
	decodeAttnFA                    bool
	faHD96                          bool
	faHD96Default                   bool
	attnFAPartial                   Buffer // [nKV][maxSplit][G][hd+2] f32 scratch
	attnFAMaxSplit                  int
	attnFASplitOverride             int
	attnFAFloorOverride             int
	attnFABlkSplit                  int
	attnFALayer                     int
	attnFANKV                       int
	uAttnFAG, uAttnFANSplit         Buffer
	encFANSplit                     Buffer
}

// gemvRowsState holds the rows-per-simdgroup int4 GEMV pipelines, configuration, and dispatch counters.
type gemvRowsState struct {
	gemvRows                                             struct{ qkv, o, gu, down int }
	pSABiasRows, pSAResidRows, pSARows, pGemvResidStaged Pipeline
	pSABiasRowsAdj, pSARowsAdj                           Pipeline
	gemvExtCoal, gemvExtSA, gemvExtResid                 [5]Pipeline
	gemvExtTGMax                                         int
	gemvExtRows                                          [3]int
}

// encodeTrunkWith encodes all decoder layers + the final norm into e using the specified
// uPos, uNKeys, and uQTempScale uniform buffers.
func (r *resident) encodeTrunkWith(e *Encoder, uPos, uNKeys, uQTempScale Buffer, uRopePos ...Buffer) {
	rp := uPos
	if len(uRopePos) > 0 && uRopePos[0] != (Buffer{}) {
		rp = uRopePos[0]
	}
	for l := 0; l < r.nL; l++ {
		r.encodeLayerWith(e, l, uPos, uNKeys, uQTempScale, rp)
	}
	if r.q4kLane { // f32 into r.aq: the swapped head pipelines read it as float
		e.Dispatch(r.pRmsF32Out, tgReduceNorm, tgReduceNorm, r.x, r.finalNorm, r.aq, r.uH, r.uEps, r.uAddOne)
		return
	}
	r.encodeNorm(e, r.x, r.finalNorm, r.finalNormBias, r.aq, r.aSc)
}

// encodeTrunkInto encodes all decoder layers + the final norm into e, leaving the quantized
// final hidden state in r.aq/r.aSc ready for an lm head. It ONLY records dispatches (referencing
// the shared buffers) — it does NOT set uPos/uNKeys or fill r.x; the caller sets those before
// commit. This value-independence is what lets the executor pre-encode token t+1 while token t
// runs (encode-ahead) — with one exception, the attention plan (attnPlan: attention_fa on or off
// and its split grid depend on the key count), which is encoded for planNKeys and which the
// executor checks against each job before committing.
func (r *resident) encodeTrunkInto(e *Encoder) {
	r.encodeTrunkWith(e, r.uPos, r.uNKeys, r.uQTempScale, r.uRopePos)
}

// encodeNorm dispatches the family's pre-GEMV norm+quant into aq/aSc — layernorm_quant (GPT-2's
// mean-centered LayerNorm, with bias) when r.layerNorm, else the default rmsnorm_quant —
// mirroring decoder/model.go's normalize() dispatch on arch.Norm. Used at every norm site that
// feeds a GEMV (pre-attn, pre-MLP, final); the Gemma sandwich post-norms are a different
// contract (rmsnorm_f32, in-place on the residual, RMS-only families) and don't route through here.
func (r *resident) encodeNorm(e *Encoder, x, w, bias, aq, aSc Buffer) {
	if r.layerNorm {
		e.Dispatch(r.pLayerNorm, tgReduceNorm, tgReduceNorm, x, w, bias, aq, aSc, r.uH, r.uEps, r.uLNHasBias)
	} else {
		e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, x, w, aq, aSc, r.uH, r.uEps, r.uAddOne)
	}
}

// encodeLayer encodes one decoder layer (attention block + FFN block) into e using default uniform buffers.
func (r *resident) encodeLayer(e *Encoder, l int) {
	r.encodeLayerWith(e, l, r.uPos, r.uNKeys, r.uQTempScale, r.uRopePos)
}

// encodeLayerWith encodes one decoder layer (attention block + FFN block) into e with parameterized uniforms.
func (r *resident) encodeLayerWith(e *Encoder, l int, uPos, uNKeys, uQTempScale Buffer, uRopePos ...Buffer) {
	r.encodeLayerResidualWith(e, l, r.x, uPos, uNKeys, uQTempScale, uRopePos...)
}

// encodeLayerResidualWith encodes one decoder layer into e with parameterized residual buffer x and uniforms: the layer's steps
// (buildLayerSteps), in order, each walked from its program operations (resident.encodeMixerWalk, resident.encodeFFNWalk).
func (r *resident) encodeLayerResidualWith(e *Encoder, l int, x Buffer, uPos, uNKeys, uQTempScale Buffer, uRopePos ...Buffer) {
	rp := uPos
	if len(uRopePos) > 0 && uRopePos[0] != (Buffer{}) {
		rp = uRopePos[0]
	}
	L := &r.layers[l]
	r.encLayer = l
	for i, st := range L.steps {
		r.encodeLayerStep(e, st, i, l, L, x, uPos, uNKeys, uQTempScale, rp, stepState{})
	}
}

// stepState is the conv window and state a recurrent step (DeltaNet, Mamba-2) runs on in place of the layer's own pair: the
// multi-sequence step passes each row's slot. The zero value is the layer's own.
type stepState struct{ win, state Buffer }

// orPair is the state's pair, or the layer's own when the state is the zero value.
func (rec stepState) orPair(win, state Buffer) (Buffer, Buffer) {
	if rec.win == (Buffer{}) {
		return win, state
	}
	return rec.win, rec.state
}

// encodeLayerStep encodes step i of layer l, st, on residual x, from its operations through the walk (resident.encodeMixerWalk,
// resident.encodeFFNWalk). rec names a recurrent step's own state.
func (r *resident) encodeLayerStep(e *Encoder, st layerStep, i, l int, L *residLayer, x, uPos, uNKeys, uQTempScale, rp Buffer, rec stepState) {
	if mixerWalks(st) {
		r.encodeMixerWalk(e, l, L, L.stepOps[i], x, uPos, uNKeys, uQTempScale, rp, rec)
		return
	}
	r.encodeFFNWalk(e, st, l, L, L.stepOps[i], x)
}

// q4kGemvRows and w8F32GemvRows are the q4k lane's rows a simdgroup for its Q4_K and int8 GEMVs (gemv_q4k_f32_rows<R>,
// gemv_w8_f32_rows<R>).
const q4kGemvRows, w8F32GemvRows = 8, 2

// q5kGemvRows and q6kGemvRows are the rows a simdgroup of the q4k lane's native Q5_K and Q6_K GEMVs (gemv_q5k_f32_rows<R>,
// gemv_q6k_f32_rows<R>). Only these R are instantiated.
const q5kGemvRows, q6kGemvRows = 4, 4

// q4kBlockPipes is the plain, residual and bias GEMV of a block-kind segment ("q4k", "q5k", "q6k") and its rows a simdgroup;
// ok is false for int8.
func (r *resident) q4kBlockPipes(kind string) ([3]Pipeline, int, bool) {
	switch kind {
	case "q4k":
		return [3]Pipeline{r.pQ4K, r.pQ4KResid, r.pQ4KBias}, r.q4kR, true
	case "q5k":
		return r.pQ5K, q5kGemvRows, true
	case "q6k":
		return r.pQ6K, q6kGemvRows, true
	}
	return [3]Pipeline{}, 0, false
}

// q4kGemv dispatches one q4k-lane projection over the f32 activation ax (K = uK), one dispatch per segment into its
// rows of out: gemv_q4k_f32_rows for Q4_K super-blocks, gemv_w8_f32_rows for int8 rows with per-row scales, at
// ceil(rows/R) simdgroups. resid accumulates into out; a non-empty bias is added per row (resid and bias are never both
// set).
func (r *resident) q4kGemv(e *Encoder, segs []q4kSeg, resid bool, bias, ax, out, uK Buffer) {
	for _, sg := range segs {
		o := out.At(sg.off * 4)
		nw := (sg.rows + r.w8F32R - 1) / r.w8F32R * 32
		if p, R, ok := r.q4kBlockPipes(sg.kind); ok {
			n := (sg.rows + R - 1) / R * 32
			switch {
			case resid:
				e.Dispatch(p[1], n, 256, sg.w, ax, o, uK, sg.uN)
			case bias != (Buffer{}):
				e.Dispatch(p[2], n, 256, sg.w, ax, o, uK, sg.uN, bias.At(sg.off*4))
			default:
				e.Dispatch(p[0], n, 256, sg.w, ax, o, uK, sg.uN)
			}
			continue
		}
		switch {
		case resid:
			e.Dispatch(r.pW8F32Resid, nw, 256, sg.w, sg.s, ax, o, uK, sg.uN)
		case bias != (Buffer{}):
			e.Dispatch(r.pW8F32Bias, nw, 256, sg.w, sg.s, ax, o, uK, sg.uN, bias.At(sg.off*4))
		default:
			e.Dispatch(r.pW8F32, nw, 256, sg.w, sg.s, ax, o, uK, sg.uN)
		}
	}
}

// encodePLE is a Gemma 4 E-model layer's Per-Layer-Embedding branch, after the FFN residual and before the
// layer scalar, as decoder/forward_gemma4.go and HF's Gemma4DecoderLayer order it:
// x += post_per_layer_input_norm( PLEProj · ( gelu_tanh(PLEGate · x) * pleIn[l] ) ). PLEGate reads the raw residual
// (no pre-norm), quantized like every other W4A8 input; the projection lands in the FFN's dO scratch, free by now.
func (r *resident) encodePLE(e *Encoder, l int, x Buffer) {
	L := &r.layers[l]
	P := r.pleP
	e.Dispatch(r.pQv, 256, 256, x, r.mq, r.mSc, r.uH)
	e.DispatchTG(r.pSA, P*32, 256, r.H*2, L.pleGW, L.pleGS, r.mq, r.mSc, r.pleG, r.uH)
	e.Dispatch(r.pPLEGeluMul, P, 64, r.pleG, r.pleIn.At(l*P*4), r.uPleP)
	e.Dispatch(r.pQv, 256, 256, r.pleG, r.pleQ, r.pleSc, r.uPleP)
	e.DispatchTG(r.pSA, r.H*32, 256, P*2, L.plePW, L.plePS, r.pleQ, r.pleSc, r.dO, r.uPleP)
	e.Dispatch(r.pRmsF32, tgReduceNorm, tgReduceNorm, r.dO, L.postPLENorm, r.uH, r.uEps, r.uAddOne)
	e.Dispatch(r.pRes, r.H, 256, x, r.dO)
}

// loadEmb stages one embedding row for a single-token forward: the hidden state into r.x and, for a Gemma 4
// E-model, the row's PLE tail into r.pleIn. The backend's entry points check the row's length
// (metalResident.embLen) before it gets here.
func (r *resident) loadEmb(emb []float32) {
	copy(r.x.Floats(), emb[:r.H])
	if r.pleP > 0 {
		copy(r.pleIn.Floats(), emb[r.H:])
	}
}

// w8RowsOffForTest keeps a native int8 resident on its one-row GEMVs instead of the rows GEMVs; read at build.
var w8RowsOffForTest bool

// w8RowsOK reports whether a native int8 resident's dense GEMVs can take the int8 rows kernels, which read their weight rows
// and the staged activation as uint4 words: the hidden width, every attention layer's q width and every FFN width are
// multiples of 16.
func (r *resident) w8RowsOK() bool {
	if r.H%16 != 0 || r.I%16 != 0 {
		return false
	}
	for _, L := range r.layers {
		if L.ffnI%16 != 0 || (L.geom != nil && (r.nH*L.geom.hd)%16 != 0) {
			return false
		}
	}
	return true
}

// ksc2OldDownForTest builds the decode library with the staged coal kernels' per-word integer sum (KSC2_OLD_DOWN), the old
// form; read at build.
var ksc2OldDownForTest bool

// gemvRowsFor is the rows-per-simdgroup count for a rows-per-simdgroup GEMV of `rows` outputs: the largest of want, want/2, ..., 2
// whose threadgroups of 8·R rows (tg 256 = 8 simdgroups) tile rows exactly, or 0 (keep the shipped kernel). The rows
// kernels index rows from full threadgroups — a partial last one under dispatchThreads reports a smaller
// threads_per_threadgroup — so an inexact tiling must not reach them.
func gemvRowsFor(rows, want int) int {
	for R := want; R >= 2; R /= 2 {
		if rows%(8*R) == 0 {
			return R
		}
	}
	return 0
}

// saRowsPick returns the SA-family pipeline and grid for `rows` outputs: the R-rows kernel at rows·32/R threads when R
// is set (resident.gemvRows), else the shipped one-row-per-simdgroup kernel at rows·32.
func saRowsPick(shipped, rowsKernel Pipeline, rows, R int) (Pipeline, int) {
	if R > 0 {
		return rowsKernel, rows * 32 / R
	}
	return shipped, rows * 32
}

// canUseF16Lane reports whether layer l's attention block and FFN gate/up can use the W4F16 decode lane
// instead of the shipped W4A8 path. It covers the plainest dense layer only: every special
// case (Gemma sandwich, Olmo postOnly/qkNorm-whole, Cohere parallelBlock, GPT-2 non-gated MLP or FeatOutBias, qGate's
// double-width Q, MoE/Gemma-4 MoE, DeltaNet, compute-time LoRA) keeps the int8 path, since each changes which kernel
// family or epilogue is dispatched and none is ported. Down-proj (the coal family, a different kernel shape) is out of
// scope regardless: encodeLayer's default branch never checks this for r.pGemvResid.
func (r *resident) canUseF16Lane(l int) bool {
	if !r.decodeLaneW4F16 {
		return false
	}
	if r.sandwich || r.postOnly || r.parallelBlock || r.nonGatedMLP || r.outBias || r.layerNorm {
		return false
	}
	if r.loraLayers != nil {
		return false
	}
	L := &r.layers[l]
	return L.moe == nil && L.g4moe == nil && !L.recurrent() && !L.qGate
}

// attnFACoreCount is the GPU core count attention_fa's split count targets (kvHead x S >= 2x the core count): 14, the
// M1 Pro it was tuned on. It sets only the legacy kernel's split (the block kernel uses attnFABlkSplit), so changing
// it changes the legacy kernel's bits. It is hardcoded, not device-queried (aikit's Device has no core-count
// accessor), although the kernel is default-on on every chip via metalAttnFAEnabled; a port to other Apple GPU core
// counts would need it read from the device, not assumed.
const attnFACoreCount = 14

// attnFABlkSplit is the split count the block kernel (attention_fa_blk) runs at. Its split response is not monotone
// and not the legacy kernel's; S = 16 was best or near-best at the depths and models measured, and was the setting of
// the confirmation run and the fidelity decision (docs/measurements/metal-decode-attn-r17-2026-09-25.md). The nKeys/32
// cap never binds above attnFADepthFloor.
const attnFABlkSplit = 16

// attnFADepthFloor is where attention_fa takes over from the shipped kernel: 1024 keys.
// Below it the two kernels were level. S=1 (no split) is not a
// shallow-depth fallback within attention_fa: it measured uniformly worse than the shipped kernel at every depth, so
// below the floor canUseAttnFA declines entirely and the shipped kernel runs.
// A revert to 1536 on the evidence of TestSpecNgram_copyOnStepVerify was a test artifact (its spec arm reused the
// cache while the cold plain arm prefilled it, and the f16 prefill is not bit-identical to decode). Before moving the
// floor, read docs/code-notes/metal.md#attnFADepthFloor.
const attnFADepthFloor = 1024

// attnFAMaxG is the most query heads per KV head attention_fa holds: its per-thread arrays are sized
// ATTN_FA_MAXG in kernels.go (TestAttnFAMaxG_matchesKernel ties the two). A wider group would index past
// them, so attnFAGroupOK keeps such a layer on the shipped kernel (docs/audit-metal-2026-09-30.md).
// The group size comes from the checkpoint, not the family (Llama 3.1 405B has 16), so this is a guard,
// not a list.
const attnFAMaxG = 8

// attnFAGroupOK reports whether a layer with nH query heads over nKV KV heads fits attention_fa's arrays.
func attnFAGroupOK(nH, nKV int) bool { return nKV > 0 && nH%nKV == 0 && nH/nKV <= attnFAMaxG }

// metalHD96DefaultMaxCtx is the resident context cap up to which the hd = 96 lane is on by default: its fidelity gate read Phi-3 mini
// at K = 3,900 within a 4,096 context, and nothing past it (docs/measurements/metal-fa-hd96-phi3-2026-11-23/README.md). A longer
// context takes the lane only when it is asked for.
const metalHD96DefaultMaxCtx = 4096

// faHD96On reports whether an hd = 96 layer may take attention_fa now: Options.FlashDecodeHD96 asked for, or set as the caller's
// default (faHD96Default) while the resident's context cap, the depth a conversation can reach, is at most metalHD96DefaultMaxCtx.
func (r *resident) faHD96On() bool {
	return r.faHD96 && (!r.faHD96Default || r.ctxCap <= metalHD96DefaultMaxCtx)
}

// attnFAHeadDimOK reports whether attention_fa can run a layer of head dim hd: 128 (attention_fa and the block kernel),
// 64 where the block kernel's hd = 64 twin is instantiated for the group size, G = 7, or 96 at G = 1 (attention_fa_blk96)
// when hd96 (Options.FlashDecodeHD96) asks for it. attention_fa itself is hd = 128 only, so a layer of another head dim
// must never reach it without its twin.
func attnFAHeadDimOK(hd, nH, nKV int, hd96 bool) bool {
	switch hd {
	case 128:
		return true
	case 64:
		return nKV > 0 && nH%nKV == 0 && nH/nKV == 7 && attnFABlk64On
	case 96:
		return hd96 && nKV > 0 && nH == nKV
	}
	return false
}

// gemvExtOn routes the int4 GEMVs gemvRows does not cover (DeltaNet's qkv and z projections, a shared expert's gate|up and
// down) through the rows-per-simdgroup kernels. Each rows kernel is bit-identical to the one it replaces (same
// lane-strided words, per-word sum and simd_sum; TestGemvExt_bitIdentical).
var gemvExtOn = true

// gemvExtKind names the shipped kernel family a gemvExt site runs: the plain coal projection, the SA projection, or
// the coal projection with a residual add.
type gemvExtKind int

const (
	gemvExtKCoal gemvExtKind = iota
	gemvExtKSA
	gemvExtKResid
)

// gemvExt encodes one int4 GEMV of `rows` outputs over K inputs at a site gemvRows does not cover: its rows-per-simdgroup
// twin when gemvExtOn and the shape allows it (rows fill whole threadgroups, K bytes of staged activations fit, an
// int4 resident), else exactly the dispatch the site always made. args are the shipped kernel's buffers in order.
func (r *resident) gemvExt(e *Encoder, kind gemvExtKind, rows, K int, args ...Buffer) {
	R := 0
	if gemvExtOn && !r.w8 && K%4 == 0 {
		R = gemvRowsFor(rows, 4)
	}
	stage := K
	if kind == gemvExtKSA {
		stage = 2 * K // the SA family stages K shorts
	}
	if R > 0 && stage > r.gemvExtTGMax {
		R = 0
	}
	if R > 0 {
		r.gemvExtRows[kind]++
	}
	switch {
	case R > 0 && kind == gemvExtKCoal:
		e.DispatchTG(r.gemvExtCoal[R], rows*32/R, 256, stage, args...)
	case R > 0 && kind == gemvExtKSA:
		e.DispatchTG(r.gemvExtSA[R], rows*32/R, 256, stage, args...)
	case R > 0 && kind == gemvExtKResid:
		e.DispatchTG(r.gemvExtResid[R], rows*32/R, 256, stage, args...)
	case kind == gemvExtKCoal:
		e.Dispatch(r.pGemv, rows*32, 32, args...)
	case kind == gemvExtKSA:
		e.DispatchTG(r.pSA, rows*32, 256, 2*K, args...)
	default:
		e.Dispatch(r.pGemvResid, rows*32, 32, args...)
	}
}

// attnFABlkAnyG selects the block kernel for the dense group sizes other than the graded 6 and 7 (G = 2, 3, 4, 5 and
// 8). The kernel agrees with g7 head for head (TestAttnFABlk_anyGMatchesG7).
var attnFABlkAnyG = true

// attnFABlk64On admits hd = 64 layers to attention_fa through the block kernel's hd = 64 twin. The twin is not
// bit-identical to the per-query-head kernel, which is why it was graded on fidelity rather than identity.
var attnFABlk64On = true

// fanSplitBuf is the attention_fa split-count uniform the command buffer being encoded binds: the greedy chain's own
// set's while it encodes one of its buffers (encFANSplit), otherwise the resident's, which setPos writes.
func (r *resident) fanSplitBuf() Buffer {
	if r.encFANSplit != (Buffer{}) {
		return r.encFANSplit
	}
	return r.uAttnFANSplit
}

// planNKeys is the key count the command buffer being encoded will run at: the executor's encNKeys while it encodes,
// otherwise the position setPos last set.
func (r *resident) planNKeys() int {
	if r.encNKeys > 0 {
		return r.encNKeys
	}
	return r.curNKeys
}

// attnPlan is the part of an encoded decode command buffer that depends on its key count or on a runtime
// toggle: whether attention_fa is dispatched (the depth gate), its split grid, and the f16 lane. Everything
// else an encode bakes in is fixed for the resident's life, and the per-token values (position, key count,
// the split uniform) are written at commit time by setPos. The pipelined executor compares a pre-encoded
// buffer's plan with the arriving job's and re-encodes on a mismatch (execLoop). nSplit is 0 when
// attention_fa is not dispatched, so the key count's effect on the cap below the floor never forces a
// re-encode. Pipeline swaps (tests) are not part of it: a test that swaps a pipeline flushes (stopExec).
type attnPlan struct {
	fa      bool
	nSplit  int
	f16Lane bool
}

// attnFAFloor is the key count at which attention_fa takes over: attnFADepthFloor unless a test set
// attnFAFloorOverride.
func (r *resident) attnFAFloor() int {
	if r.attnFAFloorOverride > 0 {
		return r.attnFAFloorOverride
	}
	return attnFADepthFloor
}

// attnPlanFor is the plan a decode command buffer running at nKeys keys needs.
func (r *resident) attnPlanFor(nKeys int) attnPlan {
	p := attnPlan{f16Lane: r.decodeLaneW4F16}
	if r.decodeAttnFA && r.attnFAPartial != (Buffer{}) && r.attnFANKV > 0 && nKeys >= r.attnFAFloor() {
		p.fa, p.nSplit = true, r.attnFASplitFor(nKeys, r.attnFANKV)
	}
	return p
}

// canUseAttnFA reports whether attention_fa may replace the shipped kernel for layer l in the command buffer being
// encoded (at planNKeys keys): dense GQA only, at a head dim attnFAHeadDimOK admits (the kernel's cooperative-load
// tiling is fixed to 32 lanes x half4). Sinks, windows and the f32-KV twin are out of scope, for the reasons of
// canUseF16Lane's family exclusions.
func (r *resident) canUseAttnFA(l int) bool {
	if !r.decodeAttnFA || r.attnFAPartial == (Buffer{}) {
		return false
	}
	if r.sandwich || r.postOnly || r.parallelBlock || r.attnSink || r.kvI8 || r.kvF32 || r.attnSoftcap != 0 {
		return false
	}
	if r.loraLayers != nil {
		return false
	}
	L := &r.layers[l]
	if L.moe != nil || L.g4moe != nil || L.recurrent() || L.geom == nil || L.qGate || L.window != 0 {
		return false
	}
	g := L.geom
	return g != nil && attnFAHeadDimOK(g.hd, r.nH, g.nKV, r.faHD96On()) && attnFAGroupOK(r.nH, g.nKV) && r.planNKeys() >= r.attnFAFloor()
}

// attnFASplitFor picks S so kvHead*S clears 2x attnFACoreCount (the attention_fa lane's registered rule),
// capped at r.attnFAMaxSplit (the partial buffer's own allocation) and at nKeys/32 (a floor on
// keys-per-split — an S so large most splits see one or zero keys wastes the second dispatch's
// combine pass for no parallelism gain).
func (r *resident) attnFASplitFor(nKeys, nKV int) int {
	want := (2*attnFACoreCount + nKV - 1) / nKV
	if r.attnFABlkSplit > 0 {
		want = r.attnFABlkSplit
	}
	if r.attnFASplitOverride > 0 {
		want = r.attnFASplitOverride
	}
	if splitCap := max(nKeys/32, 1); want > splitCap {
		want = splitCap
	}
	if want > r.attnFAMaxSplit {
		want = r.attnFAMaxSplit
	}
	if want < 1 {
		want = 1
	}
	return want
}

// encodeAttention records one layer's attention block (through the o-proj + residual/sandwich norm).
// Split from encodeLayer so the paged Gemma-4 MoE forward can put [attention + dense + router] in one
// command buffer, submit+wait, read the router idx, stage experts, then encode [experts + join] in a
// second — the value-dependent seam paging forces. Byte-identical to the old inline attention block.
func (r *resident) encodeAttention(e *Encoder, l int) {
	r.encodeAttentionWith(e, l, r.uPos, r.uNKeys, r.uQTempScale, r.uRopePos)
}

func (r *resident) encodeAttentionWith(e *Encoder, l int, uPos, uNKeys, uQTempScale Buffer, uRopePos ...Buffer) {
	r.encodeAttentionResidualWith(e, l, r.x, uPos, uNKeys, uQTempScale, uRopePos...)
}

func (r *resident) encodeAttentionResidualWith(e *Encoder, l int, x Buffer, uPos, uNKeys, uQTempScale Buffer, uRopePos ...Buffer) {
	rp := uPos
	if len(uRopePos) > 0 && uRopePos[0] != (Buffer{}) {
		rp = uRopePos[0]
	}
	// --- attention block (7 dispatches in the baseline dense case: norm, fused QKV+bias, merged Q+K RoPE, KV
	// store, attention, o-proj input quant, fused o-proj+residual; 8 at attention_fa depths, whose combine is a
	// second attention dispatch; qGate, qkNorm, kEqV, sandwich and LoRA each add their own) ---
	a := r.attnEncOf(l, x, uPos, uNKeys, uQTempScale, rp)
	form := program.NormRMS
	switch {
	case r.postOnly: // Olmo 3, Olmo Hybrid: no pre-norm at all; a DeltaNet layer never reaches this block
		form = program.NormCopy
	case r.layerNorm:
		form = program.NormLayer
	}
	r.attnInputNorm(e, &a, form)
	if a.L.qGate {
		r.attnGatedQKV(e, &a)
	} else {
		r.attnProjQKV(e, &a)
	}
	if r.qkNorm { // Qwen3: per-head Q/K RMSNorm before RoPE — qGate layers set these at build instead
		r.attnQKNorm(e, &a, r.qkNormWhole)
	}
	if r.attnVNormNeeded(&a) {
		r.attnVNorm(e, &a)
	}
	if !r.learnedPos { // GPT-2: no RoPE at all — position rides the learned embedding added at input
		r.attnRope(e, &a)
	}
	r.attnCore(e, &a)
	if a.L.qGate {
		r.attnOutGate(e, &a)
	}
	out := attnOutFused
	switch {
	case r.sandwich || r.postOnly:
		out = attnOutScratchNorm
	case r.parallelBlock:
		out = attnOutScratch
	case r.outBias:
		out = attnOutFusedBias
	}
	r.attnOut(e, &a, out)
}
