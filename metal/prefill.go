//go:build darwin

package metal

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// prefillSteelAttnOff, set only by tests, runs attention_prefill_fused where attention_prefill_steel would run: the
// baseline arm of R19's §3.2 gate and of the decomposition harness's A/B.
var prefillSteelAttnOff bool

// gemmTilePolicy picks gemmTile's rule: "" the default (D-B02's padding rule below), "a01" A-P01's rule as it shipped,
// "shipped" every GEMM on gemm_w4f16_store's 64 × 64 tile, "bm32" / "bn32" only A-P01's 32-token / 32-feature rule.
// Test-only: the arms of the grades (docs/tasks/task-metal-audit-2026-10.md); not an option or an environment variable.
var gemmTilePolicy = ""

// gemmTile is the tile selector for a rows x N prefill GEMM. Every choice is bit-identical (gemm_w4f16_tile's comment,
// TestGemmTile_bitIdentical).
//
// The default: a threadgroup costs the same whatever share of its rows is real, so a smaller token tile is taken when
// it pads the pass to strictly fewer rows: 32 tokens up to 128 rows, 16 up to 16 rows and, when N <= 8192, up to 48.
// Past those, more row tiles cost more than the padding they save, and a gate|up GEMM as wide as the 1.5B's or 7B's
// already fills the cores at 64 (docs/tasks/task-metal-audit-2026-10.md "D-B02: the 16-row tile";
// TestGemmTile16_probe). A 16-token tile always takes 32 features; otherwise 32 features when N <= 2048 and rows <=
// 64, so the narrow GEMMs fill more cores.
//
// gemmTilePolicy "a01" is the older rule: 32 tokens when the pass has at most 32 rows, 32 features when N <= 2048 and
// rows <= 64.
func (pf *prefillState) gemmTile(rows, N int) (Pipeline, int, int) {
	return pf.gemmTileFor(rows, N, pf.w8)
}

// gemmTileFor is gemmTile for a weight of a given kind: w8 picks the W8 tiles (int8 slice 2), else the int4 ones. An
// int4mix resident (w8Attn) passes true for its attention GEMMs and false for its FFN.
func (pf *prefillState) gemmTileFor(rows, N int, w8 bool) (Pipeline, int, int) {
	tm, tn := gemmTileSize(rows, N)
	if w8 { // int8 slice 2: the native int8 path's weights, the same tile
		return pf.pW8[[2]int{tm, tn}], tm, tn
	}
	switch {
	case tm == 16 && tn == 32:
		return pf.pGemmM16N32, 16, 32
	case tm == 32 && tn == 32:
		return pf.pGemmM32N32, 32, 32
	case tm == 32:
		return pf.pGemmM32N64, 32, 64
	case tn == 32:
		return pf.pGemmM64N32, 64, 32
	}
	return pf.pGemmStore, 64, 64
}

// gemmTileSize is the prefill GEMM's tile, TM tokens by TN features, for a pass of rows rows and N output features.
func gemmTileSize(rows, N int) (tm, tn int) {
	tm, tn = 64, 64
	switch gemmTilePolicy {
	case "":
		pad := (rows + 63) / 64 * 64
		if p := (rows + 31) / 32 * 32; p < pad && rows <= 128 {
			tm, pad = 32, p
		}
		if p := (rows + 15) / 16 * 16; p < pad && (rows <= 16 || rows <= 48 && N <= 8192) {
			tm = 16
		}
		if tm == 16 || (N <= 2048 && rows <= 64) {
			tn = 32
		}
	case "shipped":
	default:
		if gemmTilePolicy != "bn32" && rows <= 32 {
			tm = 32
		}
		if gemmTilePolicy != "bm32" && N <= 2048 && rows <= 64 {
			tn = 32
		}
	}
	return tm, tn
}

// swigluFusedOffForTest keeps the gate|up GEMM and swiglu_f16 as two steps with guF between them: the old arm of KS-D5's A/B.
var swigluFusedOffForTest bool

// swigluFused reports whether a pass of Mpad rows runs every dense layer's gate|up projection through gemm_w4f16_swiglu, so it
// needs no guF: the int4 weights (not the native int8 lane), the 64 x 64 tile for every layer's width, a width that is a
// multiple of 32, and no MoE layer (its expert GEMMs write guF). The pass and prefillScratchBytes both ask it.
func (r *resident) swigluFused(Mpad int) bool {
	if swigluFusedOffForTest || r.w8 || r.moe != nil {
		return false
	}
	for l := -1; l < len(r.layers); l++ {
		fI := r.I
		if l >= 0 && r.layers[l].ffnI != 0 {
			fI = r.layers[l].ffnI
		}
		if tm, tn := gemmTileSize(Mpad, 2*fI); tm != 64 || tn != 64 || fI%32 != 0 {
			return false
		}
	}
	return true
}

// prefillState holds the lazily compiled prefill pipelines (ensurePrefill); a decode-only run never builds it.
type prefillState struct {
	pGemmStore, pRms, pRes, pSw, pRope, pKv, pAttn, pQK, pRmsQ Pipeline
	pGemmSwiglu                                                Pipeline // gemm_w4f16_swiglu (KS-D5)
	pKvI8, pDeqI8                                              Pipeline // an int8 cache's store and dequantize-to-scratch (kv_i8_prefill.go)
	pLN, pLNQ                                                  Pipeline // Part B: layernorm_f16 / layernorm_quant_f16 (bias-free LayerNorm)
	pAttnImg                                                   Pipeline // S17: attention_prefill with a bidirectional image block
	pRopeM                                                     Pipeline // S16: rope_mrope_f16
	pPLEGeluMulF16, pLayerScaleF16                             Pipeline // S9 step 2: the E-model PLE gate and layer scalar
	// A-P01: gemm_w4f16_tile at the smaller tiles (tokens × features), picked by gemmTile.
	pGemmM32N64, pGemmM64N32, pGemmM32N32 Pipeline
	pGemmM16N32                           Pipeline // D-B02's 16-token tile
	// int8 slice 2 (docs/tasks/task-metal-int8-2026-10.md): a native int8 resident's GEMMs, gemm_w4f16_tile<TM, TN,
	// true> by (TM, TN); w8 selects them in gemmTile.
	w8  bool
	pW8 map[[2]int]Pipeline
	// G8 (docs/tasks/task-gpu-paths-2026-09.md): the MoE row loop's F32-scratch bridge (see
	// residual_f16_from_f32/zero_f32's own comments).
	pResF32, pZeroF32 Pipeline
	pResScaled        Pipeline // S17: residual_f16_scaled, the add into a residual stored scaled by 1/s
	// attention_prefill_fused: the simdgroup_matrix flash-attention twin of pAttn
	// (docs/completed/task-prefill-gap.md §4). On by default (metalFusedAttentionEnabled, backend.go);
	// GOINFER_METAL_FUSED_ATTENTION=0 or --exact-prefill falls back to pAttn.
	pAttnFused Pipeline
	// R19: attention_prefill_steel, the head-dim-128 prefill attention (it replaces pAttnFused there).
	pAttnSteel Pipeline
	// attention_prefill_tiled: pAttn for a pass past prefillExactAttnMaxKeys keys (exactAttnTile).
	pAttnTiled Pipeline

	// Expert-major MoE prefill pipelines
	pRouterGemm       Pipeline
	pMoeRouteBatch    Pipeline
	pRouteGptOssBatch Pipeline
	pGatherRows       Pipeline
	pScatterAdd       Pipeline
	pSharedGateAdd    Pipeline

	// D-B01: a gated-attention layer (Qwen3.5) on the batched lane; the DeltaNet mixer's own row kernels live in the
	// resident library (r.pDn*Rows), beside the decode kernels they copy.
	pQGateSplit, pCopyCols, pAttnGate Pipeline
}

// ropeF16Name is the batched prefill's rope kernel: rope_f16, or rope_mrope_f16 for the m-RoPE pass, and their pairwise
// twins for a GPT-J pairwise model (docs/tasks/task-metal-pairwise-rope-2026-10.md).
func ropeF16Name(pairwise, mrope bool) string {
	n := "rope_f16"
	if mrope {
		n = "rope_mrope_f16"
	}
	if pairwise {
		n += "_pw"
	}
	return n
}

func (r *resident) ensurePrefill() {
	if r.pf != nil {
		return
	}
	if r.pfErr != nil {
		// A prior attempt already failed: re-panic the same cached error instead of re-running the full MSL compile to
		// fail identically.
		panic(r.pfErr)
	}
	// Compile and pipeline creation here run pool-less on an unpinned thread (PrefillLast calls this before its
	// own LockOSThread). Pin and hold a pool so the autoreleased temporaries drain; the +1-owned library and
	// pipelines are tracked on the Device and freed at Close.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := NewARPool()
	defer pool.Drain()
	defer func() {
		if p := recover(); p != nil {
			err := fmt.Errorf("%v", p)
			r.pfErr = err
			panic(err)
		}
	}()
	compile := r.d.CompileLibrary
	if (r.w8 || r.w8Attn) && r.preciseMath { // the owner's precise-math decision for native int8 (w8PreciseMath) covers its prefill too
		compile = r.d.CompileLibraryPrecise
	}
	src, err := withAttnSoftcap(prefillKernels+kvI8PrefillKernels, r.attnSoftcap) // + the int8 cache's store and dequantize (kv_i8_prefill.go)
	if err != nil {
		panic(fmt.Sprintf("metal prefill source: %v", err))
	}
	lib, err := compile(src, MSL3_1)
	if err != nil {
		panic(fmt.Sprintf("metal prefill compile: %v", err))
	}
	p := func(n string) Pipeline {
		pp, e := r.d.NewComputePipeline(lib, n)
		if e != nil {
			panic(fmt.Sprintf("metal prefill pipeline %s: %v", n, e))
		}
		return pp
	}
	r.pf = &prefillState{
		pGemmStore: p("gemm_w4f16_store"), pGemmSwiglu: p("gemm_w4f16_swiglu"), pRms: p("rmsnorm_f16"),
		pGemmM32N64: p("gemm_w4f16_m32n64"), pGemmM64N32: p("gemm_w4f16_m64n32"), pGemmM32N32: p("gemm_w4f16_m32n32"), pGemmM16N32: p("gemm_w4f16_m16n32"),
		pLN: p("layernorm_f16"), pLNQ: p("layernorm_quant_f16"),
		pRes: p("residual_f16"), pSw: p("swiglu_f16"), pRope: p(ropeF16Name(r.pairwiseRoPE, false)), pRopeM: p(ropeF16Name(r.pairwiseRoPE, true)),
		pPLEGeluMulF16: p("ple_gelu_mul_f16"), pLayerScaleF16: p("layer_scale_f16"),
		pKv: p("kv_store_f16"), pKvI8: p("kv_store_i8_prefill"), pDeqI8: p("kv_dequant_i8_f16"), pAttn: p("attention_prefill"), pAttnImg: p("attention_prefill_img"), pQK: p("qk_norm_f16"),
		pRmsQ:   p("rmsnorm_quant_f16"),
		pResF32: p("residual_f16_from_f32"), pZeroF32: p("zero_f32"), pResScaled: p("residual_f16_scaled"),
		pAttnFused: p("attention_prefill_fused"), pAttnSteel: p("attention_prefill_steel"), pAttnTiled: p("attention_prefill_tiled"),

		pRouterGemm:       p("router_gemm_f16"),
		pMoeRouteBatch:    p("moe_route_batch"),
		pRouteGptOssBatch: p("route_gptoss_batch"),
		pGatherRows:       p("gather_rows_f16"),
		pScatterAdd:       p("scatter_add_weighted_f16"),
		pSharedGateAdd:    p("shared_gate_add_f16"),

		pQGateSplit: p("qgate_split_f16"), pCopyCols: p("copy_cols_f16"), pAttnGate: p("attn_gate_f16"),
	}
	if r.w8 || r.w8Attn {
		r.pf.w8, r.pf.pW8 = r.w8, map[[2]int]Pipeline{}
		for _, t := range [][2]int{{64, 64}, {32, 64}, {64, 32}, {32, 32}, {16, 32}} {
			r.pf.pW8[t] = p(fmt.Sprintf("gemm_w8f16_m%dn%d", t[0], t[1]))
		}
	}
}

// parallelEmbedsF32ToF16 converts M rows of embs (each H wide) into dst[m*H:(m+1)*H] as f16 bits, splitting across up
// to 8 workers by row. Every element is independent and f32ToF16 is a pure function of its input, so the split is
// byte-identical to the serial loop, as parallelF32ToF16's is. It does not reuse parallelF32ToF16 because embs is
// [][]float32 (one slice per row, not necessarily contiguous) where that wants one flat slice, and flattening first
// would pay its own copy; the threshold and worker shape are the same.
func parallelEmbedsF32ToF16(dst []uint16, embs [][]float32, H int) {
	parallelEmbedsF32ToF16Scaled(dst, embs, H, 1)
}

// parallelEmbedsF32ToF16Scaled is parallelEmbedsF32ToF16 with every value times inv first (S17's scaled residual; inv is
// a power of two, so the product is exact, and inv = 1 is the plain conversion bit for bit).
func parallelEmbedsF32ToF16Scaled(dst []uint16, embs [][]float32, H int, inv float32) {
	M := len(embs)
	workers := min(runtime.GOMAXPROCS(0), 8)
	if M*H < 8192 || workers <= 1 || M < 2 {
		for m := range M {
			row := embs[m]
			for i := range H {
				dst[m*H+i] = f32ToF16(row[i] * inv)
			}
		}
		return
	}
	chunk := (M + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < M; lo += chunk {
		hi := min(lo+chunk, M)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for m := lo; m < hi; m++ {
				row := embs[m]
				for i := range H {
					dst[m*H+i] = f32ToF16(row[i] * inv)
				}
			}
		}(lo, hi)
	}
	wg.Wait()
}

// prefillExactAttnMaxKeys is the exact attention_prefill kernel's bound: it keeps one score per key in `threadgroup
// float sc[4096]`, indexed by the key's absolute position, with no tiling (the decode kernels tile theirs,
// attnScoreTileBound), so above 4096 keys it writes past threadgroup memory. The exact kernel runs whenever the fused
// one cannot (head dim above 128 or not a multiple of 8, or GOINFER_METAL_FUSED_ATTENTION off), and a resident context
// reaches 32768 for an explicit -ctx or a guard-pinned load, so metalResident.PrefillLast declines such a pass to the
// sequential path. TestPrefillExactAttnBound ties this to the kernel source.
const prefillExactAttnMaxKeys = 4096

// exactAttnTiledMaxHD is attention_prefill_tiled's widest head dim: each of its tgReduceAttn threads keeps at most 4 output
// dims in registers.
const exactAttnTiledMaxHD = 4 * tgReduceAttn

// exactAttnTileForTest, when positive, makes every exact-attention pass take attention_prefill_tiled with tiles of that many
// keys, so a test compares it with attention_prefill on a short prompt.
var exactAttnTileForTest = 0

// exactAttnTile is the tile attention_prefill_tiled takes for a pass reaching keys keys, or 0 for attention_prefill: a
// pass that fits the exact kernel's score buffer runs it unchanged.
func exactAttnTile(keys int) int {
	if exactAttnTileForTest > 0 {
		return exactAttnTileForTest
	}
	if keys > prefillExactAttnMaxKeys {
		return prefillExactAttnMaxKeys
	}
	return 0
}

// prefillWidths is the batched pass's scratch widths per row: the q|k|v projection, the attention output and the
// feed-forward width, each the largest over the layers (a Gemma 4 E-model varies the attention geometry and the FFN
// width by layer) and, for an MoE, over the expert and shared-expert widths. The pass sizes its buffers from these
// and prefillScratchBytes prices them, so the two cannot drift apart.
func (r *resident) prefillWidths() (qkvDim, qDim, maxI int) {
	g0 := r.prefillGeom()
	qDim = r.nH * g0.hd
	qkvDim = qDim + 2*g0.kvDim
	maxI = r.I
	if r.moe != nil {
		maxI = max(maxI, r.moe.inter, r.moe.sharedInter)
	}
	for l := range r.layers {
		if g := r.layers[l].geom; g != nil {
			qkvDim, qDim = max(qkvDim, r.nH*g.hd+2*g.kvDim), max(qDim, r.nH*g.hd)
		}
		maxI = max(maxI, r.layers[l].ffnI)
	}
	return qkvDim, qDim, maxI
}

// prefillScratchBytes is the row-sized scratch one batched text pass over M rows allocates and frees: the half
// residual, norm, q|k|v, attention-output and feed-forward buffers and the row positions, plus an MoE's routing and
// expert buffers, an E-model's per-layer inputs and a Gated-DeltaNet hybrid's mixer buffers. It leaves out the
// pass's uniforms (a few words each) and an image turn's m-RoPE triples and DeepStack sets.
// TestPrefillScratchBytes_matchesThePass holds it to prefillScratchSeen.
func (r *resident) prefillScratchBytes(M int) int64 {
	if M <= 0 {
		return 0
	}
	m, mp, H := int64(M), int64((M+7)/8*8), int64(r.H)
	qkvDim, qDim, maxI := r.prefillWidths()
	ffnRows := int64(3) // guF (2 x maxI) and dqF (maxI)
	if r.swigluFused(int(mp)) {
		ffnRows = 1 // dqF only: gemm_w4f16_swiglu writes it with no guF
	}
	n := 2*mp*(2*H+int64(qkvDim)+int64(qDim)+ffnRows*int64(maxI)) + 4*mp
	if e := r.moe; e != nil {
		k := int64(e.k)
		n += 4*m*int64(e.nE) + 8*m*k + 4*mp*H + 8*mp*k
	}
	if r.pleP > 0 {
		n += 4*m*int64(r.nL*r.pleP) + 2*mp*int64(r.pleP)
	}
	if r.kvI8 { // the f16 K and V scratch one layer's int8 cache is dequantized into, the padded context long (kvI8Scratch)
		n += 4 * int64(r.kvI8ScratchRows()) * int64(r.prefillGeom().kvDim)
	}
	if dp := r.dnet; dp != nil {
		g0 := r.prefillGeom()
		nHhd, kvDim := int64(r.nH*g0.hd), int64(g0.kvDim)
		n += 2*mp*int64(dp.convDim+2*dp.valueDim) + 4*m*int64(4*dp.nv+dp.convDim+2*dp.keyDim+2*dp.valueDim)
		n += 2*mp*(2*nHhd+2*kvDim) + 2*m*nHhd
	}
	return n
}

// kvI8ScratchRows is the rows of an int8 cache's dequantize scratch: the cache's own padded capacity, so the attention
// kernels' whole-tile reads past the last key stay inside it, as they do in the cache.
func (r *resident) kvI8ScratchRows() int { return (r.ctxCap + 7) / 8 * 8 }

// prefillScratchSeen counts the bytes of row-sized scratch the batched pass has allocated, at the allocations
// themselves, so a test can hold prefillScratchBytes to what a pass really takes.
var prefillScratchSeen atomic.Int64

// prefillAttnKernels is which attention kernel PrefillLast dispatches for this model: fused for hd%8==0 && hd<=128
// (attention_prefill_fused's compile-time cap, ATTN_MAXHD), steel within that for hd 128 (R19: one threadgroup of 128
// threads per query head and 32 rows), and otherwise the exact attention_prefill, bounded by prefillExactAttnMaxKeys.
func (r *resident) prefillAttnKernels() (fused, steel bool) {
	g0 := r.prefillGeom()
	fused = r.prefillFusedOn() && g0.hd%8 == 0 && g0.hd <= 128
	steel = fused && g0.hd == 128 && !prefillSteelAttnOff
	return fused, steel
}

// prefillFusedOn reports whether the pass may take the fused and steel attention kernels where a layer's head
// dimension has one. A model with an attention softcap may not: the softcap is gated on the exact kernel.
func (r *resident) prefillFusedOn() bool {
	return r.attnSoftcap == 0 && metalFusedAttentionEnabled(r.knobValue("GOINFER_METAL_FUSED_ATTENTION"))
}

// prefillExactAttn reports whether any attention layer of the pass runs the exact attention kernel (no fused kernel for
// its head dim, or fused attention off), whose score buffer holds prefillExactAttnMaxKeys keys. A Gemma 4 E-model's
// layers differ (S9 step 2); every other family's are one geometry.
func (r *resident) prefillExactAttn() bool {
	fusedOn := r.prefillFusedOn()
	for l := range r.layers {
		if g := r.layers[l].geom; g != nil && !(fusedOn && g.hd%8 == 0 && g.hd <= 128) {
			return true
		}
	}
	return false
}

// exactAttnHeadDim is the widest head dim among the attention layers the exact kernel serves (prefillExactAttn), or 0.
func (r *resident) exactAttnHeadDim() int {
	fusedOn, hd := r.prefillFusedOn(), 0
	for l := range r.layers {
		if g := r.layers[l].geom; g != nil && !(fusedOn && g.hd%8 == 0 && g.hd <= 128) {
			hd = max(hd, g.hd)
		}
	}
	return hd
}

// prefillGeom is the attention geometry the pass uses for every attention layer: the first attention layer's. A
// Gated-DeltaNet hybrid's layer 0 is a DeltaNet layer, which has none (D-B01); every other admitted family's layer 0
// has it.
func (r *resident) prefillGeom() *attnGeom {
	for i := range r.layers {
		if g := r.layers[i].geom; g != nil {
			return g
		}
	}
	return nil
}

// kvCap is how many positions every K/V buffer holds: ctxCap, or metalCtxCapDefault before it is resolved (metalResident.ctxCap's rule).
func (r *resident) kvCap() int {
	if r.ctxCap == 0 {
		return metalCtxCapDefault
	}
	return r.ctxCap
}

// PrefillLast ingests M prompt embeddings at positions startPos..startPos+M-1 in one command buffer via the f16 MMA path
// (weights read once, amortized across M, unlike the token-by-token decode loop), populating the resident KV cache, and
// returns the last token's logits[V], what a generator needs to sample the first output token. Correctness-gated against
// the sequential path.
func (r *resident) PrefillLast(embs [][]float32, startPos int) []float32 {
	return r.prefillLast(embs, startPos, nil, nil)
}

// prefillDeep is Qwen3-VL's DeepStack injection for the batched pass (S16): set l is added to the image rows [start,
// start+n) of the f16 residual after decoder layer l, as decoder's addDeepstack does on the CPU.
type prefillDeep struct {
	start, n int
	sets     [][]float32
}

// prefillParallelDefectForTest is a planted defect (docs/tasks/task-metal-pairwise-followups-2026-10.md, G-B2): a
// parallel block's MLP fed from the post-attention residual instead of the layer's shared input norm. Tests set it.
var prefillParallelDefectForTest bool

// prefillResidScaleForTest, when > 0, replaces prefillResidScale's s for every family it applies to (1 turns the scale
// off: G-RS2's planted defect; a large s measures the residual's peak).
var prefillResidScaleForTest float32

// gemmaPrefillResidScale is s for the families prefillResidScale applies to: the smallest power of two that leaves 4x
// headroom under f16's 65,504 over Gemma 3 4B's residual peak (its <bos> row after layer 31;
// docs/code-notes/metal.md#gemmaPrefillResidScale).
const gemmaPrefillResidScale = 32

// prefillResidScale is the s the batched pass stores its residual divided by (S17,
// docs/tasks/task-multimodal-support-2026-10.md): Gemma 3 4B's residual outgrows f16 (inf at layer 6). It applies where
// every read of the residual is an RMSNorm, which ignores scale except through eps, and every write is an embedding
// upload or a post-normed branch's residual add: sandwich norms with no post-only norm, PLE, MoE or DeepStack rows. s is
// a power of two, so the scaled pass equals the unscaled one wherever that one does not overflow (eps and subnormals
// aside). 1 everywhere else: those families run the unchanged kernels.
func (r *resident) prefillResidScale(deep *prefillDeep) float32 {
	if !r.sandwich || r.postOnly || r.pleP != 0 || r.moe != nil || deep != nil {
		return 1
	}
	if prefillResidScaleForTest > 0 {
		return prefillResidScaleForTest
	}
	return gemmaPrefillResidScale
}

// prefillLast is PrefillLast with optional m-RoPE positions: mrope, when non-nil, holds every absolute position's
// (temporal, height, width) triple from 0 to startPos+len(embs), and the pass rotates q and k by them through
// rope_mrope_f16 and r.mropeAxis. The K/V cache rows are still placed by sequence position.
func (r *resident) prefillLast(embs [][]float32, startPos int, mrope [][3]int, deep *prefillDeep) []float32 {
	return r.prefillLastImg(embs, startPos, mrope, deep, nil)
}

// prefillLastImg is prefillLast with image blocks, each [start, end) in absolute positions, that attend bidirectionally
// (S17's Metal image prefill; several blocks for several images, S11): every layer's attention then runs
// attention_prefill_img, the exact kernel with the blocks, in place of the fused and steel kernels, which carry no block
// mask. No blocks: prefillLast exactly.
func (r *resident) prefillLastImg(embs [][]float32, startPos int, mrope [][3]int, deep *prefillDeep, blocks [][2]int) []float32 {
	return r.prefillLastImgSrc(sliceRows(embs), startPos, mrope, deep, blocks)
}

// prefillLastImgSrc is prefillLastImg over a row source (rows.go): its input rows are read rowChunk at a time straight into the
// pass's device buffers.
func (r *resident) prefillLastImgSrc(src rowSource, startPos int, mrope [][3]int, deep *prefillDeep, blocks [][2]int) []float32 {
	// Every K/V buffer holds kvCap positions, and a pass past them writes beyond the buffer. metalResident.checkCap refuses that for
	// the decoder; an internal caller that skipped it is a programming error, named here rather than left to corrupt memory.
	if c := r.kvCap(); startPos < 0 || startPos+src.n > c {
		panic(fmt.Sprintf("metal: internal prefill of %d rows at position %d is past the resident KV cap of %d positions "+
			"(metalResident.checkCap guards the decoder's path; this caller skipped it)", src.n, startPos, c))
	}
	r.ensurePrefill()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	d, pf := r.d, r.pf
	normPipe := pf.pRms // the GEMM-input norms: RMSNorm, or a bias-free LayerNorm for Cohere (Part B)
	if r.layerNorm {
		normPipe = pf.pLN
	}
	M := src.n
	Mpad := (M + 7) / 8 * 8
	H, I, V := r.H, r.I, r.V
	// Prefill runs only for uniform families (prefillOK declines Gemma's per-layer geometry), so
	// every attention layer shares one geometry.
	g0 := r.prefillGeom()
	nHhd := r.nH * g0.hd
	kvDim := g0.kvDim
	qkvDim, qDim, maxI := r.prefillWidths()

	// f16 activation scratch (per call, sized to the padded prompt).
	// xF and the five prefillScratchU16 buffers below rely on Metal's zero-fill of a new buffer for their pad rows
	// (aikit gpu.NewBufferLen* contract: Metal only; CUDA's is uninitialized) — see prefillScratchU16.
	// xF too (R-20): the embeddings are converted straight into the buffer's M rows, its pad rows left as Metal zeroed
	// them, instead of into a zeroed Go slice of Mpad rows that was then copied in whole.
	xF := prefillScratchU16(d, Mpad*H)
	residInv := 1 / r.prefillResidScale(deep)
	// The input rows, read rowChunk at a time from the source: each row's first H values into xF as f16, and a Gemma 4
	// E-model's [L·P] per-layer tail straight into the device buffer the PLE gate reads (S9 step 2), so the host holds a chunk.
	var pleInB Buffer
	if r.pleP > 0 {
		LP := r.nL * r.pleP
		pleInB = d.NewBufferLen(M * LP)
		prefillScratchSeen.Add(int64(4 * M * LP))
	}
	for lo := 0; lo < M; lo += rowChunk {
		hi := min(lo+rowChunk, M)
		rows := src.rows(lo, hi)
		parallelEmbedsF32ToF16Scaled(xF.U16s()[lo*H:hi*H], rows, H, residInv)
		if r.pleP > 0 {
			LP, pv := r.nL*r.pleP, pleInB.Floats()
			for i, row := range rows {
				copy(pv[(lo+i)*LP:(lo+i+1)*LP], row[H:H+LP])
			}
		}
	}
	normF := prefillScratchU16(d, Mpad*H)
	qkvF := prefillScratchU16(d, Mpad*qkvDim)
	ctxF := prefillScratchU16(d, Mpad*qDim)
	fusedSw := r.swigluFused(Mpad)
	var guF Buffer // the gate|up rows between the GEMM and swiglu_f16; none when the pass fuses them (swigluFused)
	if !fusedSw {
		guF = prefillScratchU16(d, Mpad*2*maxI)
	}
	dqF := prefillScratchU16(d, Mpad*maxI)
	posv := make([]uint32, Mpad)
	for m := range M {
		posv[m] = uint32(startPos + m)
	}
	posB := NewBufferUint32s(d, posv)
	prefillScratchSeen.Add(int64(4 * Mpad))
	var pos3B Buffer // S16: the rows' m-RoPE triples, row-major [Mpad*3]; pad rows repeat the last real row's
	if mrope != nil {
		p3 := make([]uint32, Mpad*3)
		for m := range Mpad {
			t := mrope[startPos+min(m, M-1)]
			p3[m*3], p3[m*3+1], p3[m*3+2] = uint32(t[0]), uint32(t[1]), uint32(t[2])
		}
		pos3B = NewBufferUint32s(d, p3)
	}

	// uniforms
	uM := NewBufferU32(d, uint32(Mpad))
	uH := r.uH
	uI := NewBufferU32(d, uint32(I))
	u2I := NewBufferU32(d, uint32(2*I))
	uQkv := NewBufferU32(d, uint32(qkvDim))
	uQDim := NewBufferU32(d, uint32(qDim))
	uHd := g0.uHd
	uStride := NewBufferU32(d, uint32(qkvDim))
	uKOff := NewBufferU32(d, uint32(nHhd))
	uVOff := NewBufferU32(d, uint32(nHhd+kvDim))
	uStartPos := NewBufferU32(d, uint32(startPos))
	uTotalQ := NewBufferU32(d, uint32(nHhd))
	uTotalK := NewBufferU32(d, uint32(kvDim))
	uBase0 := NewBufferU32(d, 0)
	uBaseK := NewBufferU32(d, uint32(nHhd))
	m0, m1, m2 := NewBufferU32(d, 0), NewBufferU32(d, 1), NewBufferU32(d, 2)
	dummyBias := NewBufferFloats(d, make([]float32, 1))
	// L2-Metal: attention_prefill_fused's own row-count uniform — REAL M (unpadded), unlike uM
	// above which holds Mpad for the GEMM grid. attention_prefill (the exact kernel) needs no
	// such uniform because its Go-side dispatch grid is already sized off the real M.
	uMReal := NewBufferU32(d, uint32(M))
	// An int8 cache: each layer's rows [0, startPos+M) are dequantized into these f16 buffers before its attention (D9 (b)).
	var kScrI8, vScrI8, uEndI8 Buffer
	if r.kvI8 {
		kScrI8, vScrI8 = prefillScratchU16(d, r.kvI8ScratchRows()*kvDim), prefillScratchU16(d, r.kvI8ScratchRows()*kvDim)
		uEndI8 = NewBufferU32(d, uint32(startPos+M))
	}
	attnTile := exactAttnTile(startPos + M) // > 0: the exact-attention layers take attention_prefill_tiled
	uTile := NewBufferU32(d, uint32(max(attnTile, 1)))
	// G8: the MoE row loop's F32 accumulate scratch (see the per-layer loop's own comment).
	// Allocated unconditionally (cheap — H floats) rather than gated on whether any layer is
	// MoE, avoiding a nil-buffer special case in the loop below.
	moeDst := NewBufferFloats(d, make([]float32, H))

	// MoE expert-major prefill scratch buffers (allocated only when model has MoE layers).
	var (
		moeLogits  Buffer
		moeIdx     Buffer
		moeWgt     Buffer
		expertIn   Buffer
		expertDown Buffer
		rowIdxBuf  Buffer
		rowWgtBuf  Buffer
	)
	if r.moe != nil {
		// R-20: zero-filled by Metal, not copied from zeroed Go slices; the router GEMM, the route kernels and the host
		// fill them before anything reads them (see prefillScratch).
		moeLogits = prefillScratch[float32](d, M*r.moe.nE)
		moeIdx = prefillScratch[uint32](d, M*r.moe.k)
		moeWgt = prefillScratch[float32](d, M*r.moe.k)
		// expertIn/expertDown rely on the same Metal zero-fill for their pad rows as the scratch above (see prefillScratchU16).
		expertIn = prefillScratchU16(d, Mpad*H)
		expertDown = prefillScratchU16(d, Mpad*H)
		rowIdxBuf = prefillScratch[uint32](d, Mpad*r.moe.k)
		rowWgtBuf = prefillScratch[float32](d, Mpad*r.moe.k)
		if prefillScratchPoison { // a test's proof that nothing reads these before the pass writes them
			for _, b := range []Buffer{moeLogits, moeIdx, moeWgt, rowIdxBuf, rowWgtBuf} {
				for i, u := 0, b.U32s(); i < len(u); i++ {
					u[i] = 0xFFFFFFFF
				}
			}
		}
	}

	// Every buffer above is per-call scratch or uniform allocated onto the device ledger, which ReleaseAll frees
	// only at Close, so each is released at the end of the call (e.End() below commits and waits, so the GPU is
	// finished with them by then). Unreleased they ratchet by a prompt's worth per call (guF alone is Mpad*2I*2)
	// until the mustBuf OOM panic kills serve, which is recovered only on the BuildResident path, not here. r.uH,
	// r.uKvDim and r.uHd are resident-owned and reused, so deliberately not in this list: releasing them would
	// corrupt the decode path.
	scratch := []Buffer{
		xF, normF, qkvF, ctxF, guF, dqF, posB, moeDst,
		uM, uI, u2I, uQkv, uQDim, uStride, uKOff, uVOff, uStartPos,
		uTotalQ, uTotalK, uBase0, uBaseK, m0, m1, m2, dummyBias, uMReal,
	}
	if r.moe != nil {
		scratch = append(scratch, moeLogits, moeIdx, moeWgt, expertIn, expertDown, rowIdxBuf, rowWgtBuf)
	}
	if r.kvI8 {
		scratch = append(scratch, kScrI8, vScrI8, uEndI8)
	}
	// S17: the scaled residual's 1/s, for residual_f16_scaled, and the eps of the norms that read the residual, eps/s², so
	// that x/s / sqrt(ms/s² + eps/s²) is x / sqrt(ms + eps) exactly (s is a power of two).
	var uResidInv Buffer
	uEpsX := r.uEps
	if residInv != 1 {
		uResidInv = NewBufferFloats(d, []float32{residInv})
		uEpsX = NewBufferFloats(d, []float32{r.uEps.Floats()[0] * residInv * residInv})
		scratch = append(scratch, uResidInv, uEpsX)
	}
	// S9 step 2: a Gemma 4 E-model's per-layer inputs, each row's [L·P] tail of its [h ‖ L·P] embedding, and the PLE
	// gate's scratch.
	var pleGF, uP, uLP Buffer
	if r.pleP > 0 {
		P, LP := r.pleP, r.nL*r.pleP
		if emodelBatchDefect == 3 { // G-S9c (3): each row's PLE inputs taken from the next row
			pv := pleInB.Floats()
			for m := 0; m+1 < M; m++ {
				copy(pv[m*LP:(m+1)*LP], pv[(m+1)*LP:(m+2)*LP])
			}
		}
		pleGF, uP, uLP = prefillScratchU16(d, Mpad*P), NewBufferU32(d, uint32(P)), NewBufferU32(d, uint32(LP))
		scratch = append(scratch, pleInB, pleGF, uP, uLP)
	}
	// D-B01: a Gated-DeltaNet hybrid's mixer scratch, and its gated-attention layers' [query ‖ gate] projection, K‖V
	// projection and split gate.
	var dn *prefillDelta
	var qgF, kvF, gateF, u2NHhd, u2KvDim, uGateN Buffer
	if r.dnet != nil {
		dn = r.newPrefillDelta(M, Mpad)
		qgF = prefillScratchU16(d, Mpad*2*nHhd)
		kvF = prefillScratchU16(d, Mpad*2*kvDim)
		gateF = prefillScratchU16(d, M*nHhd)
		u2NHhd, u2KvDim, uGateN = NewBufferU32(d, uint32(2*nHhd)), NewBufferU32(d, uint32(2*kvDim)), NewBufferU32(d, uint32(M*nHhd))
		scratch = append(append(scratch, dn.bufs...), qgF, kvF, gateF, u2NHhd, u2KvDim, uGateN)
	}
	var imgB, imgN Buffer // the bidirectional image blocks and their count, when there are any (attention_prefill_img)
	if len(blocks) > 0 {
		flat := make([]uint32, 0, 2*len(blocks))
		for _, b := range blocks {
			flat = append(flat, uint32(b[0]), uint32(b[1]))
		}
		imgB, imgN = NewBufferUint32s(d, flat), NewBufferU32(d, uint32(len(blocks)))
		scratch = append(scratch, imgB, imgN)
	}
	defer func() {
		for _, b := range scratch {
			d.ReleaseBuf(b)
		}
	}()

	// Uniform buffer cache to avoid per-expert dynamic allocations
	uConsts := make(map[int]Buffer)
	getU32 := func(val int) Buffer {
		if b, ok := uConsts[val]; ok {
			return b
		}
		b := NewBufferU32(d, uint32(val))
		uConsts[val] = b
		scratch = append(scratch, b)
		return b
	}

	// gemm: one GEMM dispatch over rows×N — a 2-D grid of (ceil(N/TN), ceil(rows/TM)) threadgroups of 128 threads, each
	// owning a TN-feature × TM-token tile: gemm_w4f16_store's 64 × 64 (R16), or for a small pass a smaller tile of the
	// same arithmetic (A-P01, gemmTile).
	gemm := func(e *Encoder, rows, N int, bufs ...Buffer) {
		p, tm, tn := pf.gemmTile(rows, N)
		e.Dispatch2D(p, (N+tn-1)/tn, (rows+tm-1)/tm, 128, 1, bufs...)
	}
	// gemmAttn is gemm for the attention projections (qkv, o), whose weights are int8 on an int4mix resident too.
	gemmAttn := func(e *Encoder, rows, N int, bufs ...Buffer) {
		p, tm, tn := pf.gemmTileFor(rows, N, r.w8 || r.w8Attn)
		e.Dispatch2D(p, (N+tn-1)/tn, (rows+tm-1)/tm, 128, 1, bufs...)
	}
	// L2-Metal: attention_prefill_fused's grid — nH×ceil(M/8) simdgroups (ATTN_SGPT=4/threadgroup,
	// prefill.go's own #define, matched here). Real M (unpadded): the tail row-tile's out-of-range
	// rows are masked in-kernel via buffer(11), not dropped from the grid.
	const attnFusedSGPT = 4
	numRowTiles := (M + 7) / 8
	attnFusedTotal := r.nH * numRowTiles
	attnFusedTotal = (attnFusedTotal + attnFusedSGPT - 1) / attnFusedSGPT * attnFusedSGPT * 32
	attnFusedTg := attnFusedSGPT * 32
	// S9 step 2: one attention geometry's uniforms, per layer. Every family before Gemma 4's E-models has one geometry, so
	// it gets one bundle with the values the single-geometry pass used; an E-model gets one per head dim, and a KV-shared
	// layer's Q-only projection its own stride. The fused/steel attention kernels hold head dims up to 128; past that the
	// exact kernel runs (its key limit is checked before the pass).
	type layerU struct {
		g                                                            *attnGeom
		qkvDim, nHhd, kvDim                                          int
		uQkv, uStride, uKOff, uVOff, uTotalQ, uTotalK, uBaseK, uQDim Buffer
		fused, steel                                                 bool
	}
	type luKey struct {
		g      *attnGeom
		shared bool
	}
	luCache := map[luKey]*layerU{}
	fusedOn := r.prefillFusedOn()
	layerUFor := func(L *residLayer) *layerU {
		k := luKey{L.geom, L.kvShared}
		if u, ok := luCache[k]; ok {
			return u
		}
		g := L.geom
		nh := r.nH * g.hd
		q := nh + 2*g.kvDim
		if L.kvShared {
			q = nh // Q only: a shared layer projects no K or V
		}
		fused := fusedOn && g.hd%8 == 0 && g.hd <= 128
		u := &layerU{g: g, qkvDim: q, nHhd: nh, kvDim: g.kvDim,
			uQkv: getU32(q), uStride: getU32(q), uKOff: getU32(nh), uVOff: getU32(nh + g.kvDim),
			uTotalQ: getU32(nh), uTotalK: getU32(g.kvDim), uBaseK: getU32(nh), uQDim: getU32(nh),
			fused: fused, steel: fused && g.hd == 128 && !prefillSteelAttnOff}
		luCache[k] = u
		return u
	}
	// G-S9c (1): a shared layer attending over an empty cache instead of its source's (zero in production).
	var zeroKV Buffer
	if emodelBatchDefect == 1 {
		zeroKV = prefillScratchU16(d, (startPos+Mpad+8)*qkvDim)
		scratch = append(scratch, zeroKV)
	}

	e := r.q.Begin()
	// S16: the DeepStack sets, uploaded once; addDeep(k) adds set k to the rows of this pass that fall in the image block.
	// Set k is added after layer k, which is done at the top of iteration k+1 and after the loop, so every branch of the
	// loop body (the MoE ones end in continue) reaches it.
	var deepB []Buffer
	if deep != nil {
		for _, set := range deep.sets {
			deepB = append(deepB, NewBufferFloats(d, set))
		}
	}
	addDeep := func(k int) {
		if k >= len(deepB) {
			return
		}
		lo, hi := max(deep.start, startPos), min(deep.start+deep.n, startPos+M)
		if hi <= lo {
			return
		}
		e.Dispatch(pf.pResF32, (hi-lo)*H, 256, xF.At((lo-startPos)*H*2), deepB[k].At((lo-deep.start)*H*4))
	}
	for l := 0; l < r.nL; l++ {
		if l > 0 {
			addDeep(l - 1)
		}
		L := &r.layers[l]
		if L.delta != nil {
			r.encodePrefillDeltaMixer(e, L, xF, normF, dn, M, Mpad, gemm, uM, m0, m2, dummyBias)
		} else {
			lu := layerUFor(L)
			// pre-attn norm — addOne (Gemma's 1+w) matters even for a plain non-sandwich family's
			// GEMV-input norm, so it is always passed (0 for every family without RMSAddOne).
			// Olmo 3 / Olmo Hybrid (postOnly): no pre-norm; GEMV reads raw residual xF directly.
			inAttn := normF
			if r.postOnly {
				inAttn = xF
			} else {
				e.Dispatch(normPipe, M*tgReduceNorm, tgReduceNorm, xF, L.preNorm, normF, uH, uEpsX, r.uAddOne)
			}
			// fused QKV (+bias)
			if L.qGate {
				// D-B01, Qwen3.5's gated softmax layer (decode's encodeAttention qGate branch): q_proj is [query ‖ gate]
				// per head at double width, split on the activation into qkv's Q slot and the gate; K‖V is its own
				// bias-free projection, copied into qkv's K and V slots.
				gemm(e, Mpad, 2*nHhd, inAttn, L.dnQw, L.dnQs, qgF, uM, u2NHhd, uH, dummyBias, m0)
				e.Dispatch(pf.pQGateSplit, M*nHhd, 256, qgF, qkvF, gateF, uTotalQ, uHd, uQkv, uMReal)
				gemm(e, Mpad, 2*kvDim, inAttn, L.qkvW, L.qkvS, kvF, uM, u2KvDim, uH, dummyBias, m0)
				e.Dispatch(pf.pCopyCols, M*2*kvDim, 256, kvF, qkvF, u2KvDim, uQkv, uKOff, uMReal)
			} else {
				gemmAttn(e, Mpad, lu.qkvDim, inAttn, L.qkvW, L.qkvS, qkvF, uM, lu.uQkv, uH, L.qkvBias, m1)
			}
			if r.qkNorm { // Qwen3 / Olmo 3: per-head Q/K RMSNorm before RoPE
				qkNH, qkNKV, qkHD, qkNHhd := r.uNH, lu.g.uNKV, lu.g.uHd, lu.g.uNHhd
				tgCount := r.nH + lu.g.nKV
				if L.kvShared {
					// Q heads only: a shared layer projects no K. The kernel maps a threadgroup to (row, head) by nH+nKV, so nKV
					// must be 0 here too, or rows shift and the "K" head normalizes the next row's Q.
					tgCount, qkNKV = r.nH, r.uZero
				}
				if r.qkNormWhole {
					qkNH, qkNKV, qkHD, qkNHhd = r.uQKWholeOne, r.uQKWholeOne, r.uQKWholeHD, r.uQKWholeHD
					tgCount = 2
				}
				e.Dispatch(pf.pQK, M*tgCount*tgReduceAttn, tgReduceAttn, qkvF, L.qNorm, L.kNorm, qkNH, qkNKV, qkHD, qkNHhd, lu.uStride, r.uEps, r.uAddOne)
			}
			// S9 step 2: Gemma 4's scale-less v_norm on every K/V-owning layer's V slot, after Q/K norm and before RoPE, as
			// decode does (encodeAttentionResidualWith): the Q/K-norm kernel over the V slot with nH = 0, a unit weight and
			// addOne 0.
			if !L.kvShared && r.g4VNorm && emodelBatchDefect != 5 {
				e.Dispatch(pf.pQK, M*lu.g.nKV*tgReduceAttn, tgReduceAttn, qkvF.At((lu.nHhd+lu.kvDim)*2), r.vNormUnit, r.vNormUnit,
					r.uZero, lu.g.uNKV, lu.g.uHd, r.uZero, lu.uStride, r.uEps, r.uZero)
			}
			// rope q, k (per-row positions): bind the per-layer RoPE table and window, exactly as decode does
			// (encodeTrunkInto), not the model-level r.invf/r.uWindow. For a mixed local/global-window arch the global
			// layers must see window=0, and each layer its own RoPE base; FeatPerLayerRoPE is in prefillFeatures, so the
			// per-layer binding is not behaviour-neutral.
			if mrope != nil { // S16: each row's own (t, h, w) rotation
				e.Dispatch(pf.pRopeM, M*r.nH*lu.g.half, 128, qkvF, L.invf, lu.g.uHd, pos3B, lu.uTotalQ, lu.uStride, uBase0, lu.g.uHalf, L.mscale, r.mropeAxis)
				if !L.kvShared {
					e.Dispatch(pf.pRopeM, M*lu.g.nKV*lu.g.half, 128, qkvF, L.invf, lu.g.uHd, pos3B, lu.uTotalK, lu.uStride, lu.uBaseK, lu.g.uHalf, L.mscale, r.mropeAxis)
				}
			} else {
				e.Dispatch(pf.pRope, M*r.nH*lu.g.half, 128, qkvF, L.invf, lu.g.uHd, posB, lu.uTotalQ, lu.uStride, uBase0, lu.g.uHalf, L.mscale)
				if !L.kvShared {
					e.Dispatch(pf.pRope, M*lu.g.nKV*lu.g.half, 128, qkvF, L.invf, lu.g.uHd, posB, lu.uTotalK, lu.uStride, lu.uBaseK, lu.g.uHalf, L.mscale)
				}
			}
			// scatter K,V to cache; a KV-shared layer stores nothing (its cache is its source's)
			if !L.kvShared && r.kvI8 {
				e.Dispatch(pf.pKvI8, M*lu.g.nKV, 32, qkvF, posB, r.kc[l], r.vc[l], r.ks[l], r.vs[l], lu.g.uNKV, lu.g.uHd, lu.uStride, lu.uKOff, lu.uVOff, uMReal)
			} else if !L.kvShared {
				e.Dispatch(pf.pKv, M*lu.kvDim, 128, qkvF, r.kc[l], r.vc[l], posB, lu.g.uKvDim, lu.uStride, lu.uKOff, lu.uVOff)
			}
			kc, vc := r.kc[l], r.vc[l]
			if r.kvI8 { // the attention kernels read f16: this layer's cache, dequantized
				e.Dispatch(pf.pDeqI8, (startPos+M)*lu.kvDim, 256, r.kc[l], r.vc[l], r.ks[l], r.vs[l], kScrI8, vScrI8, lu.g.uNKV, lu.g.uHd, uEndI8)
				kc, vc = kScrI8, vScrI8
			}
			if L.kvShared && emodelBatchDefect == 1 {
				kc, vc = zeroKV, zeroKV
			}
			// causal attention → ctx (per-layer window: 0 = full causal on a global layer)
			if imgB != (Buffer{}) {
				e.Dispatch(pf.pAttnImg, M*r.nH*tgReduceAttn, tgReduceAttn, qkvF, kc, vc, ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, uStartPos, r.uScale, lu.uStride, L.uWindow, imgB, imgN)
			} else if lu.steel {
				e.Dispatch(pf.pAttnSteel, r.nH*((M+31)/32)*128, 128, qkvF, kc, vc, ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, uStartPos, r.uScale, lu.uStride, L.uWindow, uMReal)
			} else if lu.fused {
				e.Dispatch(pf.pAttnFused, attnFusedTotal, attnFusedTg, qkvF, kc, vc, ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, uStartPos, r.uScale, lu.uStride, L.uWindow, uMReal)
			} else if attnTile > 0 {
				e.Dispatch(pf.pAttnTiled, M*r.nH*tgReduceAttn, tgReduceAttn, qkvF, kc, vc, ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, uStartPos, r.uScale, lu.uStride, L.uWindow, uTile)
			} else {
				e.Dispatch(pf.pAttn, M*r.nH*tgReduceAttn, tgReduceAttn, qkvF, kc, vc, ctxF, r.uNH, lu.g.uNKV, lu.g.uHd, uStartPos, r.uScale, lu.uStride, L.uWindow)
			}
			if L.qGate { // ctx *= sigmoid(gate), before o-proj, as decode does
				e.Dispatch(pf.pAttnGate, M*nHhd, 256, ctxF, gateF, uGateN)
			}
			// o-proj, then either the plain residual epilogue, Gemma's sandwich norm, or Olmo 3 postOnly: mode-0 write
			// into normF (free scratch here: its last use, the fused-QKV input, already ran, and its next use, the pre-MLP
			// norm's output, is below), norm the sublayer output in place (safe: rmsnorm_f16's read pass completes before
			// its write pass touches the same buffer), then a separate residual add. Other families take the fused mode-2
			// residual epilogue.
			if r.sandwich || r.postOnly {
				gemmAttn(e, Mpad, H, ctxF, L.oW, L.oS, normF, uM, uH, lu.uQDim, dummyBias, m0)
				e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, normF, L.postAttnNorm, normF, uH, r.uEps, r.uAddOne)
				if uResidInv != (Buffer{}) {
					e.Dispatch(pf.pResScaled, M*H, 256, xF, normF, uResidInv)
				} else {
					e.Dispatch(pf.pRes, M*H, 256, xF, normF)
				}
			} else {
				gemmAttn(e, Mpad, H, ctxF, L.oW, L.oS, xF, uM, uH, lu.uQDim, dummyBias, m2)
			}
		}
		if L.moe != nil {
			if r.knobValue("GOINFER_MOE_EXPERT_MAJOR") == "0" {
				// Fallback to row-by-row path for A/B testing
				for m := range M {
					row := xF.At(m * H * 2)
					e.Dispatch(pf.pRmsQ, tgReduceNorm, tgReduceNorm, row, L.postNorm, r.mq, r.mSc, uH, r.uEps, r.uAddOne)
					r.encodeMoERoute(e, L)
					if r.moeCap.idx != (Buffer{}) && m < r.moeCap.rows {
						r.encodeMoECapture(e, l*r.moeCap.rows+m)
					}
					e.Dispatch(pf.pZeroF32, r.H, 256, moeDst)
					r.encodeMoEExperts(e, L, moeDst)
					e.Dispatch(pf.pResF32, r.H, 256, row, moeDst)
				}
				continue
			}

			// --- EXPERT-MAJOR PATH ---
			// 1. Pre-norm all M rows into normF
			e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, xF, L.postNorm, normF, uH, uEpsX, r.uAddOne)

			// 2. Batched Router GEMM: normF [M, H] x routerW [nE, H] -> moeLogits [M, nE]
			totalRouterThreads := (M*r.moe.nE*32 + 255) / 256 * 256
			e.Dispatch(pf.pRouterGemm, totalRouterThreads, 256, normF, L.moe.routerW, moeLogits, uH, r.moe.uNE, uMReal)

			// 3. Batched Top-k Route: moeLogits -> moeIdx [M, k], moeWgt [M, k]
			if r.moe.isGptOss {
				e.Dispatch(pf.pRouteGptOssBatch, M*32, 32, moeLogits, L.moe.routerBias, moeIdx, moeWgt, r.moe.uNE, r.moe.uK, uMReal)
			} else {
				e.Dispatch(pf.pMoeRouteBatch, M*32, 32, moeLogits, L.moe.routerBias, moeIdx, moeWgt, r.moe.uNE, r.moe.uK, r.moe.uSigmoid, r.moe.uNorm, r.moe.uScale, r.moe.uNGroup, r.moe.uTopkGroup, uMReal)
			}

			// 4. Commit and wait so the host can read the routing decisions
			e.End()
			if db02Trace != nil {
				db02Trace(db02Event{layer: l, gpuStart: e.GPUStart(), gpuEnd: e.GPUEnd()})
			}
			r.recordExecErr(e.Err())
			if err := r.takeExecErr(); err != nil {
				return nil
			}

			// 5. Host Grouping: group (token m, rank j) by expert e
			type moeSlot struct {
				row  int
				rank int
				wgt  float32
			}
			byExpert := make([][]moeSlot, r.moe.nE)
			idxSlice := moeIdx.U32s()[:M*r.moe.k]
			wgtSlice := moeWgt.Floats()[:M*r.moe.k]
			if dg01Mutation != "" {
				dg01Mutate(idxSlice, wgtSlice, M, r.moe.k)
			}
			if r.moeCap.major != nil {
				r.moeCap.major(l, idxSlice, wgtSlice)
			}
			for m := range M {
				for j := 0; j < r.moe.k; j++ {
					eIdx := int(idxSlice[m*r.moe.k+j])
					w := wgtSlice[m*r.moe.k+j]
					if eIdx >= 0 && eIdx < r.moe.nE {
						byExpert[eIdx] = append(byExpert[eIdx], moeSlot{row: m, rank: j, wgt: w})
					}
				}
			}

			// 6. Begin new command encoder for expert dispatches
			e = r.q.Begin()

			// 7. Dispatch each active expert
			moeInter := r.moe.inter
			rowsPerExpertGu := 2 * moeInter
			rowsPerExpertD := H
			wprGu := H / 8
			gprGu := H / 32
			wprD := moeInter / 8
			gprD := moeInter / 32

			rowIdxHost := rowIdxBuf.U32s()
			rowWgtHost := rowWgtBuf.Floats()

			slotOffset := 0
			for eIdx, slots := range byExpert {
				Ce := len(slots)
				if Ce == 0 {
					continue
				}

				for i, s := range slots {
					rowIdxHost[slotOffset+i] = uint32(s.row)
					rowWgtHost[slotOffset+i] = s.wgt
				}

				CePad := (Ce + 7) / 8 * 8
				if db02Trace != nil {
					_, tmGu, _ := pf.gemmTile(CePad, rowsPerExpertGu)
					_, tmD, _ := pf.gemmTile(CePad, rowsPerExpertD)
					db02Trace(db02Event{layer: l, expert: eIdx, rows: Ce, padRows: CePad,
						tileRowsGu: (CePad + tmGu - 1) / tmGu * tmGu, tileRowsD: (CePad + tmD - 1) / tmD * tmD})
				}
				uCe := getU32(Ce)
				uCePad := getU32(CePad)
				uMoeI := getU32(moeInter)
				u2MoeI := getU32(2 * moeInter)

				expIdxBuf := rowIdxBuf.At(slotOffset * 4)
				expWgtBuf := rowWgtBuf.At(slotOffset * 4)

				// A. Gather Ce rows from normF into expertIn (with padding zeroed)
				gatherTotal := (CePad*H + 255) / 256 * 256
				e.Dispatch(pf.pGatherRows, gatherTotal, 256, normF, expertIn, expIdxBuf, uH, uCe, uCePad)

				// B. Expert Gate/Up GEMM
				guWOff := eIdx * rowsPerExpertGu * wprGu * 4
				guSOff := eIdx * rowsPerExpertGu * gprGu * 2
				if r.w8 { // int8 slice 4: H bytes of int8 codes and one f32 scale a row
					guWOff, guSOff = eIdx*rowsPerExpertGu*H, eIdx*rowsPerExpertGu*4
				}
				gemm(e, CePad, rowsPerExpertGu, expertIn, L.moe.expGuW.At(guWOff), L.moe.expGuS.At(guSOff), guF, uCePad, u2MoeI, uH, dummyBias, m0)

				// C. SwiGLU
				e.Dispatch(pf.pSw, CePad*moeInter, 256, guF, dqF, uMoeI, r.uAct)

				// D. Expert Down GEMM
				dOff := eIdx * rowsPerExpertD * wprD * 4
				dSOff := eIdx * rowsPerExpertD * gprD * 2
				if r.w8 { // moeInter bytes of int8 codes and one f32 scale a row
					dOff, dSOff = eIdx*rowsPerExpertD*moeInter, eIdx*rowsPerExpertD*4
				}
				gemm(e, CePad, rowsPerExpertD, dqF, L.moe.expDW.At(dOff), L.moe.expDS.At(dSOff), expertDown, uCePad, uH, uMoeI, dummyBias, m0)

				// E. Scatter-add weighted output into residual xF
				scatterTotal := (Ce*H + 255) / 256 * 256
				e.Dispatch(pf.pScatterAdd, scatterTotal, 256, expertDown, xF, expIdxBuf, expWgtBuf, uH, uCe, uMReal)

				slotOffset += Ce
			}

			// 8. Shared expert (if present)
			if r.moe.sharedInter > 0 && dg01Mutation != "shared-off" {
				shI := r.moe.sharedInter
				u2ShI := getU32(2 * shI)
				uShI := getU32(shI)

				gemm(e, Mpad, 2*shI, normF, L.moe.shGuW, L.moe.shGuS, guF, uM, u2ShI, uH, dummyBias, m0)
				e.Dispatch(pf.pSw, Mpad*shI, 256, guF, dqF, uShI, r.uAct)

				if r.moe.sharedUngated {
					// dst += down directly into xF
					gemm(e, Mpad, H, dqF, L.moe.shDW, L.moe.shDS, xF, uM, uH, uShI, dummyBias, m2)
				} else {
					// gated variant (Qwen2-MoE): compute sigmoid(gate)*down
					gemm(e, Mpad, H, dqF, L.moe.shDW, L.moe.shDS, expertDown, uM, uH, uShI, dummyBias, m0)
					// Gate logit for all M rows
					gateLogitsTotal := (M*32 + 255) / 256 * 256
					e.Dispatch(pf.pRouterGemm, gateLogitsTotal, 256, normF, L.moe.shGateW, moeLogits, uH, getU32(1), uMReal)
					// Combine into xF via shared_gate_add_f16
					e.Dispatch(pf.pSharedGateAdd, (M*H+255)/256*256, 256, expertDown, xF, moeLogits, uH, uMReal)
				}
			}
			continue
		}
		// pre-MLP norm. A parallel block (Cohere) has none: its MLP reads the same shared norm of the layer's input that the
		// attention read, which normF still holds (the o-proj added into xF directly), so x ends as x + attn + mlp.
		inFFN := normF
		switch {
		case r.parallelBlock && !prefillParallelDefectForTest:
		case r.parallelBlock: // Part B's planted defect: the MLP fed from the post-attention residual
			e.Dispatch(normPipe, M*tgReduceNorm, tgReduceNorm, xF, L.preNorm, normF, uH, uEpsX, r.uAddOne)
		case r.postOnly:
			inFFN = xF
		default:
			e.Dispatch(normPipe, M*tgReduceNorm, tgReduceNorm, xF, L.postNorm, normF, uH, uEpsX, r.uAddOne)
		}
		// S9 step 2: this layer's FFN width (a Gemma 4 E-model's KV-shared layers are double-wide; every other family's
		// layers are the model's I).
		fI := I
		if L.ffnI != 0 {
			fI = L.ffnI
		}
		if emodelBatchDefect == 6 && r.layers[0].ffnI != 0 { // G-S9c (6): every layer at the first layer's width
			fI = r.layers[0].ffnI
		}
		uFI, u2FI := uI, u2I
		if fI != I {
			uFI, u2FI = getU32(fI), getU32(2*fI)
		}
		// gate/up, then swiglu/geglu (G8: r.uAct selects, 0 for every family without FeatGatedGELU): one fused GEMM when the pass
		// takes it (swigluFused), else the GEMM into guF and swiglu_f16.
		if fusedSw {
			e.Dispatch2D(pf.pGemmSwiglu, (fI+31)/32, (Mpad+63)/64, 128, 1, inFFN, L.guW, L.guS, dqF, uM, uFI, uH, r.uAct)
		} else {
			gemm(e, Mpad, 2*fI, inFFN, L.guW, L.guS, guF, uM, u2FI, uH, dummyBias, m0)
			e.Dispatch(pf.pSw, M*fI, 256, guF, dqF, uFI, r.uAct)
		}
		// down-proj, same plain-vs-sandwich split as o-proj above.
		if r.sandwich || r.postOnly {
			gemm(e, Mpad, H, dqF, L.dW, L.dS, normF, uM, uH, uFI, dummyBias, m0)
			e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, normF, L.postMLPNorm, normF, uH, r.uEps, r.uAddOne)
			if uResidInv != (Buffer{}) {
				e.Dispatch(pf.pResScaled, M*H, 256, xF, normF, uResidInv)
			} else {
				e.Dispatch(pf.pRes, M*H, 256, xF, normF)
			}
		} else {
			gemm(e, Mpad, H, dqF, L.dW, L.dS, xF, uM, uH, uFI, dummyBias, m2)
		}
		// S9 step 2: a Gemma 4 E-model's PLE block, then its layer scalar, in decode's order (encodeLayerResidualWith):
		// the gate GEMM (H→P) on the raw residual, gelu(gate) times the row's own per-layer input, the projection GEMM
		// (P→H), the post-PLE norm and the residual add; then h *= layer_scalar.
		if L.pleGW != (Buffer{}) && emodelBatchDefect != 2 {
			P := r.pleP
			gemm(e, Mpad, P, xF, L.pleGW, L.pleGS, pleGF, uM, uP, uH, dummyBias, m0)
			e.Dispatch(pf.pPLEGeluMulF16, M*P, 256, pleGF, pleInB, uP, uLP, getU32(l*P))
			gemm(e, Mpad, H, pleGF, L.plePW, L.plePS, normF, uM, uH, uP, dummyBias, m0)
			e.Dispatch(pf.pRms, M*tgReduceNorm, tgReduceNorm, normF, L.postPLENorm, normF, uH, r.uEps, r.uAddOne)
			e.Dispatch(pf.pRes, M*H, 256, xF, normF)
		}
		if L.uLayerScalar != (Buffer{}) && emodelBatchDefect != 4 {
			e.Dispatch(pf.pLayerScaleF16, M*H, 256, xF, L.uLayerScalar)
		}
	}
	addDeep(r.nL - 1)
	// Final norm and LM head for the last token only, through the same int8-pinned head the decode path runs
	// (rmsnorm to int8, then gemv_w8a8): the head weights are int8 (logit-critical), and the int4 gemm_w4f16 would
	// misread them as packed nibbles and f16 scales, producing NaN logits. Norm-quant the last token's f16
	// residual row to int8, then run the decode head.
	finalNormQ := pf.pRmsQ
	if r.layerNorm {
		finalNormQ = pf.pLNQ
	}
	e.Dispatch(finalNormQ, tgReduceNorm, tgReduceNorm, xF.At((M-1)*H*2), r.finalNorm, r.aq, r.aSc, uH, uEpsX, r.uAddOne)
	e.Dispatch(r.pGemvW8, V*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
	e.End()
	// The int8 head turns a NaN row into finite zeros, which the caller's non-finite-logit check (A-C02) passes, so the
	// residual itself is checked. Its last row is enough: a row that overflows before the final layer writes NaN K/V,
	// which reaches the last row through attention; a row that overflows in the final layer writes no further K/V.
	last := xF.U16s()[(M-1)*H : M*H]
	if r.poisonPrefillResidForTest {
		last[0] = 0x7c00 // +inf
	}
	r.prefillResidNonFinite = false
	for _, h := range last {
		if h&0x7c00 == 0x7c00 {
			r.prefillResidNonFinite = true
			break
		}
	}
	if db02Trace != nil {
		db02Trace(db02Event{layer: r.nL, gpuStart: e.GPUStart(), gpuEnd: e.GPUEnd()})
	}
	r.recordExecErr(e.Err()) // C-09

	out := make([]float32, V)
	copy(out, r.logits.Floats()[:V])
	// Gemma's final-logit softcap: finalizeLogits (the decode path's entry point) applies it to r.logitsHost, but
	// this pass copies straight out of r.logits into a fresh slice and returns before finalizeLogits runs, so it
	// is applied here too. softcapParallel is a no-op for a non-softcapped family (0).
	if r.finalSoftcap > 0 {
		softcapParallel(out, r.finalSoftcap)
	}
	// Cohere's logit scale (FeatLogitScale), as finalizeLogits applies it for decode.
	if r.logitScale != 0 && r.logitScale != 1 {
		for i, v := range out {
			out[i] = v * r.logitScale
		}
	}
	return out
}

// prefillScratchU16 allocates one of a pass's f16 scratch buffers, n halves, zero-filled. Metal fills a new buffer
// with zeros itself (newBufferWithLength), so the pass does not build a zeroed Go slice of the same size and copy it
// in (prefillScratchCopy restores the copy, for the test that compares the two).
// THIS RELIES ON THE ZERO-FILL, which is a Metal property only: aikit's gpu.NewBufferLen* contract says Metal returns
// zeroed memory and CUDA uninitialized memory, and code shared across backends must not rely on zeros. These buffers
// are sized to Mpad rows and the pass writes M of them, so the pad rows hold the zeros Metal gave them. A CUDA port of
// this pattern must zero the buffer explicitly (Queue.ZeroAsync). TestPrefillScratch_zeroFilled pins the property on
// this backend.
func prefillScratchU16(d *Device, n int) Buffer {
	prefillScratchSeen.Add(int64(2 * n))
	if prefillScratchCopy {
		return NewBufferU16s(d, make([]uint16, n))
	}
	return gpu.NewBufferLenOf[uint16](d, n)
}

// prefillScratch is prefillScratchU16 for any element type: the f32 and u32 buffers that the router GEMM, the route
// kernels or the host fill before they are read (the expert-major MoE branch's moeLogits, moeIdx, moeWgt, rowIdxBuf,
// rowWgtBuf), and xF, which the embeddings are converted straight into. Same zero-fill reliance and the same copy arm,
// plus prefillR20Copy, which copies these alone so a test can time them apart from the other scratch.
func prefillScratch[T gpu.Scalar](d *Device, n int) Buffer {
	prefillScratchSeen.Add(int64(n) * int64(unsafe.Sizeof(*new(T))))
	if prefillScratchCopy || prefillR20Copy {
		return gpu.NewBufferOf(d, make([]T, n))
	}
	return gpu.NewBufferLenOf[T](d, n)
}

var prefillScratchCopy = false

// prefillR20Copy restores the copy for R-20's sites only (xF and the five MoE routing buffers), for the A/B that times
// them; tests only.
var prefillR20Copy = false

// prefillScratchPoison fills the expert-major branch's five routing buffers with 0xFF bytes (NaN as f32, an
// out-of-range expert id as u32) before the pass, set only by TestPrefillScratch_moeRoutingWrittenFirst: equal logits
// then prove the router GEMM, the route kernels and the host write every element before anything reads it, so the
// buffers need no zeros at all (R-20).
var prefillScratchPoison = false

// db02Trace is D-B02's probe (T1.12, metal/audit_db02_test.go): nil in production. The expert-major branch reports
// each command buffer it ends (gpuStart/gpuEnd, seconds; layer is the MoE layer whose routing it waits for, or nL for
// the pass's last buffer) and each active expert's rows (real, padded to 8, and padded to each GEMM's row tile).
var db02Trace func(db02Event)

type db02Event struct {
	layer, expert                        int
	gpuStart, gpuEnd                     float64
	rows, padRows, tileRowsGu, tileRowsD int
}

// dg01Mutation names a deliberate defect in the expert-major MoE prefill, for D-G01's gate to prove it can go red
// (metal/dg01_moe_gate_test.go). "" in production; set only by that test.
//   - "topk-renorm": the routing weights renormalised to sum to 1 per token, as a model with norm_topk_prob would
//     have them (Qwen1.5-MoE does not): the scatter weights wrong.
//   - "rank-swap": each token's first two experts swapped while their weights stay put, so each weight lands on the
//     other expert: the route's order wrong.
//   - "shared-off": the shared expert skipped.
var dg01Mutation string

func dg01Mutate(idx []uint32, wgt []float32, M, k int) {
	for m := range M {
		row, w := idx[m*k:(m+1)*k], wgt[m*k:(m+1)*k]
		switch dg01Mutation {
		case "topk-renorm":
			var sum float32
			for _, x := range w {
				sum += x
			}
			if sum > 0 {
				for j := range w {
					w[j] /= sum
				}
			}
		case "rank-swap":
			if k >= 2 {
				row[0], row[1] = row[1], row[0]
			}
		}
	}
}
