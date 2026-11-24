//go:build darwin

package metal

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	gpu "github.com/townsendmerino/goinfer/backend/metal/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// prefillSteelAttnOff, set only by tests, runs attention_prefill_fused where attention_prefill_steel would run: the
// baseline arm of the steel kernel's gate and of the decomposition harness's A/B.
var prefillSteelAttnOff bool

// prefillAttnSkipForTest, set only by the attention-share measurement (TestKSD1_attnShare), skips the batched pass's attention
// dispatch, so a timed pass without it prices attention. The pass's output is then meaningless.
var prefillAttnSkipForTest bool

// prefillAttnRanForTest, when a test sets it, counts the batched pass's attention launches by kernel name, so a test can
// assert which kernel ran rather than which it meant to select. Not safe for concurrent passes; nil in production.
var prefillAttnRanForTest map[string]int

// gemmTilePolicy picks gemmTile's rule: "" the default (the padding rule below), "a01" the small-tile rule as it first shipped,
// "shipped" every GEMM on gemm_w4f16_store's 64 × 64 tile, "bm32" / "bn32" only the 32-token / 32-feature tile rule.
// Test-only: the arms of the grades; not an option or an environment variable.
var gemmTilePolicy = ""

// gemmTile is the tile selector for a rows x N prefill GEMM. Every choice is bit-identical (gemm_w4f16_tile's comment,
// TestGemmTile_bitIdentical).
//
// The default: a threadgroup costs the same whatever share of its rows is real, so a smaller token tile is taken when
// it pads the pass to strictly fewer rows: 32 tokens up to 128 rows, 16 up to 16 rows and, when N <= 8192, up to 48.
// Past those, more row tiles cost more than the padding they save, and a gate|up GEMM as wide as the 1.5B's or 7B's
// already fills the cores at 64. A 16-token tile always takes 32 features; otherwise 32 features when N <= 2048 and rows <=
// 64, so the narrow GEMMs fill more cores.
//
// gemmTilePolicy "a01" is the older rule: 32 tokens when the pass has at most 32 rows, 32 features when N <= 2048 and
// rows <= 64.
func (pf *prefillState) gemmTile(rows, N int) (Pipeline, int, int) {
	return pf.gemmTileFor(rows, N, pf.w8)
}

// gemmTileFor is gemmTile for a weight of a given kind: w8 picks the W8 tiles, else the int4 ones. An
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

// swigluFusedOffForTest keeps the gate|up GEMM and swiglu_f16 as two steps with guF between them: the old arm of the fused-SwiGLU A/B.
var swigluFusedOffForTest bool

// swigluFused reports whether a pass of Mpad rows runs every dense layer's gate|up projection through the fused GEMM
// (gemm_w4f16_swiglu, or gemm_w8f16_swiglu on the native int8 lane), so it needs no guF: the 64 x 64 tile for every layer's
// width, a width that is a multiple of 32, and no MoE layer (its expert GEMMs write guF). The pass and prefillScratchBytes
// both ask it.
func (r *resident) swigluFused(Mpad int) bool {
	if swigluFusedOffForTest || r.moe != nil || r.q4kLane { // the lane's gate|up is segments of its own kinds (gemmSegs)
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
	pGemmSwiglu                                                Pipeline // gemm_w4f16_swiglu, gemm_w8f16_swiglu on a native int8 resident
	pKvI8, pDeqI8                                              Pipeline // an int8 cache's store and dequantize-to-scratch (kv_i8_prefill.go)
	pLN, pLNQ                                                  Pipeline // Part B: layernorm_f16 / layernorm_quant_f16 (bias-free LayerNorm)
	pAttnImg                                                   Pipeline // attention_prefill with a bidirectional image block
	pRopeM                                                     Pipeline // rope_mrope_f16
	pPLEGeluMulF16, pLayerScaleF16                             Pipeline // the E-model PLE gate and layer scalar
	// gemm_w4f16_tile at the smaller tiles (tokens × features), picked by gemmTile.
	pGemmM32N64, pGemmM64N32, pGemmM32N32 Pipeline
	pGemmM16N32                           Pipeline // the padding rule's 16-token tile
	// int8 slice 2: a native int8 resident's GEMMs, gemm_w4f16_tile<TM, TN,
	// true> by (TM, TN); w8 selects them in gemmTile.
	w8  bool
	pW8 map[[2]int]Pipeline
	// The MoE row loop's F32-scratch bridge (see
	// residual_f16_from_f32/zero_f32's own comments).
	pResF32, pZeroF32 Pipeline
	pResScaled        Pipeline // residual_f16_scaled, the add into a residual stored scaled by 1/s
	// attention_prefill_fused: the simdgroup_matrix flash-attention twin of pAttn.
	// On by default (metalFusedAttentionEnabled, backend.go);
	// GOINFER_METAL_FUSED_ATTENTION=0 or --exact-prefill falls back to pAttn.
	pAttnFused Pipeline
	// attention_prefill_steel, the prefill attention at the head dims steelHeadDim admits (it replaces pAttnFused there);
	// pAttnSteel64 is its head-dim-64 instance. steelFor picks by head dim.
	pAttnSteel, pAttnSteel64 Pipeline
	// attention_prefill_steel_split at head dims 256 and 512 (splitSteelHeadDim), where the exact kernel ran.
	pAttnSteel256, pAttnSteel512 Pipeline
	// attention_prefill_tiled: pAttn for a pass past prefillExactAttnMaxKeys keys (exactAttnTile).
	pAttnTiled Pipeline

	// Expert-major MoE prefill pipelines
	pRouterGemm       Pipeline
	pMoeRouteBatch    Pipeline
	pRouteGptOssBatch Pipeline
	pGatherRows       Pipeline
	pScatterAdd       Pipeline
	pSharedGateAdd    Pipeline

	// A gated-attention layer (Qwen3.5) on the batched lane; the DeltaNet mixer's own row kernels live in the
	// resident library (r.pDn*Rows), beside the decode kernels they copy.
	pQGateSplit, pCopyCols, pAttnGate Pipeline

	// The q4k lane: gemm_kq_tile by segment kind and tile (gemmKQ), and its head's f32 final norm (rmsnorm_f16_f32out).
	pKQ        map[kqTile]Pipeline
	pRmsF32Out Pipeline
}

// kqTile names one gemm_kq_tile instance: a q4kSeg kind and a tile of TM tokens by TN features.
type kqTile struct {
	kind   string
	tm, tn int
}

// gemmKQ is gemmTile for one segment of a q4k-lane projection: the same tile rule over the segment's rows (every tile is the
// same bits), on the segment kind's gemm_kq_tile.
func (pf *prefillState) gemmKQ(rows int, s q4kSeg) (Pipeline, int, int) {
	tm, tn := gemmTileSize(rows, s.rows)
	return pf.pKQ[kqTile{s.kind, tm, tn}], tm, tn
}

// ropeF16Name is the batched prefill's rope kernel: rope_f16, or rope_mrope_f16 for the m-RoPE pass, and their pairwise
// twins for a GPT-J pairwise model.
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
			mem.RepanicBug(p)
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
	if r.ksd7Old {
		src = "#define KSD7_OLD_DEQUANT 1\n" + src
	}
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
		pAttnFused: p("attention_prefill_fused"), pAttnSteel: p("attention_prefill_steel"), pAttnSteel64: p("attention_prefill_steel_hd64"),
		pAttnSteel256: p("attention_prefill_steel_hd256"), pAttnSteel512: p("attention_prefill_steel_hd512"), pAttnTiled: p("attention_prefill_tiled"),

		pRouterGemm:       p("router_gemm_f16"),
		pMoeRouteBatch:    p("moe_route_batch"),
		pRouteGptOssBatch: p("route_gptoss_batch"),
		pGatherRows:       p("gather_rows_f16"),
		pScatterAdd:       p("scatter_add_weighted_f16"),
		pSharedGateAdd:    p("shared_gate_add_f16"),

		pQGateSplit: p("qgate_split_f16"), pCopyCols: p("copy_cols_f16"), pAttnGate: p("attn_gate_f16"),
	}
	if r.w8 || r.w8Attn || r.mamba != nil { // a Mamba layer's in_proj and out_proj are int8
		r.pf.w8, r.pf.pW8 = r.w8, map[[2]int]Pipeline{}
		for _, t := range [][2]int{{64, 64}, {32, 64}, {64, 32}, {32, 32}, {16, 32}} {
			r.pf.pW8[t] = p(fmt.Sprintf("gemm_w8f16_m%dn%d", t[0], t[1]))
		}
		if r.w8 { // an int4mix resident's FFN is int4 and keeps gemm_w4f16_swiglu
			r.pf.pGemmSwiglu = p("gemm_w8f16_swiglu")
		}
	}
	if r.q4kLane {
		r.pf.pKQ, r.pf.pRmsF32Out = map[kqTile]Pipeline{}, p("rmsnorm_f16_f32out")
		for _, kind := range []string{"q4k", "q5k", "q6k", "int8"} {
			for _, t := range [][2]int{{64, 64}, {32, 64}, {64, 32}, {32, 32}, {16, 32}} {
				r.pf.pKQ[kqTile{kind, t[0], t[1]}] = p(fmt.Sprintf("gemm_kq_%s_m%dn%d", kind, t[0], t[1]))
			}
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

// parallelEmbedsF32ToF16Scaled is parallelEmbedsF32ToF16 with every value times inv first (the image prefill's scaled residual; inv is
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
		wg.Go(func() {
			for m := lo; m < hi; m++ {
				row := embs[m]
				for i := range H {
					dst[m*H+i] = f32ToF16(row[i] * inv)
				}
			}
		})
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
// expert buffers (for a paged one, the layer-major pass's batched phase 2), an E-model's per-layer inputs and a Gated-DeltaNet
// or Mamba-2 hybrid's mixer buffers. It leaves out the
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
	if e := r.moe; e != nil && e.paged && r.moeExpertBatchOK() { // the layer-major pass's batched phase 2, one chunk at a time
		n += r.moeBatchBytes(min(M, g4LayerMajorChunk))
	}
	if r.kvI8 { // the f16 K and V scratch one layer's int8 cache is dequantized into, the padded context long (kvI8Scratch)
		n += 4 * int64(r.kvI8ScratchRows()) * int64(r.prefillGeom().kvDim)
		n += 4 * int64(kvBlocksFor(r.kvI8ScratchRows())) // its identity table (kvIdentityTab)
	}
	if dp := r.dnet; dp != nil {
		g0 := r.prefillGeom()
		nHhd, kvDim := int64(r.nH*g0.hd), int64(g0.kvDim)
		n += 2*mp*int64(dp.convDim+2*dp.valueDim) + 4*m*int64(4*dp.nv+dp.convDim+2*dp.keyDim+2*dp.valueDim)
		n += 2*mp*(2*nHhd+2*kvDim) + 2*m*nHhd
	}
	if r.mamba != nil {
		n += r.prefillMambaBytes(m, mp)
	}
	return n
}

// kvIdentityTab is a block table naming blocks 0, 1, ... in order over rows rows of contiguous scratch, so a kernel that reads keys
// through a table reads that scratch as it lies. Counted with the pass's scratch (prefillScratch).
func kvIdentityTab(d *Device, rows int) Buffer {
	b := prefillScratch[uint32](d, kvBlocksFor(rows))
	for i, u := 0, b.U32s(); i < len(u); i++ {
		u[i] = uint32(i)
	}
	return b
}

// kvI8ScratchRows is the rows of an int8 cache's dequantize scratch: the cache's own padded capacity, so the attention
// kernels' whole-tile reads past the last key stay inside it, as they do in the cache.
func (r *resident) kvI8ScratchRows() int { return (r.ctxCap + 7) / 8 * 8 }

// prefillScratchSeen counts the bytes of row-sized scratch the batched pass has allocated, at the allocations
// themselves, so a test can hold prefillScratchBytes to what a pass really takes.
var prefillScratchSeen atomic.Int64

// prefillAttnKernels is which attention kernel PrefillLast dispatches for this model: fused for hd%8==0 && hd<=128
// (attention_prefill_fused's compile-time cap, ATTN_MAXHD), steel within that at the head dims steelHeadDim admits
// (one threadgroup of 128 threads per query head and 32 rows), and otherwise the exact attention_prefill, bounded by
// prefillExactAttnMaxKeys.
func (r *resident) prefillAttnKernels() (fused, steel bool) {
	g0 := r.prefillGeom()
	fused = r.prefillFusedOn() && g0.hd%8 == 0 && g0.hd <= 128
	steel = fused && steelHeadDim(g0.hd) && !prefillSteelAttnOff
	return fused, steel
}

// steelHeadDim reports whether attention_prefill_steel runs a layer of head dim hd. The kernel is built at 64, 80, 96 and
// 128, but a head dim is admitted only after a model at it passes set A's pooled gate, as 128 and 64 have.
func steelHeadDim(hd int) bool { return hd == 128 || hd == 64 }

// steelName is steelFor's kernel name.
func steelName(hd int) string {
	if hd == 64 {
		return "attention_prefill_steel_hd64"
	}
	return "attention_prefill_steel"
}

// steelFor is the steel pipeline for head dim hd, one of steelHeadDim's.
func (pf *prefillState) steelFor(hd int) Pipeline {
	if hd == 64 {
		return pf.pAttnSteel64
	}
	return pf.pAttnSteel
}

// prefillFusedOn reports whether the pass may take the fused and steel attention kernels where a layer's head
// dimension has one. A model with an attention softcap may not: the softcap is gated on the exact kernel.
func (r *resident) prefillFusedOn() bool {
	return r.attnSoftcap == 0 && metalFusedAttentionEnabled(r.knobValue("GOINFER_METAL_FUSED_ATTENTION"))
}

// prefillExactAttn reports whether any attention layer of the pass runs the exact attention kernel (no fused kernel for
// its head dim, or fused attention off), whose score buffer holds prefillExactAttnMaxKeys keys. A Gemma 4 E-model's
// layers differ; every other family's are one geometry.
func (r *resident) prefillExactAttn() bool {
	fusedOn := r.prefillFusedOn()
	for l := range r.layers {
		if g := r.layers[l].geom; g != nil && exactAttnLayer(g.hd, fusedOn) {
			return true
		}
	}
	return false
}

// splitSteelHeadDim reports whether attention_prefill_steel_split serves head dim hd.
func splitSteelHeadDim(hd int) bool { return hd == 256 || hd == 512 }

// exactAttnLayer reports whether an attention layer of head dim hd runs the exact kernel: no fused kernel (head dim over 128 or
// not a multiple of 8) and no split steel kernel, or fused attention off. prefillSteelAttnOff returns the split head dims to
// it, as the A/B's old arm.
func exactAttnLayer(hd int, fusedOn bool) bool {
	return !(fusedOn && hd%8 == 0 && hd <= 128) && !(fusedOn && splitSteelHeadDim(hd) && !prefillSteelAttnOff)
}

// exactAttnHeadDim is the widest head dim among the attention layers the exact kernel serves (prefillExactAttn), or 0.
func (r *resident) exactAttnHeadDim() int {
	fusedOn, hd := r.prefillFusedOn(), 0
	for l := range r.layers {
		if g := r.layers[l].geom; g != nil && exactAttnLayer(g.hd, fusedOn) {
			hd = max(hd, g.hd)
		}
	}
	return hd
}

// prefillGeom is the attention geometry the pass uses for every attention layer: the first attention layer's. A
// Gated-DeltaNet hybrid's layer 0 is a DeltaNet layer, which has none; every other admitted family's layer 0
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

// prefillDeep is Qwen3-VL's DeepStack injection for the batched pass: set l is added to the image rows [start,
// start+n) of the f16 residual after decoder layer l, as decoder's addDeepstack does on the CPU.
type prefillDeep struct {
	start, n int
	sets     [][]float32
}

// prefillParallelDefectForTest is a planted defect: a
// parallel block's MLP fed from the post-attention residual instead of the layer's shared input norm. Tests set it.
var prefillParallelDefectForTest bool

// prefillResidScaleForTest, when > 0, replaces prefillResidScale's s for every family it applies to (1 turns the scale
// off: G-RS2's planted defect; a large s measures the residual's peak).
var prefillResidScaleForTest float32

// gemmaPrefillResidScale is s for the families prefillResidScale applies to: the smallest power of two that leaves 4x
// headroom under f16's 65,504 over Gemma 3 4B's residual peak (its <bos> row after layer 31;
// docs/code-notes/metal.md#gemmaPrefillResidScale).
const gemmaPrefillResidScale = 32

// prefillResidScale is the s the batched pass stores its residual divided by: Gemma 3 4B's residual outgrows f16 (inf at layer 6). It applies where
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
// (the Metal image prefill; several blocks for several images): every layer's attention then runs
// attention_prefill_img, the exact kernel with the blocks, in place of the fused and steel kernels, which carry no block
// mask. No blocks: prefillLast exactly.
func (r *resident) prefillLastImg(embs [][]float32, startPos int, mrope [][3]int, deep *prefillDeep, blocks [][2]int) []float32 {
	return r.prefillLastImgSrc(sliceRows(embs), startPos, mrope, deep, blocks)
}

// prefillLastImgSrc is prefillLastImg over a row source (rows.go): its input rows are read rowChunk at a time straight into the
// pass's device buffers. It is the guards, then the pass's phases (prefillPass): its buffers, the layers, and the
// last row's head.
func (r *resident) prefillLastImgSrc(src rowSource, startPos int, mrope [][3]int, deep *prefillDeep, blocks [][2]int) []float32 {
	// Every K/V buffer holds kvCap positions, and a pass past them writes beyond the buffer. metalResident.checkCap refuses that for
	// the decoder; an internal caller that skipped it is a programming error, named here rather than left to corrupt memory.
	if c := r.kvCap(); startPos < 0 || startPos+src.n > c {
		panic(fmt.Sprintf("metal: internal prefill of %d rows at position %d is past the resident KV cap of %d positions "+
			"(metalResident.checkCap guards the decoder's path; this caller skipped it)", src.n, startPos, c))
	}
	r.ensurePrefill()
	r.stopExec() // the pass runs beside the executor: a pipelined forward may be in flight reading the table or K/V
	if err := r.kvWrite(r.kvSlot, startPos, startPos+src.n); err != nil {
		r.recordExecErr(err)
		return nil
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	ps := r.newPrefillPass(src, startPos, mrope, deep, blocks)
	defer ps.release()
	ps.begin()
	if !ps.layers() {
		return nil
	}
	return ps.lastRowLogits()
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
		return NewBufferU16sOn(mem.LineScratch, d, make([]uint16, n))
	}
	return gpu.NewBufferLenOfOn[uint16](d, mem.LineScratch, n)
}

// prefillScratch is prefillScratchU16 for any element type: the f32 and u32 buffers that the router GEMM, the route
// kernels or the host fill before they are read (the expert-major MoE branch's moeLogits, moeIdx, moeWgt, rowIdxBuf,
// rowWgtBuf), and xF, which the embeddings are converted straight into. Same zero-fill reliance and the same copy arm,
// plus prefillR20Copy, which copies these alone so a test can time them apart from the other scratch.
func prefillScratch[T gpu.Scalar](d *Device, n int) Buffer {
	prefillScratchSeen.Add(int64(n) * int64(unsafe.Sizeof(*new(T))))
	if prefillScratchCopy || prefillR20Copy {
		return gpu.NewBufferOfOn(d, mem.LineScratch, make([]T, n))
	}
	return gpu.NewBufferLenOfOn[T](d, mem.LineScratch, n)
}

var prefillScratchCopy = false

// prefillR20Copy restores the copy for the zero-filled sites only (xF and the five MoE routing buffers), for the A/B that times
// them; tests only.
var prefillR20Copy = false

// prefillScratchPoison fills the expert-major branch's five routing buffers with 0xFF bytes (NaN as f32, an
// out-of-range expert id as u32) before the pass, set only by TestPrefillScratch_moeRoutingWrittenFirst: equal logits
// then prove the router GEMM, the route kernels and the host write every element before anything reads it, so the
// buffers need no zeros at all.
var prefillScratchPoison = false

// db02Trace is the expert-major MoE prefill's probe (metal/audit_db02_test.go): nil in production. The expert-major branch reports
// each command buffer it ends (gpuStart/gpuEnd, seconds; layer is the MoE layer whose routing it waits for, or nL for
// the pass's last buffer) and each active expert's rows (real, padded to 8, and padded to each GEMM's row tile).
var db02Trace func(db02Event)

type db02Event struct {
	layer, expert                        int
	gpuStart, gpuEnd                     float64
	rows, padRows, tileRowsGu, tileRowsD int
}

// dg01Mutation names a deliberate defect in the expert-major MoE prefill, for the MoE routing gate to prove it can go red
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

// ksd7OldDequantForTest builds the prefill library with the per-weight nibble dequant in the GEMM's staging (KSD7_OLD_DEQUANT): the
// old arm of the in-GEMM dequant's A/B. The staging is in prefillKernels, so the define goes on the prefill library's source; the resident takes
// the switch when it is built (resident.ksd7Old).
var ksd7OldDequantForTest bool
