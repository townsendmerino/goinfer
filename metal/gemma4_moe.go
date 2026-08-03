//go:build darwin

package metal

import (
	"fmt"
	"os"
	"strconv"

	"github.com/townsendmerino/aikit/mmap"
	"github.com/townsendmerino/goinfer/decoder"
	"golang.org/x/sys/unix"
)

// Gemma-4 enable_moe_block (26B-A4B) MSL kernels — the parallel dense‖MoE FFN that the generic
// moe.go path (Mixtral/Qwen/GLM shape) cannot express. Kept in their OWN file/const, concatenated
// after moeKernels, so moe.go's audited MoE kernels are not touched — the same discipline as CUDA's
// separate router_f32.cu (which exists so adding a kernel doesn't rewrite the audited moe.ptx).
//
// The router-first step (9c Step 5a) added gemv_f32_f32. This block adds the remaining primitives
// the parallel dense‖MoE forward needs — each verified in isolation against a CPU oracle before the
// composition is wired (gemma4_moekernels_test.go), the same 2b→2c→2d order CUDA followed.
const gemma4MoeKernels = `
// gemv_f32_f32: pure-f32 GEMV — f32 weight [N,K] × f32 activation [K] → out[N], one simdgroup (32
// lanes) per output row. This is the Gemma-4 MoE ROUTER projection, and it quantizes NOTHING on
// purpose. The router is the one DISCRETE-failure path in the whole MoE delta: a quant error near a
// top-k tie picks a DIFFERENT expert — a cliff, not a small error. gemv_wf32_a8 (moe.go) quantizes
// the activation to int8 (~1e-2), which can flip a decision near a tie; safe only when the margin is
// wide, and the tiny fixture's margin was CONSTRUCTED wide, so "no flip" there would be circular for
// a trained 128-expert/top-8 router whose 8th-vs-9th boundary is far tighter. Quantizing nothing
// removes the perturbation entirely: this is bit-exact to the CPU f32 router modulo f32 reduction
// order (~1e-6), so routing cannot flip from activation quant at ANY expert count — routing is off
// the resident suspect list permanently. Mirrors cuda/router_f32.cu gemv_f32_f32.
kernel void gemv_f32_f32(device const float* wf[[buffer(0)]], device const float* a[[buffer(1)]],
    device float* out[[buffer(2)]], constant uint& K[[buffer(3)]],
    uint gid[[threadgroup_position_in_grid]], uint lid[[thread_index_in_threadgroup]]) {
    device const float* wr = wf + (uint)gid*K;
    float acc = 0.0f;
    for (uint k=lid; k<K; k+=32u) acc += wr[k]*a[k];
    acc = simd_sum(acc);
    if (lid==0) out[gid] = acc;
}

// rmsnorm_nw: weightless RMSNorm, OUT-OF-PLACE (src → dst). The Gemma-4 MoE router norms the RAW
// residual h WITHOUT mutating it — h still feeds the parallel dense branch, the expert branch, and
// the final residual add — so the in-place rmsnorm_f32 cannot be used here. dst[i] = src[i] *
// rsqrt(mean(src^2)+eps). The learned routerScale and hidden^-0.5 are folded into the router weight
// columns at build (RouterProjScaled), so nothing else is applied here. One threadgroup, tree
// reduction (matches rmsnorm_f32's reduction so the two norms of h agree to f32 order). Mirrors
// cuda/router_f32.cu rmsnorm_nw.
kernel void rmsnorm_nw(device const float* src[[buffer(0)]], device float* dst[[buffer(1)]],
    constant uint& H[[buffer(2)]], constant float& eps[[buffer(3)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256]; float ss=0;
    for(uint i=tid;i<H;i+=tgs) ss+=src[i]*src[i];
    red[tid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2;s>0;s>>=1){ if(tid<s) red[tid]+=red[tid+s]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    float rms=rsqrt(red[0]/float(H)+eps);
    for(uint i=tid;i<H;i+=tgs) dst[i]=src[i]*rms;
}

// scale_wgt_by_expert: fold Gemma-4's learned per-expert scale into the routed weights, AFTER
// moe_route's top-k + renormalize: wgt[k] *= perExpertScale[idx[k]] (CPU: wts[j] =
// (topv[j]/sum)*perExpertScale[idx[j]]). idx is a moe_route device output, so this must run ON-GPU:
// a host fold would read idx/wgt back per token, reintroducing the per-token sync the on-device
// router exists to avoid. K lanes (K = top_k, tiny), one dispatch. Mirrors cuda/router_f32.cu.
kernel void scale_wgt_by_expert(device float* wgt[[buffer(0)]], device const uint* idx[[buffer(1)]],
    device const float* perExpertScale[[buffer(2)]], constant uint& K[[buffer(3)]],
    uint k[[thread_position_in_grid]]) {
    if (k>=K) return;
    wgt[k] *= perExpertScale[idx[k]];
}

// scale_vec: x[i] *= s. Gemma-4's per-layer output scalar (out = (h + combined) * layerScalar),
// applied to the residual after the joint post-norm. s is a one-float buffer (Metal uniforms are
// buffers) so it can be per-layer. Mirrors cuda/router_f32.cu scale_vec.
kernel void scale_vec(device float* x[[buffer(0)]], device const float* s[[buffer(1)]],
    uint i[[thread_position_in_grid]]) { x[i] *= s[0]; }

// zero_vec: x[i] = 0. Clears the Gemma-4 expert accumulator g4x2 before the fixed-k weighted-
// accumulate loop (gemv_w4a8_moe_wacc always does out[row] += ...). A multiply-by-zero would NOT
// do — g4x2 is persistent scratch that can hold a stale NaN on the first token (NaN*0 = NaN).
kernel void zero_vec(device float* x[[buffer(0)]], uint i[[thread_position_in_grid]]) { x[i] = 0.0f; }
`

// gemma4MoeResident holds the Gemma-4 dense‖MoE pipelines, config, constant uniforms, and scratch
// shared across all enable_moe_block layers. Kept SEPARATE from moeResident (moe.go): that is the
// generic Mixtral/Qwen/GLM shape (one branch, SiLU experts, wacc straight into the residual), which
// gemma4 is not. Routing is off the suspect list (Step 5a); this is the rest of the delta.
type gemma4MoeResident struct {
	pRouterF32, pRoute, pGU, pDownWacc  Pipeline
	pRmsNW, pScaleWgt, pScaleVec, pZero Pipeline

	nE, topK, denseInter, moeInter int
	uNE, uK, uHidden               Buffer // moe_route nE/k; router GEMV K = hidden
	uDenseInter, uMoeInter         Buffer // swiglu I (dense vs expert intermediate)
	uMoeGU                         Buffer // 2*moeInter — expert fused gate|up rows/width
	uSig0, uNorm1, uScale1, uOne   Buffer // route uniforms: softmax(0), unconditional renorm(1), scale 1.0, nGroup/topkGroup=1
	uSlot                          []Buffer

	rLogits, rIdx, rWgt Buffer // router scratch: logits[nE], idx[k] (u32), wgt[k]
	g4x1, g4x2, g4rn    Buffer // dense-branch out, expert-branch accumulator, router-norm input — all [hidden]

	// Synchronous paging (GOINFER_METAL_MOE_SLOTS=N>0): the full expert set doesn't fit resident, so
	// each layer keeps N experts in a slot pool and stages the routed top-k in per token. Off (all
	// experts resident) when slots==0. idxZeros is a [topK] all-zero index buffer so the reused
	// gemv_w4a8_moe(_wacc) kernels read row 0 of a single-expert SLOT buffer while still indexing
	// rWgt by the selection slot — makes paged dispatch byte-identical to the stacked path.
	paged    bool
	slots    int
	idxZeros Buffer

	// giwFile is the re-opened .giw for pread-staging (GOINFER_MOE_PREAD=1); nil ⇒ the mmap byte-copy
	// path. Shared read-only fd across every layer's pool; closed by Resident.Close.
	giwFile *os.File
}

// gemma4MoeLayer holds one enable_moe_block layer's device weights: the parallel dense MLP
// (fused gate|up + down, W4A8), the f32 router (RouterProjScaled, scale folded) + zero bias +
// per-expert scale, the stacked all-E experts (fused gate|up + down, W4A8), the five RMSNorm
// weights, and the per-layer output scalar (a one-float buffer so scale_vec is per-layer).
type gemma4MoeLayer struct {
	routerW, routerBias, perExpertScale          Buffer
	denseGuW, denseGuS, denseDW, denseDS         Buffer
	expGuW, expGuS, expDW, expDS                 Buffer // stacked all-E (non-paged); zero when paged
	preFFN, postFFN1, preFFN2, postFFN2, postFFN Buffer
	uLayerScalar                                 Buffer
	pool                                         *expertPool // non-nil when paged: LRU slot pool + on-demand staging
}

// buildGemma4MoE builds the Resident-level Gemma-4 MoE state (pipelines, config, uniforms, scratch)
// from a model that declares enable_moe_block. Returns nil when the model has no gemma4 MoE layer,
// and an error for a shape this path cannot express (so BuildResident declines → CPU fallback rather
// than mis-running). Mirrors cuda/backend.go's isG4MoE build and its int4-width shape checks.
func buildGemma4MoE(d *Device, m *decoder.Model, pipe func(string) Pipeline, H, nL int) (*gemma4MoeResident, error) {
	if !m.HasGemma4MoEResident() {
		return nil, nil
	}
	var b decoder.Gemma4MoEResidentBundle
	var ok bool
	for l := 0; l < nL; l++ {
		if b, ok = m.Gemma4MoEResidentLayer(l); ok {
			break
		}
	}
	if !ok {
		return nil, fmt.Errorf("metal gemma4 MoE: HasGemma4MoEResident but no layer bundle")
	}
	// int4/W4A8 width checks: the GEMVs stride by the group size (hidden) and the 8-nibble word
	// (moeInter/denseInter), so all three must be multiples of 32. Same checks as cuda/backend.go.
	if H%32 != 0 || b.MoeInter%32 != 0 || b.DenseInter%32 != 0 {
		return nil, fmt.Errorf("metal gemma4 MoE int4 needs hidden(%d), moeInter(%d), denseInter(%d) all multiples of 32", H, b.MoeInter, b.DenseInter)
	}
	if b.NE > 256 {
		return nil, fmt.Errorf("metal gemma4 MoE nE=%d exceeds moe_route cap 256", b.NE)
	}
	g := &gemma4MoeResident{
		pRouterF32: pipe("gemv_f32_f32"), pRoute: pipe("moe_route"),
		pGU: pipe("gemv_w4a8_moe"), pDownWacc: pipe("gemv_w4a8_moe_wacc"),
		pRmsNW: pipe("rmsnorm_nw"), pScaleWgt: pipe("scale_wgt_by_expert"),
		pScaleVec: pipe("scale_vec"), pZero: pipe("zero_vec"),
		nE: b.NE, topK: b.TopK, denseInter: b.DenseInter, moeInter: b.MoeInter,
	}
	g.uNE, g.uK = d.NewBufferU32(uint32(b.NE)), d.NewBufferU32(uint32(b.TopK))
	g.uHidden = d.NewBufferU32(uint32(H))
	g.uDenseInter, g.uMoeInter = d.NewBufferU32(uint32(b.DenseInter)), d.NewBufferU32(uint32(b.MoeInter))
	g.uMoeGU = d.NewBufferU32(uint32(2 * b.MoeInter))
	g.uSig0, g.uNorm1 = d.NewBufferU32(0), d.NewBufferU32(1)
	g.uScale1 = d.NewBufferFloats([]float32{1})
	g.uOne = d.NewBufferU32(1)
	g.uSlot = make([]Buffer, b.TopK)
	for j := range g.uSlot {
		g.uSlot[j] = d.NewBufferU32(uint32(j))
	}
	g.rLogits = d.NewBufferLen(b.NE)
	g.rIdx = d.NewBufferUint32s(make([]uint32, b.TopK))
	g.rWgt = d.NewBufferLen(b.TopK)
	g.g4x1, g.g4x2, g.g4rn = d.NewBufferLen(H), d.NewBufferLen(H), d.NewBufferLen(H)

	// Synchronous paging: GOINFER_METAL_MOE_SLOTS=N keeps only N experts/layer resident and stages the
	// routed top-k in per token (the only way the 26B's 11.96 GB expert set runs on a 16 GB Mac). N
	// must be >= topK (a token's own top-k must fit). N==0 / unset ⇒ all experts resident (the fitting
	// path + the paged≡non-paged parity reference). idxZeros lets the paged expert GEMVs read row 0 of
	// a single-expert slot buffer while rWgt is still indexed by the selection slot (byte-identical).
	if s := os.Getenv("GOINFER_METAL_MOE_SLOTS"); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil || n < b.TopK {
			return nil, fmt.Errorf("GOINFER_METAL_MOE_SLOTS=%q invalid (need integer >= topK=%d)", s, b.TopK)
		}
		if n < b.NE { // n>=nE would hold every expert — no paging, just build the stacked path
			g.paged, g.slots = true, n
			g.idxZeros = d.NewBufferUint32s(make([]uint32, b.TopK))
		}
	}
	// Stage experts by pread'ing their nibbles straight into the slot buffers instead of a byte-copy
	// off the mmap. DEFAULT ON (cold A/B verdict: total 1892→1488 ms/tok, 0.53→0.67 tok/s, 1.26×;
	// major faults 92.8→0.0/stage — the mmap demand-fault page-in is gone). Needs a .giw-mmap'd model
	// (offsets are into that file); re-open it once, shared across layers. GOINFER_MOE_PREAD=0 opts out
	// (the mmap byte-copy baseline, kept for the A/B). If open fails or the model isn't .giw-backed,
	// buildGemma4MoELayer falls back to the byte-copy.
	if g.paged && os.Getenv("GOINFER_MOE_PREAD") != "0" {
		if p := m.GiwPath(); p != "" {
			if f, err := os.Open(p); err == nil {
				g.giwFile = f
				// GOINFER_MOE_NOCACHE=1: F_NOCACHE on the pread fd — reads bypass the unified buffer
				// cache. MEASURED AND DECLINED (cold A/B): NO effect — total 1535.3→1514.0 ms/tok
				// (−1.4%, noise), and RSS-after-build (3306→5095) and compute+coord (1032→1056) both
				// moved the WRONG way. The motivating cache-pressure hypothesis is refuted, AND its
				// evidence was a CONFOUND: the ~+2 GB RSS-after-build growth lands on whichever arm runs
				// SECOND, and RSS-after-build is sampled BEFORE the timed decode, so the pread cache flag
				// cannot cause it (ordering artifact). The pread win's +365 ms compute+coord displacement
				// remains UNEXPLAINED. Kept off by default, wired, so it isn't re-proposed.
				if os.Getenv("GOINFER_MOE_NOCACHE") == "1" {
					if _, err := unix.FcntlInt(f.Fd(), unix.F_NOCACHE, 1); err != nil {
						fmt.Fprintf(os.Stderr, "metal gemma4 MoE: F_NOCACHE failed (%v) — pread stays buffered\n", err)
					}
				}
			} else {
				fmt.Fprintf(os.Stderr, "metal gemma4 MoE: open(%s) failed (%v) — using mmap byte-copy\n", p, err)
			}
		}
	}
	return g, nil
}

// buildGemma4MoELayer packs one enable_moe_block layer's weights. The dense MLP and the experts are
// row-stacked into fused gate|up buffers via int4Concat (gate rows then up rows, per the swiglu
// gate@0/up@inter split); the router (RouterProjScaled, f32) has routerScale·hidden^-0.5 folded into
// its columns at build (decoder Gemma4MoEResidentLayer), so the resident router is rmsnorm_nw(h) →
// gemv_f32_f32(RouterProjScaled) — the algebraic dual of the CPU's scaled-rn · raw-proj.
func buildGemma4MoELayer(d *Device, m *decoder.Model, b *decoder.Gemma4MoEResidentBundle, g *gemma4MoeResident) *gemma4MoeLayer {
	ml := &gemma4MoeLayer{}
	ml.routerW = d.NewBufferFloats(b.RouterProjScaled)
	ml.routerBias = d.NewBufferFloats(make([]float32, b.NE)) // zeros → sel = score (no e_score_correction_bias)
	ml.perExpertScale = d.NewBufferFloats(b.PerExpertScale)
	ml.denseGuW, ml.denseGuS = int4Concat(d, b.MlpGate, b.MlpUp)
	var e error
	if ml.denseDW, ml.denseDS, e = int4Buf(d, b.MlpDown); e != nil {
		panic(e)
	}
	if g.paged {
		// Paged: DON'T stack the experts (that is the 11.96 GB we can't afford). Build a bounded LRU
		// slot pool + a stage fn that reads expert e's W4A8 bytes straight from the bundle's mmap-backed
		// WeightMats on demand (int4DirectWords aliases the .giw). Per-expert buffer sizes come from
		// expert 0. The .giw expert weights MUST be int4-direct (group-32) for this zero-intermediate path.
		gw0, gs0, ok1 := int4DirectWords(b.ExpertsGateUp[0])
		dw0, ds0, ok2 := int4DirectWords(b.ExpertsDown[0])
		if !ok1 || !ok2 {
			panic("metal gemma4 MoE paging: experts are not int4-direct (group-32) — cannot stage without a re-quant")
		}
		experts := b.ExpertsGateUp // capture (aliases the model mmap; kept alive by the Model)
		down := b.ExpertsDown
		stage := func(ei int) ([]byte, []uint16, []byte, []uint16) {
			gw, gs, _ := int4DirectBytes(experts[ei]) // nibble bytes aliased from mmap; no reconstruction/alloc
			dw, ds, _ := int4DirectBytes(down[ei])
			return gw, gs, dw, ds
		}
		ml.pool = newExpertPool(d, g.slots, len(gw0), len(gs0), len(dw0), len(ds0), stage)
		// GOINFER_MOE_WILLNEED=1 issues MADV_WILLNEED over the routed experts' nibble spans before
		// staging. MEASURED AND DECLINED (cold A/B, sudo purge between arms, 2026-08-03): it does NOT
		// help. major faults/stage ROSE 92.5→147.1 (readahead read MORE from disk, did not batch the
		// serial faults), and the staging-line drop (−775 ms) was an ATTRIBUTION SHIFT into the residual
		// compute+coord bucket (+700 ms), for a total move of 1851.8→1776.6 ms/tok (~4%, noise band).
		// WILLNEED targets fault LATENCY/overlap, but the bound here is fault COUNT, and darwin
		// MADV_WILLNEED on a span you touch immediately pays the read synchronously in the Advise call —
		// the synchronous per-layer Advise→copy path has no lead time to hide it behind (the routed
		// experts aren't known until phase-1 completes: the value-dependent seam). The extra faults are
		// consistent with prefetched pages being evicted under memory pressure before use (RSS fell
		// 3369→2870 MB). Kept OFF by default and wired so nobody re-proposes it without re-reading this.
		// Lesson: when you optimize a measured sub-bucket, the TOTAL is the gate — a residual bucket
		// absorbs displaced cost silently. See [[optimize-sub-bucket-total-is-the-gate]].
		if os.Getenv("GOINFER_MOE_WILLNEED") == "1" {
			ml.pool.prefetch = func(ei int) {
				if q4, _, _, ok := experts[ei].Int4(); ok {
					_ = mmap.Advise(mmap.PageAlignedInterior(q4), true)
				}
				if q4, _, _, ok := down[ei].Int4(); ok {
					_ = mmap.Advise(mmap.PageAlignedInterior(q4), true)
				}
			}
		}
		// pread staging (GOINFER_MOE_PREAD=1): resolve each expert's nibble file offset within the .giw
		// mmap (pure pointer arithmetic, no page touch), then stage by pread'ing straight into the slot's
		// UMA words — zero mmap faults, one big sequential read per expert. Scales stay f32→f16 from the
		// heap-resident q4s (giwReader.f32 COPIES them, so they never fault the mmap). Falls back to the
		// byte-copy path if the fd is absent or ANY expert's nibbles aren't .giw-mmap-backed (e.g. a
		// requantized HF load) — the offsets must all resolve for pread to be correct.
		if g.giwFile != nil {
			guOff := make([]int64, len(experts))
			dOff := make([]int64, len(down))
			resolved := true
			for ei := range experts {
				gq, _, _, ok1 := experts[ei].Int4()
				dq, _, _, ok2 := down[ei].Int4()
				if !ok1 || !ok2 {
					resolved = false
					break
				}
				go1, okg := m.MmapByteOffset(gq)
				do1, okd := m.MmapByteOffset(dq)
				if !okg || !okd {
					resolved = false
					break
				}
				guOff[ei], dOff[ei] = go1, do1
			}
			if resolved {
				fd := int(g.giwFile.Fd())
				ml.pool.stagePread = func(ei int, s expertSlot) {
					if err := preadIntoU32Buf(fd, s.guW, guOff[ei]); err != nil {
						panic(fmt.Sprintf("metal gemma4 MoE pread gate|up expert %d: %v", ei, err))
					}
					if err := preadIntoU32Buf(fd, s.dW, dOff[ei]); err != nil {
						panic(fmt.Sprintf("metal gemma4 MoE pread down expert %d: %v", ei, err))
					}
					_, gs, _ := int4DirectBytes(experts[ei]) // f16 scales from heap q4s (no mmap fault)
					_, ds, _ := int4DirectBytes(down[ei])
					copy(s.guS.U16s(), gs)
					copy(s.dS.U16s(), ds)
				}
			}
		}
	} else {
		ml.expGuW, ml.expGuS = int4Concat(d, b.ExpertsGateUp...) // per expert already fused [2*moeInter, hidden]
		ml.expDW, ml.expDS = int4Concat(d, b.ExpertsDown...)
	}
	ml.preFFN = d.NewBufferFloats(b.PreFFNNorm)
	ml.postFFN1 = d.NewBufferFloats(b.PostFFNNorm1)
	ml.preFFN2 = d.NewBufferFloats(b.PreFFNNorm2)
	ml.postFFN2 = d.NewBufferFloats(b.PostFFNNorm2)
	ml.postFFN = d.NewBufferFloats(b.PostFFNNorm)
	ml.uLayerScalar = d.NewBufferFloats([]float32{b.LayerScalar})
	return ml
}

// encodeGemma4MoEFFN records Gemma-4's parallel dense‖MoE FFN for one layer, replacing the dense
// gate/up/swiglu/down block. Unlike the generic encodeMoEFFN (one branch, wacc straight into the
// residual), TWO branches run off the SAME post-attention residual h through THREE independent
// normalizations, then join under a shared post-norm with a per-layer scalar. Mirrors
// decoder/forward_gemma4_moe.go and cuda/resident.go gemma4MoeMLP exactly:
//
//	x1 = postFFN1( mlpDown( geluTanh(mlpGate·xd)·(mlpUp·xd) ) )   xd = preFFN(h)    [dense]
//	rn = rmsnorm_nw(h); logits = RouterProjScaled·rn; idx,wgt = route; wgt *= perExpertScale[idx]
//	x2 = postFFN2( Σ_j wgt[j]·expertDown_j( geluTanh(gu_j)·up_j ) ) xe = preFFN2(h) [MoE]
//	h  = (h + postFFN(x1 + x2)) · layerScalar                                        [join]
//
// h (r.x) is read THREE times and written only at the very end. The dispatch sequence is
// value-independent (the top-k loop count is the model constant topK; each expert GEMV reads its
// own rIdx slot at execution time), so the command buffer is static every token and the encode-
// ahead executor still pre-encodes token t+1 while t runs (task-metal-moe.md).
func (r *Resident) encodeGemma4MoEFFN(e *Encoder, L *residLayer) {
	r.encodeG4Phase1(e, L)         // dense branch + router + preFFN2 quant
	r.encodeG4Phase2NonPaged(e, L) // experts from the stacked all-E buffer
	r.encodeG4Join(e, L)           // postFFN2 + join
}

// encodeG4Phase1 is the value-INDEPENDENT head of the FFN: the dense branch (→g4x1) and the router
// (→rIdx/rWgt device buffers), ending with the expert-branch input quant (preFFN2(h) → mq/mSc). In
// the paged forward this is the first command buffer; the host then reads rIdx and stages the routed
// experts before phase 2. Byte-identical to the old inline head (same dispatches, same order).
func (r *Resident) encodeG4Phase1(e *Encoder, L *residLayer) {
	g := r.g4moe
	ml := L.g4moe
	// dense branch → g4x1 (xd = preFFN(h), gelu-tanh GeGLU, own post-norm)
	e.Dispatch(r.pRms, 256, 256, r.x, ml.preFFN, r.mq, r.mSc, r.uH, r.uEps, r.uAddOne)
	e.DispatchTG(r.pSA, (2*g.denseInter)*32, 256, r.H*2, ml.denseGuW, ml.denseGuS, r.mq, r.mSc, r.gu, r.uH)
	e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(g.denseInter*4), r.dq, r.dSc, g.uDenseInter, r.uAct)
	e.Dispatch(r.pGemv, r.H*32, 32, ml.denseDW, ml.denseDS, r.dq, r.dSc, g.g4x1, g.uDenseInter)
	e.Dispatch(r.pRmsF32, 256, 256, g.g4x1, ml.postFFN1, r.uH, r.uEps, r.uAddOne)
	// router on RAW h: weightless out-of-place norm → pure-f32 proj → top-k → per-expert-scale
	e.Dispatch(g.pRmsNW, 256, 256, r.x, g.g4rn, r.uH, r.uEps)
	e.Dispatch(g.pRouterF32, g.nE*32, 32, ml.routerW, g.g4rn, g.rLogits, r.uH)
	e.Dispatch(g.pRoute, 1, 1, g.rLogits, ml.routerBias, g.rIdx, g.rWgt,
		g.uNE, g.uK, g.uSig0, g.uNorm1, g.uScale1, g.uOne, g.uOne)
	e.Dispatch(g.pScaleWgt, g.topK, g.topK, g.rWgt, g.rIdx, ml.perExpertScale, g.uK)
	// expert-branch input: xe = preFFN2(h) → mq/mSc (consumed by phase 2)
	e.Dispatch(r.pRms, 256, 256, r.x, ml.preFFN2, r.mq, r.mSc, r.uH, r.uEps, r.uAddOne)
}

// encodeG4Phase2NonPaged runs the k selected experts out of the STACKED all-E buffers (rIdx read at
// kernel-execution time — value-independent dispatch), accumulating into g4x2. The all-resident path.
func (r *Resident) encodeG4Phase2NonPaged(e *Encoder, L *residLayer) {
	g := r.g4moe
	ml := L.g4moe
	e.Dispatch(g.pZero, r.H, 256, g.g4x2)
	for j := 0; j < g.topK; j++ {
		e.DispatchTG(g.pGU, (2*g.moeInter)*32, 256, r.H*2, ml.expGuW, ml.expGuS, r.mq, r.mSc, r.gu, r.uH, g.rIdx, g.uSlot[j], g.uMoeGU)
		e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(g.moeInter*4), r.dq, r.dSc, g.uMoeInter, r.uAct)
		e.DispatchTG(g.pDownWacc, r.H*32, 256, g.moeInter*2, ml.expDW, ml.expDS, r.dq, r.dSc, g.g4x2, g.uMoeInter, g.rIdx, g.rWgt, g.uSlot[j], r.uH)
	}
}

// encodeG4Phase2Paged runs the k selected experts out of their staged SLOT buffers (one expert per
// slot), accumulating into g4x2 — the paged twin of encodeG4Phase2NonPaged. idxZeros makes each GEMV
// read row 0 of its single-expert slot (idxZeros[j]==0) while rWgt is still indexed by the selection
// slot uSlot[j], so the reused gemv_w4a8_moe(_wacc) kernels compute byte-identically to the stacked
// path (slot bytes == the stacked buffer's rows for that expert).
func (r *Resident) encodeG4Phase2Paged(e *Encoder, slots []expertSlot) {
	g := r.g4moe
	e.Dispatch(g.pZero, r.H, 256, g.g4x2)
	for j := 0; j < g.topK; j++ {
		s := slots[j]
		e.DispatchTG(g.pGU, (2*g.moeInter)*32, 256, r.H*2, s.guW, s.guS, r.mq, r.mSc, r.gu, r.uH, g.idxZeros, g.uSlot[j], g.uMoeGU)
		e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(g.moeInter*4), r.dq, r.dSc, g.uMoeInter, r.uAct)
		e.DispatchTG(g.pDownWacc, r.H*32, 256, g.moeInter*2, s.dW, s.dS, r.dq, r.dSc, g.g4x2, g.uMoeInter, g.idxZeros, g.rWgt, g.uSlot[j], r.uH)
	}
}

// forwardLogitsPaged is the SYNCHRONOUS expert-paging decode: the reference implementation that lets
// the 26B (11.96 GB experts) run on a 16 GB Mac. Per layer, dense layers encode in one command
// buffer; a paged Gemma-4 MoE layer is torn at the router (the value-dependent seam Step-0 priced at
// +43%): [attention + dense + router] → submit+wait → read rIdx → stage the routed top-k into the
// layer's LRU slot pool → [experts-from-slots + join] → submit+wait. Assumes the caller filled r.x
// with the embedding and holds the OS thread (ForwardEmb does both).
func (r *Resident) forwardLogitsPaged(pos int) []float32 {
	r.uPos.SetU32(uint32(pos))
	r.uNKeys.SetU32(uint32(pos + 1))
	g := r.g4moe
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		if L.g4moe != nil && L.g4moe.pool != nil {
			e := r.q.Begin() // phase 1: attention + dense + router → rIdx/rWgt
			r.encodeAttention(e, l)
			r.encodeG4Phase1(e, L)
			e.End() // commit + wait: rIdx/rWgt now readable
			idx := g.rIdx.U32s()
			ids := make([]int, g.topK)
			for j := 0; j < g.topK; j++ {
				ids[j] = int(idx[j])
			}
			L.g4moe.pool.prefetchAll(ids) // WILLNEED all routed experts at once (no-op when disabled)
			slots := make([]expertSlot, g.topK)
			for j := 0; j < g.topK; j++ {
				slots[j] = L.g4moe.pool.ensureResident(ids[j]) // stage on miss, evict LRU
			}
			e2 := r.q.Begin() // phase 2: experts from slots + join
			r.encodeG4Phase2Paged(e2, slots)
			r.encodeG4Join(e2, L)
			e2.End()
			continue
		}
		e := r.q.Begin()
		r.encodeLayer(e, l)
		e.End()
	}
	e := r.q.Begin()
	e.Dispatch(r.pRms, 256, 256, r.x, r.finalNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
	e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
	e.End()
	r.finalizeLogits()
	return r.logitsHost
}

// encodeG4Join is the shared tail: postFFN2 on the expert accumulator, then the join —
// h = (h + postFFN(x1 + x2)) · layerScalar (sum before the joint norm, residual after it, scalar last).
func (r *Resident) encodeG4Join(e *Encoder, L *residLayer) {
	g := r.g4moe
	ml := L.g4moe
	e.Dispatch(r.pRmsF32, 256, 256, g.g4x2, ml.postFFN2, r.uH, r.uEps, r.uAddOne)
	e.Dispatch(r.pRes, r.H, 256, g.g4x1, g.g4x2)                                 // g4x1 += g4x2
	e.Dispatch(r.pRmsF32, 256, 256, g.g4x1, ml.postFFN, r.uH, r.uEps, r.uAddOne) // g4x1 = postFFN(x1+x2)
	e.Dispatch(r.pRes, r.H, 256, r.x, g.g4x1)                                    // r.x = h + comb
	e.Dispatch(g.pScaleVec, r.H, 256, r.x, ml.uLayerScalar)                      // r.x *= layerScalar
}
