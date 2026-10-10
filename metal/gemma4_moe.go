//go:build darwin

package metal

import (
	"fmt"
	"os"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"golang.org/x/sys/unix"
)

// pagedProfile decomposes the paged forward's per-token cost into GPU-busy time per phase (the command buffer's GPU timestamps),
// wall time (GPU plus submit, wait and encode coordination) and staging (the ensureResident pread body). It accumulates across
// calls: snapshot with PagedProfile() and diff over a timed window.
type pagedProfile struct {
	p1WallNanos, p1GpuNanos       int64 // phase 1 (attention + dense branch + router): wall incl submit+wait, GPU-busy
	p1EncNanos, p1SubNanos        int64 // phase 1 wall split: encode (Begin→dispatches) vs submit+wait (End)
	p1CommitNanos, p1WaitNanos    int64 // phase 1 submit+wait split (GOINFER_MOE_PROF_SPLIT): commit() vs waitUntilCompleted()
	stageWallNanos                int64 // ensureResident (pread into slots) — cross-check vs pool.stageNanos
	idxCoordNanos                 int64 // router idx readback + prefetch + id unpack (pure host coordination)
	p2WallNanos, p2GpuNanos       int64 // phase 2 (expert GEMVs from slots + join): wall, GPU-busy
	p2EncNanos, p2SubNanos        int64 // phase 2 wall split: encode vs submit+wait
	p2CommitNanos, p2WaitNanos    int64 // phase 2 submit+wait split: commit() vs waitUntilCompleted()
	denseWallNanos, denseGpuNanos int64 // non-MoE dense layers (whole layer in one command buffer)
}

// PagedProfile returns the accumulated per-phase paging profile (see pagedProfile). Snapshot before
// and after a timed decode window and subtract for the window's breakdown.
func (r *resident) PagedProfile() pagedProfile { return r.prof }

// Gemma-4 enable_moe_block (26B-A4B) MSL kernels: the parallel dense||MoE FFN the generic moe.go path (Mixtral/Qwen/GLM shape)
// cannot express. They live in their own const, concatenated after moeKernels, so moe.go's audited kernels are not touched (as
// CUDA's separate router_f32.cu keeps moe.ptx unchanged). Each primitive was verified in isolation against a CPU oracle before
// the composition was wired (gemma4_moekernels_test.go).
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

// gemma4MoeResident holds the Gemma-4 dense||MoE pipelines, config, constant uniforms and scratch shared across all
// enable_moe_block layers. It is separate from moeResident (moe.go), the generic one-branch SiLU shape with wacc straight into the
// residual, which Gemma 4 is not.
type gemma4MoeResident struct {
	pRouterF32, pRoute, pGU, pDownWacc  Pipeline
	guR, downR                          int // D-B04: rows per simdgroup of pGU / pDownWacc (moeExpertRows)
	pRmsNW, pScaleWgt, pScaleVec, pZero Pipeline

	nE, topK, denseInter, moeInter int
	uNE, uK, uHidden               Buffer // moe_route nE/k; router GEMV K = hidden
	uDenseInter, uMoeInter         Buffer // swiglu I (dense vs expert intermediate)
	uMoeGU                         Buffer // 2*moeInter — expert fused gate|up rows/width
	uSig0, uNorm1, uScale1, uOne   Buffer // route uniforms: softmax(0), unconditional renorm(1), scale 1.0, nGroup/topkGroup=1
	uSlot                          []Buffer

	rLogits, rIdx, rWgt Buffer // router scratch: logits[nE], idx[k] (u32), wgt[k]
	g4x1, g4x2, g4rn    Buffer // dense-branch out, expert-branch accumulator, router-norm input — all [hidden]
	// Lever 3's guess at the next layer's experts (g4PrefetchOn): its router on this layer's residual, into its own
	// buffers so the true route is never touched.
	rLogitsP, rIdxP, rWgtP, g4rnP Buffer
	fence                         *pagedFence // the route fence (paged_fence.go); nil unpaged or where MSL 3.2 does not compile

	// Synchronous paging (--moe-cache-slots; metalMoESlotsRequest): the full expert set does not fit resident, so each layer keeps N
	// experts in a slot pool and stages the routed top-k in per token. Off (all experts resident) when slots==0. slotIdx is a [topK]
	// device buffer, host-written each token with the pool slot number holding each routed expert: the pool's storage is one
	// contiguous per-field buffer, so which physical row holds an expert varies token to token. The reused gemv_w4a8_moe(_wacc)
	// kernels read row slotIdx[slot] of that buffer while still indexing rWgt by the selection slot uSlot[j], which makes paged
	// dispatch byte-identical to the stacked path. slotIdx keeps one Buffer identity for the resident's lifetime (only its contents
	// change), which keeps phase 2's encode value-independent (expertpool.go).
	paged   bool
	slots   int
	slotIdx Buffer

	// giwFile is the re-opened .giw for pread staging (GOINFER_MOE_PREAD, default on); nil means the mmap byte-copy path. A shared
	// read-only fd across every layer's pool, closed by resident.Close.
	giwFile *os.File

	// alias is the S6 weight aliaser (nil when GOINFER_METAL_ALIAS=0 or the model is not .giw-mapped); buildResident sets it before the layer loop.
	alias *weightAlias
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

// buildGemma4MoE builds the resident-level Gemma-4 MoE state (pipelines, config, uniforms, scratch)
// from a model that declares enable_moe_block. Returns nil when the model has no gemma4 MoE layer,
// and an error for a shape this path cannot express (so BuildResident declines → CPU fallback rather
// than mis-running). Mirrors cuda/backend.go's isG4MoE build and its int4-width shape checks.
func buildGemma4MoE(d *Device, m *decoder.Model, pipe func(string) Pipeline, H, nL int) (*gemma4MoeResident, error) {
	if !m.HasGemma4MoEResident() {
		return nil, nil
	}
	var b decoder.Gemma4MoEResidentBundle
	var ok bool
	for l := range nL {
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
	if capE, _, _ := decoder.ResidentBackendMoECap("metal"); b.NE > capE {
		return nil, fmt.Errorf("metal gemma4 MoE nE=%d exceeds moe_route cap %d", b.NE, capE)
	}
	g := &gemma4MoeResident{
		pRouterF32: pipe("gemv_f32_f32"), pRoute: pipe("moe_route_sg"),
		pRmsNW: pipe("rmsnorm_nw"), pScaleWgt: pipe("scale_wgt_by_expert"),
		pScaleVec: pipe("scale_vec"), pZero: pipe("zero_vec"),
		nE: b.NE, topK: b.TopK, denseInter: b.DenseInter, moeInter: b.MoeInter,
	}
	g.pGU, g.pDownWacc, g.guR, g.downR = moeExpertRows(pipe, 2*b.MoeInter, H, true)
	g.uNE, g.uK = NewBufferU32(d, uint32(b.NE)), NewBufferU32(d, uint32(b.TopK))
	g.uHidden = NewBufferU32(d, uint32(H))
	g.uDenseInter, g.uMoeInter = NewBufferU32(d, uint32(b.DenseInter)), NewBufferU32(d, uint32(b.MoeInter))
	g.uMoeGU = NewBufferU32(d, uint32(2*b.MoeInter))
	g.uSig0, g.uNorm1 = NewBufferU32(d, 0), NewBufferU32(d, 1)
	g.uScale1 = NewBufferFloats(d, []float32{1})
	g.uOne = NewBufferU32(d, 1)
	g.uSlot = make([]Buffer, b.TopK)
	for j := range g.uSlot {
		g.uSlot[j] = NewBufferU32(d, uint32(j))
	}
	g.rLogits = d.NewBufferLen(b.NE)
	g.rIdx = NewBufferUint32s(d, make([]uint32, b.TopK))
	g.rWgt = d.NewBufferLen(b.TopK)
	g.g4x1, g.g4x2, g.g4rn = d.NewBufferLen(H), d.NewBufferLen(H), d.NewBufferLen(H)
	g.rLogitsP, g.rIdxP, g.rWgtP, g.g4rnP = d.NewBufferLen(b.NE), NewBufferUint32s(d, make([]uint32, b.TopK)), d.NewBufferLen(b.TopK), d.NewBufferLen(H)

	// Synchronous paging: --moe-cache-slots (GOINFER_METAL_MOE_SLOTS is the deprecated fallback; metalMoESlotsRequest) keeps only N
	// experts per layer resident and stages the routed top-k in per token. N must be >= topK (a token's own top-k must fit); N==0 or
	// unset keeps every expert resident (the fitting path and the paged-equals-non-paged parity reference).
	if s := metalMoESlotsRequest(m); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil || n < b.TopK {
			return nil, fmt.Errorf("expert-slot request %q invalid (need integer >= topK=%d)", s, b.TopK)
		}
		if n < b.NE { // n>=nE would hold every expert — no paging, just build the stacked path
			g.paged, g.slots = true, n
			g.slotIdx = NewBufferUint32s(d, make([]uint32, b.TopK))
			g.fence = newPagedFence(d, nL, b.TopK)
			for l := range nL {
				if skipPagedInt4CheckForTest {
					break
				}
				if bl, ok := m.Gemma4MoEResidentLayer(l); ok && len(bl.ExpertsGateUp) > 0 {
					_, _, ok1 := int4DirectWords(bl.ExpertsGateUp[0])
					_, _, ok2 := int4DirectWords(bl.ExpertsDown[0])
					if !ok1 || !ok2 {
						return nil, errPagedExpertsNotInt4
					}
				}
			}
		}
	}
	// Stage experts by pread'ing their nibbles straight into the slot buffers instead of a byte-copy off the mmap: default on,
	// GOINFER_MOE_PREAD=0 opts out (the mmap byte-copy baseline). It needs a .giw-mmap'd model (the offsets are into that file),
	// re-opened once and shared across layers. If the open fails or the model is not .giw-backed, buildGemma4MoELayer falls back to
	// the byte-copy.
	if g.paged && modelKnob(m, "GOINFER_MOE_PREAD") != "0" {
		if p := m.GiwPath(); p != "" {
			if f, err := os.Open(p); err == nil {
				g.giwFile = f
				// GOINFER_MOE_NOCACHE=1 sets F_NOCACHE on the pread fd, so reads bypass the unified buffer cache. Off by default and kept wired
				// so it is not re-proposed: a cold A/B measured no effect (docs/code-notes/metal.md#buildGemma4MoE.nocache).
				if modelKnob(m, "GOINFER_MOE_NOCACHE") == "1" {
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
	ml.routerW = NewBufferFloats(d, b.RouterProjScaled)
	ml.routerBias = NewBufferFloats(d, make([]float32, b.NE)) // zeros → sel = score (no e_score_correction_bias)
	ml.perExpertScale = NewBufferFloats(d, b.PerExpertScale)
	ml.denseGuW, ml.denseGuS = int4ConcatA(d, g.alias, b.MlpGate, b.MlpUp)
	var e error
	if ml.denseDW, ml.denseDS, e = int4BufA(d, g.alias, b.MlpDown); e != nil {
		panic(e)
	}
	if g.paged {
		// Paged: do not stack the experts (the all-E buffers are what cannot be afforded). Build a bounded LRU slot pool and a stage fn
		// that reads expert e's W4A8 bytes straight from the bundle's mmap-backed WeightMats on demand (int4DirectWords aliases the
		// .giw). Per-expert buffer sizes come from expert 0. The .giw expert weights MUST be int4-direct (group-32) for this
		// zero-intermediate path.
		gw0, gs0, ok1 := int4DirectWords(b.ExpertsGateUp[0])
		dw0, ds0, ok2 := int4DirectWords(b.ExpertsDown[0])
		if !ok1 || !ok2 {
			panic("metal gemma4 MoE paging: experts are not int4-direct (unreachable: buildGemma4MoE declines first, errPagedExpertsNotInt4)")
		}
		experts := b.ExpertsGateUp // capture (aliases the model mmap; kept alive by the Model)
		down := b.ExpertsDown
		// Each expert's f16 scales are read from its own WeightMat (Int4ScalesF16), which a v14 metal or v15 .giw aliases from the
		// mapping; a build-time heap cache of them would hold the same bits again.
		stage := func(ei int) ([]byte, []uint16, []byte, []uint16) {
			gw, _ := int4DirectBytesOnly(experts[ei]) // nibble bytes aliased from mmap; no reconstruction/alloc
			dw, _ := int4DirectBytesOnly(down[ei])
			return gw, experts[ei].Int4ScalesF16(), dw, down[ei].Int4ScalesF16()
		}
		ml.pool = newExpertPool(d, g.slots, len(gw0), len(gs0), len(dw0), len(ds0), stage)
		// pread staging (GOINFER_MOE_PREAD, default on): resolve each expert's nibble file offset within the .giw mmap (pointer
		// arithmetic, no page touch), then pread straight into the slot's unified-memory words: no mmap faults, one sequential read per
		// expert. The f16 scales are pread the same way when every expert's lie in the mapping (a v14 metal or v15 .giw) and copied from
		// the WeightMat otherwise. Falls back to the byte-copy path if the fd is absent or ANY expert's nibbles are not .giw-mmap-backed
		// (a requantized HF load): the offsets must all resolve for pread to be correct.
		//
		// Do not add MADV_WILLNEED over the routed experts' spans before staging. A cold A/B showed it does not help: the routed experts
		// are not known until phase 1 completes, so the synchronous Advise-then-copy path has no lead time to hide the read behind, and
		// the staging time it removed reappeared in the compute+coord bucket. When optimizing a measured sub-bucket the total is the gate
		// (docs/code-notes/metal.md#buildGemma4MoELayer.willneed).
		if g.giwFile != nil {
			guOff := make([]int64, len(experts))
			dOff := make([]int64, len(down))
			gsOff := make([]int64, len(experts))
			dsOff := make([]int64, len(down))
			resolved, scalesInFile := true, true
			for ei := range experts {
				gq, _, _, ok1 := experts[ei].Int4F16()
				dq, _, _, ok2 := down[ei].Int4F16()
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
				gso, okgs := scaleFileOffset(m, experts[ei].Int4ScalesF16())
				dso, okds := scaleFileOffset(m, down[ei].Int4ScalesF16())
				scalesInFile = scalesInFile && okgs && okds
				gsOff[ei], dsOff[ei] = gso, dso
			}
			if resolved {
				fd := int(g.giwFile.Fd())
				pool := ml.pool
				pool.stagePread = func(ei int, s expertSlot) {
					// gate|up and down are separate buffers (pool.guW, pool.dW): disjoint destinations, safe to pread concurrently (as moe.go's 3-way
					// twin). Errors panic from THIS goroutine, not the spawned ones, so BuildResident's recover() still sees them.
					//
					// preadIntoPoolSlot, not preadIntoU32Buf(fd, s.guW, ...): s.guW is a Buffer.At()-offset VIEW and U32s() ignores that offset
					// (expertSlot's doc comment), which would silently pread every expert into slot 0. preadIntoPoolSlot addresses the pool's base
					// buffer by slot NUMBER.
					var wg sync.WaitGroup
					var errGU, errD, errGS, errDS error
					wg.Add(2)
					go func() { defer wg.Done(); errGU = preadIntoPoolSlot(fd, pool.guW, s.slot, pool.nGuW, guOff[ei]) }()
					go func() { defer wg.Done(); errD = preadIntoPoolSlot(fd, pool.dW, s.slot, pool.nDW, dOff[ei]) }()
					if scalesInFile { // the scales are four more disjoint destinations' worth of the same argument
						wg.Add(2)
						go func() {
							defer wg.Done()
							errGS = preadScalesIntoPoolSlot(fd, pool.guS, s.slot, pool.nGuS, 0, gsOff[ei], 2*pool.nGuS)
						}()
						go func() {
							defer wg.Done()
							errDS = preadScalesIntoPoolSlot(fd, pool.dS, s.slot, pool.nDS, 0, dsOff[ei], 2*pool.nDS)
						}()
					}
					wg.Wait()
					for _, e := range []struct {
						what string
						err  error
					}{{"gate|up", errGU}, {"down", errD}, {"gate|up scales", errGS}, {"down scales", errDS}} {
						if e.err != nil {
							panic(fmt.Sprintf("metal gemma4 MoE pread %s expert %d: %v", e.what, ei, e.err))
						}
					}
					if scalesInFile {
						pool.scalePreads.Add(1)
						return
					}
					// An older file: the scales are the WeightMat's heap copy, converted at load.
					// copyU16sToBuf, not copy(s.guS.U16s(), ...) — same offset-view caveat as above.
					copyU16sToBuf(s.guS, experts[ei].Int4ScalesF16())
					copyU16sToBuf(s.dS, down[ei].Int4ScalesF16())
				}
			}
		}
	} else {
		ml.expGuW, ml.expGuS = int4Concat(d, b.ExpertsGateUp...) // per expert already fused [2*moeInter, hidden]
		ml.expDW, ml.expDS = int4Concat(d, b.ExpertsDown...)
	}
	ml.preFFN = NewBufferFloats(d, b.PreFFNNorm)
	ml.postFFN1 = NewBufferFloats(d, b.PostFFNNorm1)
	ml.preFFN2 = NewBufferFloats(d, b.PreFFNNorm2)
	ml.postFFN2 = NewBufferFloats(d, b.PostFFNNorm2)
	ml.postFFN = NewBufferFloats(d, b.PostFFNNorm)
	ml.uLayerScalar = NewBufferFloats(d, []float32{b.LayerScalar})
	return ml
}

// encodeGemma4MoEFFN records Gemma-4's parallel dense||MoE FFN for one layer, replacing the dense gate/up/swiglu/down block.
// Unlike the generic encodeMoEFFN (one branch, wacc straight into the residual), two branches run off the same post-attention
// residual h through three independent normalizations, then join under a shared post-norm with a per-layer scalar. Mirrors
// decoder/forward_gemma4_moe.go and cuda/resident.go gemma4MoeMLP exactly:
//
//	x1 = postFFN1( mlpDown( geluTanh(mlpGate·xd)·(mlpUp·xd) ) )   xd = preFFN(h)    [dense]
//	rn = rmsnorm_nw(h); logits = RouterProjScaled·rn; idx,wgt = route; wgt *= perExpertScale[idx]
//	x2 = postFFN2( Σ_j wgt[j]·expertDown_j( geluTanh(gu_j)·up_j ) ) xe = preFFN2(h) [MoE]
//	h  = (h + postFFN(x1 + x2)) · layerScalar                                        [join]
//
// h (r.x) is read three times and written only at the very end. The dispatch sequence is value-independent (the top-k loop count
// is the model constant topK; each expert GEMV reads its own rIdx slot at execution time), so the command buffer is static every
// token and the encode-ahead executor still pre-encodes token t+1 while t runs.
//
// This is the NON-PAGED path: encodeG4Phase2NonPaged reads the stacked all-E buffers (ml.expGuW/expGuS/expDW/expDS), which stay
// zero-value once the layer is paged. forwardLogitsPaged never reaches here for a paged layer (it tears the layer into
// encodeG4Phase1 and encodeG4Phase2Paged around a host readback); the panic below is a chokepoint against any other caller of
// encodeLayer (Forward, ForwardArgmax, forwardHiddenNoHead's encodeTrunkInto) reaching a paged layer through the non-paged
// encoder and silently computing off zero-value weights.
//
// FinishEncoding before the panic, as encodeMoEFFN does (moe.go): e already has this layer's attention dispatches recorded, and
// Metal asserts on a command encoder released without endEncoding.
func (r *resident) encodeGemma4MoEFFN(e *Encoder, L *residLayer) {
	if L.g4moe.pool != nil {
		e.FinishEncoding()
		panic("metal: encodeGemma4MoEFFN reached a paged Gemma-4 MoE layer — route through forwardLogitsPaged instead (C-02)")
	}
	r.encodeG4Phase1(e, L)         // dense branch + router + preFFN2 quant
	r.encodeG4Phase2NonPaged(e, L) // experts from the stacked all-E buffer
	r.encodeG4Join(e, L)           // postFFN2 + join
}

// encodeG4Phase1 is the value-INDEPENDENT head of the FFN: the dense branch (to g4x1) and the router (to the rIdx/rWgt device
// buffers), ending with the expert-branch input quant (preFFN2(h) to mq/mSc). In the paged forward this is the first command
// buffer; the host then reads rIdx and stages the routed experts before phase 2.
func (r *resident) encodeG4Phase1(e *Encoder, L *residLayer) {
	g := r.g4moe
	ml := L.g4moe
	// dense branch → g4x1 (xd = preFFN(h), gelu-tanh GeGLU, own post-norm)
	e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, r.x, ml.preFFN, r.mq, r.mSc, r.uH, r.uEps, r.uAddOne)
	e.DispatchTG(r.pSA, (2*g.denseInter)*32, 256, r.H*2, ml.denseGuW, ml.denseGuS, r.mq, r.mSc, r.gu, r.uH)
	e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(g.denseInter*4), r.dq, r.dSc, g.uDenseInter, r.uAct)
	e.Dispatch(r.pGemv, r.H*32, 32, ml.denseDW, ml.denseDS, r.dq, r.dSc, g.g4x1, g.uDenseInter)
	e.Dispatch(r.pRmsF32, tgReduceNorm, tgReduceNorm, g.g4x1, ml.postFFN1, r.uH, r.uEps, r.uAddOne)
	// router on RAW h: weightless out-of-place norm → pure-f32 proj → top-k → per-expert-scale
	e.Dispatch(g.pRmsNW, tgReduceNorm, tgReduceNorm, r.x, g.g4rn, r.uH, r.uEps)
	e.Dispatch(g.pRouterF32, g.nE*32, 32, ml.routerW, g.g4rn, g.rLogits, r.uH)
	e.Dispatch(g.pRoute, 32, 32, g.rLogits, ml.routerBias, g.rIdx, g.rWgt,
		g.uNE, g.uK, g.uSig0, g.uNorm1, g.uScale1, g.uOne, g.uOne)
	e.Dispatch(g.pScaleWgt, g.topK, g.topK, g.rWgt, g.rIdx, ml.perExpertScale, g.uK)
	// expert-branch input: xe = preFFN2(h) → mq/mSc (consumed by phase 2)
	e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, r.x, ml.preFFN2, r.mq, r.mSc, r.uH, r.uEps, r.uAddOne)
}

// encodeG4Phase2NonPaged runs the k selected experts out of the STACKED all-E buffers (rIdx read at
// kernel-execution time — value-independent dispatch), accumulating into g4x2. The all-resident path.
func (r *resident) encodeG4Phase2NonPaged(e *Encoder, L *residLayer) {
	g := r.g4moe
	ml := L.g4moe
	e.Dispatch(g.pZero, r.H, 256, g.g4x2)
	for j := 0; j < g.topK; j++ {
		e.DispatchTG(g.pGU, rowsGrid(2*g.moeInter, g.guR), 256, r.H*2, ml.expGuW, ml.expGuS, r.mq, r.mSc, r.gu, r.uH, g.rIdx, g.uSlot[j], g.uMoeGU)
		e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(g.moeInter*4), r.dq, r.dSc, g.uMoeInter, r.uAct)
		e.DispatchTG(g.pDownWacc, rowsGrid(r.H, g.downR), 256, g.moeInter*2, ml.expDW, ml.expDS, r.dq, r.dSc, g.g4x2, g.uMoeInter, g.rIdx, g.rWgt, g.uSlot[j], r.uH)
	}
}

// encodeG4Phase2Paged is encodeG4Phase2NonPaged over the layer's pool, one contiguous buffer per field. It always binds
// pool.guW/guS/dW/dS (fixed identity for the layer's lifetime); slotIdx, written into g.slotIdx by the caller before this encode,
// tells the reused gemv_w4a8_moe(_wacc) kernels which pool row holds each selected expert at execution time, while rWgt stays
// indexed by the selection slot uSlot[j]. Byte-identical to the stacked path (a pool row's bytes are the stacked buffer's rows
// for that expert).
func (r *resident) encodeG4Phase2Paged(e *Encoder, pool *expertPool) {
	g := r.g4moe
	e.Dispatch(g.pZero, r.H, 256, g.g4x2)
	for j := 0; j < g.topK; j++ {
		e.DispatchTG(g.pGU, rowsGrid(2*g.moeInter, g.guR), 256, r.H*2, pool.guW, pool.guS, r.mq, r.mSc, r.gu, r.uH, g.slotIdx, g.uSlot[j], g.uMoeGU)
		e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(g.moeInter*4), r.dq, r.dSc, g.uMoeInter, r.uAct)
		e.DispatchTG(g.pDownWacc, rowsGrid(r.H, g.downR), 256, g.moeInter*2, pool.dW, pool.dS, r.dq, r.dSc, g.g4x2, g.uMoeInter, g.slotIdx, g.rWgt, g.uSlot[j], r.uH)
	}
}

// forwardLogitsPaged is the synchronous expert-paging decode: the reference implementation that lets an expert set larger than
// RAM run. Dense layers encode in one command buffer; a paged Gemma-4 MoE layer is torn at the router (the value-dependent
// seam): [attention + dense + router] -> submit+wait -> read rIdx -> stage the routed top-k into the layer's LRU slot pool ->
// [experts-from-slots + join] -> submit+wait. Assumes the caller filled r.x with the embedding and holds the OS thread
// (ForwardEmb does both).
func (r *resident) forwardLogitsPaged(pos int, ropePos ...int) (logits []float32) {
	// The pread staging closure (stagePread) and the per-token MustBuf allocations panic on a transient .giw read error or OOM, deep
	// inside expertPool.ensureResident at decode time, where no other recover exists (buildResident's is build-scoped). Convert the
	// panic to execErr so the metalResident adapter surfaces a failed request and drops the stale logits instead of crashing the
	// server.
	defer func() {
		if p := recover(); p != nil {
			r.recordExecErr(fmt.Errorf("metal: paged forward aborted: %v", p))
			logits = nil
			// Accepted bounded side-effects on this rare abort path: the in-flight encoder is left un-ended (a small native cb/pool leak) and
			// a slot's stale slotExpert tag may cost one spurious re-stage on a later eviction; both are perf-only and bounded by the abort
			// count. Do NOT drain/End the live encoder here: the path mixes Begin() (own autorelease pool) and BeginNP() (nil pool, shared
			// arp), so a blind DrainPool/End would nil-panic or commit a half-encoded command buffer, which is worse than the leak.
		}
	}()
	// setPos, not a direct uPos/uNKeys write, as moe.go's paged path does.
	rp := pos
	if len(ropePos) > 0 {
		rp = ropePos[0]
	}
	r.setPos(pos, rp)
	g := r.g4moe
	p := &r.prof
	// GOINFER_MOE_PROF_SPLIT splits each End() into commit() vs waitUntilCompleted() to locate per-boundary overhead. It needs BeginNP
	// (no per-call autorelease pool) and one long-lived pool drained per token, so the sub-steps can be called individually.
	split := os.Getenv("GOINFER_MOE_PROF_SPLIT") == "1"
	async := pagedAsyncPhase2On && !split // phase 2 committed, not waited (pagedAsyncPhase2On)
	var arp ARPool
	if split || async {
		arp = NewARPool()
	}
	var pending []*Encoder   // async phase-2 buffers, waited at the token's end
	var pendingP1 []*Encoder // fenced phase-1 buffers whose route the host read from the fence, waited at the token's end
	end := func(e *Encoder, commitAcc, waitAcc *int64) {
		if !split {
			e.End()
			r.recordExecErr(e.Err()) // catch an abort in any paged per-layer submit
			return
		}
		e.FinishEncoding()
		tc := time.Now()
		e.Commit()
		*commitAcc += time.Since(tc).Nanoseconds()
		tw := time.Now()
		e.WaitDone() // waitUntilCompleted + ReadTimes
		r.recordExecErr(e.Err())
		*waitAcc += time.Since(tw).Nanoseconds()
	}
	begin := func() *Encoder {
		if split || async {
			return r.q.BeginNP()
		}
		return r.q.Begin()
	}
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		if L.g4moe != nil && L.g4moe.pool != nil {
			w1 := time.Now()
			e := begin() // phase 1: attention + dense + router → rIdx/rWgt
			r.encodeAttention(e, l)
			r.encodeG4Phase1(e, L)
			var next *gemma4MoeLayer // lever 3: guess the next paged MoE layer's experts from this residual
			if g4PrefetchOn && l+1 < r.nL {
				if nl := r.layers[l+1].g4moe; nl != nil && nl.pool != nil {
					next = nl
					e.Dispatch(g.pRmsNW, tgReduceNorm, tgReduceNorm, r.x, g.g4rnP, r.uH, r.uEps)
					e.Dispatch(g.pRouterF32, g.nE*32, 32, nl.routerW, g.g4rnP, g.rLogitsP, r.uH)
					e.Dispatch(g.pRoute, 32, 32, g.rLogitsP, nl.routerBias, g.rIdxP, g.rWgtP,
						g.uNE, g.uK, g.uSig0, g.uNorm1, g.uScale1, g.uOne, g.uOne)
				}
			}
			fenced := g.fence != nil && async // the route mirrored and fenced, the host spins (paged_fence.go)
			var seq uint32
			if fenced {
				var idx2 Buffer
				if next != nil {
					idx2 = g.rIdxP
				}
				seq = g.fence.encode(e, l, g.rIdx, idx2)
			}
			encDone := time.Now()
			if fenced {
				e.FinishEncoding()
				e.Commit()
				done := g.fence.wait(e, seq)
				if pagedFenceCheckForTest {
					if !done {
						e.WaitDone()
						done = true
					}
					if !slices.Equal(g.fence.ids(g.topK, false), u32sToInts(g.rIdx.U32s()[:g.topK])) ||
						next != nil && !slices.Equal(g.fence.ids(g.topK, true), u32sToInts(g.rIdxP.U32s()[:g.topK])) {
						g.fence.stale++
					}
				}
				if done {
					r.recordExecErr(e.Err())
					p.p1GpuNanos += int64((e.GPUEnd() - e.GPUStart()) * 1e9)
				} else {
					pendingP1 = append(pendingP1, e)
				}
			} else {
				end(e, &p.p1CommitNanos, &p.p1WaitNanos) // commit + wait: rIdx/rWgt now readable
				p.p1GpuNanos += int64((e.GPUEnd() - e.GPUStart()) * 1e9)
			}
			p.p1EncNanos += encDone.Sub(w1).Nanoseconds()
			p.p1SubNanos += time.Since(encDone).Nanoseconds()
			p.p1WallNanos += time.Since(w1).Nanoseconds()
			c0 := time.Now()
			var ids []int
			if fenced {
				ids = g.fence.ids(g.topK, false)
			} else {
				ids = u32sToInts(g.rIdx.U32s()[:g.topK])
			}
			if next != nil { // start the guess's reads first, so they queue on the disk beside this layer's misses
				var all []int
				if fenced {
					all = g.fence.ids(g.topK, true)
				} else {
					all = u32sToInts(g.rIdxP.U32s()[:g.topK])
				}
				next.pool.prefetchAsync(all[:min(g.topK, g4PrefetchTop)]) // the guess's highest-scoring experts (moe_route's rank order)
			}
			p.idxCoordNanos += time.Since(c0).Nanoseconds()
			s0 := time.Now()
			// Stage every miss in this token's top-k concurrently rather than one pread at queue depth 1 per expert (ensureResidentBatch's
			// doc comment says why that is safe).
			slots := L.g4moe.pool.ensureResidentBatch(ids)
			p.stageWallNanos += time.Since(s0).Nanoseconds() // cross-check vs pool.stageNanos (same body)
			// Tell the GPU which pool row holds each routed expert: a small host-to-device write, not a re-encode (gemma4MoeResident.slotIdx).
			gIdx := g.slotIdx.U32s()
			for j, s := range slots {
				gIdx[j] = uint32(expertPoolSlotForTest(s.slot, len(L.g4moe.pool.slotExpert)))
			}
			w2 := time.Now()
			e2 := begin()                        // phase 2: experts from slots + join
			if r.residency != (ResidencySet{}) { // per-encoder attach, phase 2 only
				e2.UseResidencySet(r.residency)
			}
			r.encodeG4Phase2Paged(e2, L.g4moe.pool)
			r.encodeG4Join(e2, L)
			enc2 := time.Now()
			if async {
				e2.FinishEncoding()
				e2.Commit()
				pending = append(pending, e2)
			} else {
				end(e2, &p.p2CommitNanos, &p.p2WaitNanos)
			}
			p.p2EncNanos += enc2.Sub(w2).Nanoseconds()
			p.p2SubNanos += time.Since(enc2).Nanoseconds()
			p.p2WallNanos += time.Since(w2).Nanoseconds()
			if !async {
				p.p2GpuNanos += int64((e2.GPUEnd() - e2.GPUStart()) * 1e9)
			}
			continue
		}
		w0 := time.Now()
		e := begin()
		r.encodeLayer(e, l)
		var d0, d1 int64
		end(e, &d0, &d1)
		p.denseWallNanos += time.Since(w0).Nanoseconds()
		p.denseGpuNanos += int64((e.GPUEnd() - e.GPUStart()) * 1e9)
	}
	for _, pe := range pendingP1 {
		pe.WaitDone()
		r.recordExecErr(pe.Err())
		p.p1GpuNanos += int64((pe.GPUEnd() - pe.GPUStart()) * 1e9)
	}
	for _, pe := range pending { // already complete or nearly: the queue ran them before the phase 1s that followed
		pe.WaitDone()
		r.recordExecErr(pe.Err())
		p.p2GpuNanos += int64((pe.GPUEnd() - pe.GPUStart()) * 1e9)
	}
	if split || async {
		arp.Drain()
	}
	if r.pagedNoHead { // a prompt token: no logits wanted (ForwardEmbNoLogitsPipe)
		return nil
	}
	e := r.q.Begin()
	e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, r.x, r.finalNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
	e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
	e.End()
	r.recordExecErr(e.Err())
	r.finalizeLogits()
	return r.logitsHost
}

// encodeG4Join is the shared tail: postFFN2 on the expert accumulator, then the join —
// h = (h + postFFN(x1 + x2)) · layerScalar (sum before the joint norm, residual after it, scalar last).
func (r *resident) encodeG4Join(e *Encoder, L *residLayer) {
	g := r.g4moe
	ml := L.g4moe
	e.Dispatch(r.pRmsF32, tgReduceNorm, tgReduceNorm, g.g4x2, ml.postFFN2, r.uH, r.uEps, r.uAddOne)
	e.Dispatch(r.pRes, r.H, 256, g.g4x1, g.g4x2)                                                   // g4x1 += g4x2
	e.Dispatch(r.pRmsF32, tgReduceNorm, tgReduceNorm, g.g4x1, ml.postFFN, r.uH, r.uEps, r.uAddOne) // g4x1 = postFFN(x1+x2)
	e.Dispatch(r.pRes, r.H, 256, r.x, g.g4x1)                                                      // r.x = h + comb
	e.Dispatch(g.pScaleVec, r.H, 256, r.x, ml.uLayerScalar)                                        // r.x *= layerScalar
}

// pagedAsyncPhase2On commits a paged MoE layer's phase 2 (experts and join) without waiting for it (M-11,
// docs/audit-metal-2026-09-30.md): the host reads nothing back from phase 2, and the next layer's phase 1 runs on the
// same queue, ordered after it by Metal's hazard tracking on the residual it reads. The slot table the host rewrites for
// the next layer is written only after that phase 1 is seen complete, so after this phase 2. Half the token's
// submit-and-wait round trips go; the arithmetic is unchanged. Both pagers (forwardLogitsPaged, forwardLogitsMoEPaged).
var pagedAsyncPhase2On = true

// g4PrefetchOn guesses a paged Gemma 4 MoE layer's experts from the layer before it and reads the misses in the background:
// phase 1 of layer l also runs layer l+1's router on l's residual, the host starts reading those experts into l+1's pool, and the
// reads overlap the GPU's phase 2 of l and phase 1 of l+1. The true router still picks; a wrong guess costs an evicted slot,
// never a wrong bit. OFF: the extra reads compete with demand reads on the SSD and it measured slower than not guessing
// (docs/tasks/task-m26-mac-2026-10.md, "Lever 3"). Kept, gated bit-exact, for a better guess.
var g4PrefetchOn = false

// g4PrefetchTop is how many of the guess's experts are prefetched, highest score first.
var g4PrefetchTop = 2
