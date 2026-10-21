//go:build darwin

package metal

// Mixture-of-experts (MoE) resident decode for the Metal backend (FeatMoE). Mirrors the WebGPU implementation
// (gpu/moe.go): the router runs entirely on-GPU and writes idx[k]/wgt[k] device buffers, then the host records a fixed
// k dispatches per projection, each reading its expert index dynamically (weightRow = idx[slot]*rowsPerExpert + n).
// The command buffer is static every token, so the encode-ahead executor (ForwardEmbPipe) still pre-encodes token t+1
// while t runs; a host-side top-k readback would have broken that pipeline. See docs/completed/task-metal-moe.md.

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// moeLayer holds one MoE layer's device weights: the f32 router (+bias), the stacked all-E
// fused gate|up and down expert buffers (W4A8), and the optional shared expert (dense W4A8
// gate|up + down, plus the f32 sigmoid gate for the qwen2_moe gated variant).
type moeLayer struct {
	routerW    Buffer // f32 [nE, H]
	routerBias Buffer // f32 [nE] (zeros when the arch has no e_score_correction_bias)
	expGuW     Buffer // stacked all-E fused gate|up: rows nE*2*inter (int4 words)
	expGuS     Buffer // ... f16 group scales
	expDW      Buffer // stacked all-E down: rows nE*H
	expDS      Buffer
	// shared expert (sharedInter > 0)
	shGuW, shGuS Buffer // dense fused gate|up: rows 2*sharedInter
	shDW, shDS   Buffer // dense down: rows H
	shGateW      Buffer // f32 [1, H] sigmoid gate (gated variant only; zero if ungated)
	// gpt-oss: per-expert stacked biases, packed to match the weight addressing swiglu_quant_gptoss
	// / gemv_w4a8_moe_wacc_bias read (biasOff = idx[slot]*2*inter / idx[slot]*H). Zero-value Buffer
	// for every other family — the gpt-oss dispatch path in encodeMoEExperts is the only reader.
	expGuBias, expDBias Buffer
	// pool is non-nil when mo.paged: a bounded LRU slot pool with on-demand staging (expertpool.go), shared with
	// gemma4_moe.go's paging; this file supplies the stageFn. expGuW/expGuS/expDW/expDS stay zero-value when
	// paged: the stacked all-E buffer is the multi-GB set paging exists to avoid.
	pool *expertPool
}

// moeResident holds the MoE pipelines, config, constant uniforms and scratch shared across
// all MoE layers (config is uniform per model). rIdx/rWgt are the on-GPU router outputs.
type moeResident struct {
	pRouter, pRoute, pGU, pDownWacc, pSharedGate Pipeline
	guR, downR                                   int // D-B04: R18 rows per simdgroup of pGU / pDownWacc, 0 = one row (moeExpertRows)

	// gpt-oss: its own router/activation/down-combine kernels (route_gptoss,
	// swiglu_quant_gptoss, gemv_w4a8_moe_wacc_bias) replace pRoute/pGU+pSw/pDownWacc in
	// encodeMoERoute and encodeMoEExperts — the bias semantics genuinely differ (see the kernel docs in moeKernels),
	// not just an added parameter. Zero-value Pipelines/Buffers for every other family.
	isGptOss                 bool
	pRouteGptOss, pActGptOss Pipeline
	pDownWaccBias            Pipeline
	uAlpha, uLimit, uHasBias Buffer

	nE, k, inter, sharedInter     int
	sharedUngated                 bool
	uNE, uK, uSigmoid, uNorm      Buffer // route uniforms
	uScale                        Buffer // f32
	uNGroup, uTopkGroup           Buffer
	uInter, uInter2, uSharedInter Buffer
	uSlot                         []Buffer // k compile-time slot-index constants

	rLogits, rIdx, rWgt Buffer // router scratch: logits[nE], idx[k] (u32), wgt[k]

	// w8: the experts and shared expert run on their int8 weights (int8 slice 4); set by buildResident once w8Eligible
	// has admitted the model, before the layers are built.
	w8 bool

	// D-P03: the k selected experts in four dispatches (gate|up, SwiGLU, down, combine) instead of 3k. kSlots is
	// false for gpt-oss (its own activation and biased down kernels) and for a shape the k-slot grids do not divide;
	// then encodeMoEExperts runs the per-slot loop. guK [k][2I] f32, dqK [k][I] int8, dScK [k], accK [k][H].
	kSlots                          bool
	pGUk, pSwRows, pDownK, pCombine Pipeline
	guK, dqK, dScK, accK            Buffer
	shGl, shDown                    Buffer // shared-expert scratch: gate logit[1], down out[H]

	// Synchronous paging (--moe-cache-slots; GOINFER_METAL_MOE_SLOTS is the deprecated fallback,
	// metalMoESlotsRequest): gemma4_moe.go's paging generalized to this generic MoE shape, with the same
	// expertPool. N must be >= k (a token's own top-k must fit); N==0 or unset means every expert resident.
	// slotIdx is host-written each token with the pool row holding each routed expert, so the paged expert GEMVs
	// read the right row of the pool's contiguous buffer while rWgt stays indexed by the selection slot: the
	// reused gemv_w4a8_moe(_wacc[_bias]) kernels then compute byte-identically to the stacked path
	// (gemma4_moe.go's gemma4MoeResident.slotIdx comment has the full reason).
	paged   bool
	slots   int
	slotIdx Buffer

	// Two index spaces, diverging only when paging is on, as in cuda/resident.go's expIdx / expertBiasIdx pair:
	//
	//	the weight index  WHERE the weights live  → slotIdx when this LAYER is paged (the pool row
	//	                                            currently holding the routed expert), rIdx
	//	                                            otherwise; passed explicitly, see biasIdx's note
	//	biasIdx()         WHICH expert is running → ALWAYS rIdx
	//
	// With paging off the two are the same buffer, so a site that binds the wrong one is correct in every unpaged
	// configuration and wrong, silently with plausible logits, in the paged one: gpt-oss's stacked
	// expGuBias/expDBias tables must be addressed by the real expert id (biasIdx), never a constant or the slot
	// index.

	// giwFile is the re-opened .giw for pread staging; nil means the mmap byte-copy path. A shared read-only fd
	// across every layer's pool, closed by resident.Close. It has the same GOINFER_MOE_PREAD knob (0 opts out) and
	// default as gemma4_moe.go's.
	giwFile *os.File

	// alias is the S6 weight aliaser (nil when GOINFER_METAL_ALIAS=0 or the model is not .giw-mapped); buildResident sets it before the layer loop.
	alias *weightAlias
}

// errPagedExpertsNotInt4 is the expert cache's decline for experts it cannot stage. Paging copies each expert's int4
// bytes into a slot as they are stored; a non-paged resident re-quantizes int8 experts to int4 while it stacks them,
// which a per-token stage cannot afford. Checked before the build, so it is a named decline rather than the panic
// buildMoELayer and buildGemma4MoELayer would raise.
var errPagedExpertsNotInt4 = errors.New("the expert cache (--moe-cache-experts / --moe-cache-slots) pages int4 experts only, and this model's experts are not int4 (group-32): load it with --quant int4, or drop --moe-cache-experts and --moe-cache-slots to hold every expert resident")

// skipPagedInt4CheckForTest drops the errPagedExpertsNotInt4 check, so G-2 can show the gate goes red without it.
var skipPagedInt4CheckForTest bool

// moePagedAllSlotsForTest pages a model even when its slot request holds every expert, so a test can run the paged
// forward with no eviction possible (the option-path admission doc's finding 2, step 0). False in production.
var moePagedAllSlotsForTest bool

// buildMoE builds the resident-level MoE state (pipelines, config, uniforms, scratch) from a dense-shaped model that
// declares an MoE FFN. It returns nil when the model is not MoE, and an error for an MoE variant this path does not
// implement, so BuildResident declines to the CPU rather than running wrong.
func buildMoE(d *Device, m *decoder.Model, pipe func(string) Pipeline, H int) (*moeResident, error) {
	nE, k, inter, sharedInter, sigmoid, norm, sharedUngated, scale, nGroup, topkGroup, ok := m.MoEResidentParams()
	if !ok {
		return nil, nil // dense model — no MoE
	}
	// Gemma 4's enable_moe_block (gemma4_text, 26B-A4B) sets arch.MoE, so MoEResidentParams reports ok, but it is
	// not the generic Mixtral/Qwen shape this path implements: it is a parallel dense||MoE FFN with gelu-tanh
	// experts (gemv_w4a8_moe's epilogue is SiLU), a weightless router pre-norm with a learned [hidden] scale and
	// per_expert_scale, and seven norms. Building it here would run a wrong MoE. It runs encodeGemma4MoEFFN
	// (gemma4_moe.go) instead, routed around here via HasGemma4MoEResident as CUDA's isG4MoE split does. Dense
	// Gemma 4 has no MoE, so it returns nil,nil above and admits normally.
	if m.HasGemma4MoEResident() {
		// Return nil (not MoE for this builder): r.moe stays nil, BuildResident builds r.g4moe instead, and the
		// per-layer loop routes on the bundle.
		return nil, nil
	}
	// The router's fixed-size score/group arrays (moe_route); the cap is decoder's declaration, which the
	// load path's admission gate already applied — this is the build's own backstop, not a second copy.
	if capE, capG, _ := decoder.ResidentBackendMoECap("metal"); nE > capE || nGroup > capG {
		return nil, fmt.Errorf("metal MoE: nE=%d / nGroup=%d exceeds the router cap (%d / %d)", nE, nGroup, capE, capG)
	}
	_, _, isGptOss := m.GptOssActResident()
	mo := &moeResident{
		pRouter: pipe("gemv_wf32_a8"), pRoute: pipe("moe_route_sg"),
		pSharedGate:   pipe("shared_gate_combine"),
		isGptOss:      isGptOss,
		nE:            nE,
		k:             k,
		inter:         inter,
		sharedInter:   sharedInter,
		sharedUngated: sharedUngated,
	}
	mo.pGU, mo.pDownWacc, mo.guR, mo.downR = moeExpertRows(pipe, 2*inter, H, !isGptOss)
	if isGptOss {
		alpha, limit, _ := m.GptOssActResident()
		mo.pRouteGptOss = pipe("route_gptoss_sg")
		mo.pActGptOss = pipe("swiglu_quant_gptoss")
		mo.pDownWaccBias = pipe("gemv_w4a8_moe_wacc_bias")
		mo.uAlpha, mo.uLimit = NewBufferFloats(d, []float32{alpha}), NewBufferFloats(d, []float32{limit})
		mo.uHasBias = NewBufferU32(d, 1)
	}
	b1 := func(b bool) uint32 {
		if b {
			return 1
		}
		return 0
	}
	mo.uNE, mo.uK = NewBufferU32(d, uint32(nE)), NewBufferU32(d, uint32(k))
	mo.uSigmoid, mo.uNorm = NewBufferU32(d, b1(sigmoid)), NewBufferU32(d, b1(norm))
	mo.uScale = NewBufferFloats(d, []float32{float32(scale)})
	mo.uNGroup, mo.uTopkGroup = NewBufferU32(d, uint32(nGroup)), NewBufferU32(d, uint32(topkGroup))
	mo.uInter, mo.uInter2 = NewBufferU32(d, uint32(inter)), NewBufferU32(d, uint32(2*inter))
	mo.uSharedInter = NewBufferU32(d, uint32(sharedInter))
	mo.uSlot = make([]Buffer, k)
	for j := range mo.uSlot {
		mo.uSlot[j] = NewBufferU32(d, uint32(j))
	}
	mo.rLogits = d.NewBufferLen(nE)
	mo.rIdx = NewBufferUint32s(d, make([]uint32, k))
	mo.rWgt = d.NewBufferLen(k)
	mo.shGl, mo.shDown = d.NewBufferLen(1), d.NewBufferLen(H)
	if !isGptOss && (2*inter)%8 == 0 && H%8 == 0 { // 8 rows per 256-thread threadgroup, per slot
		mo.kSlots = true
		mo.pGUk, mo.pDownK = pipe("dp03_gemv_w4a8_moe_k"), pipe("dp03_gemv_w4a8_moe_down_k")
		mo.pSwRows, mo.pCombine = pipe("mc3_swiglu_quant_rows"), pipe("moe_combine_k")
		mo.guK, mo.dqK = d.NewBufferLen(k*2*inter), byteBuf(d, k*inter)
		mo.dScK, mo.accK = d.NewBufferLen(k), d.NewBufferLen(k*H)
	}
	// Synchronous paging: --moe-cache-slots / GOINFER_METAL_MOE_SLOTS (deprecated fallback,
	// metalMoESlotsRequest) keeps only N experts/layer resident and stages the routed top-k in
	// per token — same knob and semantics as gemma4_moe.go's (shared mechanism, generalized).
	// N==0/unset ⇒ all experts resident (today's behavior).
	if s := metalMoESlotsRequest(m); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil || n < k {
			return nil, fmt.Errorf("expert-slot request %q invalid (need integer >= topK=%d)", s, k)
		}
		if n < nE || moePagedAllSlotsForTest { // n>=nE would hold every expert — no paging, just build the stacked path
			mo.paged, mo.slots = true, n
			mo.slotIdx = NewBufferUint32s(d, make([]uint32, k))
			w := m.Weights()
			for l := range w.Layers {
				if skipPagedInt4CheckForTest {
					break
				}
				if ex := w.Layers[l].Experts; len(ex) > 0 {
					_, _, okg := int4DirectWords(&ex[0].Gate)
					_, _, oku := int4DirectWords(&ex[0].Up)
					_, _, okd := int4DirectWords(&ex[0].Down)
					if !okg || !oku || !okd {
						return nil, errPagedExpertsNotInt4
					}
				}
			}
		}
	}
	// Stage experts by pread'ing their nibbles straight into the slot buffers instead of byte-copying off the
	// mmap: the refinement gemma4_moe.go measured (the demand-fault page-in gone), applied to this shape. On by
	// default; needs a .giw-mmap'd model, since the offsets are into that file; re-opened once, shared across
	// layers. GOINFER_MOE_PREAD=0 opts out (the mmap byte-copy baseline, kept for the A/B). If the open fails or
	// the model is not .giw-backed, buildMoELayer falls back to the byte-copy.
	//
	// GOINFER_MOE_NOCACHE is deliberately not wired here: gemma4 measured it and declined it (no effect, and its
	// motivating evidence was an ordering confound; see buildGemma4MoE). Porting a declined flag would
	// re-open a settled question.
	if mo.paged && modelKnob(m, "GOINFER_MOE_PREAD") != "0" {
		if p := m.GiwPath(); p != "" {
			if f, err := os.Open(p); err == nil {
				mo.giwFile = f
			} else {
				fmt.Fprintf(os.Stderr, "metal MoE: open(%s) failed (%v) — using mmap byte-copy\n", p, err)
			}
		}
	}
	return mo, nil
}

// buildMoELayer packs one MoE layer's weights. Experts are row-stacked into ONE buffer per
// projection via the existing int4Concat (all-E gate|up interleaved gate_e,up_e; all-E down),
// so expert e lives at weight row e*2*inter (gate|up) / e*H (down). The router (+bias) stays
// f32; the shared expert is packed like the dense FFN.
func buildMoELayer(d *Device, m *decoder.Model, l int, lw *decoder.LayerWeights, mo *moeResident) *moeLayer {
	ml := &moeLayer{}
	ml.routerW = f32Mat(d, &lw.Router)
	if len(lw.RouterBias) > 0 {
		ml.routerBias = NewBufferFloats(d, lw.RouterBias)
	} else {
		ml.routerBias = NewBufferFloats(d, make([]float32, mo.nE)) // zeros → sel = score
	}
	if mo.paged {
		// Paged: do not stack the experts (the multi-GB set paging exists to avoid). Build a bounded LRU slot pool and
		// a stage fn reading expert e's W4A8 bytes from the layer's own WeightMats on demand; per-expert buffer sizes
		// come from expert 0. Mirrors buildGemma4MoELayer's paged branch, with byte-copy staging as the baseline and
		// the pread fast path below on top of it.
		gw0, gs0, ok1 := int4DirectWords(&lw.Experts[0].Gate)
		uw0, us0, ok1u := int4DirectWords(&lw.Experts[0].Up)
		dw0, ds0, ok2 := int4DirectWords(&lw.Experts[0].Down)
		if !ok1 || !ok1u || !ok2 {
			panic("metal MoE paging: experts are not int4-direct (unreachable: buildMoE declines first, errPagedExpertsNotInt4)")
		}
		nGuW, nGuS := len(gw0)+len(uw0), len(gs0)+len(us0)
		experts := lw.Experts // capture (aliases the model's own weights; kept alive by the Model)
		// Each expert's f16 scales are read from its own WeightMats (Int4ScalesF16), which a .giw
		// aliases from the mapping, rather than from a build-time heap cache that would hold the same bits again.
		stage := func(e int) ([]byte, []uint16, []byte, []uint16) {
			gw, _ := int4DirectBytesOnly(&experts[e].Gate)
			uw, _ := int4DirectBytesOnly(&experts[e].Up)
			dw, _ := int4DirectBytesOnly(&experts[e].Down)
			// gate|up fused per row-block: the stacked (non-paged) layout is gate rows THEN up rows
			// per expert (int4Concat(gate,up) order), so a slot must reproduce that concatenation —
			// gemv_w4a8_moe reads row r < inter as gate, r >= inter as up, off the SAME buffer.
			guBytes := append(append([]byte(nil), gw...), uw...)
			guS := append(append([]uint16(nil), experts[e].Gate.Int4ScalesF16()...), experts[e].Up.Int4ScalesF16()...)
			return guBytes, guS, dw, experts[e].Down.Int4ScalesF16()
		}
		ml.pool = newExpertPool(d, mo.slots, nGuW, nGuS, len(dw0), len(ds0), stage)
		// pread staging: resolve each expert's nibble file offsets within the .giw mmap (pure pointer arithmetic:
		// MmapByteOffset does not touch the pages), then stage by pread'ing straight into the slot's UMA words: zero
		// mmap faults, one large sequential read per projection.
		//
		// Three reads per expert, not gemma4's two: this shape keeps gate and up as separate tensors (gemma4's bundle
		// hands over one already-fused gate|up span), so the slot's fused gate|up buffer is filled in two ranges, gate
		// at byte 0 and up at byte len(gate), reproducing the byte-copy path's append(gw, uw...) and hence the stacked
		// layout gemv_w4a8_moe reads. The f16 scales are pread the same way, into the two halves of the slot's gate|up
		// scale range and the down scale buffer, when every expert's lie in the mapping (a .giw);
		// otherwise they are copied from the WeightMats (an older file converts them onto the heap at load).
		//
		// Falls back to the byte-copy path if the fd is absent or any expert's nibbles are not .giw-mmap-backed (a
		// requantized HF/safetensors load): every offset must resolve for pread to be correct, so this is
		// all-or-nothing per layer, never per expert.
		if mo.giwFile != nil {
			type expertSpan struct {
				gOff, uOff, dOff    int64
				gLen, uLen, dLen    int
				gsOff, usOff, dsOff int64 // the f16 scales' file offsets (C-P01), when scalesInFile
			}
			spans := make([]expertSpan, len(experts))
			resolved, scalesInFile := true, true
			for ei := range experts {
				gq, _, _, okg := experts[ei].Gate.Int4F16()
				uq, _, _, oku := experts[ei].Up.Int4F16()
				dq, _, _, okd := experts[ei].Down.Int4F16()
				if !okg || !oku || !okd {
					resolved = false
					break
				}
				gOff, ok1 := m.MmapByteOffset(gq)
				uOff, ok2 := m.MmapByteOffset(uq)
				dOff, ok3 := m.MmapByteOffset(dq)
				if !ok1 || !ok2 || !ok3 {
					resolved = false
					break
				}
				spans[ei] = expertSpan{gOff: gOff, uOff: uOff, dOff: dOff, gLen: len(gq), uLen: len(uq), dLen: len(dq)}
				var okgs, okus, okds bool
				spans[ei].gsOff, okgs = scaleFileOffset(m, experts[ei].Gate.Int4ScalesF16())
				spans[ei].usOff, okus = scaleFileOffset(m, experts[ei].Up.Int4ScalesF16())
				spans[ei].dsOff, okds = scaleFileOffset(m, experts[ei].Down.Int4ScalesF16())
				scalesInFile = scalesInFile && okgs && okus && okds
			}
			if resolved {
				fd := int(mo.giwFile.Fd())
				pool := ml.pool
				pool.stagePread = func(ei int, sl expertSlot) {
					sp := spans[ei]
					// The three spans write disjoint destination ranges (gate and up are different byte offsets within the slot's
					// gate|up region; down is a separate buffer), and pread on a shared fd is positional, so they are issued
					// concurrently. Errors are collected and panicked from this goroutine, not the spawned ones, so
					// BuildResident's recover() still sees them; a panic in an unrecovered goroutine would crash the whole
					// process.
					//
					// Use preadRangeIntoPoolSlot, not preadRangeIntoU32Buf(fd, sl.guW, ...): sl.guW is a Buffer.At()-offset view,
					// and U32s() ignores that offset (expertSlot's comment), which would silently pread every expert into slot 0.
					// preadRangeIntoPoolSlot addresses the pool's base buffer by slot number (sl.slot) plus the within-slot
					// sub-offset (0 for gate, sp.gLen for up).
					var wg sync.WaitGroup
					var errG, errU, errD error
					wg.Add(3)
					go func() {
						defer wg.Done()
						errG = preadRangeIntoPoolSlot(fd, pool.guW, sl.slot, pool.nGuW, 0, sp.gOff, sp.gLen)
					}()
					go func() {
						defer wg.Done()
						errU = preadRangeIntoPoolSlot(fd, pool.guW, sl.slot, pool.nGuW, sp.gLen, sp.uOff, sp.uLen)
					}()
					go func() {
						defer wg.Done()
						errD = preadRangeIntoPoolSlot(fd, pool.dW, sl.slot, pool.nDW, 0, sp.dOff, sp.dLen)
					}()
					// C-P01: with the scales in the file, they are three more disjoint ranges: gate's and
					// up's fill the two halves of the slot's gate|up scale range, down's its own buffer.
					var errGS, errUS, errDS error
					gsLen := 2 * len(experts[ei].Gate.Int4ScalesF16())
					if scalesInFile {
						wg.Add(3)
						go func() {
							defer wg.Done()
							errGS = preadScalesIntoPoolSlot(fd, pool.guS, sl.slot, pool.nGuS, 0, sp.gsOff, gsLen)
						}()
						go func() {
							defer wg.Done()
							errUS = preadScalesIntoPoolSlot(fd, pool.guS, sl.slot, pool.nGuS, gsLen, sp.usOff, 2*pool.nGuS-gsLen)
						}()
						go func() {
							defer wg.Done()
							errDS = preadScalesIntoPoolSlot(fd, pool.dS, sl.slot, pool.nDS, 0, sp.dsOff, 2*pool.nDS)
						}()
					}
					wg.Wait()
					for _, e := range []struct {
						what string
						err  error
					}{{"gate", errG}, {"up", errU}, {"down", errD}, {"gate scales", errGS}, {"up scales", errUS}, {"down scales", errDS}} {
						if e.err != nil {
							panic(fmt.Sprintf("metal MoE pread %s expert %d: %v", e.what, ei, e.err))
						}
					}
					if scalesInFile {
						pool.scalePreads.Add(1)
						return
					}
					// An older file: the scales are the WeightMats' heap copies, converted at load. Gate's
					// then up's fill the slot's gate|up scale range; the views are built from the pool's
					// base buffer by slot number, because Buffer.At sets an absolute offset rather than
					// adding to one.
					gs := experts[ei].Gate.Int4ScalesF16()
					base := sl.slot * pool.nGuS * 2
					copyU16sToBuf(pool.guS.At(base), gs)
					copyU16sToBuf(pool.guS.At(base+2*len(gs)), experts[ei].Up.Int4ScalesF16())
					copyU16sToBuf(sl.dS, experts[ei].Down.Int4ScalesF16())
				}
			}
		}
	} else {
		guMats := make([]*linalg.WeightMat, 0, 2*len(lw.Experts))
		for e := range lw.Experts {
			guMats = append(guMats, &lw.Experts[e].Gate, &lw.Experts[e].Up)
		}
		downMats := make([]*linalg.WeightMat, 0, len(lw.Experts))
		for e := range lw.Experts {
			downMats = append(downMats, &lw.Experts[e].Down)
		}
		if mo.w8 { // slice 4: every expert's int8 rows stacked, one f32 scale a row
			ml.expGuW, ml.expGuS = int8Concat(d, mo.alias, guMats...)
			ml.expDW, ml.expDS = int8Concat(d, mo.alias, downMats...)
		} else {
			ml.expGuW, ml.expGuS = int4Concat(d, guMats...)
			ml.expDW, ml.expDS = int4Concat(d, downMats...)
		}
	}
	if mo.sharedInter > 0 {
		var e error
		if mo.w8 {
			ml.shGuW, ml.shGuS = int8Concat(d, mo.alias, &lw.SharedExpert.Gate, &lw.SharedExpert.Up)
			ml.shDW, ml.shDS, e = int8BufA(d, mo.alias, &lw.SharedExpert.Down)
		} else {
			ml.shGuW, ml.shGuS = int4ConcatA(d, mo.alias, &lw.SharedExpert.Gate, &lw.SharedExpert.Up)
			ml.shDW, ml.shDS, e = int4BufA(d, mo.alias, &lw.SharedExpert.Down)
		}
		if e != nil {
			panic(e)
		}
		if !mo.sharedUngated {
			ml.shGateW = f32Mat(d, &lw.SharedGate)
		}
	}
	if mo.isGptOss {
		ml.expGuBias = NewBufferFloats(d, m.GptOssExpertBiasResident(l))
		ml.expDBias = NewBufferFloats(d, m.GptOssExpertDownBiasResident(l))
	}
	return ml
}

// f32Mat uploads a full-precision WeightMat (router / shared-gate) to a Metal f32 buffer.
func f32Mat(d *Device, w *linalg.WeightMat) Buffer {
	f, ok := w.F32()
	if !ok {
		panic(fmt.Sprintf("metal MoE: weight kind %q is not f32 (router/shared-gate must be full precision)", w.Kind()))
	}
	return NewBufferFloats(d, f)
}

// encodeMoEFFNWithX records the MoE FFN for one layer, replacing the dense gate/up/swiglu/down dispatches: post-attn
// norm, router logits, on-GPU top-k, per-selected-expert gate|up/swiglu/weighted-down, then the optional shared
// expert. Value-independent (idx/wgt are read at kernel-execution time), so the encode-ahead executor still
// pre-encodes it. x is the post-attention hidden state to route from: r.x for decode, a caller-owned buffer for
// batched prefill.
//
// This is the non-paged path: it reads the stacked all-E buffers (ml.expGuW/expGuS/expDW/expDS), which stay zero-value
// once the layer is paged (see moeLayer's comment on pool). forwardLogitsMoEPaged never reaches here for a paged layer
// (it tears the layer into encodeMoERouter and encodeMoEExpertsPaged around a host readback), so the panic below is a
// chokepoint against every other caller of encodeLayer (Forward, ForwardArgmax, forwardHiddenNoHead's encodeTrunkInto)
// reaching a paged layer through the non-paged encoder and silently computing off zero-value weights.
//
// FinishEncoding before the panic: e already holds this layer's attention or mixer dispatches, and Metal asserts if a
// command encoder is released without endEncoding ("[_MTLCommandEncoder dealloc]: failed assertion"), as an
// uncommitted, never-End()'d Encoder left for the Go GC to finalize would. Ending encoding without committing discards
// the partial work cleanly, with no GPU execution and no side effect on r.x.
func (r *resident) encodeMoEFFNWithX(e *Encoder, L *residLayer, x Buffer) {
	if L.moe.pool != nil {
		e.FinishEncoding()
		panic("metal: encodeMoEFFNWithX reached a paged MoE layer — route through forwardLogitsMoEPaged instead (C-02)")
	}
	r.encodeMoERouterWithX(e, L, x)
	r.encodeMoEExperts(e, L, x)
}

// encodeMoERouter is the value-INDEPENDENT head of the FFN: post-attn norm → router logits →
// on-GPU top-k, ending with rIdx[k]/rWgt[k] written on-device. In the paged forward this is the
// first command buffer; the host then reads rIdx and stages the routed experts before the second.
func (r *resident) encodeMoERouter(e *Encoder, L *residLayer) {
	r.encodeMoERouterWithX(e, L, r.x)
}

func (r *resident) encodeMoERouterWithX(e *Encoder, L *residLayer, x Buffer) {
	// post-attn RMSNorm → quantized activation mq/mSc (same as the dense FFN entry).
	e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, x, L.postNorm, r.mq, r.mSc, r.uH, r.uEps, r.uAddOne)
	r.encodeMoERoute(e, L)
}

// encodeMoERoute is encodeMoERouter's route-selection half, split out so the batched-prefill MoE row loop (prefill.go)
// can reuse it after its own norm step: rmsnorm_quant_f16 reading a row of the f16 residual directly into r.mq/r.mSc
// (r.pRms above is f32-only, so prefill cannot call encodeMoERouter). Reads r.mq/r.mSc (already normed and quantized
// by either caller); writes mo.rIdx/mo.rWgt. No r.x involvement.
func (r *resident) encodeMoERoute(e *Encoder, L *residLayer) {
	mo := r.moe
	ml := L.moe
	// router logits: routerW[nE,H] × mq → rLogits[nE]
	e.Dispatch(mo.pRouter, mo.nE*32, 32, ml.routerW, r.mq, r.mSc, mo.rLogits, r.uH)
	// on-GPU top-k → rIdx[k], rWgt[k]. gpt-oss's router bias reaches the logits BEFORE top-k and
	// the weight is softmax over the SELECTED biased logits — a genuinely different kernel from
	// the generic DeepSeek/GLM/Qwen-MoE shape (moe_route), not a parameter swap (see route_gptoss).
	if mo.isGptOss {
		e.Dispatch(mo.pRouteGptOss, 32, 32, mo.rLogits, ml.routerBias, mo.rIdx, mo.rWgt, mo.uNE, mo.uK)
	} else {
		e.Dispatch(mo.pRoute, 32, 32, mo.rLogits, ml.routerBias, mo.rIdx, mo.rWgt,
			mo.uNE, mo.uK, mo.uSigmoid, mo.uNorm, mo.uScale, mo.uNGroup, mo.uTopkGroup)
	}
	if r.moeForce.idx != (Buffer{}) { // test-only: this layer's routing replaced by the caller's (moeForce)
		off := r.encLayer * mo.k * 4
		e.Dispatch(r.pCopyU32, mo.k, 32, r.moeForce.idx.At(off), mo.rIdx, mo.uK)
		e.Dispatch(r.pCopyVec, mo.k, 32, r.moeForce.wgt.At(off), mo.rWgt, mo.uK)
	}
}

// moeKSlotsOn selects the k-slot dispatches (the k selected experts in four dispatches instead of 3k) where the shape
// admits them. Off: parked, bit-identical but too small a gain (docs/tasks/task-metal-audit-2026-10.md, "D-P03").
// Tests turn it on for the k-slot arm.
var moeKSlotsOn = false

// encodeMoEExperts runs the k selected experts out of the stacked all-E buffer (ml.expGuW/expDW, indexed by rIdx at
// kernel-execution time) plus the optional shared expert: the non-paged path.
//
// dst is the down-projection's accumulate target. Decode's caller (encodeMoEFFNWithX) passes x, the shared f32 residual
// stream; the batched-prefill MoE row loop (prefill.go) accumulates the same expert-loop math into an isolated f32
// scratch instead, because prefill's own residual (xF) is f16 and these kernels are f32-only (`device float*` versus
// `half*`), and adds that scratch into xF's row afterward with a separate small kernel.
func (r *resident) encodeMoEExperts(e *Encoder, L *residLayer, dst Buffer) {
	mo := r.moe
	ml := L.moe
	// k selected experts: fused gate|up (overwrite) → clamped-swiglu/swiglu → down (weighted-
	// accumulate into dst). gpt-oss fuses its per-expert gate‖up bias into the activation step and
	// its per-expert down bias into the combine step (both INSIDE the router-weight scale — see
	// swiglu_quant_gptoss / gemv_w4a8_moe_wacc_bias's docs), so it swaps both dispatches, not just
	// the router.
	if mo.kSlots && moeKSlotsOn {
		// D-P03: the same kernels' bodies over all k slots at once (batch_rows.go), then the per-slot weighted
		// accumulate as one combine in slot order: the same bytes in dst as the loop below.
		e.DispatchTG(mo.pGUk, mo.k*(2*mo.inter)*32, 256, r.H*2, ml.expGuW, ml.expGuS, r.mq, r.mSc, mo.guK, r.uH, mo.rIdx, mo.uK, mo.uInter2)
		e.Dispatch(mo.pSwRows, mo.k*256, 256, mo.guK, mo.guK, mo.dqK, mo.dScK, mo.uInter, r.uAct)
		e.DispatchTG(mo.pDownK, mo.k*r.H*32, 256, mo.inter*2, ml.expDW, ml.expDS, mo.dqK, mo.dScK, mo.accK, mo.uInter, mo.rIdx, mo.rWgt, mo.uK, r.uH)
		e.Dispatch(mo.pCombine, r.H, 256, dst, mo.accK, mo.rWgt, mo.dScK, mo.uK, r.uH)
		r.encodeMoESharedExpert(e, L, dst)
		return
	}
	for j := 0; j < mo.k; j++ {
		e.DispatchTG(mo.pGU, rowsGrid(2*mo.inter, mo.guR), 256, r.H*2, ml.expGuW, ml.expGuS, r.mq, r.mSc, r.gu, r.uH, mo.rIdx, mo.uSlot[j], mo.uInter2)
		if mo.isGptOss {
			e.Dispatch(mo.pActGptOss, 256, 256, r.gu, r.gu.At(mo.inter*4), r.dq, r.dSc, mo.uInter,
				ml.expGuBias, mo.biasIdx(), mo.uSlot[j], mo.uHasBias, mo.uAlpha, mo.uLimit)
			e.DispatchTG(mo.pDownWaccBias, r.H*32, 256, mo.inter*2, ml.expDW, ml.expDS, r.dq, r.dSc, dst, mo.uInter, mo.rIdx, mo.rWgt, mo.uSlot[j], r.uH, ml.expDBias, mo.biasIdx())
		} else {
			e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(mo.inter*4), r.dq, r.dSc, mo.uInter, r.uAct)
			e.DispatchTG(mo.pDownWacc, rowsGrid(r.H, mo.downR), 256, mo.inter*2, ml.expDW, ml.expDS, r.dq, r.dSc, dst, mo.uInter, mo.rIdx, mo.rWgt, mo.uSlot[j], r.uH)
		}
	}
	r.encodeMoESharedExpert(e, L, dst)
}

// biasIdx is the index for PER-EXPERT TABLES that stay STACKED for all experts and are addressed
// on the device — today gpt-oss's [nExpert][2*I] gate‖up bias and [nExpert][H] down bias. It is
// always the router's real expert ids, NEVER the slot index: those tables do not move when an
// expert is staged into a slot, so paging must not renumber their rows.
func (mo *moeResident) biasIdx() Buffer { return mo.rIdx }

// THERE IS DELIBERATELY NO SYMMETRIC weightIdx(). It was written and removed: keying it on
// mo.paged is WRONG, because paging is decided PER LAYER — forwardLogitsMoEPaged branches on
// L.moe.pool != nil, so a layer with no pool encodes through the ordinary non-paged path while
// mo.paged is true, and such an accessor would hand it slotIdx for its weights. That is the same
// silent-plausible-logits failure this pair exists to prevent, reintroduced by the symmetry. The
// weight index stays explicit at each call site, where which encoder you are in settles it.

// encodeMoEExpertsPaged is encodeMoEExperts' paged twin: the k selected experts run out of the layer's pool (one
// contiguous buffer per field) instead of the stacked all-E buffer. Dispatches always bind pool.guW/guS/dW/dS (fixed
// identity for the layer's lifetime); slotIdx (written into mo.slotIdx by the caller, one host write before this encode)
// tells the reused gemv_w4a8_moe(_wacc[_bias]) kernels which pool row holds each selected expert at kernel-execution
// time, while rWgt stays indexed by the selection slot uSlot[j]. That computes byte-identically to the stacked path: a
// pool row's bytes equal the stacked buffer's rows for that expert (see buildMoELayer's stage fn). Mirrors
// gemma4_moe.go's encodeG4Phase2Paged.
func (r *resident) encodeMoEExpertsPaged(e *Encoder, L *residLayer, pool *expertPool) {
	mo := r.moe
	ml := L.moe
	for j := 0; j < mo.k; j++ {
		e.DispatchTG(mo.pGU, rowsGrid(2*mo.inter, mo.guR), 256, r.H*2, pool.guW, pool.guS, r.mq, r.mSc, r.gu, r.uH, mo.slotIdx, mo.uSlot[j], mo.uInter2)
		if mo.isGptOss {
			// mo.rIdx, not slotIdx: this kernel's idx feeds ONLY biasOff, and the gate/up bias
			// table is the stacked all-expert one even on the paged path (C-09).
			e.Dispatch(mo.pActGptOss, 256, 256, r.gu, r.gu.At(mo.inter*4), r.dq, r.dSc, mo.uInter,
				ml.expGuBias, mo.biasIdx(), mo.uSlot[j], mo.uHasBias, mo.uAlpha, mo.uLimit)
			e.DispatchTG(mo.pDownWaccBias, r.H*32, 256, mo.inter*2, pool.dW, pool.dS, r.dq, r.dSc, r.x, mo.uInter, mo.slotIdx, mo.rWgt, mo.uSlot[j], r.uH, ml.expDBias, mo.biasIdx())
		} else {
			e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(mo.inter*4), r.dq, r.dSc, mo.uInter, r.uAct)
			e.DispatchTG(mo.pDownWacc, rowsGrid(r.H, mo.downR), 256, mo.inter*2, pool.dW, pool.dS, r.dq, r.dSc, r.x, mo.uInter, mo.slotIdx, mo.rWgt, mo.uSlot[j], r.uH)
		}
	}
	// Paging is never used by prefill (its own tiny/simple scope, G8), so this stays hardcoded to
	// r.x — unlike encodeMoEExperts' non-paged twin, there is no second caller to parameterize for.
	r.encodeMoESharedExpert(e, L, r.x)
}

// encodeMoESharedExpert is the always-on shared expert (Qwen2-MoE gated / GLM ungated) — never
// paged (it is one dense expert, always resident), so identical in both the paged and non-paged
// tail. dst is the same redirectable accumulate target encodeMoEExperts takes (see its own
// comment) — decode's two callers both pass r.x; prefill's row loop passes an isolated F32 scratch.
func (r *resident) encodeMoESharedExpert(e *Encoder, L *residLayer, dst Buffer) {
	mo := r.moe
	ml := L.moe
	if mo.sharedInter > 0 {
		r.gemvExt(e, gemvExtKSA, 2*mo.sharedInter, r.H, ml.shGuW, ml.shGuS, r.mq, r.mSc, r.gu, r.uH)
		e.Dispatch(r.pSw, 256, 256, r.gu, r.gu.At(mo.sharedInter*4), r.dq, r.dSc, mo.uSharedInter, r.uAct)
		if mo.sharedUngated {
			r.gemvExt(e, gemvExtKResid, r.H, mo.sharedInter, ml.shDW, ml.shDS, r.dq, r.dSc, dst, mo.uSharedInter) // dst += down
		} else {
			r.gemvExt(e, gemvExtKCoal, r.H, mo.sharedInter, ml.shDW, ml.shDS, r.dq, r.dSc, mo.shDown, mo.uSharedInter) // down → scratch
			e.Dispatch(mo.pRouter, 32, 32, ml.shGateW, r.mq, r.mSc, mo.shGl, r.uH)                                     // gate logit (1 row)
			e.Dispatch(mo.pSharedGate, r.H, 256, dst, mo.shDown, mo.shGl)                                              // dst += sigmoid(gl)*down
		}
	}
}

// forwardLogitsMoEPaged is the SYNCHRONOUS expert-paging decode for the generic MoE shape (the
// mechanism gemma4_moe.go's forwardLogitsPaged pioneered, generalized here to also cover a
// DeltaNet mixer layer's FFN half — qwen3_5_moe/qwen3_next pair a recurrent mixer with THIS FFN
// shape, not gemma4's). Per layer: dense/attention-only layers (including a DeltaNet layer whose
// FFN isn't paged, or when GOINFER_METAL_MOE_SLOTS is unset) encode in one command buffer via the
// existing encodeLayer; a paged MoE layer is torn at the router — the value-dependent seam —
// [mixer/attention + router] -> submit+wait -> read rIdx -> stage the routed top-k into the
// layer's LRU slot pool -> [experts-from-slots] -> submit+wait. Assumes the caller filled r.x with
// the embedding and holds the OS thread (ForwardEmb does both).
func (r *resident) forwardLogitsMoEPaged(pos int, ropePos ...int) (logits []float32) {
	// Same abort discipline as forwardLogitsPaged (audit R-02/C-09): a transient staging error or
	// OOM deep inside expertPool.ensureResident at decode time must not crash the process.
	defer func() {
		if p := recover(); p != nil {
			r.recordExecErr(fmt.Errorf("metal: paged MoE forward aborted: %v", p))
			logits = nil
		}
	}()
	// setPos, not a direct uPos/uNKeys write, so uQTempScale (FeatAttnTemp) refreshes on this path as setPos's
	// "called at every forward entry point" contract says. Currently a no-op (no paged family has FeatAttnTemp:
	// r.attnTempBeta==0 gives scale=1), but it would silently go stale the day one does.
	rp := pos
	if len(ropePos) > 0 {
		rp = ropePos[0]
	}
	r.setPos(pos, rp)
	mo := r.moe
	// Consecutive dense layers share one command buffer instead of a submit and wait each: only a paged MoE
	// layer's router readback is a value-dependent seam (idx/rWgt must land host-side before staging). dense is
	// opened on first use and closed at the next MoE layer or the loop's end, as encodeTrunkInto does for the
	// pure-dense path.
	async := pagedAsyncPhase2On // M-11: phase 2 committed, not waited (gemma4_moe.go, pagedAsyncPhase2On)
	var arp ARPool
	if async {
		arp = NewARPool()
	}
	var pending []*Encoder
	var dense *Encoder
	closeDense := func() {
		if dense == nil {
			return
		}
		dense.End()
		r.recordExecErr(dense.Err())
		dense = nil
	}
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		if L.moe != nil && L.moe.pool != nil {
			closeDense()
			e := r.q.Begin() // phase 1: mixer/attention + router -> rIdx/rWgt
			switch {
			case L.delta != nil:
				r.encodeDeltaNetMixer(e, L)
			case L.mamba != nil:
				r.encodeMambaMixer(e, L, r.x)
			case L.geom != nil:
				r.encodeAttention(e, l)
			}
			r.encodeMoERouter(e, L)
			e.End()
			r.recordExecErr(e.Err())
			idx := mo.rIdx.U32s()
			ids := make([]int, mo.k)
			for j := 0; j < mo.k; j++ {
				ids[j] = int(idx[j])
			}
			// Stage every miss in this token's top-k concurrently instead of one pread at queue depth 1 per expert
			// (ensureResidentBatch's comment says why that is safe).
			slots := L.moe.pool.ensureResidentBatch(ids)
			// Tell the GPU which pool row holds each routed expert: a small host-to-device write, not a re-encode
			// (moeResident.slotIdx).
			moIdx := mo.slotIdx.U32s()
			for j, s := range slots {
				moIdx[j] = uint32(expertPoolSlotForTest(s.slot, len(L.moe.pool.slotExpert)))
			}
			var e2 *Encoder // phase 2: experts from slots (+ shared expert)
			if async {
				e2 = r.q.BeginNP()
			} else {
				e2 = r.q.Begin()
			}
			if r.residency != (ResidencySet{}) { // M-14: per-encoder attach, phase 2 only
				e2.UseResidencySet(r.residency)
			}
			r.encodeMoEExpertsPaged(e2, L, L.moe.pool)
			if async {
				e2.FinishEncoding()
				e2.Commit()
				pending = append(pending, e2)
			} else {
				e2.End()
				r.recordExecErr(e2.Err())
			}
			continue
		}
		if dense == nil {
			dense = r.q.Begin()
		}
		r.encodeLayer(dense, l)
	}
	closeDense()
	for _, pe := range pending {
		pe.WaitDone()
		r.recordExecErr(pe.Err())
	}
	if async {
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

// encodeMoECapture copies this layer's routing (rIdx, rWgt) into slot `slot` of the test capture (resident.moeCap).
func (r *resident) encodeMoECapture(e *Encoder, slot int) {
	mo := r.moe
	off := slot * mo.k * 4
	e.Dispatch(r.pCopyU32, mo.k, 32, mo.rIdx, r.moeCap.idx.At(off), mo.uK)
	e.Dispatch(r.pCopyVec, mo.k, 32, mo.rWgt, r.moeCap.wgt.At(off), mo.uK)
}

// moeExpertRowsOn takes the routed-expert GEMVs through R18's rows form (D-B04's routed-expert half,
// gemv_w4a8_moe_rows / gemv_w4a8_moe_wacc_rows), bit-identical to the one-row kernels.
var moeExpertRowsOn = true

// moeExpertRows picks the routed-expert gate|up and down pipelines and their rows per simdgroup: the rows form when
// moeExpertRowsOn, ok and the output row counts admit it (gemvRowsFor, full threadgroups), else the one-row kernels.
func moeExpertRows(pipe func(string) Pipeline, guRows, downRows int, ok bool) (gu, down Pipeline, guR, downR int) {
	if moeExpertRowsOn && ok {
		guR, downR = gemvRowsFor(guRows, 4), gemvRowsFor(downRows, 4)
	}
	gu, down = pipe("gemv_w4a8_moe"), pipe("gemv_w4a8_moe_wacc")
	if guR > 0 {
		gu = pipe(fmt.Sprintf("gemv_w4a8_moe_rows%d", guR))
	}
	if downR > 0 {
		down = pipe(fmt.Sprintf("gemv_w4a8_moe_wacc_rows%d", downR))
	}
	return gu, down, guR, downR
}

// rowsGrid is a GEMV's thread count for `rows` outputs at R rows per simdgroup (0: one row).
func rowsGrid(rows, R int) int {
	if R > 0 {
		return rows * 32 / R
	}
	return rows * 32
}
