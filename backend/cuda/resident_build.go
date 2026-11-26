//go:build cuda

package cuda

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"

	"github.com/townsendmerino/aikit/linalg"
	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/backend/cuda/internal/launchgeom"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/decoder/fit"
	"github.com/townsendmerino/goinfer/internal/runtime/logs"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// buildDecline marks an error from a build phase as a decline: buildResidentMode's declined closure turns it into the typed
// decoder.DeclineResident, where any other error from a phase is returned as it is.
type buildDecline struct{ error }

func decline(err error) error { return buildDecline{err} }

// buildState is what a resident build resolves before it allocates anything, handed from one phase to the next. A phase fills the
// fields its successors read.
type buildState struct {
	m     *decoder.Model
	w     *decoder.Weights
	plans bool // plan-only or memory plan: the build stops before it allocates
	// kq reads the file's K-quant blocks for planProj; the build closes it when it returns.
	kq *decoder.KQuantReader

	shape program.Shape

	H, nLayers, nH, I, vocab int
	pleP                     int

	nE, topK, moeInter, sharedInter, nGroup, topkGroup int
	moeSig, moeNorm, sharedUngated, isMoE              bool
	moeScale                                           float64
	isG4MoE, isGemma4                                  bool

	qLoRA, kvLoRA, qkNope, qkRope, vHead int
	interleave                           bool
	mlaAttnScale, mlaRopeScale           float64
	mlaOK                                bool

	dnConvK, dnHK, dnHV, dnNK, dnNV int
	dnAttnGate, dnetOK              bool
	dnetP                           *dnetParams
	mambaP                          *mambaParams

	sandwich, postOnly, qkNormWhole, layerNorm, parallelBlock bool
	logitScale                                                float32
	actG32                                                    bool

	maxQDim, maxHd int // the widest attention width and head dim over the softmax layers, set by sizeAttnScratch

	hls       []hlayer // each layer's planned weights, filled by planLayers
	hlm       hostW    // the head's
	loadAbort <-chan struct{}
}

// planProj plans a projection or the head: the file's own Q5_K or Q6_K blocks where the q4k lane uploads them natively
// (decoder.Model.KQuantNative, which the fit pricing also reads), else planWeight. s.kq reads the file until the build returns.
// A weight the load already holds as Q5_K or Q6_K (a cuda .giw carries them, decoder.giwCarriesKQuant56) is planned from its
// own blocks: no int8 copy exists, and the fit pricing counts it at its native bytes.
func (s *buildState) planProj(src *linalg.WeightMat) (hostW, error) {
	m := s.m
	if k := src.Kind(); k == "q5k" || k == "q6k" {
		if !s.actG32 {
			return hostW{}, decline(fmt.Errorf("a native %s weight needs per-32 activations (a q4k load's), and this load has none", k))
		}
		return planKQuantHeld(src)
	}
	kind := m.KQuantNative("cuda", src)
	if kind == "" {
		return planWeight(src)
	}
	if s.kq == nil {
		var e error
		if s.kq, e = m.OpenKQuant(); e != nil || s.kq == nil {
			return hostW{}, fmt.Errorf("cuda: the file's %s blocks are unreadable: %v", kind, e)
		}
	}
	row, e := s.kq.RowSource(src)
	if e != nil {
		return hostW{}, e
	}
	h, e := planKQuant(src, kind, row)
	if e == nil && kquantKeepInt8ForTest {
		alt, e2 := planWeight(src)
		if e2 != nil {
			return hostW{}, e2
		}
		h.alt = &alt
	}
	return h, e
}

// admit resolves the geometry the build dispatches on and declines what this backend does not implement: the features outside
// the program, a layer shape or operation CUDA's table lacks, and the per-family and per-32 activation sub-shapes. It reads the
// model and allocates nothing.
func (s *buildState) admit() error {
	m := s.m
	bv := m.BackendView()
	w := bv.Weights()
	if w == nil || len(w.Layers) == 0 {
		return decline(fmt.Errorf("the model has no loaded layer weights"))
	}
	// Admission: anything this backend does not implement must decline, or it is silently dropped and the logits are wrong. The
	// operations and parameters are checked against CUDA's table (program.CUDA) below; the features the program does not express
	// (decoder.Model.FeaturesOutsideProgram) are checked here.
	if missing := m.FeaturesOutsideProgram(); len(missing) > 0 {
		return decline(fmt.Errorf("cuda does not implement %v, which this model needs", missing))
	}
	// The layer shape the build dispatches on comes from the program (OP4cu-b1); a program whose layers disagree on it, or ask
	// for two placements at once, declines here, naming the layer.
	shape, shapeErr := program.ShapeOf(m.Program())
	if shapeErr != nil {
		return decline(shapeErr)
	}
	// Every operation and parameter must be one CUDA's walker implements (program.CUDA, OP4cu-c1); the decline names the first
	// layer, operation and parameter outside the table.
	if tableErr := program.CUDA.Check(m.Program()); tableErr != nil && legacyLayerForTest == nil {
		return decline(tableErr)
	}

	dims := bv.Dims() // nKV, hd are per-layer now (KVHeadsAtResident/HeadDimAtResident); model-level unused
	H, nLayers, nH, I, vocab := dims.Hidden, dims.Layers, dims.Heads, dims.Inter, dims.Vocab
	// Gemma 4 E-model PLE width P (0 for every other model). The resident embedding row is then [hidden ‖ nLayers*P] (EmbedLen):
	// the host computes the per-layer inputs, and each layer's PLE branch reads its slice of the tail.
	pleP := bv.Gemma4PLEDim()
	if want := H + nLayers*pleP; pleP > 0 && bv.EmbedLen() != want {
		return decline(fmt.Errorf("gemma4 E-model: the decoder's resident embedding row is %d floats, but hidden+layers*P is %d", bv.EmbedLen(), want))
	}

	// ---- MoE knobs (ok=false for a dense model; every field then stays zero) ----
	// sharedUngated picks which shared-expert combine runs: GLM/DeepSeek add the shared output
	// straight into the residual, Qwen-MoE scales it by sigmoid(SharedGate·h) first. Both are
	// the same kernel (shared_gate_combine, `ungated` flag); only the gated one needs the extra
	// [1,hidden] weight.
	mo, isMoE := bv.MoE()
	nE, topK, moeInter, sharedInter := mo.NumExperts, mo.TopK, mo.IntermediateDim, mo.SharedIntermediateDim
	moeSig, moeNorm, sharedUngated, moeScale := mo.RouterSigmoid, mo.NormTopKProb, mo.SharedUngated, mo.RoutedScale
	nGroup, topkGroup := mo.NGroup, mo.TopkGroup
	// Gemma-4 enable_moe_block sets arch.MoE (so isMoE is true) but is NOT the generic MoE: it runs
	// gemma4MoeMLP (parallel dense‖MoE + join), so it bypasses the generic-MoE admission checks below
	// (gelu-tanh act, the attention sandwich norm) and gets its own int4-shape checks. Its nE/topK/
	// moeInter still come from arch.MoE (MoE), matching the bundle.
	isG4MoE := bv.HasGemma4MoE()
	isGemma4 := bv.IsGemma4()
	if isMoE && !isG4MoE {
		// Decline anything the dispatch does not implement loudly rather than dropping it; FeatMoE is one flag and cannot
		// express these sub-shapes. The gated shared expert (Qwen-MoE) is a derived feature (FeatMoEGatedShared,
		// decoder/features.go) that CUDA does not declare, so the admission check above declines it before this switch; a copy
		// of that decline here could drift from the taxonomy. moe_route's MOE_MAX_E / MOE_MAX_G come from decoder's declaration
		// (ResidentBackendMoECap), which the load path's admission gate already applied; do not restate the numbers here.
		capE, capG, _ := decoder.ResidentBackendMoECap("cuda")
		switch {
		case nE > capE:
			return decline(fmt.Errorf("MoE nE=%d exceeds moe_route's MOE_MAX_E=%d", nE, capE))
		case nGroup > capG:
			return decline(fmt.Errorf("MoE nGroup=%d exceeds moe_route's MOE_MAX_G=%d", nGroup, capG))
		case bv.GatedAct() != 1: // decoder.ActSiLU — decoder/mlp.go errors on any other
			return decline(fmt.Errorf("MoE experts are SwiGLU-only, arch act=%d", bv.GatedAct()))
		case shape.Sandwich:
			return decline(fmt.Errorf("MoE + sandwich norms: the combine accumulates straight into " +
				"the residual, so there is no seam to normalize the block output at"))
		case moeInter%32 != 0 || H%32 != 0:
			return decline(fmt.Errorf("MoE int4 needs moeInter(%d) and hidden(%d) both multiples of 32", moeInter, H))
		case sharedInter > 0 && sharedInter%32 != 0:
			return decline(fmt.Errorf("MoE int4 shared expert needs sharedInter(%d) a multiple of 32", sharedInter))
		}
	}
	if isG4MoE {
		// gemma4MoeMLP's indexed-expert + dense GEMVs are int4/W4A8: the widths the kernels stride by
		// must be multiples of the group size (hidden) / 8-nibble word (moeInter, denseInter).
		if moeInter%32 != 0 || H%32 != 0 {
			return decline(fmt.Errorf("gemma4 MoE int4 needs moeInter(%d) and hidden(%d) both multiples of 32", moeInter, H))
		}
		if capE, _, _ := decoder.ResidentBackendMoECap("cuda"); nE > capE {
			return decline(fmt.Errorf("gemma4 MoE nE=%d exceeds moe_route's MOE_MAX_E=%d", nE, capE))
		}
	}

	// ---- host pack all weights (CPU; any incompatible shape → decline) ----
	// MLA geometry (DeepSeek / Kimi).
	mla, mlaOK := bv.MLA()
	qLoRA, kvLoRA, qkNope, qkRope, vHead := mla.QLoRARank, mla.KVLoRARank, mla.QKNopeHeadDim, mla.QKRopeHeadDim, mla.VHeadDim
	interleave, mlaAttnScale, mlaRopeScale := mla.RopeInterleave, mla.AttnScale, mla.RopeScale

	// Gated-DeltaNet hybrid geometry. Resolved before the per-layer pack because BOTH layer kinds
	// branch on it: the linear layers take the recurrence, and the softmax ones carry the fused
	// double-width q_proj that no other family has.
	dn, dnetOK := bv.Qwen35()
	dnConvK, dnHK, dnHV, dnNK, dnNV, dnAttnGate := dn.ConvKernel, dn.KeyHeadDim, dn.ValueHeadDim, dn.NumKeyHeads, dn.NumValueHeads, dn.AttnGate
	var dnetP *dnetParams
	if dnetOK {
		keyDim, valueDim := dnNK*dnHK, dnNV*dnHV
		dnetP = &dnetParams{
			convK: dnConvK, hk: dnHK, hv: dnHV, nk: dnNK, nv: dnNV, rep: dnNV / dnNK,
			keyDim: keyDim, valueDim: valueDim, convDim: 2*keyDim + valueDim,
			stateElems: dnNV * dnHV * dnHK,
			qScale:     float32(1 / math.Sqrt(float64(dnHK))),
		}
	}

	// Mamba-2 hybrid geometry (Nemotron-H). A block of that family carries one operation (cudaMixerOf), and its non-gated MoE blocks are not built
	// here: they decline by name, as does Granite-4.0-H, and an int8 load (buildMambaParams).
	mambaP, mambaErr := buildMambaParams(m)
	if mambaErr != nil {
		return decline(mambaErr)
	}
	if mambaP != nil && isMoE {
		return decline(fmt.Errorf("Nemotron-H's non-gated relu² MoE blocks are not built on CUDA"))
	}

	sandwich := shape.Sandwich
	// Olmo 3 / Olmo Hybrid: post-only norms.
	postOnly := shape.PostOnly
	qkNormWhole := shape.QKNormWhole
	// Cohere/Command-R and Cohere2/Command-R7B: LayerNorm and parallel blocks.
	layerNorm := shape.LayerNorm
	parallelBlock := shape.Parallel
	logitScale, _ := bv.LogitScale() // ok=false ⇒ 1, already the applyLogitScale no-op value
	// Per-32 ACTIVATION quantization (decoder.Options.ActQuantGroup, actgroup.cu): implemented for the
	// plain dense RMSNorm decode path. Every path whose kernels read ONE activation scale per vector
	// declines by name rather than reading a per-group buffer as a per-vector one.
	actG32 := m.ActQuantGroup() == 32
	if actG32 {
		switch {
		case isMoE && !moeActG32DeclineOffForTest:
			return decline(fmt.Errorf("per-32 activations: the MoE expert/router kernels read per-vector activation scales"))
		case mlaOK:
			return decline(fmt.Errorf("per-32 activations: the MLA path is not implemented"))
		case isGemma4:
			return decline(fmt.Errorf("per-32 activations: the Gemma 4 path is not implemented"))
		case dnetOK:
			return decline(fmt.Errorf("per-32 activations: the Gated DeltaNet path is not implemented"))
		case layerNorm, postOnly, parallelBlock:
			return decline(fmt.Errorf("per-32 activations: LayerNorm / post-norm-only / parallel-block families are not implemented"))
		case H%32 != 0 || I%32 != 0:
			return decline(fmt.Errorf("per-32 activations need hidden (%d) and FFN width (%d) to be multiples of 32", H, I))
		}
	}

	s.w, s.shape = w, shape
	s.H, s.nLayers, s.nH, s.I, s.vocab, s.pleP = H, nLayers, nH, I, vocab, pleP
	s.nE, s.topK, s.moeInter, s.sharedInter, s.nGroup, s.topkGroup = nE, topK, moeInter, sharedInter, nGroup, topkGroup
	s.moeSig, s.moeNorm, s.sharedUngated, s.isMoE, s.moeScale = moeSig, moeNorm, sharedUngated, isMoE, moeScale
	s.isG4MoE, s.isGemma4 = isG4MoE, isGemma4
	s.qLoRA, s.kvLoRA, s.qkNope, s.qkRope, s.vHead, s.interleave = qLoRA, kvLoRA, qkNope, qkRope, vHead, interleave
	s.mlaAttnScale, s.mlaRopeScale, s.mlaOK = mlaAttnScale, mlaRopeScale, mlaOK
	s.dnConvK, s.dnHK, s.dnHV, s.dnNK, s.dnNV, s.dnAttnGate, s.dnetOK, s.dnetP = dnConvK, dnHK, dnHV, dnNK, dnNV, dnAttnGate, dnetOK, dnetP
	s.mambaP = mambaP
	s.sandwich, s.postOnly, s.qkNormWhole, s.layerNorm, s.parallelBlock = sandwich, postOnly, qkNormWhole, layerNorm, parallelBlock
	s.logitScale, s.actG32 = logitScale, actG32
	return nil
}

// planLayers validates and host-plans every layer's weights and the head, declining what CUDA cannot hold, and leaves s.hls and
// s.hlm for the upload. With s.plans set the experts are validated instead of packed. It allocates nothing on the device.
func (s *buildState) planLayers() error {
	m, w, planProj, plans := s.m, s.w, s.planProj, s.plans
	bv := m.BackendView()
	nLayers := s.nLayers
	if planStartForTest != nil && !plans {
		planStartForTest()
	}
	hls := make([]hlayer, nLayers)
	// Under the C' expert cache the routed experts are packed straight into their pinned host copy when the build uploads, so planning only validates them.
	expertStack, sharedStack, sharedOne := packWeightStack, packWeightStack, packWeight
	if m.MoECacheExperts() {
		expertStack = planWeightStack
	}
	if plans {
		expertStack, sharedStack, sharedOne = planWeightStack, planWeightStack, planWeight
	}
	loadAbort := bv.LoadAbort() // the swap tripwire's channel (Options.LoadAbort); nil never fires
	for l := range nLayers {
		if e := s.planLayer(l, hls, loadAbort, expertStack, sharedStack, sharedOne); e != nil {
			return e
		}
	}
	lmSrc := &w.LMHead
	if w.LMHead.Rows() == 0 {
		lmSrc = &w.Embed // tied embeddings
	}
	hlm, e := planProj(lmSrc)
	if e != nil {
		return decline(e)
	}
	if planCompleteForTest != nil && !plans {
		planCompleteForTest()
	}
	s.hls, s.hlm, s.loadAbort = hls, hlm, loadAbort
	return nil
}

// newResident builds the cudaResident the setup job fills: its fields from the planned geometry, the executor goroutine, the layer
// geometry and the KV context plan (r.ctxCap), declining what it cannot hold. Plan-only and the memory plan end here, returning
// planOnlySlack and errMemPlanned. A MoE build whose context is planned opens its device here (openForPlan), and every error
// return after that closes it.
func (s *buildState) newResident(planOnly bool, mp *memPlanRun) (*cudaResident, error) {
	m, shape := s.m, s.shape
	bv := m.BackendView()
	H, nLayers, nH, I, vocab, pleP := s.H, s.nLayers, s.nH, s.I, s.vocab, s.pleP
	nE, topK, moeInter, sharedInter, nGroup, topkGroup := s.nE, s.topK, s.moeInter, s.sharedInter, s.nGroup, s.topkGroup
	moeSig, moeNorm, isMoE, moeScale := s.moeSig, s.moeNorm, s.isMoE, s.moeScale
	isG4MoE, isGemma4 := s.isG4MoE, s.isGemma4
	qLoRA, kvLoRA, qkNope, qkRope, vHead, interleave := s.qLoRA, s.kvLoRA, s.qkNope, s.qkRope, s.vHead, s.interleave
	mlaAttnScale, mlaRopeScale, mlaOK := s.mlaAttnScale, s.mlaRopeScale, s.mlaOK
	dnetP, mambaP := s.dnetP, s.mambaP
	sandwich, postOnly, qkNormWhole, layerNorm, parallelBlock := s.sandwich, s.postOnly, s.qkNormWhole, s.layerNorm, s.parallelBlock
	logitScale, actG32 := s.logitScale, s.actG32
	hls, hlm, loadAbort := s.hls, s.hlm, s.loadAbort
	attnTempBeta, attnTempOrigMaxPos := m.BackendView().AttnTempParams() // Ministral 3, FeatAttnTemp; 0 for every other family
	// Computed once: both ctxCap and extraBytes below need it, and a struct literal cannot read a sibling field set in the
	// same literal.
	kvSlotsReq := cudaKVSlotsRequest(m)
	// The driver's 2 MiB rounding over the weight buffers is the plan's figure (decoder.Model.PlanSlack, this package's
	// report): the plan prices it as a fixed term, so the context it picks already leaves room for it, and the build adds no
	// copy of its own. Plan-only, it is computed here, over the buffers just planned.
	if planOnly {
		return nil, planOnlySlack(packedAllocSlack([]any{hls, hlm}))
	}
	allocSlack := m.PlanSlack("cuda")
	if allocSlackSeenForTest != nil && mp == nil {
		allocSlackSeenForTest(allocSlack, packedAllocSlack([]any{hls, hlm}))
	}
	attnCap := bv.AttnLogitSoftcap()
	if attnSoftcapDropForTest {
		attnCap = 0
	}
	if attnCap != 0 && mlaOK {
		return nil, decline(fmt.Errorf("an attention softcap on a latent-attention model: mla_attn has no cap"))
	}
	r := &cudaResident{
		prog:      m.Program(),
		loadAbort: loadAbort,
		knob:      m.Knob,
		kvF16:     m.KVCacheF16(),
		kvI8:      m.KVDeliveredFor("cuda") == mem.KVI8,
		hidden:    H, nLayers: nLayers, nH: nH, inter: I, vocab: vocab,
		eps: bv.NormEps(), attnScale: bv.AttnScale(), finalSoftcap: bv.FinalLogitSoftcap(), attnCap: attnCap, faSoftcap: m.FlashDecodeSoftcap(), faHD96: m.FlashDecodeHD96(), faHD96Default: m.FlashDecodeHD96Default(),
		attnTempBeta: attnTempBeta, attnTempOrigMaxPos: attnTempOrigMaxPos,
		qkNorm: shape.QKNorm, qkNormWhole: qkNormWhole, rmsAddOne: bv.RMSAddOne(),
		act: int32(bv.GatedAct()), sandwich: sandwich, postOnly: postOnly,
		layerNorm: layerNorm, parallelBlock: parallelBlock, logitScale: logitScale, actG32: actG32,
		moe: isMoE, nE: nE, topK: topK, moeInter: moeInter,
		moeScale: float32(moeScale), nGroup: nGroup, topkGroup: topkGroup,
		sharedInter: sharedInter,
		gemma4Moe:   isG4MoE, gemma4Dense: isGemma4, g4cap: os.Getenv("GOINFER_G4_CAPTURE") != "",
		// C′: DMA the routed int4 experts host→VRAM slots per token (device read, correct). The
		// path to running a model whose experts exceed VRAM. Off by default; byte-identical when off.
		cacheExperts: m.MoECacheExperts(),
		// MoEExpertsUnpinned: the cache's slots upload from pageable pages of a packed-experts file beside the model, not a pinned copy.
		unpinnedExperts: m.MoEExpertsUnpinned(), unpinnedDir: modelDir(m.SourcePath()),
		cacheProf: os.Getenv("GOINFER_MOE_CACHE_PROF") != "",
		dnet:      dnetP,
		mamba:     mambaP,
		isMLA:     mlaOK, mlaRank: kvLoRA, mlaLatDim: kvLoRA + qkRope,
		mlaQKHead: qkNope + qkRope, mlaQKNope: qkNope, mlaQKRope: qkRope,
		mlaVHead: vHead, mlaQLoRA: qLoRA, mlaInterleave: interleave,
		mlaRopeScale:    float32(mlaRopeScale),
		allocSlackBytes: allocSlack,
		ctxExplicit:     ctxExplicitFor(m),
		// One conversation may use the whole KV pool (KV4a-3) unless a context was asked for or pinned by the fit guard, or a companion's
		// device K/V scales with the context (a --drafter's, priced at the planned one).
		kvRaise:    m.ResidentContextRequest() <= 0 && m.ExtraResidentKVPerPosition() == 0,
		modelCtx:   m.Config().MaxPositions,
		kvSlotsReq: kvSlotsReq,
	}
	if moeSig {
		r.moeSigmoid = 1
	}
	if moeNorm {
		r.moeNormTopK = 1
	}
	if mlaOK && mlaAttnScale != 0 {
		r.attnScale = float32(mlaAttnScale)
	}
	// C' device expert slots per layer: an LRU cache of nSlots experts (clamped [topK, nE]), costing
	// nLayers*nSlots*perExpert of VRAM; more slots trade VRAM for fewer per-token DMAs. The request comes from
	// --moe-cache-slots (GOINFER_MOE_CACHE_SLOTS is still honoured).
	//
	// With caching on and no explicit request the default is 8*topK (capped at nE): a bounded multiple, neither topK nor the
	// "ask for all, let allocSlots cap to free VRAM" that MoECacheSlotsRequest documents for 0. The floor is topK (one
	// token's routed set must be resident at once) and allocSlots still caps to measured free VRAM. Do not default to "ask
	// for all": allocSlots's headroom (marginBytes) covers per-token costs, not what the forward allocates after it, so on
	// the real 26B it capped 128 slots to 34 and the warm forward then died with CUDA_ERROR_OUT_OF_MEMORY. 8*topK sits just
	// above the knee of one measured sweep; it is a heuristic, not a derived constant, because where the knee falls depends
	// on the model's routing entropy. The sweep and the history:
	// docs/code-notes/cuda.md#BuildResident: MoE cache slots default.
	r.cacheSlots = topK
	r.cacheSlotsReq = topK
	if r.cacheExperts && nE > 0 {
		if d := 8 * topK; d > topK {
			r.cacheSlots = min(d, nE)
			r.cacheSlotsReq = r.cacheSlots
		}
	}
	if r.cacheExperts {
		// The request comes from Options (--moe-cache-slots); unset keeps the 8*topK default above, not the accessor's "ask for
		// all". A request BELOW topK cannot be honoured, since one token's own top-k must fit, so it is refused rather than
		// floored: flooring left the 8*topK default in place, and a gate asking for fewer slots than experts got the identity
		// mapping and discriminated by routing luck.
		if req := m.MoECacheSlotsRequest(); req > 0 && req < topK {
			// A hard error, not declined(): declined() swallows the error and falls back to the staged path, which suits a shape
			// this backend does not implement but not an operator flag that cannot mean what it says.
			return nil, fmt.Errorf("cuda: --moe-cache-slots %d is below top-k %d — one "+
				"token's own routed experts must all be resident at once, so this cannot be "+
				"honoured", req, topK)
		} else if req >= topK {
			r.cacheSlots = req
			r.cacheSlotsReq = req
			if nE > 0 && r.cacheSlots > nE {
				r.cacheSlots = nE
			}
		}
	}
	// The context margin, on the planned geometry, before the context is planned (ctxMarginFor): the setup job replaces r.layers,
	// and the fused-attention flags are set again where the modules load. Read recurrent floats before starting the executor
	// goroutine so an early error never leaks the locked OS thread.
	r.layers = planLayerGeometry(m, hls, nH, r)
	if e := r.readRecurrentFloats(m); e != nil {
		return nil, decline(e)
	}
	if mp == nil { // the memory plan runs no job: it opens no device
		r.reqCh = make(chan func() error)
		r.ackCh = make(chan error)
		go func() {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			for j := range r.reqCh {
				r.ackCh <- runJob(j)
			}
		}()
	}
	r.pleP, r.eModel = pleP, pleP > 0
	for l := range r.layers {
		r.eModel = r.eModel || r.layers[l].kvShared
	}
	r.fastAttn, r.fastGemm = fastPrefillEnabled(r.knobValue("GOINFER_CUDA_FAST_PREFILL"))
	if m.ExactPrefill() {
		r.fastAttn, r.fastGemm = false, false
	}
	r.planning = true
	// A MoE build whose context is planned opens its device and pays moe_route's local-memory pool (payRoutePool) before the plan
	// reads free VRAM, on the context it keeps: the startup self-test's pool is released with its own context before the build, so
	// nothing else has paid it. It checks the weights
	// first, against the margin at the floor, so a model that does not fit gets checkWeightsFit's decline rather than a warm-up's
	// failure. The setup job keeps this device and stream, and checks the weights again against the margin it plans.
	if _, floor, plans := ctxPlanRange(m, m.ResidentContextRequest(), m.Config().MaxPositions); r.moe && plans {
		r.ctxMarginBytes = r.ctxMarginFor(floor)
		if mp != nil {
			mp.out.Margin, mp.out.ExtraBytes, mp.out.KVSlotsReq = r.ctxMargin(), m.ExtraResidentBytes(), r.kvSlotsReq
			if e := mp.weightsCheck(m, "weightsOpen", r.ctxMargin()); e != nil {
				return nil, e
			}
			mp.out.PoolPaid = true // openForPlan pays the pool only after this check passes
			if mp.derive {
				mp.in.Plan, mp.in.Weights = mp.in.WeightsOpen-mp.pool, mp.in.WeightsOpen-mp.pool
			}
		} else if e := r.do(func() error { return r.openForPlan(m) }); e != nil {
			r.Close()
			return nil, decline(e)
		}
	}
	owners, mlaPools := kvLayerCounts(r.layers)
	// Resolve the resident KV capacity HERE, at construction, not at the KV allocation site: several buffers are sized from it
	// earlier (the split-KV score scratch among them), and a zero-value ctxCap makes those 0-byte allocations that fail the whole
	// resident build. cap = min(model context window, request); request 0 ⇒ the 4096 default, so a caller who did not ask
	// allocates exactly what they always did.
	// The plan reads free VRAM: a MoE build has paid its local-memory pool above, on the context it keeps.
	readFree := liveCtxFree
	if mp != nil {
		readFree = func() (int64, bool) {
			if mp.derive {
				mp.in.PlanOK = true // called only when the context is planned
			}
			return mp.in.Plan, mp.in.PlanOK
		}
	}
	if recurrentStatePlannedForTest != nil && mp == nil {
		recurrentStatePlannedForTest(m.Plan("cuda", 1<<40, fit.PlanRequest{Ctx: 1}).StateBytes) // the state does not grow with the context
	}
	r.ctxCap, r.ctxMarginBytes = resolveCtxCapFitFree(m, m.ResidentContextRequest(), m.Config().MaxPositions, kvSlotsReq,
		func(ctx int) int64 { r.ctxMarginCtx = ctx; return r.ctxMarginFor(ctx) },
		kvRoundingBound(owners, mlaPools), readFree)
	planned := r.layers
	r.planning, r.layers = false, nil
	r.ctxPlanned = r.ctxCap
	// A drafter's device K/V is priced at the FINAL ctxCap. resolveCtxCapFit's ExtraBytes consult priced it at candidate, the
	// widest ctx that call considered, and the real allocation can only be <= candidate, so this is never a smaller price than the
	// one planned against; checkKVFits enforces it.
	r.extraBytes = m.ExtraResidentBytes() + m.ExtraResidentKVPerPosition()*int64(r.ctxCap)
	if mp != nil {
		return nil, r.finishMemPlan(m, hls, planned, mp)
	}
	return r, nil
}

// bindKernels creates the device on the pinned thread, binds it to this thread, checks that the fixed weights fit, and compiles and
// binds every kernel the model's features need, recording on r which optional paths (batched prefill, split-KV, flash decode, MoE,
// the family modules) loaded. It runs inside the setup job and uploads no weights; the device is not drained to refusal here
// (docs/code-notes/cuda.md#BuildResident: device exhaustion).
func (s *buildState) bindKernels(r *cudaResident) error {
	m, nLayers := s.m, s.nLayers
	bv := m.BackendView()
	var e error
	if r.dev == nil { // a MoE build opened it before the plan (openForPlan)
		if r.dev, e = CreateSystemDefaultDevice(); e != nil {
			return e
		}
	}
	// launch issues kernels from this thread (Queue.LaunchDirect), which needs the context current here.
	if e := r.dev.BindCurrentThread(); e != nil {
		return e
	}
	// checkWeightsFit runs first, right after the device exists and before the kernel compiles and uploads, and before this
	// backend's other declines, so a model whose fixed (non-expert) weights exceed free VRAM gets a clean decline naming
	// that rather than whichever raw driver error the first oversized allocation produced.
	if e := r.checkWeightsFit(m); e != nil {
		return e
	}
	// The module and pipeline handles cached here do not survive device exhaustion. Once the device has been drained to
	// refusal, a later launch through a cached handle returns SUCCESS and executes nothing (an all-zero output, no error,
	// and cuFuncGetAttribute answers normally); only re-loading the module and re-resolving the function immediately before
	// the launch restores it. goinfer avoids this because BuildResident declines on exhaustion and issues no more CUDA, and
	// that decline is a safety property, not an incidental fallback.
	//
	// Any change that drives the device to refusal and then keeps using the same context breaks this, silently, as zeros. A
	// retry loop, an eviction-and-rebuild or a "just try a smaller cache" must re-load rather than reuse these handles.
	// Evidence, path by path: docs/code-notes/cuda.md#BuildResident: device exhaustion.
	if e := s.bindGlueKernels(r, bv); e != nil {
		return e
	}
	if e := s.bindSideKernels(r); e != nil {
		return e
	}
	s.bindPrefillKernels(r, bv)
	s.bindFastPrefill(r, m)
	// Split-KV decode attention: a high-occupancy, bit-identical alternative to the attn_batched(M=1) decode launch (it
	// replaces exactly that launch, so it needs prefillReady). Own module. skMinKeys -1 means use the per-geometry table;
	// set it before the load so a partial split-KV load cannot leave the zero value, which would mean "always split".
	r.skMinKeys = -1
	s.bindGptOss(r, bv)
	s.bindSplitKV(r, bv, nLayers)
	if r.prefillReady {
		r.loadFlashDecode(m, nLayers)
		r.loadPrefillSplitKV()
	}
	if e := s.bindMoEKernels(r, nLayers); e != nil {
		return e
	}
	return nil
}

// bindFamilyState loads the kernels of the families that carry their own (gpt-oss, Gated DeltaNet, Mamba-2, MLA) and allocates their
// per-token scratch and the per-layer tables they upload once: attention sinks, gpt-oss's expert biases, and the recurrent and
// latent-attention scratch. It runs inside the setup job after bindKernels and uploads no layer weights.
func (s *buildState) bindFamilyState(r *cudaResident) error {
	m, nLayers, dnetP, mambaP, hls := s.m, s.nLayers, s.dnetP, s.mambaP, s.hls
	bv := m.BackendView()
	// gpt-oss per-layer device state: the attention sinks and the per-expert gate‖up
	// Per-head attention sinks (gpt-oss and Flint) and gpt-oss's per-expert bias
	// tables. Uploaded once here rather than per token — they are weights, not
	// activations. Both are nil for every other family, and r.af/r.up32 are not called
	// at all in that case (Alloc(0) is an error, not a no-op).
	if bv.HasAttnSinks() {
		// The glue `attention` fallback (taken when !prefillReady) has no sink parameter, so a model whose
		// prefill_batched.ptx fails to JIT would decode through a kernel that silently omits the learned per-head sink. Refuse
		// that case: decoder/features.go says the sink reaches both paths, and this is what makes that true.
		if !r.prefillReady {
			return fmt.Errorf("cuda: attention sinks need the batched attention kernel, " +
				"and prefill_batched.ptx did not load — the glue attention fallback has no " +
				"sink parameter, so decoding here would silently drop it")
		}
		r.gptOssSinks = make([]Buffer, nLayers)
		for l := range nLayers {
			if sk := bv.Layer(l).AttnSinks(); len(sk) > 0 {
				r.gptOssSinks[l] = r.upOn(mem.LineDense, sk)
			}
		}
	}
	if _, _, isGptOss := bv.GptOssAct(); isGptOss {
		r.gptOssExpBias = make([]Buffer, nLayers)
		r.gptOssDownBias = make([]Buffer, nLayers)
		for l := range nLayers {
			if gb := bv.Layer(l).GptOssExpertBias(); len(gb) > 0 {
				r.gptOssExpBias[l] = r.upOn(mem.LineDense, gb)
			}
			// The per-expert DOWN bias, consumed by gemv_w4a8_moe_wacc_bias inside the router-weight product; omitting it costs min
			// cosine 0.75 against 0.997.
			if db := bv.Layer(l).GptOssExpertDownBias(); len(db) > 0 {
				r.gptOssDownBias[l] = r.upOn(mem.LineDense, db)
			}
		}
	}
	if dnetP != nil {
		// Gated-DeltaNet: its own module (nothing else here is recurrent) and its own
		// per-token scratch. convDim is 2*nk*hk + nv*hv and bears no relation to
		// qDim/kvDim, so none of the attention scratch can be reused for it.
		dmod, de := r.dev.CompileLibrary(deltaNetPTX)
		if de != nil {
			return fmt.Errorf("cuda: JIT deltanet.ptx: %w", de)
		}
		var lerr error
		loadD := func(dst *Pipeline, name string) {
			// Loader closure, not a plain assignment: see the pipeline-lint note at the L3 load above.
			pl, pe := r.dev.NewComputePipeline(dmod, name)
			if pe != nil {
				lerr = fmt.Errorf("cuda: deltanet kernel %q: %w", name, pe)
				return
			}
			*dst = pl
		}
		loadD(&r.dnConv, "delta_conv")
		loadD(&r.dnGates, "delta_gates")
		loadD(&r.dnNorm, "delta_norm")
		loadD(&r.dnRule, "delta_rule")
		loadD(&r.dnGNorm, "delta_gnorm")
		// Row-batched twins for prompt prefill.
		loadD(&r.dnConvRows, "delta_conv_rows")
		loadD(&r.dnGatesRows, "delta_gates_rows")
		loadD(&r.dnNormRows, "delta_norm_rows")
		loadD(&r.dnRuleRows, "delta_rule_rows")
		loadD(&r.dnRuleRows128, "delta_rule_rows_128")
		loadD(&r.dnGNormRows, "delta_gnorm_rows")
		// The speculative verify's twins (recurrent_spec.go).
		loadD(&r.dnConvRowsSpec, "delta_conv_rows_spec")
		loadD(&r.dnRuleRowsSpec, "delta_rule_rows_spec")
		loadD(&r.dnRuleRows128Spec, "delta_rule_rows_128_spec")
		loadD(&r.dnQSplit, "delta_qsplit")
		loadD(&r.dnAttnGate, "delta_attn_gate")
		if lerr != nil {
			// Hard error, not a silent degrade: unlike gpt-oss's optional epilogue, every
			// linear layer of this family NEEDS these. A missing pipeline would dispatch
			// null and produce garbage rather than fall back to anything.
			return lerr
		}
		dp := dnetP
		r.dnMixed, r.dnConvOut = r.af(dp.convDim), r.af(dp.convDim)
		r.dnQn, r.dnKn = r.af(dp.keyDim), r.af(dp.keyDim)
		r.dnHeadP, r.dnBt, r.dnAt = r.af(dp.nv*2), r.af(dp.nv), r.af(dp.nv)
		r.dnZOut, r.dnCore, r.dnGated = r.af(dp.valueDim), r.af(dp.valueDim), r.af(dp.valueDim)
		r.dnGq, r.dnGSc = r.ai(dp.valueDim/4), r.af(1)
	}
	if mambaP != nil {
		needNonGated := false
		for l := range hls {
			needNonGated = needNonGated || hls[l].mixer == decoder.MixerNone
		}
		if e := r.loadMambaKernels(mambaP, needNonGated); e != nil {
			return e
		}
	}
	if r.isMLA {
		mlamod, me := r.compileKV(r.kvPTX(mlaPTX, mlaKV16PTX))
		if me != nil {
			return fmt.Errorf("cuda: JIT mla.ptx: %w", me)
		}
		var lerr error
		loadMLA := func(dst *Pipeline, name string) {
			pl, pe := r.dev.NewComputePipeline(mlamod, name)
			if pe != nil {
				lerr = fmt.Errorf("cuda: mla kernel %q: %w", name, pe)
				return
			}
			*dst = pl
		}
		loadMLA(&r.fMlaStore, "mla_latent_store")
		loadMLA(&r.fMlaHeadMV, "mla_head_matvec")
		loadMLA(&r.fMlaQRope, "mla_q_rope")
		loadMLA(&r.fMlaAttn, "mla_attn")
		if lerr != nil {
			return lerr
		}
		r.mlaKVDown = r.af(r.mlaLatDim)
		r.qAbs = r.af(r.nH * r.mlaLatDim)
		r.wsum = r.af(r.nH * r.mlaRank)
		if r.mlaQLoRA > 0 {
			r.mlaQAOut = r.af(r.mlaQLoRA)
			r.mlaQAQ = r.ai((r.mlaQLoRA + 3) / 4)
			r.mlaQASc = r.af(1)
		}
	}
	return nil
}

// uploadLayers uploads every layer's planned weights and per-layer state, in layer order, into r.layers, charging them to the dense
// line, and stops at a layer if the load was aborted. It runs inside the setup job after bindFamilyState and allocates no KV or
// scratch.
func (s *buildState) uploadLayers(r *cudaResident) error {
	m, nLayers, nH, pleP, hls, hlm, hFinal := s.m, s.nLayers, s.nH, s.pleP, s.hls, s.hlm, s.w.FinalNorm
	bv := m.BackendView()
	r.layers = make([]cudaLayer, nLayers)
	endWeights := r.onLine(mem.LineDense)
	if r.loadAborted() {
		return fmt.Errorf("cuda: resident build aborted before uploading its weights: %w", decoder.ErrLoadAborted)
	}
	r.openWeightArena([]any{hls, hlm})
	defer r.closeWeightArena()
	for l := range nLayers {
		if r.loadAborted() {
			return fmt.Errorf("cuda: resident build aborted before uploading layer %d of %d: %w", l, nLayers, decoder.ErrLoadAborted)
		}
		h := &hls[l]
		lw, _ := program.DecodeLayerWalk(m.Program().Layers[l], "CUDA")
		L := cudaLayer{idx: l, mixer: h.mixer, ffn: h.ffn, steps: h.steps, prefillSteps: cudaPrefillStepsOf(m.Program().Layers[l]), walk: lw}
		if l == prefillStepsEmptyLayerForTest {
			L.prefillSteps = nil
		}
		if !r.postOnly || h.mixer == decoder.MixerDeltaNet {
			// postOnly (Olmo 3/Olmo Hybrid) has no pre-norm weight for its non-DeltaNet layers (0-length, and Alloc(0) is a hard
			// error), but Olmo Hybrid's DeltaNet layers reach NormPre2 through NormPlacementLinear and carry a real one, required
			// non-empty above.
			L.preNorm = r.up32(h.preNorm)
		}
		if h.ffn && h.mixer != decoder.MixerNone && ((!r.postOnly && !r.parallelBlock) || h.mixer == decoder.MixerDeltaNet) {
			// parallelBlock (Cohere/Command-R) has no pre-MLP norm tensor (h.postNorm is 0-length, the same Alloc(0) hazard);
			// segBFFN reuses segA's r.aq/r.aSc as the MLP input (see Model.ParallelBlockResident).
			L.postNorm = r.up32(h.postNorm)
		}
		if h.mixer == decoder.MixerDeltaNet {
			// Gated-DeltaNet mixer layer: no q/k/v/o, rope table or KV cache. It has two persistent state buffers instead, zeroed at
			// build and re-zeroed per generation: the recurrence compounds, so unlike a KV cache the next sequence cannot overwrite
			// it.
			L.dnQKV, L.dnZ, L.dnOut = r.upW(h.dnQKV), r.upW(h.dnZ), r.upW(h.dnOut)
			L.dnB, L.dnA = r.upW(h.dnB), r.upW(h.dnA)
			L.dnConvW, L.dnDtBias = r.up32(h.dnConvW), r.up32(h.dnDtBias)
			L.dnNegExpA, L.dnNormW = r.up32(h.dnNegExpA), r.up32(h.dnNormW)
			L.dnBetaScale, L.dnGNormEps = h.dnBetaScale, r.eps
			if h.dnONormEps != 0 {
				L.dnGNormEps = h.dnONormEps
			}
			L.dnWin = r.upOn(mem.LineState, make([]float32, r.recFloats[l].win))
			L.dnState = r.upOn(mem.LineState, make([]float32, r.recFloats[l].state))
		} else if h.mixer == decoder.MixerMamba {
			// Nemotron-H Mamba-2 block: the two int8 projections, the small tables, and the conv window and SSM state, zeroed at build and by resetState.
			L.mbIn, L.mbOut = r.upW(h.mamba.in), r.upW(h.mamba.out)
			L.mbConvW, L.mbConvB = r.up32(h.mamba.convW), r.up32(h.mamba.convB)
			L.mbHeadP, L.mbNormW = r.up32(h.mamba.headP), r.up32(h.mamba.normW)
			L.mbWin = r.upOn(mem.LineState, make([]float32, r.recFloats[l].win))
			L.mbSSM = r.upOn(mem.LineState, make([]float32, r.recFloats[l].state))
		} else if h.mixer == decoder.MixerNone {
			// A feed-forward-only block has no attention weights to upload (u and d follow below).
		} else if h.mixer == decoder.MixerMLA {
			if r.mlaQLoRA > 0 {
				L.mlaQA = r.upW(h.mlaQA)
				L.mlaQB = r.upW(h.mlaQB)
				L.mlaQANorm = r.up32(h.mlaQANorm)
			} else {
				L.mlaQ = r.upW(h.mlaQ)
			}
			L.mlaKVA = r.upW(h.mlaKVA)
			L.mlaKVANorm = r.up32(h.mlaKVANorm)
			L.mlaWUK = r.up32(h.mlaWUK)
			L.mlaWUV = r.up32(h.mlaWUV)
			L.o = r.upW(h.o)
			L.invF = r.up32(h.invFreq)
			L.mscale = float32(m.BackendView().Layer(l).RopeMscale())
		} else {
			L.q, L.o = r.upW(h.q), r.upW(h.o)
			if bv.Layer(l).KVSrc() == l {
				L.k = r.upW(h.k) // a KV-shared layer has no k_proj; v below (K=V layers have none)
			}
			L.invF = r.up32(h.invFreq)
			L.mscale = float32(m.BackendView().Layer(l).RopeMscale()) // YaRN attention_factor; 1.0 for every family without it
			L.qGate = h.qGate
		}
		if h.mixer == decoder.MixerNone {
			L.u, L.d = r.upW(h.u), r.upW(h.d)
			L.ffnI = L.u.N
		} else if !h.isMoE && h.ffn {
			// A routed layer has no dense FFN to upload: its hostW's are empty, and
			// Alloc(0) is an error rather than a harmless no-op.
			L.g, L.u, L.d = r.upW(h.g), r.upW(h.u), r.upW(h.d)
			L.ffnI = L.g.N
			if g4OneFFNWidthForTest && l > 0 {
				L.ffnI = r.layers[0].ffnI
			}
			if pleP > 0 {
				L.pleGate, L.pleProj, L.postPLENorm = r.upW(h.pleGate), r.upW(h.pleProj), r.up32(h.postPLENorm)
			}
			// Gemma 4's per-layer output scalar (out = h*layerScalar after the dense MLP residual add): present on every dense
			// gemma4 layer (1 when the checkpoint's tensor is absent, decoder/weights.go); g4moe layers override it below with their
			// bundle's value. Zero, the skip sentinel segB's dense tail checks, for every non-gemma4 family.
			L.layerScalar = bv.Layer(l).Gemma4DenseScalar()
		}
		if (r.sandwich || r.postOnly) && h.mixer != decoder.MixerDeltaNet {
			// h.mixer == decoder.MixerDeltaNet: postOnly is model-level, but Olmo Hybrid's DeltaNet layers use NormPre2 and were never given
			// postAttnNorm/postMLPNorm (see the validation loop's DeltaNet continue).
			L.postAttnNorm, L.postMLPNorm = r.up32(h.postAttnNorm), r.up32(h.postMLPNorm)
		}
		// Both of these are ATTENTION side tables: a DeltaNet mixer layer has neither, and
		// its nil slices would become 0-byte allocations (a hard error, not a no-op).
		if h.hasBias && h.mixer != decoder.MixerDeltaNet && h.mixer != decoder.MixerMLA {
			L.qb, L.kb, L.vb, L.hasBias = r.up32(h.qb), r.up32(h.kb), r.up32(h.vb), true
		}
		if h.hasOBias && h.mixer != decoder.MixerDeltaNet {
			L.ob, L.hasOBias = r.up32(h.ob), true
		}
		if r.qkNorm && h.mixer != decoder.MixerDeltaNet && h.mixer != decoder.MixerMLA {
			L.qNorm = r.up32(h.qNorm)
			if len(h.kNorm) > 0 { // empty on a KV-shared layer, and Alloc(0) is an error
				L.kNorm = r.up32(h.kNorm)
			}
		}
		if h.isMoE {
			L.isMoE = true
			L.routerW, L.routerB = r.up32(h.router), r.up32(h.routerBs)
			L.expGU, L.expDown = r.upExperts(h.expGU), r.upExperts(h.expDown)
			if h.hasShared {
				L.hasShared = true
				L.shGU, L.shDown = r.upW(h.shGU), r.upW(h.shDown)
				if h.shGate.N > 0 {
					L.shGateW = r.upW(h.shGate)
				}
			}
		}
		if h.g4moe {
			// Gemma-4 parallel dense‖MoE: router (folded f32 proj) + experts + the 5 norms +
			// per-expert scale + layerScalar. Dense g/u/d were uploaded via the !h.isMoE branch.
			L.g4moe = true
			L.routerW, L.routerB = r.up32(h.router), r.up32(h.routerBs)
			L.expGU, L.expDown = r.upExperts(h.expGU), r.upExperts(h.expDown)
			L.g4preFFN, L.g4postFFN1 = r.up32(h.g4preFFN), r.up32(h.g4postFFN1)
			L.g4preFFN2, L.g4postFFN2, L.g4postFFN = r.up32(h.g4preFFN2), r.up32(h.g4postFFN2), r.up32(h.g4post)
			L.perExpertScaleB = r.up32(h.perExpertScale)
			L.layerScalar = h.layerScalar
		}
		L.window = h.window
		// Per-layer attention geometry, from the same accessors the CPU forward uses (headDimAt/kvHeadsAt/isGlobalLayer through
		// the *Resident wrappers), never a recomputed interleave, so the runner cannot drift from the Gemma 4 program walk. Uniform
		// families collapse to the model-level fields; Gemma 4's global layers report the wide head, fewer KV heads and partial
		// rotary.
		L.hd, L.nKV, L.rhalf = bv.Layer(l).HeadDim(), bv.Layer(l).KVHeads(), bv.Layer(l).RotaryDim()/2
		L.qDim, L.kvDim = nH*L.hd, L.nKV*L.hd
		if h.mixer == decoder.MixerDeltaNet || h.mixer == decoder.MixerMamba || h.mixer == decoder.MixerNone {
			// A recurrent mixer or a feed-forward-only block has no attention geometry to validate or rope table to bind. A non-zero kvDim would
			// make the KV allocator below size a cache this layer never reads, wasting VRAM on the family that most needs it.
			L.hd, L.nKV, L.rhalf, L.qDim, L.kvDim = 0, 0, 0, 0, 0
			r.layers[l] = L
			continue
		}
		if h.mixer == decoder.MixerMLA {
			L.hd = r.mlaQKHead
			L.nKV = 1
			L.rhalf = r.mlaQKRope / 2
			L.qDim = nH * r.mlaVHead
			L.kvDim = r.mlaLatDim
			r.layers[l] = L
			continue
		}
		if src := bv.Layer(l).KVSrc(); src != l {
			if g4KVSrcOffForTest {
				// Test defect: the source one OWNING layer of the same attention type earlier (when there is one).
				for j := src - 1; j >= 0; j-- {
					if bv.Layer(j).KVSrc() == j && bv.Layer(j).HeadDim() == bv.Layer(src).HeadDim() {
						src = j
						break
					}
				}
			}
			// Gemma 4 E-model KV sharing: this layer reads layer src's cache and owns no K/V. The geometry must be the
			// source's, or its attention would index the aliased buffers with the wrong stride.
			so := &r.layers[src]
			if src > l || so.kvShared || so.hd != L.hd || so.nKV != L.nKV || so.kvDim != L.kvDim {
				return fmt.Errorf("cuda: layer %d shares layer %d's KV cache but its geometry (hd %d, nKV %d, kvDim %d) is not an owning layer's (hd %d, nKV %d, kvDim %d)",
					l, src, L.hd, L.nKV, L.kvDim, so.hd, so.nKV, so.kvDim)
			}
			L.kvShared, L.kvSrc = true, src
		}
		L.kEqV = bv.Layer(l).VFromK() && !L.kvShared
		// Gemma 4 applies v_norm on every layer that owns its K/V; K=V layers always did. g4DropVNormForTest re-drops it on the
		// non-K=V layers so a test can reproduce the pre-fix numbers.
		L.vNorm = L.walk.Attn.G4VNorm && (L.kEqV || !g4DropVNormForTest)
		if !L.kEqV && !L.kvShared {
			L.v = r.upW(h.v) // non-K=V layers have a real v_proj weight; K=V derives V from k; a shared layer has neither
		}
		// Per-layer rope-table invariant: rope_kv rotates L.rhalf pairs per head reading invFreq[0..rhalf), so the bound
		// per-layer table must have exactly rhalf entries. Gemma 4's global and local tables differ in length and the generic
		// finalizeRoPE check cannot see that, so assert it per layer at build.
		if len(h.invFreq) != L.rhalf {
			return fmt.Errorf("cuda: layer %d rope table len=%d != rhalf=%d — per-layer invFreq must match the rotated-pair count the kernel indexes", l, len(h.invFreq), L.rhalf)
		}
		r.layers[l] = L
	}
	r.lmW = r.upW(hlm)
	r.closeWeightArena() // the final norm is not in the walk that sized it
	r.finalNorm = r.up32(hFinal)
	endWeights()
	return nil
}

// checkSublayers declines a layer whose uploaded buffers are not the program's sublayers, whose walk CUDA's table lacks, or whose
// batched prefill steps are not the ones its kind runs. It runs inside the setup job after uploadLayers and allocates nothing.
func (s *buildState) checkSublayers(r *cudaResident) error {
	m := s.m
	// Each layer's program sublayers must be what its upload holds (loadedSublayers reads the built layer's buffers), so a
	// block the walk would run on missing or miswired weights declines here (OP4cu-b2).
	for l := range r.layers {
		if legacyLayerForTest != nil {
			break
		}
		prog, progParallel, err := cudaSublayersOf(m.Program().Layers[l])
		if err == nil {
			err = checkLoadedSublayers(prog, progParallel, loadedSublayers(&r.layers[l]), r.parallelBlock)
		}
		if err == nil {
			err = r.checkCUDALayerWalk(&r.layers[l], m.Program().Layers[l])
		}
		// The batched prefill's steps, where it has any, must be the blocks the layer's kind chose before (OP4cu-d1).
		if ps := r.layers[l].prefillSteps; err == nil && !prefillStepsBuildChecksOffForTest && len(ps) > 0 && !slices.Equal(ps, prefillStepsFromFlags(&r.layers[l])) {
			err = fmt.Errorf("the program's batched prefill steps are %v and the layer's kind runs %v", ps, prefillStepsFromFlags(&r.layers[l]))
		}
		if err != nil {
			return fmt.Errorf("cuda: layer %d: %w", l, err)
		}
	}
	return nil
}

// sizeAttnScratch allocates the activation and KV-projection scratch the layers share, sized to the widest layer (a hybrid's
// recurrent layers excluded), declares each layer's KV buffers against the model, and loads the int8 KV kernels. It leaves the widest
// attention width and head dim in s.maxQDim and s.maxHd for the later scratch. It runs inside the setup job after checkSublayers.
func (s *buildState) sizeAttnScratch(r *cudaResident) error {
	m, nLayers, nH, H, pleP := s.m, s.nLayers, s.nH, s.H, s.pleP
	bv := m.BackendView()
	dnetP, mambaP, dnAttnGate := s.dnetP, s.mambaP, s.dnAttnGate
	// Scratch (Q/K/V projections, attention context) is allocated ONCE and shared across
	// layers, so it must fit the WIDEST per-layer geometry — Gemma 4's 512-global head, not
	// a model-level value. maxQDim/maxKVDim reduce to nH*hd / nKV*hd for uniform families.
	maxQDim, maxKVDim, maxHd := 0, 0, 0
	for l := range r.layers {
		if q := r.layers[l].qDim; q > maxQDim {
			maxQDim = q
		}
		if k := r.layers[l].kvDim; k > maxKVDim {
			maxKVDim = k
		}
		if d := bv.Layer(l).HeadDim(); d > maxHd {
			maxHd = d
		}
	}
	// The scratch MUST size to the widest head over LAYERS, not m.Dims() — which reports the
	// LOCAL head_dim (256 for the real 26B), not the 512 the global layers need. Cross-check
	// maxQDim (from per-layer qDim) against nH*maxHd (from the accessors independently): a
	// mismatch means the per-layer geometry drifted from its source. GPU OOB writes don't
	// reliably fault, so assert at plan time rather than wait for a parity red.
	if dnetP != nil || mambaP != nil {
		// A DeltaNet layer reports zero attention geometry, so it must not drag maxQDim/maxHd
		// to zero for the SOFTMAX layers that share this scratch. Recompute over the
		// attention layers only, and check the invariant against those.
		maxQDim, maxKVDim, maxHd = 0, 0, 0
		for l := range r.layers {
			if r.layers[l].noKV() {
				continue
			}
			if q := r.layers[l].qDim; q > maxQDim {
				maxQDim = q
			}
			if k := r.layers[l].kvDim; k > maxKVDim {
				maxKVDim = k
			}
			if d := bv.Layer(l).HeadDim(); d > maxHd {
				maxHd = d
			}
		}
		if maxQDim == 0 {
			return fmt.Errorf("cuda: every layer is a recurrent mixer or a feed-forward block — no softmax layer to size attention scratch from; a hybrid with none is a loader bug, not a valid shape")
		}
		if dnAttnGate {
			// The softmax layers' fused q_proj is double width; both halves live in dnQg
			// until delta_qsplit separates them into qB and the gate.
			r.dnQg, r.dnAGate = r.af(2*maxQDim), r.af(maxQDim)
		}
	}
	if !r.isMLA && maxQDim != nH*maxHd {
		return fmt.Errorf("cuda: scratch maxQDim=%d != nH*maxHd=%d*%d=%d — per-layer geometry inconsistent with the accessors", maxQDim, nH, maxHd, nH*maxHd)
	}
	// r.x carries the embedding row: for an E-model that is [hidden ‖ nLayers*P PLE inputs], uploaded in ONE copy, and each layer's
	// PLE branch reads its slice with an element offset (glu_quant's uOff), so no buffer view is needed. Every other model: hidden floats.
	r.pleP, r.embLen = pleP, H+nLayers*pleP
	r.eModel = pleP > 0
	for l := range r.layers {
		r.eModel = r.eModel || r.layers[l].kvShared
	}
	r.x, r.aSc, r.aq = r.af(r.embLen), r.af(r.actScaleLen(H)), r.ai(H/4)
	if pleP > 0 {
		r.pleG = r.af(pleP)
	}
	qBufDim := maxQDim
	if r.isMLA {
		qBufDim = nH * r.mlaQKHead
	}
	r.qB, r.kB, r.vB = r.af(qBufDim), r.af(maxKVDim), r.af(maxKVDim)
	r.kc, r.vc = make([]Buffer, nLayers), make([]Buffer, nLayers)
	r.ksc, r.vsc = make([]Buffer, nLayers), make([]Buffer, nLayers)
	for l := range r.layers {
		r.layers[l].kvHalf = r.kvF16 && !r.layers[l].noKV() && !r.layers[l].kvShared
		r.layers[l].kvI8 = r.kvI8 && !r.layers[l].noKV() && !r.layers[l].kvShared
		if e := kvCheckDeclared(l, m.LayerBuffers(l), &r.layers[l]); e != nil {
			return e
		}
	}
	if r.kvI8 {
		if e := r.loadKV8(); e != nil {
			return e
		}
	}
	s.maxQDim, s.maxHd = maxQDim, maxHd
	return nil
}

// allocKV prices the KV cache against the free VRAM, checks each layer's cache width against the model, and allocates the pools and
// every slot's tables, keeping the slots that fit. It runs inside the setup job after sizeAttnScratch; a resident context the
// operator asked for that does not fit returns errKVWontFit, which the build returns as a hard error rather than a decline.
func (s *buildState) allocKV(r *cudaResident) error {
	m := s.m
	bv := m.BackendView()
	r.kvWindowSetup(m) // before checkKVFits: windowed layers price and allocate at window+slack, not at the context
	// r.ctxCap was resolved at construction (several earlier buffers size from it). The fit check
	// belongs HERE, though: it needs the per-layer kvDims, and running it immediately before the
	// caches are allocated is what makes `free` mean "what is actually left for KV".
	if e := r.payRoutePool(); e != nil {
		return e
	}
	if e := r.checkKVFits(); e != nil {
		return e
	}
	for l := range r.kc {
		// Each layer's KV pool is sized by its own kvDim (Gemma 4's local 2048 vs global 1024), matching the row*Ly.kvDim stride
		// the kernels index it with. The kvDim the pool is sized with must equal the one the accessors derive: a stale model-level
		// kvDim would index off the end into garbage output rather than panic, so fail loudly at plan time.
		if r.layers[l].noKV() {
			// No KV cache for a recurrent mixer. Allocating one would be pure waste on the
			// family least able to afford it: 3 of every 4 layers are this kind.
			continue
		}
		if r.layers[l].isMLA() {
			if r.layers[l].kvDim != r.mlaLatDim {
				return fmt.Errorf("cuda: layer %d MLA latent cache kvDim=%d != latDim=%d", l, r.layers[l].kvDim, r.mlaLatDim)
			}
			continue
		}
		if want := bv.Layer(l).KVHeads() * bv.Layer(l).HeadDim(); r.layers[l].kvDim != want {
			return fmt.Errorf("cuda: layer %d KV cache kvDim=%d != nKV*hd=%d (accessor-derived) — geometry/cache-size mismatch", l, r.layers[l].kvDim, want)
		}
	}
	// The pools and every slot's tables (kvpool.go). checkKVFits granted r.kvSlotsN of the request against the free VRAM read before
	// the build's own scratch, with the margin as the only slack, so the pools can miss by the scratch: keep the slots that fit
	// instead of dropping the whole resident to the CPU; one slot that does not fit still declines.
	if e := r.kvAllocPoolsFit(); e != nil {
		return e
	}
	if kvPoolsDoneForTest != nil {
		kvPoolsDoneForTest()
	}
	return nil
}

// allocDecodeScratch allocates the activation, projection, MoE, sampling and readback scratch the decode and prefill paths share,
// sized from the widest attention layer (s.maxQDim, s.maxHd), and loads the float-activation GEMV kernels when the model takes them.
// It runs inside the setup job after allocKV, before the recurrent checkpoint and the expert slots are sized from what is left.
func (s *buildState) allocDecodeScratch(r *cudaResident) error {
	var e error
	H, I, vocab, nE, topK, moeInter, sharedInter, isMoE := s.H, s.I, s.vocab, s.nE, s.topK, s.moeInter, s.sharedInter, s.isMoE
	maxQDim, maxHd := s.maxQDim, s.maxHd
	if r.actG32 && maxQDim%32 != 0 {
		return fmt.Errorf("per-32 activations need the attention width (%d) to be a multiple of 32", maxQDim)
	}
	r.cctx, r.cSc, r.cq = r.af(maxQDim), r.af(r.actScaleLen(maxQDim)), r.ai(maxQDim/4)
	// v_norm (Gemma 4: every K/V-owning layer; K=V layers also derive V = v_norm(k)) reuses qk_norm with a UNIT
	// weight [maxHd] (so x*inv*w = x*inv, scale-less; addOne=0). Allocate it only when needed.
	for l := range r.layers {
		if r.layers[l].vNorm {
			ones := make([]float32, maxHd)
			for i := range ones {
				ones[i] = 1.0
			}
			r.vNormUnit = r.upOn(mem.LineDense, ones)
			break
		}
	}
	r.oO = r.af(H)
	r.mSc, r.mq = r.af(r.actScaleLen(H)), r.ai(H/4)
	if r.layerNorm {
		r.zeroBias = r.af(H)
		if err := gpu.Upload(r.zeroBias, make([]float32, H)); err != nil {
			return err
		}
	}
	// Dense-FFN scratch, only if the model has a dense FFN: a model whose every layer is routed reports intermediate_size 0
	// (Qwen3.6-35B-A3B's config omits the key), and these would be 0-byte allocations, which this driver rejects as "invalid
	// length". The dense branch of segBFFN is unreachable for such a model. Scratch is sized to the widest dense layer: an
	// E-model's layers differ, and the launches take each layer's own width (cudaLayer.ffnI).
	maxI := I
	for l := range r.layers {
		maxI = max(maxI, r.layers[l].ffnI)
	}
	if maxI > 0 {
		r.gO, r.uO = r.af(maxI), r.af(maxI)
		r.dSc, r.dScr, r.dq = r.af(r.actScaleLen(maxI)), r.af(maxI), r.ai(maxI/4)
	} else if !isMoE {
		return fmt.Errorf("cuda: intermediate_size is 0 on a DENSE model — no FFN to run")
	}
	if r.moe {
		// Sized to the MoE expert width, not the dense one (Mellum's moe_intermediate_size
		// differs from intermediate_size).
		r.rLogits, r.rIdx, r.rWgt = r.af(nE), r.au32(topK), r.af(topK)
		if r.cacheExperts { // C′: per-token slot-id buffer (uploaded per layer) + readback scratch
			r.slotIdx = r.au32(topK)
			r.hostIdx = make([]uint32, topK)
			r.hostSlot = make([]uint32, topK)
			if hb, e := gpu.NewHostBufferOn[uint8](r.dev, mem.LineScratch, topK*4); e != nil {
				return e
			} else {
				r.slotIdxHost = hb
			}
		}
		r.moeGU = r.af(2 * moeInter)
		r.moeSc, r.moeScr, r.moeQ = r.af(1), r.af(moeInter), r.ai(moeInter/4)
		if sharedInter > 0 {
			r.shGUout = r.af(2 * sharedInter)
			r.shSc, r.shScr, r.shQ = r.af(1), r.af(sharedInter), r.ai(sharedInter/4)
			r.shDownOut = r.af(H)
			r.shGl = r.af(1) // the sigmoid gate logit; allocated unconditionally (one float)
		}
	}
	if r.gemma4Moe { // parallel dense‖MoE branch scratch
		r.g4x1, r.g4x2, r.g4rn = r.af(H), r.af(H), r.af(H)
	}
	if why := r.floatActDecline(); why == "" {
		fmod, e2 := r.dev.CompileLibrary(gemvW4F32PTX)
		if e2 != nil {
			return e2
		}
		for _, f := range []struct {
			dst  *Pipeline
			name string
		}{
			{&r.fW4F32, "gemv_w4f32"}, {&r.fMoEGemvF32, "gemv_w4f32_moe"}, {&r.fMoEWaccF32, "gemv_w4f32_moe_wacc"}, {&r.fGluF32, "glu_f32"},
		} {
			if *f.dst, e = r.dev.NewComputePipeline(fmod, f.name); e != nil {
				return e
			}
		}
		r.xF, r.dF, r.moeGLUF = r.af(H), r.af(max(maxI, 1)), r.af(moeInter)
		r.floatAct = true
		logs.Debugf("cuda: decode activations: f32 into every int4 projection (gemv_w4f32)")
	} else if r.gemma4Moe {
		logs.Debugf("cuda: decode activations: int8 (W4A8): %s", why)
	}
	r.dO, r.logits = r.af(H), r.af(vocab)
	r.argIdx, r.argVal = r.ai(1), r.af(1) // greedy fast-path readback (4 B vs 594 KB)
	r.topkOut = r.ai(2*topkMaxK + 2)      // sampled fast-path readback (~2 KB vs 594 KB)
	r.topkCV, r.topkCI, r.topkMx, r.topkPart = r.af(topkCap), r.ai(topkCap), r.af(1), r.af(topkThreads)
	r.topkCnt, r.topkDone = r.ai(1), r.ai(1)
	if e := gpu.Upload(r.topkCnt, []int32{0}); e != nil {
		return e
	}
	if e := gpu.Upload(r.topkDone, []int32{0}); e != nil {
		return e
	}
	nGb := launchgeom.GumbelBlocks(vocab)
	r.gbKey, r.gbIdx, r.gbOut = r.af(nGb), r.ai(nGb), r.ai(1) // Gumbel-max sampling scratch
	if hb, e := gpu.NewHostBufferOn[float32](r.dev, mem.LineScratch, vocab); e != nil {
		return e
	} else {
		r.logitsPinned, r.logitsHost = hb, hb.Slice()
	}
	return nil
}

// grantSlots takes what is left of the device after the KV, weights and scratch: the recurrent checkpoint, then the expert slot
// cache (sized net of the windowed-KV grant's floor), then the windowed-KV grant itself. It runs last in the setup job, so every
// allocation these must not cost has been made. It is the one place the grants are ordered.
func (s *buildState) grantSlots(r *cudaResident) error {
	// The recurrent checkpoint comes out of what is left before the expert cache and the windowed grant are sized from it.
	if postPoolDoneForTest != nil {
		postPoolDoneForTest()
	}
	r.allocRecurrentCheckpoint()
	// C′: the core + KV + scratch are now up, so free VRAM reflects them — size the expert slot
	// cache to what actually fits (cap-and-log, never OOM) and allocate it.
	if r.cacheExperts {
		r.kvGrantRollback(r.expertCacheFloorBytes()) // before the cache is sized: the grant comes out of it, never below its minimum
	}
	if e := r.allocSlots(); e != nil {
		return e
	}
	if !r.cacheExperts {
		r.kvGrantRollback(0) // last: everything it must not cost is allocated
	}
	return nil
}

// selectFusion fixes the build's dispatch flags from the uploaded layers: whether the fused QKV and per-32 paths run, whether decode
// replays captured graphs, whether the expert DMA overlaps compute, and whether the batched prefill has steps on every layer. A
// decline here returns before the width choice, and the caller closes r. It runs after the setup job, which has already released the
// packing blocks.
func (s *buildState) selectFusion(r *cudaResident) error {
	H, postOnly, parallelBlock, hls := s.H, s.postOnly, s.parallelBlock, s.hls
	// The fused path needs int4 projections on the layers it runs on (layerFusable). Anything else falls back to the unfused
	// chain, which handles mixed quant correctly.
	r.fuseQKV = true
	for l := range hls {
		h := &hls[l]
		if !layerFusable(
			h.q.kind == "int4" && h.k.kind == "int4" && h.v.kind == "int4",
			h.isMoE,
			h.g.kind == "int4" && h.u.kind == "int4" && h.d.kind == "int4") {
			r.fuseQKV = false
			break
		}
	}
	if r.knobValue("GOINFER_CUDA_NO_FUSE") != "" {
		r.fuseQKV = false
	}
	if postOnly {
		// fQKV bakes a real pre-norm weight into its rmsnorm+quant dispatch, and a postOnly arch has none (Ly.preNorm is never
		// uploaded), so it takes the unfused segA chain, which quantizes the raw residual.
		r.fuseQKV = false
	}
	if parallelBlock {
		// fQKV goes straight from x to Q/K/V and never materializes the normed, quantized activation that segBFFN's MLP branch
		// reuses as its own input (FeatParallelBlock: segA's r.aq/r.aSc); the fused path has nowhere to hand it back.
		r.fuseQKV = false
	}
	if r.isMLA {
		r.fuseQKV = false
	}
	if r.actG32 {
		// fused_rms_qkv / fused_rms_gu quantize per vector internally.
		r.fuseQKV = false
		// Their per-32 twins (fused_rms_qkv_g32 / fused_rms_gu_g32) take any per-32 weight kind, per projection, with fuseQKV's
		// structural exclusions (no pre-norm, parallel block, MLA, gated q, MoE layers). Size-gated (fusedG32MaxHidden): every
		// block redoes the rmsnorm and per-32 quant, which grows with H while the launches saved matter less as the GEMVs grow.
		// Fusion paid off at H=1536 and lost at 3072 and 3584.
		r.fuseG32 = r.knobValue("GOINFER_CUDA_NO_FUSE") == "" && !postOnly && !parallelBlock && !r.isMLA &&
			H <= fusedG32MaxHidden
		for l := range hls {
			h := &hls[l]
			if h.isMoE || h.qGate || !g32Fusable(h.q.kind, h.k.kind, h.v.kind, h.g.kind, h.u.kind) {
				r.fuseG32 = false
				break
			}
		}
	}
	// CUDA graphs (GOINFER_CUDA_GRAPHS): capture each layer's static launch segments now that fuseQKV is final (segA/segB
	// branch on it), so decode replays them instead of re-issuing the launches. Off, launchToken is unchanged. Incompatible
	// with the g4cap diagnostic, which syncs inside a segment. admitGraphs gates it: replay is bit-exact only under
	// EXCLUSIVE_PROCESS tenancy or MPS, so a shared-GPU box under DEFAULT declines to the live path
	// (docs/cuda-graphs-investigation.md).
	// A windowed layer's K/V pointer moves with its base, so a captured graph would replay a stale one: graphs decline under windowed KV.
	r.graphs = os.Getenv("GOINFER_CUDA_GRAPHS") != "" && !r.g4cap && !r.isMLA && !r.kvWin && r.mamba == nil &&
		!(r.dnet != nil && r.KVSlots() > 1) // a captured segment holds one slot's window and state
	r.graphsSync = os.Getenv("GOINFER_CUDA_GRAPHS_SYNC") != "" // debug: serialize replays (bisect ordering hazards)
	r.graphMask = os.Getenv("GOINFER_CUDA_GRAPHS_ONLY")        // debug: replay only these segments (A/B/C), rest live
	if e := r.admitGraphs(); e != nil {
		return decline(fmt.Errorf("cuda: %w", e))
	}
	// C′ compute/DMA overlap (see the overlap field). Not under graphs: a captured segB cannot record
	// the router event mid-segment and a captured segC cannot wait per miss.
	r.overlap = r.cacheExperts && !r.graphs && r.knobValue("GOINFER_MOE_DMA_OVERLAP") != "0"
	if r.overlap {
		if e := r.initOverlap(); e != nil {
			return decline(fmt.Errorf("cuda: overlap: %w", e))
		}
	}
	// A model the batched prefill's guards batch has steps on every layer; a layer without would send it to the sequential
	// path with nothing said, so the build declines naming the layer.
	if !prefillStepsBuildChecksOffForTest && r.prefillFlagsDecline() == nil {
		for l := range r.layers {
			if len(r.layers[l].prefillSteps) == 0 {
				return decline(fmt.Errorf("cuda: layer %d: the batched prefill's guards admit the model and the program gives the layer no batched steps", l))
			}
		}
	}
	// KS-B8: the default pass width, from what the build left free.
	if r.prefillReady {
		if e := r.do(func() error { r.choosePrefillWidth(); return nil }); e != nil {
			return decline(fmt.Errorf("cuda: prefill width: %w", e))
		}
	}
	return nil
}

// vsumSplitForTest, when above 1, loads the flash-decode V-sum spike at build, splitting the decode V-sum into that many key
// chunks. It is not bit-identical to the shipped single fold and refuses speculative decoding (DecodeVerifyDivergence), so
// only tests set it.
var vsumSplitForTest int
