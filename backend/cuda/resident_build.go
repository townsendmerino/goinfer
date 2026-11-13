//go:build cuda

package cuda

import (
	"fmt"
	"math"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// buildDecline marks an error from a build phase as a decline: buildResidentMode's declined closure turns it into the typed
// decoder.DeclineResident, where any other error from a phase is returned as it is.
type buildDecline struct{ error }

func decline(err error) error { return buildDecline{err} }

// buildState is what a resident build resolves before it allocates anything, handed from one phase to the next. A phase fills the
// fields its successors read.
type buildState struct {
	m *decoder.Model
	w *decoder.Weights
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
}

// planProj plans a projection or the head: the file's own Q5_K or Q6_K blocks where the q4k lane uploads them natively
// (decoder.Model.KQuantNative, which the fit pricing also reads), else planWeight. s.kq reads the file until the build returns.
func (s *buildState) planProj(src *linalg.WeightMat) (hostW, error) {
	m := s.m
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
	w := m.Weights()
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

	H, nLayers, nH, _, _, I, vocab := m.Dims() // nKV, hd are per-layer now (KVHeadsAtResident/HeadDimAtResident); model-level unused
	// Gemma 4 E-model PLE width P (0 for every other model). The resident embedding row is then [hidden ‖ nLayers*P] (ResidentEmbedLen):
	// the host computes the per-layer inputs, and each layer's PLE branch reads its slice of the tail (S1 on CUDA).
	pleP := m.Gemma4PLEDimResident()
	if want := H + nLayers*pleP; pleP > 0 && m.ResidentEmbedLen() != want {
		return decline(fmt.Errorf("gemma4 E-model: the decoder's resident embedding row is %d floats, but hidden+layers*P is %d", m.ResidentEmbedLen(), want))
	}

	// ---- MoE knobs (ok=false for a dense model; every field then stays zero) ----
	// sharedUngated picks which shared-expert combine runs: GLM/DeepSeek add the shared output
	// straight into the residual, Qwen-MoE scales it by sigmoid(SharedGate·h) first. Both are
	// the same kernel (shared_gate_combine, `ungated` flag); only the gated one needs the extra
	// [1,hidden] weight.
	nE, topK, moeInter, sharedInter, moeSig, moeNorm, sharedUngated, moeScale, nGroup, topkGroup, isMoE := m.MoEResidentParams()
	// Gemma-4 enable_moe_block sets arch.MoE (so isMoE is true) but is NOT the generic MoE: it runs
	// gemma4MoeMLP (parallel dense‖MoE + join), so it bypasses the generic-MoE admission checks below
	// (gelu-tanh act, the attention sandwich norm) and gets its own int4-shape checks. Its nE/topK/
	// moeInter still come from arch.MoE (MoEResidentParams), matching the bundle.
	isG4MoE := m.HasGemma4MoEResident()
	isGemma4 := m.IsGemma4Resident()
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
		case m.GatedActResident() != 1: // decoder.ActSiLU — decoder/mlp.go errors on any other
			return decline(fmt.Errorf("MoE experts are SwiGLU-only, arch act=%d", m.GatedActResident()))
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
	qLoRA, kvLoRA, qkNope, qkRope, vHead, interleave, mlaAttnScale, mlaRopeScale, mlaOK := m.MLAResidentParams()

	// Gated-DeltaNet hybrid geometry. Resolved before the per-layer pack because BOTH layer kinds
	// branch on it: the linear layers take the recurrence, and the softmax ones carry the fused
	// double-width q_proj that no other family has.
	dnConvK, dnHK, dnHV, dnNK, dnNV, dnAttnGate, dnetOK := m.Qwen35ResidentParams()
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
	logitScale, _ := m.LogitScaleResident() // ok=false ⇒ 1, already the applyLogitScale no-op value
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
