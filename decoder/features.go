package decoder

import (
	"fmt"
	"maps"
	"strings"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// RequiredResidentFeatures returns the features a resident backend must implement to run this
// model correctly. Derived from the arch — see ResidentFeature.
func (m *Model) RequiredResidentFeatures() []ResidentFeature { return m.w.Arch.ResidentFeatures() }

// MissingResidentFeatures returns the required features that `implemented` does not contain,
// sorted. Empty ⇒ a backend declaring that set can run m correctly, and admission is granted.
// Backends call this instead of hand-rolling their own decline list, so the taxonomy has one
// source of truth (three copies is how the bug recurs).
func (m *Model) MissingResidentFeatures(implemented map[darch.ResidentFeature]bool) []darch.ResidentFeature {
	return missingFeatures(m.w.Arch.ResidentFeatures(), implemented)
}

func missingFeatures(required []darch.ResidentFeature, implemented map[darch.ResidentFeature]bool) []darch.ResidentFeature {
	var missing []darch.ResidentFeature
	for _, r := range required {
		if !implemented[r] {
			missing = append(missing, r)
		}
	}
	return missing
}

// ResidentEligible reports whether backend can run architecture a on its resident (GPU) decode path: the capability predicate, from arch
// flags alone, read from Admit's runner shape and capability gates with the default activations. It composes the two gates every resident admission applies: the arch is a shape the runner supports
// (decodeRunnerEligible) and the backend implements every feature the arch needs (residentGateReason). The runtime and the
// hardware-matrix generator derive from the same pieces, so the published table cannot disagree with what a backend can run.
//
// It is capability, not runtime admission: load-time policy such as Nemotron's int4-only precision gate lives in
// Model.DecodeRunnerEligible, and the matrix footnotes it. "Arch flags only" is also not strictly true of granite: its admission gate is
// inside decodeRunnerEligible and reads GOINFER_SSM_RESIDENT from the live environment (ssmResidentEnv). The matrix generator pins that variable empty, so the published table shows granite CPU-only on every
// backend (docs/hardware-matrix.md footnote); docs/code-notes/decoder.md#ResidentEligible.
func ResidentEligible(a *darch.Architecture, backend string) bool {
	if !residentDeclared(backend) {
		return false
	}
	ad := Admit(archAdmitFacts(a, backend))
	return ad.Shape == "" && len(ad.Gates) == 0
}

// residentGateReason is the first of residentGateReasons with the default activations: why backend declines a after the runner
// shape, "" when every gate admits. Admit and the matrices read residentGateReasons; this is the one-reason form tests read.
func residentGateReason(a *darch.Architecture, backend string) string {
	return residentGateReasonAct(a, backend, false)
}

// residentGateReasonAct is residentGateReason for a model whose resident projections run W8A8 with per-32 activation scales (actSafe):
// that clears ActivationQuantHazard on a backend that implements it (cuda's actgroup.cu, webgpu's per-32 twins), since the hazard is the per-vector activation
// scale.
func residentGateReasonAct(a *darch.Architecture, backend string, actSafe bool) string {
	if g := residentGateReasons(a, nil, backend, actSafe); len(g) > 0 {
		return g[0].Reason
	}
	return ""
}

// residentGate is one of residentGateReasons' refusals: which gate, and the sentence the load path reports. Missing is set on the
// feature gate only.
type residentGate struct {
	Kind    string // residentGateBackend, residentGateActHazard, residentGateFeatures, residentGateOperations, residentGateMoECap, residentGateGeometry or residentGateGemma4MoE
	Reason  string
	Missing []darch.ResidentFeature
}

const (
	residentGateBackend    = "backend"
	residentGateActHazard  = "activation-hazard"
	residentGateFeatures   = "features"
	residentGateOperations = "operations"
	residentGateMoECap     = "moe-capacity"
	residentGateGeometry   = "per-layer-geometry"
	residentGateGemma4MoE  = "gemma4-moe"
)

// residentGateReasons is every gate after the runner-shape one that refuses a on backend, in the order admission applies them; empty
// when all admit. Admission reports the first (residentGateReasonAct). The hardware matrix prints all of them, because the first
// refusal says nothing about the rest: a family that lacks one feature may also exceed the router's capacity. prog is a's program
// when the caller has it built; nil builds it here.
func residentGateReasons(a *darch.Architecture, prog *layerProgram, backend string, actSafe bool) []residentGate {
	impl, ok := residentBackendFeatures[backend]
	table := residentBackendTables[backend]
	if !ok && table == nil {
		return []residentGate{{Kind: residentGateBackend, Reason: fmt.Sprintf("backend %q declares no resident feature set", backend)}}
	}
	var out []residentGate
	// No resident GPU backend has an f32-activation projection, so a family whose output int8 activations destroy runs on the CPU at
	// weight-only precision instead of fast and wrong. Checked here rather than in decodeRunnerDecline so the generated matrix shows it.
	if why := ActivationQuantHazard(a.Name); why != "" && !(actSafe && (backend == "cuda" || backend == "metal" || backend == "webgpu")) {
		out = append(out, residentGate{Kind: residentGateActHazard,
			Reason: "every resident " + backend + " projection quantizes activations to int8, and " + why})
	}
	if table != nil {
		if prog == nil {
			prog = buildProgram(a, configExperts(a))
		}
		if gaps := table.Gaps(prog); len(gaps) > 0 {
			out = append(out, residentGate{Kind: residentGateOperations, Reason: strings.Join(gaps, "; ")})
		}
		if missing := a.FeaturesOutsideProgram(); len(missing) > 0 {
			out = append(out, residentGate{Kind: residentGateFeatures, Missing: missing,
				Reason: fmt.Sprintf("%s does not implement %v, which this model needs", backend, missing)})
		}
	} else if missing := missingFeatures(a.ResidentFeatures(), impl); len(missing) > 0 {
		out = append(out, residentGate{Kind: residentGateFeatures, Missing: missing,
			Reason: fmt.Sprintf("%s does not implement %v, which this model needs", backend, missing)})
	}
	if !residentMoECapacityOK(a, backend) {
		c := residentBackendMoECap[backend]
		out = append(out, residentGate{Kind: residentGateMoECap,
			Reason: fmt.Sprintf("MoE with %d experts / %d expert groups exceeds %s's router capacity (%d / %d)",
				a.MoE.NumExperts, a.MoE.NGroup, backend, c.experts, c.groups)})
	}
	if !residentPerLayerGeomOK(a, backend) {
		out = append(out, residentGate{Kind: residentGateGeometry,
			Reason: fmt.Sprintf("per-layer attention geometry (head_dim %d on local layers, %d on global) is not implemented on %s",
				a.HeadDim, a.Gemma4.GlobalHeadDim, backend)})
	}
	if !residentGemma4MoEOK(a, backend) {
		out = append(out, residentGate{Kind: residentGateGemma4MoE,
			Reason: fmt.Sprintf("Gemma 4's parallel dense+MoE FFN is not implemented on %s", backend)})
	}
	return out
}

// residentBackendTables names the backends whose admission reads an operation-and-parameter table instead of a feature set.
// The backend's walker checks the same
// table at build.
var residentBackendTables = map[string]*program.Support{"metal": program.Metal, "cuda": program.CUDA}

// residentDeclared reports whether backend declares what its resident implements: a feature set or a table.
func residentDeclared(backend string) bool {
	_, features := residentBackendFeatures[backend]
	return features || residentBackendTables[backend] != nil
}

// residentCovers reports whether backend implements everything a needs: its feature set, or its table and the features outside
// the program. prog is a's program, nil to build it.
func residentCovers(a *darch.Architecture, prog *layerProgram, backend string) bool {
	if t := residentBackendTables[backend]; t != nil {
		if prog == nil {
			prog = buildProgram(a, configExperts(a))
		}
		return t.Check(prog) == nil && len(a.FeaturesOutsideProgram()) == 0
	}
	return len(missingFeatures(a.ResidentFeatures(), residentBackendFeatures[backend])) == 0
}

// FeaturesOutsideProgram is the features m needs that its program does not express (featuresOutsideProgram). A backend whose
// admission reads an operation table declines a build when it is not empty.
func (m *Model) FeaturesOutsideProgram() []ResidentFeature { return m.w.Arch.FeaturesOutsideProgram() }

// residentBackendMoECap is the router-kernel capacity of each backend whose MoE scoreboard is a fixed-size array: the numeric twin of
// the feature sets (a feature is "implemented at all", a cap "implemented up to N"). A model past the cap would route on only the first
// N experts or index groups out of bounds, with plausible-looking wrong output and no error, so it declines to the staged/CPU path. An
// absent entry means no fixed-size router. Each value mirrors a constant in the backend's kernel source (gpu/moe.go MAXE, cuda/moe.cu
// MOE_MAX_E/MOE_MAX_G, metal/moe.go score arrays): change both together. Raising cuda's cap changes moe_route's scratch in the frozen
// cuda/testdata/moe.ptx, which must be regenerated at the pinned toolchain (cuda/testdata/REGEN.md); raising metal's needs Mac
// validation. History: docs/code-notes/decoder.md#residentBackendMoECap.
var residentBackendMoECap = map[string]struct{ experts, groups int }{
	"webgpu": {experts: 512, groups: 32}, // gpu/moe.go: MAXE 512, array<f32,512> score/sel / array<f32,32> gscore
	"cuda":   {experts: 512, groups: 64}, // cuda/moe.cu: MOE_MAX_E 512 / MOE_MAX_G 64. 256→512 raised deliberately (see below)
	"metal":  {experts: 256, groups: 64}, // metal/moe.go: float score[256]/sel[256], gscore[64]/keep[64]; guarded at build (moe.go:375-376)
}

// ResidentBackendMoECap returns backend's declared router-kernel capacity; ok is false for a backend with no fixed-size router.
//
// Exported for the backends to read in their own admission, instead of hardcoding a copy: two copies drift, and ResidentEligible then
// admits what BuildResident declines. History: docs/code-notes/decoder.md#ResidentBackendMoECap.
func ResidentBackendMoECap(backend string) (experts, groups int, ok bool) {
	c, ok := residentBackendMoECap[backend]
	return c.experts, c.groups, ok
}

// residentMoECapacityOK reports whether backend's router kernel can route arch's MoE; true for a dense arch or a backend without a
// fixed-size router.
func residentMoECapacityOK(a *darch.Architecture, backend string) bool {
	cap, ok := residentBackendMoECap[backend]
	if !ok || a.MoE == nil {
		return true
	}
	if cap.experts > 0 && a.MoE.NumExperts > cap.experts {
		return false
	}
	if cap.groups > 0 && a.MoE.NGroup > cap.groups {
		return false
	}
	return true
}

// residentPerLayerGeomBackends declares which resident backends implement per-layer attention geometry: a layer's own head_dim and KV
// head count genuinely differing from another's (Gemma 4's local/global split, HeadDim vs gemma4.GlobalHeadDim), not just a per-layer
// RoPE table (FeatPerLayerRoPE, which this is not).
//
// It is not a ResidentFeature: no other family needs it, and Gemma 3 and dense Gemma 4 otherwise derive identical feature sets, so the
// taxonomy cannot name it. A backend without the seam would be admitted by the feature check alone and fail on upload, which is the
// decline this predicate makes deliberate. History: docs/code-notes/decoder.md#residentPerLayerGeomBackends.
var residentPerLayerGeomBackends = map[string]bool{"cuda": true, "metal": true, "webgpu": true}

// residentPerLayerGeomOK reports whether backend implements the per-layer geometry a's layers
// actually need. true for every arch that doesn't vary (the overwhelming majority); false only
// when the arch's layers genuinely differ AND the backend lacks the seam.
func residentPerLayerGeomOK(a *darch.Architecture, backend string) bool {
	if a.Gemma4 == nil || a.Gemma4.GlobalHeadDim <= 0 || a.Gemma4.GlobalHeadDim == a.HeadDim {
		return true // uniform head_dim across every layer — no backend needs a special seam
	}
	return residentPerLayerGeomBackends[backend]
}

// residentGemma4MoEBackends declares which resident backends implement Gemma 4's parallel dense+MoE FFN (enable_moe_block,
// gemma4MoeMLP): the joint dense‖MoE bridge, not just the per-layer attention geometry of dense Gemma 4.
var residentGemma4MoEBackends = map[string]bool{"cuda": true, "metal": true, "webgpu": true}

func residentGemma4MoEOK(a *darch.Architecture, backend string) bool {
	if a.Gemma4 == nil || a.MoE == nil {
		return true
	}
	return residentGemma4MoEBackends[backend]
}

// PerLayerGeomOK is residentPerLayerGeomOK's Model-level twin, exported so a backend's BuildResident can check it directly: that
// admission is hand-rolled from MissingResidentFeatures, not ResidentEligible, and the feature check alone would admit dense Gemma 4
// onto a backend without the per-layer geometry seam (see residentPerLayerGeomBackends).
func (m *Model) PerLayerGeomOK(backend string) bool { return residentPerLayerGeomOK(m.w.Arch, backend) }

// Gemma4MoEOK is residentGemma4MoEOK's Model-level twin, exported so a resident backend's own
// BuildResident can check it directly.
func (m *Model) Gemma4MoEOK(backend string) bool { return residentGemma4MoEOK(m.w.Arch, backend) }

// HasPerLayerGeometry reports whether the model's layers genuinely differ in attention geometry
// (e.g. Gemma 4's local/global head_dim split: 256 vs 512).
func (m *Model) HasPerLayerGeometry() bool {
	a := m.w.Arch
	return a.Gemma4 != nil && a.Gemma4.GlobalHeadDim > 0 && a.Gemma4.GlobalHeadDim != a.HeadDim
}

// ResidentBackendFeatures returns a copy of the feature set a resident backend implements (nil if the backend is unknown). The copy keeps
// the source map read-only from outside the package: a caller or third-party init() cannot add a feature claim its kernels do not
// implement, nor trigger a concurrent map write during a Load. The package's own admission path reads the unexported map.
func ResidentBackendFeatures(backend string) map[darch.ResidentFeature]bool {
	return maps.Clone(residentBackendFeatures[backend])
}

// residentBackendFeatures declares what each resident backend's decode path implements. It lives here, not in the backends, for one
// source of truth (hand-maintained copies are how the silent-wrong-output bug recurs) and for testability: the backends are build-tagged,
// so a test that could see their sets could not run in CI, while the registry-driven admission gate (features_test.go) checks every
// (arch × backend) pair with no GPU present.
//
// A backend adds an entry only when it ships the kernel that implements it, and only with an end-to-end gate on the real path:
// overclaiming here is exactly the lie the gate exists to catch. Per-entry history: docs/code-notes/decoder.md#residentBackendFeatures and the sections named after it.
var residentBackendFeatures = map[string]map[darch.ResidentFeature]bool{

	// WebGPU (gpu/): the richest runner — the levers in docs/gpu-residency-coverage.md.
	"webgpu": {
		darch.FeatQKNorm:         true, // per-head QK-norm before RoPE
		darch.FeatPartialRotary:  true, // rotary_dim < head_dim
		darch.FeatSlidingWindow:  true, // per-layer windowed start
		darch.FeatPerLayerRoPE:   true, // differing invFreq per layer type
		darch.FeatRopeMscale:     true, // YaRN attention_factor
		darch.FeatMoE:            true, // router / stacked experts / shared expert
		darch.FeatMoEGatedShared: true, // sharedGatedCombine — sigmoid-gated shared expert (gpu/moe.go)
		darch.FeatMLA:            true, // latent-KV attention
		darch.FeatSSM:            true, // Mamba-2 engine (Granite-4.0-H, Nemotron-H)
		darch.FeatDeltaNet:       true, // Gated-DeltaNet engine + fused attn output gate (gpu/deltanet.go)
		darch.FeatNonGatedMLP:    true, // relu2Quant (Nemotron-H squared-ReLU)
		darch.FeatNonGatedReLU2:  true, // relu2Quant is the only non-gated activation kernel
		darch.FeatLogitScale:     true, // Granite logits_scaling — folded into the lm_head weight scale at BuildResident, not a host-side postcap
		darch.FeatRMSAddOne:      true, // (1+w) RMS offset
		// The Gemma3/Gemma4/gpt-oss set. FeatEmbedScale needs no gpu/ code: embedResident applies √hidden host-side for every backend before
		// resident.Forward.
		darch.FeatEmbedScale: true,
		// FeatFinalLogitSoftcap: applySoftcap in gpu/softcap.go, host-side after readback.
		darch.FeatFinalLogitSoftcap: true,
		// FeatSandwichNorm reuses the rmsnormF32 closure; it defeats the fused gemvAdd residual epilogue at the o-proj and MLP down-proj sites.
		darch.FeatSandwichNorm: true,
		// FeatGatedGELU: the gegluShaderWGSL/gegluQuantWGSL kernel pair (gpu/layer.go, gpu/decodefuse.go). It clamps the tanh argument to ±15
		// before calling tanh, as Metal's does: without the clamp tanh overflows.
		darch.FeatGatedGELU: true,
		// FeatOutBias: gpt-oss's o_proj bias, composed from a bare gemv then biasAdd (the general vec[i] += other[i] kernel); no new kernel.
		darch.FeatOutBias: true,
		// FeatGemma4EModel: shared-KV layers bind their source's cache and project Q only, each layer's FFN runs at its own width, and the
		// per-layer-embedding branch is composed from the gated-product and gemv kernels (gpu/decoderunner.go, runLayer.kvShared and
		// pleGate). The resident row is [hidden ‖ layers·pleDim] and DecodeRunner.writeInputs refuses any other length.
		darch.FeatGemma4EModel: true,
		// FeatAttnSink: the learned sink threaded through every attention pipeline (attn, attn-keys, attn-f16, attn-i8 and the wide variants),
		// the clamped interleaved-SwiGLU expert (gptossGluQuantWGSL) and a router whose bias reaches the mixing weight (routeGptOssWGSL;
		// moeRouteWGSL's contract is wrong for this family). The sink needs its own per-layer uniform (HS in gpu/attention.go): hasSink is a
		// per-layer property that the geometry-cached P uniform's geomFor dedup cannot carry. Declared after a real gpt-oss forward ran
		// resident end to end (TestGptOssResidentParityWebGPU).
		darch.FeatAttnSink: true,
		// Declared after each ran resident against the CPU, with a plant on each red (TestOlmoHybridResidentParityWebGPU).
		darch.FeatNoPE:         true,
		darch.FeatPostOnlyNorm: true,
		darch.FeatQKNormWhole:  true,
		// FeatAttnTemp: the decode runner folds the query temperature into each token's attention score scale, and the batched
		// prefill declines to row-by-row decode (TestMinistral3ResidentParityWebGPU).
		darch.FeatAttnTemp: true,
		// FeatPairwiseRoPE: the decode rope kernels' pairwise twins (gpu/pairwise.go), bound in place of the NeoX ones. FeatPairwiseMRoPE:
		// the batched prefill, the image turn's m-RoPE pass included, declines on a pairwise model, so an image prompt takes the CPU
		// prefill and its upload, and decode past it rotates at ForwardMRoPE's position through the same twins
		// (TestGlmOcrResidentParityWebGPU).
		darch.FeatPairwiseRoPE:  true,
		darch.FeatPairwiseMRoPE: true,
		// FeatLayerNorm (bias-free; a bias declines at the build) and FeatParallelBlock: LayerNormQuantWGSL in RMSNorm's place,
		// and the MLP reading the attention's quantized input (TestCohereResidentParityWebGPU).
		darch.FeatLayerNorm:     true,
		darch.FeatParallelBlock: true,
	},
}
