package decoder

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime/debug"
	"sort"
	"strings"
	"sync/atomic"

	"github.com/townsendmerino/aikit/linalg"

	"github.com/townsendmerino/goinfer/internal/runtime/adapter"
	"github.com/townsendmerino/goinfer/internal/runtime/logs"
)

// isWebGPUBackend reports whether a Backend.Name() string names the webgpu backend. NOT a plain
// equality check: gpu.webgpuBackend's Name() is "webgpu:" + the underlying GPU surface ("webgpu:metal",
// "webgpu:vulkan"), while every other backend's is a bare string ("cpu", "cuda", "metal"), so
// a `== "webgpu"` comparison never matches a real webgpu backend.
// History: docs/code-notes/decoder.md#isWebGPUBackend.
func isWebGPUBackend(name string) bool { return strings.HasPrefix(name, "webgpu") }

// GPU full-residency decode support. When a backend implements ResidencyBackend (cuda, metal, webgpu) and admits the
// model (withResidency → Admit), the whole per-token forward runs on the device through the backend's
// ResidentForward. This file is the decoder-side seam: the interfaces the backends implement, the admission gates, and
// (in generateInto) the routing. Sampler, constrain and Session bookkeeping stay CPU-side and consume the logits (or
// device-sampled ids) that come back.
//
// The stateless Model.Generate path runs resident, reusing the prefix already committed to the resident KV
// (resident_reuse.go) and prefilling the rest with the backend's batched Prefiller where it has one. A plain Session
// drives a CPU-side KVCache the resident KV cannot share, so its turns stay on the CPU/staged path (serve sends
// resident models down the stateless path: loadedModel.drive). Speculative verify runs through ForwardN.

// Prefiller is an OPTIONAL ResidentForward extension: a backend that can ingest a whole prompt
// in one batched pass (populating the resident KV for positions startPos..startPos+len-1) and
// return the LAST token's logits, instead of the token-by-token Forward loop. Much faster TTFT
// for long prompts when the backend has a real batched (e.g. MMA) forward. generateInto type-
// asserts for this; backends without it fall back to the sequential prefill.
type Prefiller interface {
	// ctx is checked at whatever granularity the backend can honour: between passes, and inside
	// them where the backend's own loop allows. A batched prefill is the one place a cancelled
	// request keeps a GPU busy with nothing watching (the sequential fallback checks ctx.Err() per
	// token; a backend that ingests the prompt in one pass has no equivalent unless handed the
	// context). Cancellation returns ctx.Err(); the caller stops, rather than falling back to the
	// sequential loop it was also cancelling.
	PrefillLast(ctx context.Context, embeddings [][]float32, startPos int) (logits []float32, err error)
}

// PrefillPartial is the error a batched prefill returns when one of its passes declined or failed after earlier passes of the same call
// committed their rows: positions [startPos, Next) are written and the caller continues from Next, not from startPos. A windowed resident
// requires it, because the committed passes may have moved its window past startPos, so a restart there reads history it has dropped.
// Unwrap is the failing pass's own error, so errors.Is(err, a decline sentinel) still holds.
type PrefillPartial struct {
	Next int
	Err  error
}

func (e *PrefillPartial) Error() string {
	return fmt.Sprintf("%v (positions before %d are written)", e.Err, e.Next)
}
func (e *PrefillPartial) Unwrap() error { return e.Err }

// prefillResume is where a prefill continues after a batched call that started at from returned err: PrefillPartial's Next when it names a
// later position, else from.
func prefillResume(from int, err error) int {
	var p *PrefillPartial
	if errors.As(err, &p) && p.Next > from {
		return p.Next
	}
	return from
}

// ResidentAdapterProj and ResidentAdapterLayer are a compute-time LoRA delta as plain data (package internal/adapter has
// their layout and semantics). They live there so a backend's kernel layer can take them without importing decoder.
type (
	ResidentAdapterProj  = adapter.ResidentAdapterProj
	ResidentAdapterLayer = adapter.ResidentAdapterLayer
)

// residentAdapterProj converts one internal loraDelta to its exported ResidentAdapterProj twin
// (nil ⇒ nil, matching loraLayerDelta's own untargeted-projection convention).
func residentAdapterProj(d *loraDelta) *ResidentAdapterProj {
	if d == nil {
		return nil
	}
	return &ResidentAdapterProj{A: d.a, B: d.b, R: d.r, In: d.in, Out: d.out, Scale: float32(d.scale)}
}

// residentAdapterLayers converts a loaded loraRuntime into the exported per-layer shape a
// resident backend's SetAdapter consumes.
func residentAdapterLayers(rt *loraRuntime) []ResidentAdapterLayer {
	out := make([]ResidentAdapterLayer, len(rt.layers))
	for i, l := range rt.layers {
		out[i] = ResidentAdapterLayer{
			Q: residentAdapterProj(l.q), K: residentAdapterProj(l.k), V: residentAdapterProj(l.v),
			O: residentAdapterProj(l.o), Gate: residentAdapterProj(l.gate), Up: residentAdapterProj(l.up),
			Down: residentAdapterProj(l.down),
		}
	}
	return out
}

// ResidentHybridMRoPEPrefill is an OPTIONAL marker for a MRoPEPrefillAvailable implementer: its batched m-RoPE prefill is correct for a
// model with RECURRENT state (the Gated-DeltaNet hybrids, Qwen3.5+), building that state itself from the spliced embeddings. Without
// it GenerateQwenVL keeps a recurrent family on the CPU prefill, because the CPU-prefill-then-UploadKV bridge copies only the layers
// that have a KV and would leave the recurrent state zeroed (docs/multimodal.md, P8 record, item 5). HybridMRoPEPrefill reports
// whether it applies to THIS loaded model (the kernel is present, the layout is supported, and no layer is one the implementer has not
// been gated on).
type ResidentHybridMRoPEPrefill interface {
	HybridMRoPEPrefill() bool
}

// PrefillPathReporter is an OPTIONAL Prefiller extension: report at LOAD time whether the batched prefill will be
// taken for THIS model, and when it won't, why and what that costs. Prefiller declines per call and generateInto's
// fallback is silent by design, so a decline is a large, invisible TTFT regression; serve states the resolved path at
// startup. reason names the ACTUAL condition and its cost, not "declined", e.g. "batched prefill requires int4
// projections (int8int8 at layer 0)".
type PrefillPathReporter interface {
	PrefillPath() (batched bool, reason string)
}

// VerifyPathReporter is an OPTIONAL resident extension: report at LOAD time whether the
// speculative verify path (ForwardN) is batched or falls back to a sequential loop.
// reason names the condition when sequential (e.g. "sequential — paged MoE requires per-layer host staging").
type VerifyPathReporter interface {
	VerifyPath() (batched bool, reason string)
}

// residentCapsOf is rf's ResidentCaps, or the zero value when rf is nil or reports none.
func residentCapsOf(rf ResidentForward) ResidentCaps {
	return SubsystemsOf(rf).Runtime.ResidentCaps()
}

// prefillBlocks is the resident's PrefillBlocksAvailable, or nil when it reports none.
func (m *Model) prefillBlocks() ResidentPrefillSubsystem {
	pf := m.resPrefill()
	if !pf.PrefillBlocksAvailable() {
		return nil
	}
	return pf
}

// prefillBlockRows is the pass length residentPrefillSeed splits a prompt of promptLen tokens into, or 0 when the
// resident's prefill is not split: it does not report a block, or its continuation is not exact. A block under the
// resident's kernel-class floor is raised to it, as mc3Prefill raises its chunk, so the first pass runs the class the
// whole prompt would.
func (m *Model) prefillBlockRows(promptLen int) int {
	pb := m.prefillBlocks()
	if pb == nil {
		return 0
	}
	if !m.resPrefill().PrefillTailExact() {
		return 0
	}
	B := pb.PrefillBlockRows()
	if f := m.prefillKernelFloor(); B > 0 && f > 0 && promptLen >= f && B < f {
		B = f
	}
	return max(B, 0)
}

// residentPrefillScratch is the memory a resident request's prefill takes at its largest pass, when the resident
// splits its prefill (prefillBlockRows): that pass's scratch by the resident's own figure, plus the host embeddings
// built for it. The last pass is the longest, at most a block and the resident's tail minimum less one rows. A resident that
// chunks its own prefill (PrefillChunksAvailable) is priced at one chunk and the embeddings of the whole suffix. ok is false for a
// resident that does neither, whose request is priced by prefillScratchBytes.
func (m *Model) residentPrefillScratch(promptTokens int) (bytes int64, ok bool) {
	B := m.prefillBlockRows(promptTokens)
	if B <= 0 || promptTokens <= 0 {
		return m.residentChunkedScratch(promptTokens)
	}
	rows := min(promptTokens, B+m.prefillTailMinRows()-1)
	return m.prefillBlocks().PrefillScratchBytes(rows) + int64(rows)*4*int64(m.BackendView().EmbedLen()), true
}

// prefillKernelFloor is the resident's kernel-class boundary for chunking: PrefillKernelFloor when the resident
// reports it, else the fast-prefill floor (the same boundary, as CUDA reports it), else 0.
func (m *Model) prefillKernelFloor() int {
	fast, hasFast, kernel, hasKernel := m.residentFloors()
	switch {
	case hasKernel:
		return kernel
	case hasFast:
		return fast
	}
	return 0
}

// residentFloors is the resident's two kernel-class reports (ResidentCaps), each with whether the resident gives it. A caller that
// keys on a report's presence reads has, not a 0.
func (m *Model) residentFloors() (fast int, hasFast bool, kernel int, hasKernel bool) {
	c := residentCapsOf(m.resident)
	return c.FastPrefillFloor, c.HasFastPrefill, c.KernelFloor, c.HasKernelFloor
}

// ResidencyBackend is the optional capability a Backend advertises to build a
// ResidentForward from a loaded Model. The cuda, metal and webgpu backends implement it.
// withResidency applies Admit's gates BEFORE calling it; the backend's own build can
// still decline (memory, or a runtime check of its own).
//
// To DECLINE — this model does not run resident here — return ok=false with a *ResidentDeclineError
// (DeclineResident) naming the reason: the load path records it as the model's ResidentDecline, which
// DecodePath and `serve check` print. Any other non-nil err is a FAILURE (no usable device, a driver
// error). (nil, false, nil) still means "declined, no reason given" for backends written before this.
type ResidencyBackend interface {
	BuildResident(m *Model) (rf ResidentForward, ok bool, err error)
}

// ResidencyPricer is a ResidencyBackend that can say what BuildResident's own checks before it allocates (its features, its
// context, its memory guard) decide for m, without building: FitQuery's decode path on that backend. err is what BuildResident
// would return for that decision (a DeclineResident, or the failure it would report). It must not touch the device or keep
// state on m.
type ResidencyPricer interface {
	PriceResident(m *Model) (ResidentPrice, error)
}

// ResidentPrice is a ResidencyPricer's answer for a build its checks admit.
type ResidentPrice struct {
	// Quant is the resident quant DecodePath would report, "" for residentQuantLabel's. Unknown, when set, is why only the build
	// can name it.
	Quant, Unknown string
	Ctx            int // the context the resident would hold
}

// ResidentDeclineError is a backend's reason for not building a resident path: an ordinary decline
// (unsupported shape, too big for the router, doesn't fit memory), not a device failure.
type ResidentDeclineError struct{ Reason string }

func (e *ResidentDeclineError) Error() string { return "resident path declined: " + e.Reason }

// DeclineResident is the error a BuildResident returns to decline with a reason.
func DeclineResident(format string, args ...any) error {
	return &ResidentDeclineError{Reason: fmt.Sprintf(format, args...)}
}

// IsResidentDecline reports whether err is a decline (DeclineResident) rather than a failure.
func IsResidentDecline(err error) bool {
	var d *ResidentDeclineError
	return errors.As(err, &d)
}

// DecodeRunnerEligible reports whether this model passes the FIRST resident-admission gate: its arch is a shape the
// uniform-layer resident runners can express (decodeRunnerEligible) and the load-time precision policy admits it.
// The per-backend gates (implemented features, MoE router capacity, per-layer geometry) are residentGateReason's
// (features.go); Admit (admit.go) runs both. A decline → CPU (or webgpu's staged path).
func (m *Model) DecodeRunnerEligible() bool { return m.decodeRunnerDecline() == "" }

// decodeRunnerDecline is DecodeRunnerEligible with its reason: "" when the arch is a shape the resident
// runner supports and the load-time precision policy admits it.
func (m *Model) decodeRunnerDecline() string {
	f := m.admitFacts()
	return admitRunner(f, f.Arch.runnerShapeDeclineSSM(f.SSMResident))
}

// actGroupResidentDecline is the ActQuantGroup gate (option-path admission finding 1, 2026-10-08,
// ): per-group activation scales are implemented by
// CUDA residency only (actgroup.cu). Metal's and WebGPU's projections read per-vector scales and nothing
// in either backend reads the group, so a load that asked for per-32 ran resident at per-vector, silently,
// whenever the family had no activation hazard for the gate below to catch. The Options doc has always
// said such a load declines to the CPU; this makes it so. Metal's q4k lane is exempt: it quantizes no
// activation at all, and every q4k load carries the group (a direct load's and a q4k .giw's alike). quant
// is the loaded weights' (Model.Quant), as Metal's build reads it, not the --quant option a .giw does not keep.
func actGroupResidentDecline(backend string, actGroup int, quant string) string {
	if actGroup == 0 || backend == "cuda" || backend == "metal" && quant == "q4k" {
		return ""
	}
	return fmt.Sprintf("ActQuantGroup %d: %s's resident projections use per-vector activation scales (per-group "+
		"scales are implemented by CUDA residency only), so the load runs on the CPU, which honours the group", actGroup, backend)
}

// residentProjsInt4 reports whether the loaded projection weights are int4 (W4A8) — the gate for
// Nemotron-H default-on residency. Scans for the first attention/MLP projection (the mamba
// in/out_proj are always f32 in the model and quantized at build, so they don't indicate the
// load precision).
// ResidentProjsInt4 is residentProjsInt4, for a backend that runs Nemotron-H resident at int4 only (Metal: it has no native
// int8 path for a recurrent family, and would otherwise re-quantize int8 projections to int4).
func (m *Model) ResidentProjsInt4() bool { return m.residentProjsInt4() }

func (m *Model) residentProjsInt4() bool {
	for i := range m.w.Layers {
		lw := &m.w.Layers[i]
		if lw.UpProj.Rows() > 0 {
			return lw.UpProj.Kind() == "int4"
		}
		if lw.QProj.Rows() > 0 {
			return lw.QProj.Kind() == "int4"
		}
	}
	return false
}

// decodeRunnerEligible is the arch-only predicate behind DecodeRunnerEligible —
// usable directly from a resolved Architecture (e.g. the capability matrix)
// without a loaded Model.
func (a *Architecture) decodeRunnerEligible() bool { return a.runnerShapeDecline() == "" }

// Runner-shape decline reasons. The hardware matrix prints them, so they are short and name no backend.
const (
	runnerShapeOwnForward = "own forward, not bridged"
	runnerShapeBringUp    = "bring-up gate (GOINFER_SSM_RESIDENT unset)"
	runnerShapeNemoMoE    = "Nemotron MoE block"
	runnerShapeMoE        = "MoE shape"
	runnerShapeRoPE       = "local and global RoPE tables differ in length"
)

// runnerShapeDecline is decodeRunnerEligible with its reason: "" when the uniform-layer resident runners can express
// the arch's shape. It is the first of the admission gates and says nothing about the later ones, so a caller that
// reports why a model is not resident reads residentGateReasons as well. It reads GOINFER_SSM_RESIDENT from the arch's knob
// snapshot, or the environment for an arch built without one.
func (a *Architecture) runnerShapeDecline() string {
	return a.runnerShapeDeclineSSM(a.ssmResidentKnob())
}

// ssmResidentKnob reports whether GOINFER_SSM_RESIDENT is set for this arch: its knob snapshot, or the environment for an arch
// built without one.
func (a *Architecture) ssmResidentKnob() bool { return a.knobs.get(knobSSMResident) != "" }

// runnerShapeDeclineSSM is runnerShapeDecline with the GOINFER_SSM_RESIDENT knob given (ssmResident), not read: Admit's form.
func (a *Architecture) runnerShapeDeclineSSM(ssmResident bool) string {
	// Own-forward families (their non-uniform forward the uniform-layer runners can't express).
	// Only Gemma 4 (dense, env-gated) is bridged so far; the rest keep the staged path. A
	// bridged family does NOT early-return true — it FALLS THROUGH to the common feature/shape
	// checks below, so a later decline COMPOSES instead of being short-circuited past. That is
	// the trap the old blanket `return false` set for whoever narrows the next family here: move
	// one line to a conditional admit and it silently skips every check below it. Structuring the
	// bridged case as a fall-through makes that mistake un-writable (cf. the geometry port).
	switch {
	case a.gptoss != nil:
		// gpt-oss falls through to the common checks below, same discipline as gemma4: the sink
		// (FeatAttnSink) and the MoE bias-in-combine kernels are backend feature gates, not an
		// arch-level decline, so a backend without them declines there instead of here.
	case a.gemma4 != nil:
		// Gemma 4 is admitted UNCONDITIONALLY: do not re-gate a family on an env var without a gate behind it (the flag reads
		// as caution and functions as a coverage hole). The gates are cuda.TestGemma4MoEScaled_residentParity and
		// TestGemma4DenseScaled_residentParity. Dense and MoE both fall through to the common checks below; softcap and the
		// E-model's PLE (FeatGemma4EModel) are per-backend feature gates. The parallel dense‖MoE FFN runs on cuda's own path,
		// routed around the generic MoE checks via HasGemma4MoE.
	case a.qwen35 != nil:
		// Gated-DeltaNet hybrids (qwen3_5, qwen3_5_moe, qwen3_next) fall through, same discipline as gemma4/gpt-oss: the
		// recurrence and the fused output gate are a BACKEND capability (FeatDeltaNet), so a backend without them declines at
		// the feature gate, and falling through is what keeps the checks below composing. The MoE siblings also need FeatMoE,
		// from the ordinary MoE checks.
	case a.llama4 != nil:
		return runnerShapeOwnForward
	case a.lfm2 != nil:
		// LFM2/LFM2.5: its short-convolution layers (opShortConv) are not bridged. Declining HERE and not only via FeatShortConv is
		// deliberate: decodeRunnerEligible is an ARCH-SHAPE predicate, and 22 of 30 layers run a gated short
		// convolution with a rolling window that no uniform-layer runner can express, a property of the family
		// rather than of any backend's kernels. It also keeps the capability matrix honest: its GPU-resident
		// column is generated from this predicate, and without this case LFM2 would fall through to
		// ropeResidentCompatible and read "yes".
		return runnerShapeOwnForward
	}
	// Granite-4.0-H resident SSM hybrid (P5b): its own mixer-kind path (Mamba-2 ⊕
	// attention), so it bypasses the standard GQA gates below. Guarded during the
	// parity bring-up (P5b.3/P5b.4); P6 makes this unconditional.
	if a.granite != nil {
		if !ssmResident {
			return runnerShapeBringUp
		}
		return ""
	}
	// Nemotron-H resident (dense squared-ReLU hybrid): single-op-per-block Mamba-2 / NoPE-GQA / relu² MLP,
	// reusing the granite SSM engine. Arch-eligible; the int4-default-vs-int8-opt-in precision gate is applied
	// at the Model level (DecodeRunnerEligible).
	//
	// Nemotron 3 Nano adds a fourth block kind (MoE FFN, arch.MoE != nil). cuda and metal build no MoE block
	// and their per-layer switch has no default, so an unhandled kind would leave that layer's buffers nil: a
	// crash-or-garbage risk, not a clean decline. Decline here, at the one predicate every backend's admission
	// funnels through; decodeRunnerDecline overrides it for webgpu, which implements the block.
	if a.nemotron != nil {
		if a.MoE != nil {
			return runnerShapeNemoMoE
		}
		return ""
	}
	if a.MoE != nil && !a.moeResidentEligible() {
		return runnerShapeMoE
	}
	// FFN / norm / head constraints common to both attention paths (sandwich norms, logit softcap, NonGatedMLP,
	// LearnedPosEmbed, OutBias) are all representable, and a backend that lacks one declines via the feature gate
	// (decoder/features.go), not here: this stays an ARCH-shape predicate, because a capability blocked here for everyone
	// blocks the family for a backend that implements it. GPT-2, the only LearnedPosEmbed family, has no RoPE, so
	// ropeResidentCompatible passes vacuously.
	// MLA (DeepSeek/Kimi) runs its own latent-attention path (gpu/mla.go), so the GQA/RoPE/QK-norm/sliding-window checks
	// don't apply; its decoupled RoPE rides a separate qk_rope slice, not HeadDim.
	if a.mla != nil {
		return ""
	}
	// Standard GQA attention: QK-norm (Qwen3/GLM/Mellum) is handled (Lever C1 — per-head
	// RMSNorm before RoPE), q/k/v bias (Qwen2) and the (1+w) RMS offset too. Partial RoPE
	// (GLM/Phi rotary_dim < HeadDim) is handled (Lever C5), sliding window (Lever C6), and
	// per-layer-type RoPE (Mellum YaRN-on-global vs default-local invFreq + mscale, Lever
	// C7). The last gate is ropeResidentCompatible below.
	if !a.ropeResidentCompatible() {
		return runnerShapeRoPE
	}
	return ""
}

// ropeResidentCompatible is a coarse ARCH-level screen on the GENERIC (finalizeRoPE) local/global inv-freq tables:
// they must share a length. Gemma passes it trivially (both come from one model-level rd), and that is NOT proof the
// runner ropes Gemma correctly: its REAL per-layer tables (Layer(i).RopeInvFreq) differ in length, an inequality
// this check never sees. The real invariant (each bound invFreq buffer has exactly rhalf entries) is asserted per layer
// where the buffer meets the geometry (cuda/backend.go).
func (a *Architecture) ropeResidentCompatible() bool {
	return len(a.ropeInvFreqLocal) == len(a.ropeInvFreqGlobal)
}

// moeResidentEligible reports whether this arch's MoE is a shape the GPU resident
// runner handles. The route kernel now covers every routing flavor — softmax/sigmoid,
// optional selection bias, norm_topk_prob, routed scaling, and DeepSeek group-limited
// top-k (NGroup > 1, Lever C4c) — plus the always-on shared expert (C3d) and dense
// prefix layers (FirstKDense > 0), so any int8-stackable MoE qualifies.
func (a *Architecture) moeResidentEligible() bool {
	return a.MoE != nil
}

// ResidentMoESpec holds the MoE geometry and routing knobs for a resident backend or planner.
type ResidentMoESpec struct {
	NumExperts            int
	TopK                  int
	IntermediateDim       int
	SharedIntermediateDim int
	RouterSigmoid         bool
	NormTopKProb          bool
	SharedUngated         bool
	RoutedScale           float64
	NGroup                int
	TopkGroup             int
}

// ResidentMLASpec holds the DeepSeek/Kimi MLA geometry and scales for a resident backend or planner.
type ResidentMLASpec struct {
	QLoRARank      int
	KVLoRARank     int
	QKNopeHeadDim  int
	QKRopeHeadDim  int
	VHeadDim       int
	RopeInterleave bool
	AttnScale      float64
	RopeScale      float64
}

// ResidentQwen35Spec holds the Gated-DeltaNet and gated-softmax geometry for Qwen3.5/Qwen3-Next/Olmo-Hybrid.
type ResidentQwen35Spec struct {
	ConvKernel    int
	KeyHeadDim    int
	ValueHeadDim  int
	NumKeyHeads   int
	NumValueHeads int
	AttnGate      bool
}

// ResidentNemotronSpec holds the Mamba-2 and GQA geometry for Nemotron-H.
type ResidentNemotronSpec struct {
	NHeads     int
	HeadDim    int
	DState     int
	NGroups    int
	DConv      int
	NormGroups int
	AttnScale  float32
}

// ResidentGraniteSpec holds the Mamba-2 geometry and four scalar multipliers for Granite-4.0-H.
type ResidentGraniteSpec struct {
	NHeads     int
	HeadDim    int
	DState     int
	NGroups    int
	DConv      int
	EmbMul     float32
	ResidMul   float32
	LogitScale float32
	AttnScale  float32
}

// ResidentSpec bundles the family-specific resident geometry of a Model into named sub-structs so callers
// (such as fitplan.go and backend builders) can inspect named fields instead of unpacking positional 7-to-11-tuples.
type ResidentSpec struct {
	MoE      *ResidentMoESpec
	MLA      *ResidentMLASpec
	Qwen35   *ResidentQwen35Spec
	Nemotron *ResidentNemotronSpec
	Granite  *ResidentGraniteSpec
}

// ResidentSpec returns the structured resident geometry of m. Sub-structs are non-nil only when the model's
// architecture carries that subsystem.
func (m *Model) ResidentSpec() ResidentSpec {
	var spec ResidentSpec
	a := m.w.arch
	if mo := a.MoE; mo != nil {
		spec.MoE = &ResidentMoESpec{
			NumExperts:            mo.NumExperts,
			TopK:                  mo.TopK,
			IntermediateDim:       mo.IntermediateDim,
			SharedIntermediateDim: mo.SharedIntermediateDim,
			RouterSigmoid:         mo.RouterSigmoid,
			NormTopKProb:          mo.NormTopKProb,
			SharedUngated:         mo.SharedUngated,
			RoutedScale:           mo.RoutedScale,
			NGroup:                mo.NGroup,
			TopkGroup:             mo.TopkGroup,
		}
	}
	if p := a.mla; p != nil {
		spec.MLA = &ResidentMLASpec{
			QLoRARank:      p.QLoRARank,
			KVLoRARank:     p.KVLoRARank,
			QKNopeHeadDim:  p.QKNopeHeadDim,
			QKRopeHeadDim:  p.QKRopeHeadDim,
			VHeadDim:       p.VHeadDim,
			RopeInterleave: p.ropeInterleave,
			AttnScale:      a.AttnScale,
			RopeScale:      a.ropeMscale(0),
		}
	}
	if g := a.qwen35; g != nil {
		spec.Qwen35 = &ResidentQwen35Spec{
			ConvKernel:    g.ConvKernel,
			KeyHeadDim:    g.KeyHeadDim,
			ValueHeadDim:  g.ValueHeadDim,
			NumKeyHeads:   g.NumKeyHeads,
			NumValueHeads: g.NumValueHeads,
			AttnGate:      g.AttnGate,
		}
	}
	if np := a.nemotron; np != nil {
		spec.Nemotron = &ResidentNemotronSpec{
			NHeads:     np.NHeads,
			HeadDim:    np.HeadDim,
			DState:     np.DState,
			NGroups:    np.NGroups,
			DConv:      np.DConv,
			NormGroups: np.NGroups,
			AttnScale:  float32(a.AttnScale),
		}
	}
	if g := a.granite; g != nil {
		ls := float32(1)
		if a.LogitScale != 0 {
			ls = float32(a.LogitScale)
		}
		em, rm := g.EmbMul, g.ResidMul
		if os.Getenv("GOINFER_SSM_NOMUL") != "" {
			em, rm, ls = 1, 1, 1
		}
		spec.Granite = &ResidentGraniteSpec{
			NHeads:     g.NHeads,
			HeadDim:    g.HeadDim,
			DState:     g.DState,
			NGroups:    g.NGroups,
			DConv:      g.DConv,
			EmbMul:     em,
			ResidMul:   rm,
			LogitScale: ls,
			AttnScale:  float32(a.AttnScale),
		}
	}
	return spec
}

// Gemma4MoEResidentBundle is everything the CUDA resident build needs to construct one gemma4 enable_moe_block layer
// (the parallel dense‖MoE FFN). RouterProjScaled has the learned routerScale and the hidden^-0.5 factor FOLDED into its
// columns (exact in f32), so the resident router is rmsnorm_nw(h) → gemv_f32_f32(RouterProjScaled) → moe_route with
// scores bit-identical to the CPU's. Weight-matrix pointers alias the model's tensors (read-only, for packing).
type Gemma4MoEResidentBundle struct {
	PreFFNNorm, PostFFNNorm1, PreFFNNorm2, PostFFNNorm2, PostFFNNorm []float32           // the 5 RMSNorm weights
	MlpGate, MlpUp, MlpDown                                          *linalg.WeightMat   // parallel dense branch
	RouterProjScaled                                                 []float32           // [nE*hidden] row-major, scale folded
	PerExpertScale                                                   []float32           // [nE]
	ExpertsGateUp, ExpertsDown                                       []*linalg.WeightMat // per expert (gate‖up fused; down)
	LayerScalar                                                      float32
	DenseInter, MoeInter, NE, TopK                                   int
}

// RopeInvFreqLayer returns layer i's RoPE inverse-frequency table as float32: the global or local table per
// the layer's attention type (Mellum YaRN-on-global vs default-local).
//
// A NoPE layer (SmolLM3's no_rope_layers, Cohere2's every-Nth global layer: arch.isNoPELayer, FeatNoPE) gets an
// all-ZERO table of the same width. The resident rope kernels compute cos=cos(pos·invFreq)·mscale and
// sin=sin(pos·invFreq)·mscale, so invFreq==0 is the identity rotation ONLY when mscale==1 (Layer(i).RopeMscale).
// That holds for every family that sets layerNoPE today; a family combining YaRN with NoPE needs this revisited.
// A RESIDENT-ONLY accessor: the CPU forward skips ropeAt for a NoPE layer.
//
// A whole-model no-RoPE arch (rope_theta absent, arch.NoPositionEncoding) never populates ropeInvFreqGlobal/Local,
// so ropeInvFreq(i) is empty for every layer. The table is then RotaryDim()/2 wide, the width every resident rope
// kernel launch expects (rhalf): a zero-length table would be a 0-byte device allocation, a hard error on CUDA
// ( G5; docs/code-notes/decoder.md#Model.RopeInvFreqLayer).
func (m *Model) RopeInvFreqLayer(i int) []float32 {
	a := m.w.arch
	inv := a.ropeInvFreq(i)
	width := len(inv)
	if width == 0 {
		width = a.rotaryDim() / 2
	}
	out := make([]float32, width)
	if a.isNoPELayer(i) {
		return out // all-zero: identity rotation, given mscale==1 on every NoPE layer today
	}
	for j, v := range inv {
		out[j] = float32(v)
	}
	return out
}

// AttnScale is the attention softmax q·k multiplier the resident backends (CUDA/Metal/WebGPU) apply: the
// registry-RESOLVED arch.AttnScale, which is 1/√headDim for the common case but deliberately not for several
// archs (Gemma-3 uses query_pre_attn_scalar^-0.5, == headDim for 1B/4B/12B but hidden/heads for 27B; Granite uses
// attention_multiplier; MLA uses the qk head-dim). Never hardcode 1/√headDim here: the small parity fixtures
// (scalar == headDim) cannot see the difference. Falls back to 1/√headDim only if an arch left AttnScale unset.
func (m *Model) AttnScale() float32 {
	if s := m.w.arch.AttnScale; s != 0 {
		return float32(s)
	}
	return float32(1.0 / math.Sqrt(float64(m.w.arch.HeadDim)))
}

// Dims returns the model shape from the resolved Architecture — authoritative,
// unlike Config() (Cfg), which a GGUF-loaded or .giw model may leave partly
// zero. The forward pass reads these.
func (m *Model) Dims() (hidden, nLayers, nH, nKV, hd, inter, vocab int) {
	return m.BackendView().Dims()
}

// NormEps is the RMSNorm epsilon (arch-backed).
func (m *Model) NormEps() float32 {
	if e := float32(m.w.arch.NormEps); e != 0 {
		return e
	}
	return 1e-6
}

// withResidency builds the GPU resident decoder if the backend supports it and
// the arch is eligible, then returns m. A no-op for the CPU backend / ineligible
// archs (m.resident stays nil → staged/CPU path). Called at every model-load site.
// attnSoftcapOffForTest removes the attention-score softcap from every model loaded while it is set, on the CPU and on
// a resident alike, before the resident is built: OP1's real Gemma 2 gate measures its own baseline with it
// ( G2-R's arm B). Set only through SetAttnSoftcapOffForTest
// (goinfer_testhooks).
var attnSoftcapOffForTest atomic.Bool

func (m *Model) withResidency() *Model {
	defer m.releaseKQuantFile() // the q4k load's GGUF mapping is read by this build only (KQuantReader)
	if attnSoftcapOffForTest.Load() {
		m.w.arch.AttnLogitSoftcap = 0
		m.prog = buildProgram(m.w.arch, m.w.hasExperts) // the CPU reads the softcap from the program
	}
	if m.planResidency {
		return m // FitQuery's open: plannedDecodePath asks the backend instead
	}
	if applyNormULPNoiseDiag(m.w); m.knobs.get(knobNoResidency) != "" { // normnoise.go diagnostic first: it must precede any resident upload of the norms
		return m // force the per-matmul staged path (decision-matrix measurement)
	}
	rb, why := m.residencyGate()
	if why != "" {
		m.resDecline = why
		return m
	}
	// H2: the backend's startup self-test, once per backend and quant per process, BEFORE the build, so the device memory its
	// fixture builds take (the local-memory pool their kernels' first launches reserve) is released or already held when the
	// build's plan and fit check read free memory, never taken after them. A mismatch skips the build and continues on the CPU path with the reason on the decode path, instead of
	// producing wrong numbers silently.
	if r, ran := gpuProbeBeforeResident(m); ran && r.Status == SelfTestDeclined {
		m.resDecline = describeProbe(r)
		logs.Infof("[resident] %s — continuing on the CPU/staged path", m.resDecline)
		return m
	}
	rf, ok, err := rb.BuildResident(m)
	if errors.Is(err, ErrLoadAborted) {
		// The swap tripwire stopped the build: not a decline (which would continue on the CPU and keep loading), but the load's error.
		m.loadAbortErr = err
		return m
	}
	if err != nil || !ok {
		// Unconditional, for the same reason as the cuda side: a decline moves the entire forward to
		// CPU, and the runtime's contract is to say so rather than quietly get slower.
		logs.Warnf("[resident] BuildResident declined (ok=%v err=%v) — continuing on the CPU/staged path", ok, err)
		// Falls back for correctness (a decline must never be fatal mid-load) but keeps the REASON: on a GPU backend this
		// is the whole forward moving to CPU.
		m.resDecline = buildDeclineReason(err)
		return m
	}
	if why := autoMetalKeepsPrecision(m.backendAuto, m.be.Name(), m.Quant(), rf); why != "" {
		_ = rf.Close()
		m.resDecline = why
		return m
	}
	m.resident = rf
	if kv := m.resKV(); kv.KVDemoterAvailable() && m.kvSnaps != nil && m.kvDemote && !m.kvShareOff {
		kv.SetKVHostBudget(m.kvSnaps)
	}
	// BuildResident's host-side work (the CUDA backend packs every layer's weights on the Go heap before uploading,
	// and stages the expert stacks in pinned memory) is garbage the moment it returns, but Go hands freed heap back
	// to the OS only when its background scavenger gets round to it. Release it now, once, at load time: a resident
	// model's whole point is that the host copy is no longer needed.
	releaseHostMemory()
	return m
}

// withResidencyOrAbort is withResidency for Load: a resident build that the swap tripwire aborted (Options.LoadAbort closed during it)
// closes the model and returns the build's error, which wraps ErrLoadAborted, instead of continuing on the CPU path.
func (m *Model) withResidencyOrAbort() (*Model, error) {
	m = m.withResidency()
	if err := m.loadAbortErr; err != nil {
		_ = m.Close()
		return nil, err
	}
	if err := m.kvDeliveredOrRefuse(); err != nil {
		_ = m.Close()
		return nil, err
	}
	if err := m.backendDeclinedErr(); err != nil {
		_ = m.Close()
		return nil, err
	}
	return m, nil
}

// backendDeclinedErr is Options.RequireBackend's refusal: an ErrBackendDeclined error naming the backend and the decline when
// the load asked for a backend other than the CPU and built no resident runner on it, else nil.
func (m *Model) backendDeclinedErr() error {
	be := m.kvOpts.Backend
	if !m.kvOpts.RequireBackend || be == "" || be == "cpu" || m.ResidentActive() {
		return nil
	}
	reason := m.ResidentDecline()
	if reason == "" {
		reason = "no reason recorded"
	}
	return errorOfKind(ErrBackendDeclined, "decoder: backend %q built no resident runner and RequireBackend refuses the CPU fallback: %s", be, reason)
}

// LoadAbort is Options.LoadAbort as the load that built this model passed it: a backend's resident build polls it between layers and
// between upload blocks and returns an error wrapping ErrLoadAborted once it is closed. Nil (never fires) when the load was not guarded.
// It is meaningful during BuildResident only; nothing reads it after Load returns.
func (m *Model) LoadAbort() <-chan struct{} { return m.loadAbort }

// residencyGate is withResidency's checks before it asks the backend to build: the backend's residency, the quant's resident
// kernels, -backend auto's precision rule and admission, all Admit's (admit.go) over the model's facts. why is the decline they record, "" with the backend to build on.
func (m *Model) residencyGate() (rb ResidencyBackend, why string) {
	if why := Admit(m.admitFacts()).Why; why != "" {
		return nil, why
	}
	return m.be.(ResidencyBackend), ""
}

// buildDeclineReason is the ResidentDecline a failed or declined BuildResident leaves: a decline's own reason, or the failure
// (a driver or device error: `--backend cuda` on a box with no usable GPU lands here, since the device is touched only at
// build time).
func buildDeclineReason(err error) string {
	var d *ResidentDeclineError
	switch {
	case errors.As(err, &d):
		return d.Reason
	case err != nil:
		return "backend failed to build a resident path: " + err.Error()
	}
	return "backend declined to build a resident path (no reason given)"
}

// plannedDecodePath is the decode path a load of m would report, decided by the checks withResidency and the backend's
// BuildResident make before the build allocates (ResidencyPricer), on a model opened with Options.planResidency. "" with the
// reason when only the build can tell. A decline found inside the build itself, or by the startup self-test after it, is not
// foreseen.
func (m *Model) plannedDecodePath() (path, unknown string, ctx int) {
	if m.knobs.get(knobNoResidency) != "" {
		return m.DecodePath(), "", 0
	}
	rb, why := m.residencyGate()
	if why == "" {
		p, ok := rb.(ResidencyPricer)
		if !ok {
			return "", m.be.Name() + " decides its residency only by building it", 0
		}
		price, err := p.PriceResident(m)
		if err == nil {
			if price.Unknown != "" {
				return "", price.Unknown, price.Ctx
			}
			label := price.Quant
			if label == "" {
				label = residentQuantLabel(m.be.Name(), m.Quant())
			}
			return fmt.Sprintf("%s-resident (%s)", m.be.Name(), label), "", price.Ctx
		}
		why = buildDeclineReason(err)
	}
	saved := m.resDecline
	m.resDecline = why
	defer func() { m.resDecline = saved }()
	return m.DecodePath(), "", 0
}

// releaseHostMemory returns freed heap to the OS. A variable so a test can observe that a successful resident
// build calls it (and a declined one does not); production is always debug.FreeOSMemory.
var releaseHostMemory = debug.FreeOSMemory

// --- Granite-4.0-H resident SSM bridge (P5b: resident Mamba decode, docs/ssm-residency-scope.md) ---
// These expose the hybrid's per-layer mixer-kind + Mamba-2 mixer weights + scalar multipliers
// to the gpu resident builder. Additive accessors; the eligibility flip (P6) is separate and
// still gated, so production routing is unchanged until then.

// ResidentActive reports whether the GPU full-residency decode path is built and
// will run for a plain stateless Generate (webgpu backend + eligible arch).
func (m *Model) ResidentActive() bool { return m.resident != nil }

// ResidentContextCap returns the resident backend's fixed KV/attention capacity in positions, or 0
// when the model is not GPU-resident or its backend exposes no cap. A stateless prompt longer than
// this cannot be prefilled on the resident path, so a caller can reject it up front rather than
// fail mid-prefill (audit R-10). clampToResidentCap applies the same cap to maxTokens.
func (m *Model) ResidentContextCap() int { return residentCapsOf(m.resident).ContextCap }

// clampToResidentCap is maxTokens cut so that a turn starting at gpuPos ends at the resident's context cap, and whether it was cut.
// A resident without a cap, or a cap of 0, cuts nothing; the result may be 0 when the prompt fills the cap.
func (m *Model) clampToResidentCap(gpuPos, maxTokens int) (int, bool) {
	if c := m.ResidentContextCap(); c > 0 && gpuPos+maxTokens > c {
		return c - gpuPos, true
	}
	return maxTokens, false
}

// ResidentContextNote is the active resident's ContextNote ("fit: 2 slots, 1.9 GB KV"), or "" when there is no resident or it gives none.
func (m *Model) ResidentContextNote() string { return residentCapsOf(m.resident).ContextNote }

// ResidentContextKVBytes is the KV one resident slot holds at ctx positions on the active resident's backend, at the precision
// that backend allocates: the figure FitBudgetSummary reports at the resolved context. 0 when no resident is active.
func (m *Model) ResidentContextKVBytes(ctx int) int64 {
	if m.resident == nil || m.be == nil {
		return 0
	}
	f16, i8 := m.residentKVFlags(m.be.Name())
	return m.ResidentKVBytes(m.be.Name(), ctx, f16, i8)
}

// residentKVFlags are the kvF16 and kvI8 ResidentKVBytes is handed for this model's own load on backend: on CUDA the precision its KV policy
// delivers (int8 only where ResolveKV gives it), elsewhere the load's f16 decision and the CPU cache's int8 request, as before.
func (m *Model) residentKVFlags(backend string) (kvF16, kvI8 bool) {
	if backend == "cuda" {
		return m.kvF16, m.KVDeliveredFor("cuda") == kvPrecI8
	}
	return m.kvF16, m.kvI8
}

// ResidentKVPrecision is the KV-cache precision the resident runner actually allocates — "f32", "f16" or
// "i8" — or "" when no resident runner is active. It reports what runs, not what was requested: Metal
// ships f16 (or int8) KV only, its f32 KV kernels being compiled out (metal/model.go's r.kvF32), so a
// Metal resident model runs f16 KV even when nothing asked for it (
// found serve's banner printing "KV f32" there).
func (m *Model) ResidentKVPrecision() string {
	if m.resident == nil || m.be == nil || m.w == nil {
		return ""
	}
	got, _ := ResolveKV(m.be.Name(), m.w.arch, m.kvOpts)
	return got
}

// HeadTable is the precision of the token-embedding / LM-head table this model loaded with: "int4" (--embed-int4), "int8" (the pin), "f32" or "q4k"; "" when the CPU copy is not held.
// The head is the matrix every decoded token streams in full, so serve's banner names it: the same checkpoint is a different model under a different table (docs/quantization.md).
func (m *Model) HeadTable() string {
	if m.w == nil {
		return ""
	}
	if m.w.arch != nil && m.w.arch.TiedLMHead {
		return m.w.Embed.Kind()
	}
	return m.w.LMHead.Kind()
}

// DecodePath names the decode path this model actually resolved to — "<backend>-resident" when the
// full-residency runner built, "<backend>-staged" when the backend runs per-matmul under the CPU
// forward, "cpu" otherwise — with the resident weight quant in parens. A staged GPU path also names
// WHY residency declined, because that decline is the larger of the two silent regressions: the
// whole forward is on CPU, not just the prefill. Display/diagnostic only.
func (m *Model) DecodePath() string {
	be := "cpu"
	if m.be != nil {
		be = m.be.Name()
	}
	switch {
	case m.resident != nil:
		label := residentQuantLabel(be, m.Quant())
		if q := residentCapsOf(m.resident).Quant; q != "" {
			label = q
		}
		return fmt.Sprintf("%s-resident (%s)", be, label)
	case be == "cpu":
		s := fmt.Sprintf("cpu (%s)", m.Quant())
		// The repacked-only int4 policy (wantsCanonicalInt4, aikit audit M-22) only ever
		// activates when Options.Backend is the literal "cpu" promise — appended here rather
		// than in every DecodePath case, since a GPU-backend case always keeps canonical(+row4)
		// by construction and has nothing new to say.
		if layout := m.int4LayoutSummary(); layout != "" {
			s += " [" + layout + "]"
		}
		return s
	case !isWebGPUBackend(be):
		// The cuda and metal staged paths never reach the device: both backends' MatmulBT are bare CPU calls and neither
		// implements QuantBackend, so int4, int8 and f32 all run the same CPU kernels. There is no real "cuda-staged" or
		// "metal-staged" path, so report it the way BackendSummary reports a backend that failed to build: requested vs.
		// actually executing (docs/measurements/cold-user-2026-09-06-nobara-pc.md, R9).
		req := be
		if m.backendAuto { // the user asked for auto, which chose this backend (R17)
			req = "auto (" + be + ")"
		}
		return fmt.Sprintf("cpu (%s) — %s", m.Quant(), BackendSummary(req, "cpu", declinedToCPUReason(be, m.resDecline)))
	case m.resDecline != "":
		return fmt.Sprintf("webgpu-staged (%s)%s — %s", m.Quant(), stagedDeviceNote(m.Quant()), m.resDecline)
	default:
		return fmt.Sprintf("webgpu-staged (%s)%s", m.Quant(), stagedDeviceNote(m.Quant()))
	}
}

// residentQuantLabel is DecodePath's resident-quant string, split out so it is testable without a live Model
// (decoder/staged_device_note_test.go). Metal has no int8 GEMV kernel of its own: unless the resident reports
// otherwise (ResidentCaps.Quant), an Int8()-kind weight is re-quantized to W4A8 at resident-build time, under
// `--quant int8` and `int8int8` and for `int4mix`'s int8-kind tensors, so the label says "→int4". f32 never produces
// a resident build there. Every other (backend, quant) pair echoes the requested quant.
func residentQuantLabel(backend, quant string) string {
	if backend == "metal" && (quant == "int8" || quant == "int8int8" || quant == "int4mix") {
		return quant + "→int4, no Metal int8 GEMV kernel"
	}
	return quant
}

// autoMetalPrecision is why a model stays on the CPU when -backend auto chose metal, "" when it may go resident. A
// backend the user did not name must not change the precision they loaded at (such as the int8int8 bundle a
// model-included goinfer-chat carries; R17). Metal runs int4 resident, dense
// int8int8 natively, and q4k on its q4k lane, which keeps the file's K-quant
// blocks; it re-quantizes int8, int4mix and the int8int8 models native int8 does
// not cover (MoE, DeltaNet) to int4, and takes no f32. So int8int8 may try the resident here and
// autoMetalKeepsPrecision checks afterwards that it ran natively; the others stay on the CPU. A named -backend
// metal still runs them re-quantized.
func autoMetalPrecision(auto bool, backend, quant string) string {
	if !auto || backend != "metal" || quant == "int4" || quant == "int8int8" || quant == "q4k" {
		return ""
	}
	switch quant {
	case "int8", "int4mix":
		return "-backend auto keeps this " + quant + " model on the CPU: metal runs only int4 resident and would re-quantize it (-backend metal does that)"
	case "", "native": // Quant() of an f32 load
		quant = "f32"
	}
	return "-backend auto keeps this " + quant + " model on the CPU: metal runs only int4 resident"
}

// autoMetalKeepsPrecision is why a resident -backend auto built on metal must be dropped for the CPU, "" when it keeps
// it: an int8int8 model the resident did not run natively (ResidentCaps.Quant is not int8int8) was re-quantized
// to int4, a precision the user did not ask for.
func autoMetalKeepsPrecision(auto bool, backend, quant string, rf ResidentForward) string {
	if !auto || backend != "metal" || quant != "int8int8" {
		return ""
	}
	if residentCapsOf(rf).Quant == quant {
		return ""
	}
	return "-backend auto keeps this int8int8 model on the CPU: metal would re-quantize it to int4 (its int8 kernels cover dense models only; -backend metal does that)"
}

// declinedToCPUReason builds the reason BackendSummary needs when cuda or metal has no staged
// path at all for a resident-ineligible architecture — split out so it is testable without a
// live Model (see decoder/staged_device_note_test.go).
func declinedToCPUReason(backend, resDecline string) string {
	reason := backend + " has no staged decode path"
	if resDecline != "" {
		reason = resDecline + ", and " + reason
	}
	return reason
}

// stagedDeviceNote names when webgpu's own staged path (the only one that has one, see DecodePath) still reaches no
// device for a given quant, so the path name alone does not overclaim device use. Its Backend.MatmulBT and
// QuantBackend.MatmulW8A8 (gpu/backend.go) do real device work, unlike cuda/metal; its int4 dispatch,
// QuantBackend4.MatmulW4A8, covers M=1 only, so decode runs int4 on the device and a multi-token prefill declines to
// the host W4A8 kernel (no int4 GEMM). int4mix splits the same way (int4 FFN, int8 attention). f32/int8/int8int8
// reach the device and get no note. TestStagedDeviceNote pins this text.
func stagedDeviceNote(quant string) string {
	switch quant {
	case "int4":
		return " [int4 decode (one token) runs on webgpu's int4 kernel; prefill (several tokens) has no multi-token int4 kernel here and runs on the host]"
	case "int4mix":
		return " [decode (one token) runs on webgpu, int4 FFN included; in prefill (several tokens) the int4 FFN tensors run on the host — no multi-token int4 kernel here — and only attention (int8) reaches webgpu]"
	}
	return ""
}

// ResidentDecline returns why the resident decode path is absent, or "" when it is active (or was
// never requested). withResidency falls back silently by design; this keeps the reason so serve can
// print it and -require-backend can refuse to start on it.
func (m *Model) ResidentDecline() string { return m.resDecline }

// PrefillPath reports how a long prompt will actually be ingested (one batched pass, or the sequential per-token
// loop) plus the reason when it is sequential. It is the load-time answer to a decline that generateInto otherwise
// takes silently: the batched prefill is an OPTIONAL backend capability that falls back per call, so a model can
// lose most of its TTFT without emitting anything. serve prints this at startup and -require-backend refuses to
// start on a decline.
//
// Both halves are covered: the resident path asks the backend (PrefillPathReporter), and the staged/CPU path
// reports canBatchN, which excludes the families with their own sequential forward (Gemma 4, qwen3_5, Mamba-2
// hybrids, MLA, Llama 4, gpt-oss).
func (m *Model) PrefillPath() (batched bool, reason string) {
	if m.resident == nil {
		if !m.canBatchN(2) {
			return false, "sequential — this arch has its own per-token forward (no batched CPU prefill)"
		}
		// "batched" names the SHAPE (weights stream once per prompt at M=K instead of once per token), not a throughput
		// claim: the CPU prefill path is single-threaded and runs at about the model's decode rate. Say what it is and
		// what it is not.
		return true, "batched shape (CPU forwardLayersN, one weight stream for the whole prompt) — " +
			"describes weight streaming, not throughput; this path is single-threaded"
	}
	if m.knobs.get(knobBatchedPrefill) == "0" {
		return false, "sequential — GOINFER_BATCHED_PREFILL=0 forces the per-token loop"
	}
	if !canStep(m.resident, stepBatch{Pass: passRows, Emb: stepProbe}) {
		return false, "sequential — this backend has no batched prefill (per-token resident forward)"
	}
	if batched, reason, ok := m.resPrefill().PrefillPathInfo(); ok {
		return batched, reason
	}
	return true, "batched (backend reports no load-time declines; a per-prompt fallback is still possible)"
}

// embedResident returns the input embedding [hidden] for token id — the CPU half of the
// residency forward, including any embedding scale the arch applies before layer 0. Always
// allocates fresh: safe for callers that collect several embeddings into a slice that outlives
// the call (batch prefill assembly), where a shared buffer would corrupt every entry but the
// last. The per-token decode loop wants embedResidentInto instead (P-08, audit-2026-09-10).
func (m *Model) embedResident(id int) []float32 { return m.embedResidentInto(id, nil) }

// embedIsTableRow reports whether embedResidentInto is exactly m.w.Embed.Row: no embedding multiplier of either kind is
// applied. A resident that gathers the next token's embedding itself (GreedyChainAvailable) is exact only then.
func (m *Model) embedIsTableRow() bool {
	if sc := m.w.arch.EmbedScale; sc != 0 && sc != 1 {
		return false
	}
	if g := m.w.arch.granite; g != nil && g.EmbMul != 0 && g.EmbMul != 1 {
		return false
	}
	return true
}

// embedResidentInto is embedResident with a caller-owned destination: dst is reused when its
// capacity already fits (grown once otherwise, the same discipline as decodeScratch's buffers),
// eliminating the make() embedResident pays every call. Safe ONLY where the caller consumes the
// result synchronously before requesting the next one — the resident decode loop's actual shape
// (embed → Forward → discard, one token at a time) — never where more than one live embedding
// must exist at once.
//
// A Gemma 4 E-model's row is [h ‖ PLE inputs], EmbedLen() long: the per-layer inputs need the token id,
// which the resident interface does not carry, so the host computes them here with the CPU's own helper
// (gemma4PLEInputs) and the backend reads them from the row's tail.
func (m *Model) embedResidentInto(id int, dst []float32) []float32 {
	return m.embedResidentScratch(id, dst, nil)
}

// embedResidentScratch is embedResidentInto with a Gemma 4 E-model's per-layer-input scratch (gemma4PLEInputsScratch) supplied,
// so embedding row after row allocates nothing per row.
func (m *Model) embedResidentScratch(id int, dst, pleScratch []float32) []float32 {
	hidden := m.w.arch.HiddenDim
	n := m.BackendView().EmbedLen()
	var row []float32
	if cap(dst) >= n {
		row = dst[:n]
	} else {
		row = make([]float32, n)
	}
	h := row[:hidden]
	m.w.Embed.Row(id, h)
	// Gemma's √hidden embedding multiplier (Architecture.EmbedScale), mirroring runLayers.
	// Applied here rather than on the GPU so the resident stream starts where the CPU's does.
	if sc := m.w.arch.EmbedScale; sc != 0 && sc != 1 {
		f := float32(sc)
		for i := range h {
			h[i] *= f
		}
	}
	// Granite embedding_multiplier — the resident residual stream starts at emb·EmbMul
	// (P5b; the only Granite scalar applied outside the resident runner / weight folds).
	if g := m.w.arch.granite; g != nil && g.EmbMul != 0 && g.EmbMul != 1 {
		for i := range h {
			h[i] *= g.EmbMul
		}
	}
	if n > hidden {
		tail := row[hidden:]
		m.gemma4PLEInputsScratch(h, id, tail, pleScratch)
		if gemma4PLEDropTokenForTest {
			// G2 defect (2), resident row only: take the token-identity term back out, leaving context_aware/√2.
			tok := make([]float32, len(tail))
			m.w.PerLayerTokenEmbed.Row(id, tok)
			sc := float32(math.Sqrt(float64(m.w.arch.gemma4.HiddenSizePerLayerInput)) / math.Sqrt2)
			for i := range tail {
				tail[i] -= tok[i] * sc
			}
		}
	}
	return row
}

// gemma4PLEDropTokenForTest is S1's G2 planted defect (2): the
// token-identity term missing from the resident row's PLE inputs, which G1 (resident against the CPU) must catch.
// Set only through SetGemma4PLEDropTokenForTest (goinfer_testhooks).
var gemma4PLEDropTokenForTest bool

// ResidentMixer is the sequence-mixing operation of one layer in a resident's layer walk.
type ResidentMixer uint8

const (
	MixerAttention ResidentMixer = iota // softmax attention over a KV cache
	MixerDeltaNet                       // the Gated-DeltaNet recurrence (Qwen3.5/3.6, Qwen3-Next, Olmo Hybrid)
	MixerMamba                          // the Mamba-2 recurrence (Nemotron-H, Granite-4.0-H)
	MixerNone                           // no mixer: a feed-forward-only block (Nemotron-H's MLP and MoE blocks)
	MixerMLA                            // multi-head latent attention over a latent cache (DeepSeek-V2/V3, Kimi K2)
)

// Recurrent reports whether the mixer keeps per-sequence recurrent state instead of a KV cache.
func (x ResidentMixer) Recurrent() bool { return x == MixerDeltaNet || x == MixerMamba }

// BackendReport renders the load banner's backend field. It NEVER names a backend that is not executing: when the
// requested one was unavailable it reads "requested metal → running on cpu: <reason>" instead of "metal" (see
// backendNames). Appends int4LayoutSummary when non-empty (the repacked-only int4 policy, wantsCanonicalInt4 in
// decoder/weightmat.go): the visible counterpart to a load-time decision that is otherwise invisible until a
// resident or staged build declines for a reason that traces back to it.
func (m *Model) BackendReport() string {
	s := BackendSummary(m.reqBackend, m.effBackend, m.beDecline)
	if layout := m.int4LayoutSummary(); layout != "" {
		s += " [" + layout + "]"
	}
	return s
}

// int4LayoutSummary reports which int4 layout each tensor CLASS resolved to (aikit's WeightMat.Int4Layout(), one
// representative per class, since every tensor in a class shares one needCanonical decision): "int4 layout: row4-only
// (embed, qkv, gate/up); canonical (down)". Diagnosable from this line instead of a tok/s comparison when a resident or
// staged build declines for a reason that traces back to wantsCanonicalInt4 (M-22). "" when the model has no layers or
// nothing in it is int4-resident.
func (m *Model) int4LayoutSummary() string {
	if m.w == nil || len(m.w.Layers) == 0 {
		return ""
	}
	type class struct {
		label string
		wm    linalg.WeightMat
	}
	classes := []class{{"embed", m.w.Embed}}
	if m.w.LMHead.Rows() > 0 {
		classes = append(classes, class{"lm_head", m.w.LMHead})
	}
	l0 := m.w.Layers[0]
	classes = append(classes,
		class{"qkv", l0.QProj},
		class{"gate/up", l0.GateProj},
		class{"down", l0.DownProj},
	)
	byLayout := map[string][]string{}
	for _, c := range classes {
		if !c.wm.IsInt4() {
			continue
		}
		byLayout[c.wm.Int4Layout()] = append(byLayout[c.wm.Int4Layout()], c.label)
	}
	if len(byLayout) == 0 {
		return ""
	}
	layouts := make([]string, 0, len(byLayout))
	for l := range byLayout {
		layouts = append(layouts, l)
	}
	sort.Strings(layouts)
	parts := make([]string, len(layouts))
	for i, l := range layouts {
		name := l
		if l == "row4" || l == "splithalf" {
			name += "-only" // distinguish from the canonical+row4/canonical+splithalf "both" policy
		}
		sort.Strings(byLayout[l])
		parts[i] = fmt.Sprintf("%s (%s)", name, strings.Join(byLayout[l], ", "))
	}
	return "int4 layout: " + strings.Join(parts, "; ")
}

// EffectiveBackend is the backend actually executing ("cpu" when a request fell back).
func (m *Model) EffectiveBackend() string { return m.effBackend }

// RequestedBackend is what the caller asked for, normalised ("" → "cpu").
func (m *Model) RequestedBackend() string { return m.reqBackend }
