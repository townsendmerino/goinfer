//go:build goinfer_testhooks

// Test-only hooks, compiled only under -tags goinfer_testhooks so they are not part of the public API (RELEASING.md).
// Hand-maintained: nothing generates this file (there is no //go:generate directive); edit it directly like any other file.
// Mirrors gpu/testhooks.go and cuda/testhooks.go.

package decoder

import (
	"context"
	"fmt"
	"os"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// ForwardSubCaptureLogitsForTest is ForwardSubCapture plus the token's logits from the same forward: the gpu package's
// per-layer resident-vs-CPU parity gate needs both, and running the token twice would append it to the KV cache twice. Same
// arch guard and byte-identical-output contract as ForwardSubCapture. See docs/completed/task-webgpu-nogqa-decode-bug.md.
func (m *Model) ForwardSubCaptureLogitsForTest(id int, cache *KVCache) (logits []float32, attn, mlp, ctx, mlpPre [][]float32, err error) {
	a := m.w.arch
	if _, own := a.ownForward(); own {
		return nil, nil, nil, nil, nil, fmt.Errorf("decoder.ForwardSubCaptureLogitsForTest: not wired for arch %q (own runLayers)", a.Name)
	}
	nL := a.NumLayers
	cache.subCapture = true
	cache.subAttn = make([][]float32, nL)
	cache.subMLP = make([][]float32, nL)
	cache.subMLPpre = make([][]float32, nL)
	cache.subCtx = make([][]float32, nL)
	defer func() {
		cache.subCapture = false
		cache.subAttn, cache.subMLP, cache.subMLPpre, cache.subCtx = nil, nil, nil, nil
	}()
	lg, ferr := m.forward(id, cache)
	if ferr != nil {
		return nil, nil, nil, nil, nil, ferr
	}
	return append([]float32(nil), lg...), cache.subAttn, cache.subMLP, cache.subCtx, cache.subMLPpre, nil
}

// FinalNormForTest applies the model's final normalization to a copy of h: the CPU reference for the seam between the trunk
// and the LM head in a resident bisect. It does not project to logits; that is the head.
func (m *Model) FinalNormForTest(h []float32) []float32 {
	out := append([]float32(nil), h...)
	normalize(m.w.arch, out, m.w.FinalNorm, m.w.FinalNormBias, m.w.arch.HiddenDim)
	return out
}

// EmbedResidentForTest is the resident input embedding (including Granite's EmbMul): a CPU-side reference seam for the gpu-package
// resident parity gates.
func (m *Model) EmbedResidentForTest(id int) []float32 { return m.embedResident(id) }

// Gemma4ResidentMediaRowForTest is gemma4ResidentMediaRow: the resident row GenerateGemma4VL's E-model resident prefill
// builds for an image or audio position.
func (m *Model) Gemma4ResidentMediaRowForTest(feature []float32) []float32 {
	return m.gemma4ResidentMediaRow(feature)
}

// ForwardForTest is the CPU per-token logits: a reference seam for the gpu-package resident parity gates, which cannot live
// in package decoder (import cycle via gpu).
func (m *Model) ForwardForTest(id int, cache *KVCache) ([]float32, error) {
	return m.forward(id, cache)
}

// ResidentForwardForTest exposes the resident forward (nil when not resident) for the GPU ForwardN-vs-Forward parity gate.
// Production code routes through Generate / GenerateSpeculative, not this.
func (m *Model) ResidentForwardForTest() ResidentForward { return m.resident }

// SplitQGateForTest and QGateContextForTest are the CPU's attn_output_gate split and output gate
// (decoder/forward_qwen35.go), so the resident kernels that mirror them can be checked against the code the CPU runs.
func SplitQGateForTest(qg []float32, nH, hd int) (q, gate []float32) {
	q, gate = make([]float32, nH*hd), make([]float32, nH*hd)
	splitQGate(qg, q, gate, nH, hd)
	return q, gate
}

// QGateContextForTest applies the output gate to ctx in place (see SplitQGateForTest).
func QGateContextForTest(ctx, gate []float32) { qGateContext(ctx, gate) }

// AutoPinResidentContextForTest makes the model's resident context request ctx without the caller having chosen it: the
// state the load-time fit guard's auto-pin leaves, which only a load under memory pressure produces.
func (m *Model) AutoPinResidentContextForTest(ctx int) { m.resCtxReq, m.resCtxPinned = ctx, false }

// ResidentImagePrefillForTest wraps residentImagePrefill, the primitive GenerateVL's resident image-prefill fast path calls,
// so a real-checkpoint gate (cuda/) can compare its logits with PrefillLogitsVLForTest's CPU reference at matched precision
// on the same model instance and the same real image.
func (m *Model) ResidentImagePrefillForTest(ctx context.Context, rip ResidentImagePrefill, ids []int, imageEmbeds []float32, imgPos, imgLen int) ([]float32, int, error) {
	return m.residentImagePrefill(ctx, rip, ids, imageEmbeds, []ImageSpan{{Pos: imgPos, Len: imgLen}})
}

// ResidentImagePrefillSpansForTest is ResidentImagePrefillForTest for several images.
func (m *Model) ResidentImagePrefillSpansForTest(ctx context.Context, rip ResidentImagePrefill, ids []int, imageEmbeds []float32, spans []ImageSpan) ([]float32, int, error) {
	return m.residentImagePrefill(ctx, rip, ids, imageEmbeds, spans)
}

// ResidentMRoPEPrefillForTest wraps residentMRoPEPrefillDeep (no DeepStack sets), the primitive GenerateQwenVL's resident
// m-RoPE prefill fast path calls, mirroring ResidentImagePrefillForTest for a real-checkpoint gate comparing its logits
// against PrefillLogitsQwenVLForTest.
func (m *Model) ResidentMRoPEPrefillForTest(ctx context.Context, rmp ResidentMRoPEPrefill, ids []int, imageFeats []float32, imgPos, imgLen int, mropePos [][3]int) ([]float32, int, error) {
	return m.residentMRoPEPrefillDeep(ctx, rmp, ids, imageFeats, []ImageSpan{{Pos: imgPos, Len: imgLen}}, mropePos, nil)
}

// ResidentMRoPEDeepstackPrefillForTest wraps residentMRoPEPrefillDeep with Qwen3-VL's DeepStack sets, for the Metal gate
// that compares it with the CPU prefill.
func (m *Model) ResidentMRoPEDeepstackPrefillForTest(ctx context.Context, rmp ResidentMRoPEPrefill, ids []int, imageFeats []float32, imgPos, imgLen int, mropePos [][3]int, deep [][]float32) ([]float32, int, error) {
	return m.residentMRoPEPrefillDeep(ctx, rmp, ids, imageFeats, []ImageSpan{{Pos: imgPos, Len: imgLen}}, mropePos, deep)
}

// SetDeepstackForTest gives c the DeepStack sets its next prefill adds after each decoder layer (GenerateQwenVLDeepstack's
// own setting), so a cross-package gate can run the CPU prefill with them; nil clears them.
func (c *KVCache) SetDeepstackForTest(start, n int, rows [][]float32) {
	if rows == nil {
		c.deepstack = nil
		return
	}
	c.deepstack = &deepstackRows{spans: []ImageSpan{{Pos: start, Len: n}}, rows: rows}
}

// ResidentUploadPrefillForTest wraps residentUploadPrefill, the generic per-layer KVCache.LayerKV -> ResidentForward.UploadKV
// bridge GenerateGemma4VL's resident branch uses after a CPU bidirectional prefill, so a real-hardware gate (cuda/) can upload
// a CPU-computed cache and continue decode on the resident backend, on the same model instance whose CPU decode it is
// compared against.
func (m *Model) ResidentUploadPrefillForTest(cache *KVCache) error {
	return m.residentUploadPrefill(cache)
}

// ApplyMRoPEForTest wraps applyMRoPE, the CPU m-RoPE reference a resident kernel's own rotation must match bit-for-bit.
// Production always reaches applyMRoPE through ropeAt, never directly. interleaved selects Qwen3-VL's per-index component
// layout (mropeComponentInterleaved) over Qwen2.5-VL's contiguous-block one (mropeComponent); pass false for Qwen2.5-VL.
func ApplyMRoPEForTest(vec []float32, heads, headDim int, pos [3]int, section []int, invFreq []float64, scale float64, interleaved bool) {
	applyMRoPE(vec, heads, headDim, pos, section, invFreq, scale, interleaved)
}

// Gemma4MoEExpertForTest computes one gemma4 MoE expert's output on a caller-supplied input xe ([hidden]): edown = Down ·
// (geluTanh(gate)·up) with gate‖up = GateUp·xe, returned with the expert's fused gate‖up and down weight matrices. The cuda
// single-expert gate packs those weights and runs the resident chain (gemv_w4a8_moe → glu_quant act=GELU_TANH → down) on the
// same xe: the gelu-tanh epilogue × indexed-expert GEMV is a combination no other family ships (Gemma-3 is dense, Mixtral and
// GLM are indexed but SiLU), so it is verified directly, not argued by composition. ok=false for a non-gemma4 model, a dense
// layer, an e out of range, or len(xe) != hidden.
func (m *Model) Gemma4MoEExpertForTest(layer, e int, xe []float32) (edown []float32, gateUp, down *linalg.WeightMat, moeInter, hidden int, ok bool) {
	if m.w.arch.gemma4 == nil || layer < 0 || layer >= len(m.w.Layers) {
		return nil, nil, nil, 0, 0, false
	}
	gm := m.w.Layers[layer].gemma4moe
	if gm == nil || e < 0 || e >= len(gm.expertsGateUp) || len(xe) != m.w.arch.HiddenDim {
		return nil, nil, nil, 0, 0, false
	}
	be := &cpuBackend{}
	gu := make([]float32, 2*gm.moeInter)
	matmul(be, &gm.expertsGateUp[e], xe, gu, 1)
	mid := make([]float32, gm.moeInter)
	for i := 0; i < gm.moeInter; i++ {
		mid[i] = geluTanh(gu[i]) * gu[gm.moeInter+i]
	}
	out := make([]float32, m.w.arch.HiddenDim)
	matmul(be, &gm.expertsDown[e], mid, out, 1)
	return out, &gm.expertsGateUp[e], &gm.expertsDown[e], gm.moeInter, m.w.arch.HiddenDim, true
}

// Gemma4MoERouterForTest exposes a gemma4 MoE layer's f32 router projection (row-major [nE, hidden]) and its selection bias
// (zeros: gemma4 has no router bias) for the cuda resident-router idx-equality test. That test replays captured router inputs
// (routerRnBuf) through the CUDA selection kernels and compares resident idx[] with the CPU idx[], which isolates a routing flip
// from any expert-GEMV numeric difference. ok=false for a non-gemma4 model, a dense (non-MoE) layer, an out-of-range layer, or a
// router not held as f32.
func (m *Model) Gemma4MoERouterForTest(layer int) (proj, bias []float32, nE, topK, hidden int, ok bool) {
	if m.w.arch.gemma4 == nil || layer < 0 || layer >= len(m.w.Layers) {
		return nil, nil, 0, 0, 0, false
	}
	gm := m.w.Layers[layer].gemma4moe
	if gm == nil {
		return nil, nil, 0, 0, 0, false
	}
	f, has := gm.routerProj.F32()
	if !has {
		return nil, nil, 0, 0, 0, false
	}
	return f, make([]float32, gm.nE), gm.nE, gm.topK, m.w.arch.HiddenDim, true
}

// gemma4HiddenBuf accumulates a copy of the residual stream after each layer (index 0 = post-embedding, from g4traceHidden's
// layer -1; index i+1 = after layer i) on every runLayersGemma4 call while capture is on; it backs the per-layer localization of
// a resident-vs-CPU gate. It lives here, not beside g4traceHidden, because only SetGemma4HiddenCaptureForTest appends to it
// (production sees only the nil-by-default g4traceHidden seam): off-tag it would be an unused package var (staticcheck U1000).
var gemma4HiddenBuf [][]float32

// Gemma4HiddenCaptureForTest returns the captured residual stream per layer, in call order:
// index 0 = post-embedding, index i+1 = the stream after layer i (pre-final-norm, matching what a
// resident runner's trunk-to-N-layers produces).
func Gemma4HiddenCaptureForTest() [][]float32 { return gemma4HiddenBuf }

// SetGemma4HiddenCaptureForTest toggles per-layer hidden capture, clearing the buffer when enabled.
// Exported so a resident test in another package (metal/cuda) can drive a CPU gemma4 forward and
// read the per-layer residual stream — it cannot touch the unexported g4traceHidden directly.
// Capture only a SINGLE token's forward (drive one ForwardForTest between on and off): the buffer
// accumulates across calls, so a multi-token pass concatenates every token's per-layer trace.
func SetGemma4HiddenCaptureForTest(on bool) {
	if on {
		gemma4HiddenBuf = nil
		g4traceHidden = func(_ int, h []float32) {
			gemma4HiddenBuf = append(gemma4HiddenBuf, append([]float32(nil), h...))
		}
	} else {
		g4traceHidden = nil
		gemma4HiddenBuf = nil
	}
}

// RouterMarginForTest returns the per-decision top-k boundary margin (the smallest selected expert's softmax prob minus the
// largest rejected expert's), in RouterCaptureForTest's order and index. It is the MoE robustness signal a noise-floor check
// reads: a fixture whose margin sits below the int4-vs-f32 routing perturbation can flip top-k under quant and cannot gate a
// resident router, however correct the port.
func RouterMarginForTest() []float32 { return routerMarginBuf }

// Gemma4MoECaptureForTest returns the CPU per-decision wts / x1 / x2 (same order as RouterCaptureForTest).
func Gemma4MoECaptureForTest() (wts, x1, x2 [][]float32) {
	return routerWtsBuf, routerX1Buf, routerX2Buf
}

// RouterCaptureForTest returns the captured per-decision selected experts and router inputs (same
// order/index). Sibling to SetRouterCaptureForTest for the cross-package (cuda) router-first gate.
func RouterCaptureForTest() (idx [][]int, rn [][]float32) { return routerCaptureBuf, routerRnBuf }

// SetRouterCaptureForTest toggles router capture and (when enabling) clears the buffers. Exported so
// the CUDA resident-router unit test (package cuda) can drive a CPU forward, capture idx/rn, and
// replay rn through the device kernels — it cannot touch these unexported vars directly.
func SetRouterCaptureForTest(on bool) {
	routerCapture = on
	if on {
		routerCaptureBuf, routerRnBuf, routerMarginBuf = nil, nil, nil
		routerWtsBuf, routerX1Buf, routerX2Buf = nil, nil, nil
	}
}

// LayerKVForTest is KVCache.LayerKV under the name the cross-package (cuda, metal) test call sites use: the cross-backend
// attention confirmer injects goinfer's exact K/V into another engine.
func (c *KVCache) LayerKVForTest(layer int) (k, v []float32, base int) { return c.LayerKV(layer) }

// SetSSMQ8CPU makes the CPU Mamba reference round-trip its projections through int8, so it carries the same quantization error as
// the resident W8A8 path (the confirmation seam GOINFER_SSM_Q8CPU also turns on).
func SetSSMQ8CPU(v bool) { ssmQ8CPU = v }

// SetSSMForceF32 toggles the CPU reference's f32 SSM accumulation (the gated-norm precision-localization seam). It and
// SetSSMQ8CPU are exports because gpu/ssm_kernel_control_test.go needs the staged webgpu backend and so lives in package gpu,
// which cannot reach decoder's unexported variables.
func SetSSMForceF32(v bool) { ssmForceF32 = v }

// SetMambaCapHook installs/clears the capture hook (the gpu package can't import decoder-internal
// state directly, so it drives this via the export, like SetSSMQ8CPU).
func SetMambaCapHook(f func(proj, gated []float32)) { mambaCapHook = f }

// DraftBlockCPUForTest runs a block drafter's trunk on the CPU backend, for a GPU backend's parity gate. The trunk's own
// DraftBlock takes a Backend and cpuBackend is unexported, so a backend package (cuda/metal/gpu) cannot drive the reference
// it must match without this. A seam for gates, not API.
func DraftBlockCPUForTest(d BlockDrafterWeights, fused, blockIn [][]float32) ([][]float32, error) {
	t, ok := d.(interface {
		DraftBlock(Backend, [][]float32, [][]float32) ([][]float32, error)
	})
	if !ok {
		return nil, fmt.Errorf("decoder: %T has no DraftBlock", d)
	}
	return t.DraftBlock(&cpuBackend{}, fused, blockIn)
}

// FuseContextCPUForTest is the same seam for the fusion step.
func FuseContextCPUForTest(d BlockDrafterWeights, ctxCat [][]float32) ([][]float32, error) {
	t, ok := d.(interface {
		FuseContext(Backend, [][]float32) ([][]float32, error)
	})
	if !ok {
		return nil, fmt.Errorf("decoder: %T has no FuseContext", d)
	}
	return t.FuseContext(&cpuBackend{}, ctxCat)
}

// LoadTokenizerForTest and EncodeChatForTest give a backend package the same chat-templated token ids the CPU acceptance
// sweep used, so a GPU wall-clock number is comparable to it without re-implementing the template. The non-thinking suffix
// in particular must match.
func LoadTokenizerForTest(dir string) (*tokenizer.Tokenizer, error) { return tokenizer.Load(dir) }

func EncodeChatForTest(tk *tokenizer.Tokenizer, prompt string) ([]int, error) {
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		return nil, err
	}
	ids, err := tk.EncodeSegments(tmpl.RenderSegments("", []chat.Turn{{Role: "user", Content: prompt}}), false)
	if err != nil {
		return nil, err
	}
	if noThink := os.Getenv("GOINFER_TEST_NOTHINK") != "0"; !noThink {
		// GOINFER_TEST_NOTHINK=0 reproduces what the server renders: Qwen3's template with thinking left on. The drafter was trained
		// on non-thinking output, so that is the off-distribution case.
		return ids, nil
	}
	if _, ok := tk.TokenID("<think>"); ok {
		sfx, e := tk.Encode("<think>\n\n</think>\n\n", false)
		if e != nil {
			return nil, e
		}
		ids = append(ids, sfx...)
	}
	return ids, nil
}

// DeltaNetStepForTest runs one Gated-DeltaNet decode step on the CPU reference and returns the layer output: the parity target
// for a backend's DeltaNet kernel (gpu/deltanet_test.go).
//
// It is a hook rather than a reimplementation in the backend's test because the CPU recurrence is already gated against HF
// (TestGatedDeltaNet_parity, against transformers' torch_recurrent_gated_delta_rule), so a kernel compared with this one is
// kernel ≡ CPU ≡ HF; a reference rewritten inside package gpu would be a second unvalidated implementation
// (docs/parity-coverage-policy.md).
//
// The caller supplies the weights and the state; st is mutated in place across calls, so a multi-token drift test calls this in
// a loop. A nil backend means the CPU one, which is unexported, as in DraftBlockCPUForTest; matmul dereferences it
// unconditionally.
func DeltaNetStepForTest(be Backend, h []float32, w *DeltaNetWeightsForTest, hidden int, eps float64, st *DeltaStateForTest) []float32 {
	if be == nil {
		be = &cpuBackend{}
	}
	return gatedDeltaNetStep(be, h, w.w, w.p, hidden, eps, st.s)
}

// DeltaNetWeightsForTest builds the CPU weight struct from plain slices, so a backend test can
// construct one without the decoder package's unexported types.
type DeltaNetWeightsForTest struct {
	w *deltaNetWeights
	p qwen35Params
}

// DeltaStateForTest wraps the per-layer recurrent state (conv window + delta state).
type DeltaStateForTest struct{ s *deltaState }

// NewDeltaNetForTest assembles the CPU DeltaNet layer + a zeroed state at the given geometry.
// convW/dtBias/negExpA/normW and the projections are caller-supplied so the test controls every
// value the kernel will see.
func NewDeltaNetForTest(convKernel, keyHeadDim, valueHeadDim, numKeyHeads, numValueHeads, hidden int,
	inProjQKV, inProjZ, inProjB, inProjA, convW, dtBias, negExpA, normW, outProj []float32,
) (*DeltaNetWeightsForTest, *DeltaStateForTest) {
	p := qwen35Params{
		ConvKernel: convKernel, KeyHeadDim: keyHeadDim, ValueHeadDim: valueHeadDim,
		NumKeyHeads: numKeyHeads, NumValueHeads: numValueHeads,
	}
	keyDim, valueDim := keyHeadDim*numKeyHeads, valueHeadDim*numValueHeads
	convDim := 2*keyDim + valueDim
	w := &deltaNetWeights{
		inProjQKV: linalg.WrapF32(inProjQKV, convDim, hidden),
		inProjZ:   linalg.WrapF32(inProjZ, valueDim, hidden),
		inProjB:   inProjB,
		inProjA:   inProjA,
		convW:     convW,
		dtBias:    dtBias,
		negExpA:   negExpA,
		normW:     normW,
		outProj:   linalg.WrapF32(outProj, hidden, valueDim),
	}
	return &DeltaNetWeightsForTest{w: w, p: p}, &DeltaStateForTest{s: newDeltaState(p)}
}

// SetDeltaCapHook installs/clears the DeltaNet step-3 capture seam. A backend package cannot
// reach deltaCapHook directly (it is decoder-internal), so it drives it through here — the same
// arrangement SetMambaCapHook has for the Mamba-2 kernel gate.
func SetDeltaCapHook(f func(mixed, conv, gateIn, betaGate, corePre, gated, z []float32)) {
	deltaCapHook = f
}

// L2NormScaledForTest exposes the DeltaNet q/k normalizer (FLA's l2norm, eps 1e-6) so a backend's
// norm kernel is gated against the reference rather than against a five-line reimplementation of
// it. The epsilon is load-bearing for a zero-magnitude head, which is the case worth gating.
func L2NormScaledForTest(x []float32, s float32) []float32 { return l2normScaled(x, s) }

// GumbelDrawForTest is decoder's reference temperature-only draw for an explicit (seed, draw): the host
// implementation every device kernel (CUDA, and the other backends') is held to. Sequential and
// chunk-independent.
func GumbelDrawForTest(logits []float32, temperature float64, seed, draw uint64) int {
	_, idx := gumbelArgmaxRange(logits, 0, len(logits), float32(1/temperature), [2]uint32{uint32(seed), uint32(seed >> 32)}, draw)
	if idx < 0 {
		return argmax(logits)
	}
	return idx
}

// GumbelKeyForTest is the host's score for token i under the same (seed, draw) — what the device must agree
// with; a device/host mismatch is legitimate only when two tokens' host keys are within a few ulps.
func GumbelKeyForTest(logits []float32, temperature float64, seed, draw uint64, i int) float32 {
	r := philox4x32([4]uint32{uint32(i >> 2), uint32(draw), uint32(draw >> 32), 0}, [2]uint32{uint32(seed), uint32(seed >> 32)})
	return gumbelKey(logits[i], float32(1/temperature), r[i&3])
}

// SetSSMStopLayerForTest sets the granite forward's layer truncation (ssmStopLayer in forward_granite.go) and returns a func
// that restores the previous value. Keep it a variable: an environment variable read once at init would ignore a sweep's
// per-iteration change.
func SetSSMStopLayerForTest(n int) (restore func()) {
	prev := ssmStopLayer
	ssmStopLayer = n
	return func() { ssmStopLayer = prev }
}

// SetGGUFRowFilterForTest installs f as the GGUF loader's per-row filter over layer-matmul weights
// (ggufRowFilter); nil removes it. Not safe to change while a load is in flight.
func SetGGUFRowFilterForTest(f func(name string, row []float32)) { ggufRowFilter = f }

// SetHostRAMAvailableForTest makes the memory probe behind the load-time fit guard and the per-request prefill
// admission (AdmitPrefillMemory, AdmitPrefillMemoryShare) report v bytes available, and returns the restore. It clears
// the probe's TTL cache both ways, so neither the override nor the real value is masked by a stale read. Set it after
// a model loads, or the load's own fit guard sees v too.
func SetHostRAMAvailableForTest(v int64) (restore func()) {
	prev := hostRAMAvailable
	hostRAMAvailable = func() int64 { return v }
	resetAvailProbeCache()
	return func() {
		hostRAMAvailable = prev
		resetAvailProbeCache()
	}
}

// SetMemoryProbeForTest makes the named backend's registered memory probe (RegisterMemoryProbe: "cuda", "metal") report
// freeBytes and ok, and returns the restore. It reaches the residency decisions that read FreeBytesFor: the resident context a
// load picks, how many KV slots fit, whether Plan places or declines the model. A budget below the real card runs end to end on
// it (the model shrinks its context or declines to the CPU); a budget above it only changes the decision, and a resident build
// that then asked the real card for the memory would fail, so a test should not force a figure the model cannot really be given.
// Set it before the model loads. A name with no registered probe is created, so a test can also force a backend that is not
// linked.
func SetMemoryProbeForTest(name string, freeBytes int64, ok bool) (restore func()) {
	memProbeMu.Lock()
	prev, had := memProbes[name]
	memProbes[name] = func() (int64, bool) { return freeBytes, ok }
	memProbeMu.Unlock()
	return func() {
		memProbeMu.Lock()
		defer memProbeMu.Unlock()
		if had {
			memProbes[name] = prev
		} else {
			delete(memProbes, name)
		}
	}
}

// ProbeBackendForTest runs the resident self-test (selftest_gpu.go) for backend and quant, uncached and unrecorded, and returns its result.
func ProbeBackendForTest(backend, quant string) SelfTestResult { return probeBackend(backend, quant) }

// SetProbeGPUConfigMutatorForTest makes the resident self-test hand each fixture's GPU model a config.json rewritten by mutate while the CPU reference keeps the true one, and returns
// the restore. It is the mutation proof that the probe can go red: a wrong rope base or norm epsilon in the resident path looks exactly like this from the outside.
func SetProbeGPUConfigMutatorForTest(mutate func(name string, cfg []byte) []byte) (restore func()) {
	prev := probeGPUConfigMutator
	probeGPUConfigMutator = mutate
	return func() { probeGPUConfigMutator = prev }
}

// ResetSelfTestCachesForTest clears the per-(backend, quant) probe cache and the recorded results, so a test sees a first-use probe.
func ResetSelfTestCachesForTest() { resetSelfTestCaches() }

// SetGemma4PLEDropTokenForTest plants a defect: the resident embedding row's PLE inputs lose their token-identity term
// (docs/tasks/task-multimodal-support-2026-10.md), so a gate can be shown to go red. The CPU forward is unaffected.
func SetGemma4PLEDropTokenForTest(on bool) { gemma4PLEDropTokenForTest = on }

// SetGemma4VLRowsDefectForTest selects a planted defect in the E-model image-row builder (0 clears it; see gemma4VLRowsDefectForTest).
func SetGemma4VLRowsDefectForTest(d int) { gemma4VLRowsDefectForTest = d }

// Gemma4EModelImageRowsForTest exposes the E-model image-row builder; a gate builds the rows, then hands them to the resident's PrefillLast.
func (m *Model) Gemma4EModelImageRowsForTest(ids []int, imageEmbeds []float32, imgPos, imgLen int) ([][]float32, error) {
	return m.gemma4EModelImageRowsSpans(ids, imageEmbeds, []ImageSpan{{Pos: imgPos, Len: imgLen}})
}

// SetCaptureLayersForTest arms the batched hidden-state capture (the seam forwardn.go's layer loop fills after layer l's residual
// add and DeepStack add) on c for the next CPU prefill: CapturedForTest then holds one [rows*hidden] float32 copy per requested
// layer, the residual after that layer, as HF's hidden_states[l+1] is. nil disarms.
func (c *KVCache) SetCaptureLayersForTest(layers []int) {
	if layers == nil {
		c.captureLayers, c.captured = nil, nil
		return
	}
	c.captureLayers = append([]int(nil), layers...)
	c.captured = make([][]float32, len(layers))
}

// CapturedForTest returns the rows SetCaptureLayersForTest armed the cache to record, one slice per requested layer.
func (c *KVCache) CapturedForTest() [][]float32 { return c.captured }

// SetStreamDirSwapForTest turns on StreamTranscodeDir's planted defect (layers 0 and 1 written in each other's place) and
// returns the restore (docs/tasks/task-prequant-dir-streaming-2026-10.md).
func SetStreamDirSwapForTest(on bool) func() {
	prev := streamDirSwapForTest
	streamDirSwapForTest = on
	return func() { streamDirSwapForTest = prev }
}
