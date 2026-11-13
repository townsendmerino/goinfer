//go:build gpu

//lint:file-ignore SA1019 Release is Close's deprecated alias, called here as before; it is deprecated for new code only.

package webgpu

import (
	"context"
	"fmt"
	"slices"

	"github.com/townsendmerino/goinfer/backend/webgpu/internal/kernels"
	"github.com/townsendmerino/goinfer/internal/runtime/program"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// GPU full-residency bridge: builds a resident DecodeRunner from a loaded decoder.Model so decoder.Generate's per-token
// forward runs entirely on the device. webgpuBackend satisfies decoder.ResidencyBackend: the decoder calls BuildResident
// when the arch is eligible, then routes the per-token forward through the returned ResidentForward
// (decoder/residency.go).

// uploadProj uploads one projection to the device at its native precision and returns a decodeWeight the DecodeRunner can
// GEMV. int4 and int8 upload as they are; an f32 projection is quantized row-wise to int8. It reads the projection through
// linalg.WeightMat's exported accessors (Kind/Rows/Cols/Int4/Int8).
func uploadProj(c *Context, w *linalg.WeightMat) (kernels.DecodeWeight, error) {
	N, K := w.Rows(), w.Cols()
	switch w.Kind() {
	case "int4":
		q4, q4s, group, _ := decoder.Int4F32(w)
		if group != kernels.W4a8GroupSize {
			return nil, fmt.Errorf("gpu: residency int4 group %d != %d", group, kernels.W4a8GroupSize)
		}
		// Fast path: when K%32==0 the decoder's 2-nibble/byte int4 is byte-identical to the GPU packed layout
		// (TestInt4LayoutMatch), so upload the bytes straight, with no per-element unpack and re-pack.
		if K%kernels.W4a8GroupSize == 0 {
			return c.UploadW4A8Packed(q4, q4s, N, K)
		}
		// Fallback (K not a multiple of 32 → row padding differs): unpack 2-nibble/byte to
		// one nibble (0..15) per element and let UploadW4A8 re-pack. Values preserved.
		nib := make([]uint8, N*K)
		for r := range N {
			row := q4[r*((K+1)/2):]
			dst := nib[r*K : r*K+K]
			for k := range K {
				b := row[k>>1]
				if k&1 == 0 {
					dst[k] = b & 0x0F
				} else {
					dst[k] = b >> 4
				}
			}
		}
		return c.UploadW4A8(nib, q4s, N, K)
	case "int8":
		q8, scales, _, _ := w.Int8()
		return c.UploadW8A8(q8, scales, N, K)
	case "f32":
		// Some projections stay f32 even under an int8 load (the MoE router is kept
		// full-precision for selection stability). Quantize row-wise to int8 here so the
		// resident GEMV can read it — the router logits feed a top-k, so int8 is ample.
		f32, ok := w.F32()
		if !ok {
			return nil, fmt.Errorf("gpu: residency f32 projection has no data")
		}
		q8, scales := linalg.QuantizeRowsInt8(f32, N, K)
		return c.UploadW8A8(q8, scales, N, K)
	default:
		return nil, fmt.Errorf("gpu: residency unsupported projection precision %q", w.Kind())
	}
}

var (
	_ decoder.ResidentForward   = (*residentDecoder)(nil)
	_ decoder.ResidentAdapter   = (*residentDecoder)(nil)
	_ decoder.ResidentPrefillKV = (*residentDecoder)(nil)
)

// residentDecoder is the gpu side of decoder.ResidentForward: a persistent
// DecodeRunner + the runModel (for KV upload), built once per model.
type residentDecoder struct {
	c                        *Context
	runner                   *DecodeRunner
	rm                       kernels.RunModel
	nKV, hd                  int      // KV grouping — UploadKV needs it to per-head quantize int8
	hidden, inter, nH, vocab int      // the shape PrefillScratchBytes prices a pass by
	ctxCap                   int      // resident KV capacity in positions; writes past it corrupt
	keep                     []func() // release the resident buffers (norms, biases, KV, projections)

	// finalSoftcap is Gemma 2/3/4's final-logit softcap (FeatFinalLogitSoftcap), applied host-side after readback. 0 ⇒ none.
	// Forward and ForwardN are this backend's only readback sites (it has no argmax fast path), so both apply it.
	finalSoftcap float32

	// Batched verify (ForwardN): extra DecodeRunner instances sharing rm (the same
	// resident weights + KV caches) with their own scratch/uniforms, built lazily up
	// to the largest K seen. batch[0] aliases runner. newRunner builds one more.
	newRunner func() (*DecodeRunner, error)
	batch     []*DecodeRunner

	// runnerFor builds a DecodeRunner over a given runModel (newRunner is runnerFor(rd.rm)); buildKVSlots uses it for
	// the slots beyond the first, whose runModels share rd.rm's weights and own their KV.
	runnerFor func(rm kernels.RunModel) (*DecodeRunner, error)

	// slots holds the resident KV slots (gpu/kv_slots.go), nil with one. slots[kvSlot] is the bound slot, and rm / runner /
	// batch above are always ITS, so binding a slot swaps the three. adapter is what SetAdapter last bound, so a switch can
	// move it to the new runner.
	slots   []webgpuKVSlot
	kvSlot  int
	adapter []decoder.ResidentAdapterLayer

	// prefillLast is decoder.Prefiller's batched-M forward, a closure over BuildResident's hidden/nH/inter/eps/scale/addOne
	// locals like newRunner. It reads rd.rm lazily, at call time (not a snapshot), and converts it with runModelToPrefill to the
	// plain dense shape the batched pass takes, declining models outside that scope.
	prefillLast func(xs [][]float32, startPos int, mrope [][3]int) ([]float32, error)
	// mropeAxis is the model's m-RoPE axis table for a head half; it returns nil for a model without m-RoPE.
	mropeAxis func(half int) []int

	// resetErr holds the first error from Reset's state re-zero. Reset satisfies the cross-backend ResidentForward interface
	// and returns nothing, so the error is recorded here and surfaced by the next Forward (the same shape metal/ uses for its
	// encode errors); dropping it would let the next generation continue from the previous sequence's recurrent state with no
	// sign anything went wrong.
	resetErr error
}

// quantKVDeclineOffForTest lets the layouts without a quantized KV cache past their decline (the planted defect of
// TestBackendAdmissionCensus_webgpu_plant).
var quantKVDeclineOffForTest bool

// staticResidentDecline is every decline BuildResident makes from the model alone, in its order, before it reads memory or
// touches the device: the runner's eligibility, the shared feature taxonomy, per-layer geometry, the router-kernel cap, Gemma 4's
// MoE FFN, the layouts without a quantized KV cache and the routed experts' weight kind. It returns a typed decline
// (decoder.DeclineResident) or nil. SB2's layer B-webgpu census (gpu/sb2_census_test.go) records it per fixture and option
// cell, and Plan's webgpu check (decoder.Model.Plan) mirrors the last two.
func staticResidentDecline(m *decoder.Model) error {
	// The Granite-4.0-H and Nemotron-H SSM hybrids are buildable independent of DecodeRunnerEligible (Granite's admission
	// there is knob-gated), so they can be parity-gated first. Production routing still goes through withResidency()'s
	// admission.
	_, _, _, _, _, _, _, _, _, granOK := m.GraniteResidentParams()
	_, _, _, _, _, _, _, nemoOK := m.NemotronResidentParams()
	if !m.DecodeRunnerEligible() && !granOK && !nemoOK {
		return decoder.DeclineResident("arch is not eligible for the webgpu resident decode runner")
	}
	// Admission: refuse any arch needing a feature this runner does not implement, rather than dropping it silently and
	// emitting wrong logits (the bug class in docs/metal-model-coverage.md). The taxonomy is shared (decoder/features.go) so
	// CUDA, WebGPU and Metal cannot drift apart; an arch that arrives with a feature nobody implemented declines here.
	if missing := m.MissingResidentFeatures(decoder.ResidentBackendFeatures("webgpu")); len(missing) > 0 {
		return decoder.DeclineResident("webgpu does not implement %v, which this model needs", missing)
	}
	// Per-layer attention geometry (a local/global head_dim split, as in dense Gemma 4) is not a ResidentFeature, so
	// MissingResidentFeatures cannot catch a backend without the seam; decoder.Model.PerLayerGeomOK checks it directly. The
	// per-layer builder below populates runLayer.ghd/gnKV/ghalf.
	if !m.PerLayerGeomOK("webgpu") {
		return decoder.DeclineResident("per-layer attention geometry is not implemented on webgpu")
	}
	// Router-kernel capacity (gpu/moe.go): the router's score/sel and group arrays are fixed-size, so a model with more experts
	// or groups than the kernel handles would route on only the first experts (the rest silently never considered) or index out
	// of bounds: plausible-looking wrong output. Decline to the staged path, before any allocation. The cap is READ from
	// decoder's declaration (decoder.ResidentBackendMoECap), never restated here: a private copy once disagreed with the map and
	// both published matrices (docs/code-notes/gpu.md#webgpuBackend.BuildResident.routercap).
	if capE, capG, capOK := decoder.ResidentBackendMoECap("webgpu"); capOK {
		if nE, _, _, _, _, _, _, _, nGroup, _, moeOK := m.MoEResidentParams(); moeOK &&
			((capE > 0 && nE > capE) || (capG > 0 && nGroup > capG)) {
			return decoder.DeclineResident("MoE nE=%d/nGroup=%d exceeds webgpu's router-kernel cap (%d experts / %d groups)",
				nE, nGroup, capE, capG)
		}
	}
	if m.HasGemma4MoEResident() && !m.Gemma4MoEOK("webgpu") {
		return decoder.DeclineResident("Gemma 4's parallel dense+MoE FFN is not implemented on webgpu")
	}
	// Only the generic GQA branch honours the KV precision flag: the Nemotron, gated Qwen3.5 and MLA branches always allocate f32
	// caches, while ctxCap below is raised by the flag and the kernel selection is model-wide. With --kv i8 the missing scale
	// buffers would surface as "device allocation failed (VRAM exhausted?)" and the model would silently run on CPU for a false
	// reason; with --kv f16 each cache would be ctxCap×kvDim×4 bytes at the RAISED cap, twice the intended footprint. Decline,
	// naming the flag, rather than invent unvalidated quantized KV for those layouts; the operator can drop the flag and get the
	// resident path back.
	kvDelivered := m.KVDeliveredFor("webgpu")
	if kvI8, kvF16 := kvDelivered == "i8", kvDelivered == "f16"; kvI8 || kvF16 {
		_, _, _, _, _, _, _, _, mlaOK := m.MLAResidentParams()
		_, _, _, _, _, attnGate, dnetOK := m.Qwen35ResidentParams()
		if family := map[bool]string{true: "nemotron"}[nemoOK] + map[bool]string{true: "qwen3_5"}[dnetOK && attnGate] +
			map[bool]string{true: "mla"}[mlaOK]; family != "" && !quantKVDeclineOffForTest {
			flag := map[bool]string{true: "--kv i8"}[kvI8] + map[bool]string{true: "--kv f16"}[kvF16]
			return decoder.DeclineResident("%s does not implement %s KV on webgpu (only the generic GQA path does); "+
				"drop the flag to use the resident path", family, flag)
		}
	}
	// The stacked-expert kernels take int8 or int4 (buildStacked); an f32 or other expert kind would fail mid-upload.
	if k := m.MoEExpertKind(); k != "" && k != "int8" && k != "int4" {
		return decoder.DeclineResident("the routed experts are %s, and webgpu's MoE kernels take int8 or int4 (load with Quant int4 or int8int8)", k)
	}
	return nil
}

// The plants of the Olmo and SmolLM3 parity tests: each drops one setting at the build while the CPU reference keeps the true
// program. minCtxCapForTest raises webgpuCtxBeforeSlots' clamp to the model's positions, so a 128-token run fits on a tiny fixture
// with fewer. Never set outside tests.
var (
	dnetNegEigvalOffForTest, dnetONormEpsOffForTest               bool
	qkNormWholeOffForTest, postOnlyNormOffForTest, noPEOffForTest bool
	minCtxCapForTest                                              int
)

// deltaNetOpParams reads the DeltaNet recurrence parameters from m's compiled program.
// The runner's DeltaNet block holds one set of uniforms (BetaScale, Eps), so if two
// OpDeltaNet layers in the program disagree, the build declines naming the layer.
func deltaNetOpParams(m *decoder.Model) (betaScale, oNormEps float32, err error) {
	var p program.DeltaNetOpParams
	seen := false
	for l, ops := range m.Program().Layers {
		for i := range ops {
			if ops[i].Kind != program.OpDeltaNet {
				continue
			}
			if seen && ops[i].Delta != p {
				return 0, 0, fmt.Errorf("gpu: layer %d's DeltaNet parameters %+v differ from an earlier layer's %+v; the block holds one set", l, ops[i].Delta, p)
			}
			p, seen = ops[i].Delta, true
		}
	}
	betaScale = 1
	if p.NegEigval && !dnetNegEigvalOffForTest {
		betaScale = 2
	}
	oNormEps = m.NormEps()
	if p.ONormEps != 0 && p.ONormEps != p.Eps && !dnetONormEpsOffForTest {
		oNormEps = float32(p.ONormEps)
	}
	return betaScale, oNormEps, nil
}

// BuildResident builds a resident DecodeRunner from m, or (nil,false,nil) when
// the arch is ineligible / a projection is f32 (caller uses the staged path).
func (b *webgpuBackend) BuildResident(m *decoder.Model) (decoder.ResidentForward, bool, error) {
	if err := staticResidentDecline(m); err != nil {
		return nil, false, err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	avail0 := availBeforeBuild(m) // MC1: read before the first upload (darwinKVSlots)

	// No-GQA models (nKV == nH, e.g. phi3-mini) are NOT declined on WebGPU. A resident-vs-CPU cosine bar does not transfer to a
	// real int4 checkpoint: the CPU forward is itself that sensitive to a ±1-ULP perturbation of its f32 norms (the int8
	// activation quantizer flips rounding decisions, and the flips compound per layer), so a divergence near that floor is not
	// evidence of a kernel bug. Do not re-add a no-GQA decline, or hold a real int4 checkpoint to a fixed 0.999 floor, on that
	// evidence. The tests that establish this: TestDecodeRunnerW4A8_geometries (the production plan at phi3-mini's shape against
	// a bit-exact oracle), TestResidentCaptureParityWebGPU (per-sublayer differencing on a real checkpoint) and
	// TestCPUQuantSensitivity (the CPU against itself). Full account:
	// docs/code-notes/gpu.md#webgpuBackend.BuildResident.nogqa and docs/completed/task-webgpu-nogqa-decode-bug.md.

	bs := b.newBuild(m, avail0)
	rd := bs.rd
	fail := func(err error) (decoder.ResidentForward, bool, error) { rd.release(); return nil, false, err }

	if err := bs.uploadModelLevel(); err != nil {
		return fail(err)
	}

	if err := bs.uploadLayers(); err != nil {
		return fail(err)
	}

	if err := bs.bindRunner(); err != nil {
		return fail(err)
	}
	rd.buildKVSlots(m, kvSlotsRequest(m, &rd.rm), avail0)
	return rd, true, nil
}

// ContextCap is the resident KV capacity in positions, queryable so the decode loop can clamp or refuse up front instead of
// hitting the mid-generation cap error.
func (rd *residentDecoder) ContextCap() int { return rd.ctxCap }

// ResidentCaps is decoder.ResidentCapsReporter's: the context cap, and nothing else.
func (rd *residentDecoder) ResidentCaps() decoder.ResidentCaps {
	return decoder.ResidentCaps{ContextCap: rd.ContextCap()}
}

var _ decoder.ResidentCapsReporter = (*residentDecoder)(nil)

// checkCap refuses a write that would land at/past the resident cap. Every KV buffer is
// sized ctxCap positions; a write for absolute position p lands at [p*kvDim…], so p ≥
// ctxCap is an out-of-bounds device write — WGSL robust-access silently clamps it and
// attention then reads garbage. Return an error instead; the decode loop stops (model.go)
// and the caller can fall back to the staged path, which handles longer contexts.
func (rd *residentDecoder) checkCap(pos, n int) error {
	if pos < 0 || pos+n > rd.ctxCap {
		return fmt.Errorf("gpu: KV position %d(+%d) exceeds resident context cap %d — use the staged path for longer contexts", pos, n, rd.ctxCap)
	}
	return nil
}

var _ decoder.ResidentMRoPE = (*residentDecoder)(nil)

func (rd *residentDecoder) Forward(embedding []float32, pos int) ([]float32, error) {
	if err := rd.checkCap(pos, 1); err != nil {
		return nil, err
	}
	if pos == 0 {
		// Fresh sequence: re-zero the compounding recurrent state so it does not carry over from a prior Generate on this
		// *Model. Reset cannot return an error (the cross-backend interface returns nothing), so it records one in resetErr;
		// surface it HERE rather than continue from the previous sequence's state.
		rd.Reset()
		if rd.resetErr != nil {
			return nil, rd.resetErr
		}
	}
	logits, err := rd.runner.Run(embedding, pos, pos)
	if err != nil {
		return nil, err
	}
	kernels.ApplySoftcap(logits, rd.finalSoftcap)
	return logits, nil
}

// ForwardMRoPE is decoder.ResidentMRoPE: like Forward, but the rope-angle position (ropePos)
// is supplied separately from the KV-cache/attention position (pos) — Qwen2.5-VL decode past
// an image block needs them to differ (posUni's doc comment, decoderunner.go). Forward(pos) ==
// ForwardMRoPE(pos, pos) exactly, so this does not duplicate Forward's body.
func (rd *residentDecoder) ForwardMRoPE(embedding []float32, pos, ropePos int) ([]float32, error) {
	if err := rd.checkCap(pos, 1); err != nil {
		return nil, err
	}
	if pos == 0 {
		rd.Reset()
		if rd.resetErr != nil {
			return nil, rd.resetErr
		}
	}
	logits, err := rd.runner.Run(embedding, pos, ropePos)
	if err != nil {
		return nil, err
	}
	kernels.ApplySoftcap(logits, rd.finalSoftcap)
	return logits, nil
}

// ForwardNoLogits implements decoder.ResidentPrefillKV: run one prefill token to populate
// the resident KV cache, skipping the LM head GEMV dispatch, logits staging copy, D2H transfer, and softcap.
func (rd *residentDecoder) ForwardNoLogits(embedding []float32, pos int) error {
	if err := rd.checkCap(pos, 1); err != nil {
		return err
	}
	if pos == 0 {
		rd.Reset()
		if rd.resetErr != nil {
			return rd.resetErr
		}
	}
	return rd.runner.RunNoLogits(embedding, pos, pos)
}

// ForwardN runs K tokens at startPos..startPos+K-1 in one command buffer. It lazily
// grows a pool of K DecodeRunners that share rd.rm (the resident weights + KV caches)
// but own their scratch/uniforms, then records all K into a single submit (runBatch).
// Causal across the shared KV: row i's kv-store precedes row i+1's attention read.
func (rd *residentDecoder) ForwardN(embeddings [][]float32, startPos int) ([][]float32, error) {
	n := len(embeddings)
	if n == 0 {
		return nil, nil
	}
	if err := rd.checkCap(startPos, n); err != nil {
		return nil, err
	}
	// Batched verify is ONLY sound when every position this runner writes can be rolled back. The resident KV can (TruncateTo
	// is a no-op because the cache is positional and Forward re-sets nKeys=pos+1), but the Mamba {win,ssm} and DeltaNet state is
	// NOT positional: running K rows advances it K times, and a partial accept cannot undo the rejected rows, so the next round
	// would decode from over-advanced state, silently wrong. decoder.specRollbackSafe already refuses the recurrent families;
	// the guard is also HERE because the invariant is a property of this runner's own state, and the other check could be
	// relaxed without anyone reading this function. Decline (the callers fall back to plain decode); n==1 stays allowed.
	if n > 1 && (rd.rm.Mamba != nil || rd.rm.Dnet != nil) {
		return nil, fmt.Errorf("gpu: batched ForwardN (K=%d) declines on a recurrent model — its mixer state is not positional and cannot be rolled back after a partial accept", n)
	}
	if startPos == 0 {
		rd.Reset() // fresh sequence (prefill from 0): re-zero Mamba {win,ssm}
		if rd.resetErr != nil {
			return nil, rd.resetErr // see Forward
		}
	}
	if len(rd.batch) == 0 {
		rd.batch = append(rd.batch, rd.runner) // batch[0] aliases the M=1 runner
	}
	for len(rd.batch) < n {
		r, err := rd.newRunner()
		if err != nil {
			return nil, err
		}
		rd.batch = append(rd.batch, r)
	}
	out, err := kernels.RunBatch(rd.c, rd.batch, embeddings, startPos)
	if err != nil {
		return nil, err
	}
	if rd.finalSoftcap > 0 {
		for _, logits := range out {
			kernels.ApplySoftcap(logits, rd.finalSoftcap)
		}
	}
	return out, nil
}

// VerifyPath (decoder.VerifyPathReporter) reports that ForwardN is a sequential verify pass: the marginal cost of an extra
// row is about one full step. runBatch records every row into one Submit/Poll, but that removes Go-side dispatch and sync
// overhead between rows, not GPU compute: each runner still issues its own full set of per-layer dispatches. The decoder's
// verifyTheta reads this instead of an unmeasured default Theta (a smaller Theta drafts DEEPER, not shallower). Always
// sequential: no configuration measured otherwise, and no code path here varies by model shape. Evidence:
// docs/measurements/theta-webgpu-2026-09-23.md (gpu/theta_probe_test.go).
func (rd *residentDecoder) VerifyPath() (bool, string) {
	return false, "sequential — measured Theta 0.978-1.028, indistinguishable from a per-token loop despite the single-submit structure"
}

var _ decoder.Prefiller = (*residentDecoder)(nil)

// PrefillLast is decoder.Prefiller: process the whole embeddings slice in ONE on-device batched pass
// (docs/completed/task-gpu-batched-prefill.md) and return only the LAST position's logits, the sub-linear-TTFT alternative
// to residentPrefillSeed's per-token Forward loop. It declines (the caller falls back to that loop; warnPrefillDeclined logs
// it) when the model is outside the plain dense pass (rd.prefillLast's runModelToPrefill guard); a decline is not fatal.
func (rd *residentDecoder) PrefillLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err // checked once, up front — the whole pass has no finer-grained cancel point
	}
	n := len(embeddings)
	if n == 0 {
		return nil, fmt.Errorf("gpu: PrefillLast: no embeddings")
	}
	if err := rd.checkCap(startPos, n); err != nil {
		return nil, err
	}
	if startPos == 0 {
		// Fresh sequence: re-zero recurrent state, mirroring Forward/ForwardN, even though runModelToModelW currently declines
		// every model that HAS such state; keeping the call means it does not silently stop resetting if that scope widens.
		rd.Reset()
		if rd.resetErr != nil {
			return nil, rd.resetErr
		}
	}
	logits, err := rd.prefillBlocks(ctx, embeddings, startPos, nil)
	if err != nil && (rd.rm.PleDim > 0 || rd.rm.G4moe) && ctx.Err() == nil {
		// A Gemma 4 E-model or a Gemma 4 MoE model that the pass declines is still prefilled here: the rows are walked in
		// order, which is the sequential forward and overwrites whatever a declined pass stored. Their image turns count
		// on a resident prefill (decoder's gemma4VLResidentPrefill).
		for i, row := range embeddings[:n-1] {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if err := rd.runner.RunNoLogits(row, startPos+i, startPos+i); err != nil {
				return nil, err
			}
		}
		logits, err = rd.runner.Run(embeddings[n-1], startPos+n-1, startPos+n-1)
	}
	if err != nil {
		return nil, err
	}
	kernels.ApplySoftcap(logits, rd.finalSoftcap)
	return logits, nil
}

// prefillBlocks runs the batched pass over embeddings in blocks. The pass's scratch grows with its row count, so a
// long prompt goes through in several; a block attends to the K/V the blocks before it wrote, as it would in one pass,
// so the result is the same bits, and only the last block's logits are kept. mrope is nil, or each row's three rotary
// positions.
func (rd *residentDecoder) prefillBlocks(ctx context.Context, embeddings [][]float32, startPos int, mrope [][3]int) ([]float32, error) {
	var logits []float32
	for from := 0; from < len(embeddings); from += prefillBlockRows {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		to := min(from+prefillBlockRows, len(embeddings))
		var rows [][3]int
		if mrope != nil {
			rows = mrope[from:to]
		}
		var err error
		if logits, err = rd.prefillLast(embeddings[from:to], startPos+from, rows); err != nil {
			return nil, err
		}
	}
	return logits, nil
}

// mropePrefillBackends lists the WebGPU backends on which an image turn's m-RoPE prefill runs on the device. On every
// other PrefillMRoPELast declines and the turn keeps the CPU's prefill and the upload. Vulkan is in it on a fidelity
// gate against the float32 reference: the device prefill and the CPU's are two int4 readings of the model, and the
// device's is not the further one. They are not each other's reference, so a reply can differ between them. Metal is
// out because the CPU route is the faster one there. A backend goes in on its own run of that gate
// (docs/completed/task-multimodal-support-2026-10.md, "G-S12.4b-4a-5").
var mropePrefillBackends = map[string]bool{"vulkan": true}

var _ decoder.ResidentMRoPEPrefill = (*residentDecoder)(nil)

// PrefillMRoPELast is decoder.ResidentMRoPEPrefill: PrefillLast for rows whose rotary position is three numbers.
// mropePos is prompt-absolute and covers [0, startPos+len(embeddings)). It declines on a backend outside
// mropePrefillBackends, on a model without m-RoPE, and wherever PrefillLast would.
func (rd *residentDecoder) PrefillMRoPELast(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !mropePrefillBackends[rd.c.Backend()] {
		return nil, fmt.Errorf("gpu: PrefillMRoPELast declines on backend %q: the CPU prefill and upload is the faster route there", rd.c.Backend())
	}
	n := len(embeddings)
	if n == 0 || len(mropePos) != startPos+n {
		return nil, fmt.Errorf("gpu: PrefillMRoPELast declines %d rows at %d with %d positions: the positions must cover the prompt up to the last row", n, startPos, len(mropePos))
	}
	if rd.mropeAxis == nil || rd.mropeAxis(1) == nil {
		return nil, fmt.Errorf("gpu: PrefillMRoPELast declines: the model has no m-RoPE sections")
	}
	if err := rd.checkCap(startPos, n); err != nil {
		return nil, err
	}
	if startPos == 0 {
		rd.Reset()
		if rd.resetErr != nil {
			return nil, rd.resetErr
		}
	}
	logits, err := rd.prefillBlocks(ctx, embeddings, startPos, mropePos[startPos:])
	if err != nil {
		return nil, err
	}
	kernels.ApplySoftcap(logits, rd.finalSoftcap)
	return logits, nil
}

// prefillBlockRows is the most rows the batched pass takes in one call. A variable so a test can cross a block boundary
// on a tiny fixture.
var prefillBlockRows = 512

// TruncateTo is a no-op on the resident cache: it is positional and Forward sets
// nKeys=pos+1, so entries past pos are never read and get overwritten next round
// (see the ResidentForward.TruncateTo contract).
func (rd *residentDecoder) TruncateTo(pos int) {}

// SetAdapter implements decoder.ResidentAdapter: generateInto calls it to bind/clear a compute-time LoRA adapter for an
// admitted session. It forwards to rd.runner only: the adapter-admission path calls Forward exclusively, and the ForwardN
// verify runners (rd.batch) are speculative-decode-only, a path this backend's LoRA support does not cover (see the file
// comment of gpu/lora.go). The adapter is recorded too, because generateInto binds it BEFORE acquiring a KV slot and
// UseKVSlot must move it onto the new slot's runner. A failed bind leaves none bound.
func (rd *residentDecoder) SetAdapter(layers []decoder.ResidentAdapterLayer) error {
	rd.adapter = nil
	if err := rd.runner.SetAdapter(layers); err != nil {
		return err
	}
	rd.adapter = layers
	return nil
}

// UploadKV writes a layer's post-RoPE K and raw V into the resident caches at absolute positions base..base+n-1, the
// prefill bridge. keys/vals are [n*kvDim] f32, kvDim being the layer's own (its KV heads times its head_dim), packed to the cache's precision (f32 raw, f16 via packF16Pairs, int8
// per-head + scales) so the upload matches what an on-device decode would have written. base is normally 0; a wrapped
// CPU-side sliding-window ring's live K/V can start later (KVCache.LayerKV's base return). Element widths: f32 raw = 4
// B/elem; f16 (2 values/u32) = 2 B/elem; int8 (4 values/u32) = 1 B/elem for the KV buffers, and nKV f32 scales per
// position (4 B/elem) for the scale buffers.
func (rd *residentDecoder) UploadKV(layer, base int, keys, vals []float32) error {
	if layer < 0 || layer >= len(rd.rm.Layers) {
		return fmt.Errorf("gpu: UploadKV layer %d out of range", layer)
	}
	// A layer with its own attention geometry (runLayer.ghd != 0) stores rows of its own width and head grouping.
	l := rd.rm.Layers[layer]
	nKV, hd := rd.nKV, rd.hd
	if l.Ghd != 0 {
		nKV, hd = l.GnKV, l.Ghd
	}
	if kernels.UploadKVModelGeomForTest {
		nKV, hd = rd.nKV, rd.hd
	}
	kvDim := nKV * hd
	if kvDim > 0 {
		if len(keys)%kvDim != 0 || len(vals) != len(keys) {
			return fmt.Errorf("gpu: UploadKV layer %d: %d key and %d value floats are not whole rows of %d", layer, len(keys), len(vals), kvDim)
		}
		if err := rd.checkCap(base, len(keys)/kvDim); err != nil {
			return err
		}
	}
	switch {
	case rd.rm.KvI8:
		kw, ks := kernels.PackKVInt8(keys, nKV, hd)
		vw, vs := kernels.PackKVInt8(vals, nKV, hd)
		kvOff := uint64(base) * uint64(kvDim)
		scOff := uint64(base) * uint64(nKV) * 4
		if err := (*kernels.ContextQueue(rd.c)).TryWriteBuffer(l.KCache, kvOff, wgpu.ToBytes(kw)); err != nil {
			return err
		}
		if err := (*kernels.ContextQueue(rd.c)).TryWriteBuffer(l.KScale, scOff, wgpu.ToBytes(ks)); err != nil {
			return err
		}
		if err := (*kernels.ContextQueue(rd.c)).TryWriteBuffer(l.VCache, kvOff, wgpu.ToBytes(vw)); err != nil {
			return err
		}
		return (*kernels.ContextQueue(rd.c)).TryWriteBuffer(l.VScale, scOff, wgpu.ToBytes(vs))
	case rd.rm.KvF16:
		off := uint64(base) * uint64(kvDim) * 2
		if err := (*kernels.ContextQueue(rd.c)).TryWriteBuffer(l.KCache, off, wgpu.ToBytes(kernels.PackF16Pairs(keys))); err != nil {
			return err
		}
		return (*kernels.ContextQueue(rd.c)).TryWriteBuffer(l.VCache, off, wgpu.ToBytes(kernels.PackF16Pairs(vals)))
	default:
		off := uint64(base) * uint64(kvDim) * 4
		if err := (*kernels.ContextQueue(rd.c)).TryWriteBuffer(l.KCache, off, wgpu.ToBytes(keys)); err != nil {
			return err
		}
		return (*kernels.ContextQueue(rd.c)).TryWriteBuffer(l.VCache, off, wgpu.ToBytes(vals))
	}
}

// Reset clears resident state for a fresh generation. KV positions are overwritten
// (caller tracks pos), but the RECURRENT state COMPOUNDS — Mamba's {win, ssm} and
// DeltaNet's {win, dnState} alike — so it must be re-zeroed each generation or the
// recurrence carries over from the prior sequence.
func (rd *residentDecoder) Reset() {
	// Every WriteBuffer error is recorded in resetErr: a failed re-zero would leave the previous sequence's recurrent state
	// resident, and the next generation would continue from it with no sign anything went wrong.
	rd.resetErr = nil
	wr := func(b *wgpu.Buffer, data []byte) {
		if err := (*kernels.ContextQueue(rd.c)).TryWriteBuffer(b, 0, data); err != nil && rd.resetErr == nil {
			rd.resetErr = fmt.Errorf("gpu: resident Reset failed to re-zero recurrent state: %w", err)
		}
	}
	if mp := rd.rm.Mamba; mp != nil {
		winZ := make([]float32, (mp.DConv-1)*mp.ConvDim)
		ssmZ := make([]float32, mp.NHeads*mp.Hp*mp.Dn)
		for i := range rd.rm.Layers {
			if lw := &rd.rm.Layers[i]; lw.IsMamba {
				wr(lw.MambaWin, wgpu.ToBytes(winZ))
				wr(lw.MambaSSM, wgpu.ToBytes(ssmZ))
			}
		}
	}
	if dp := rd.rm.Dnet; dp != nil {
		winZ := make([]float32, (dp.ConvK-1)*dp.ConvDim)
		stZ := make([]float32, dp.StateElems)
		for i := range rd.rm.Layers {
			if lw := &rd.rm.Layers[i]; lw.IsDeltaNet {
				wr(lw.MambaWin, wgpu.ToBytes(winZ))
				wr(lw.DnState, wgpu.ToBytes(stZ))
			}
		}
	}
}

func (rd *residentDecoder) Close() error { kernels.ContextReleaseOwned(rd.c, rd.release); return nil }

func (rd *residentDecoder) release() {
	// Every KV slot's runner and verify pool (MC1); the bound slot's are rd.runner / rd.batch.
	slots := rd.slots
	if slots == nil {
		slots = []webgpuKVSlot{{}}
	}
	slots[rd.kvSlot].runner, slots[rd.kvSlot].batch = rd.runner, rd.batch
	for _, s := range slots {
		// batch[0] aliases the slot's runner; release the extra verify runners (batch[1:]) —
		// they own scratch but share the slot's weights/KV (freed via rd.keep below).
		for i := 1; i < len(s.batch); i++ {
			s.batch[i].Release()
		}
		if s.runner != nil {
			s.runner.Release()
		}
	}
	rd.slots, rd.kvSlot, rd.batch, rd.runner, rd.adapter = nil, 0, nil, nil, nil
	for _, v := range slices.Backward(rd.keep) {
		v()
	}
	rd.keep = nil
}

// webgpuCtxBeforeSlots is the resident context before the KV slots are priced (slotsBeforeContext): the per-precision ceiling,
// lowered to the model's window and to a context the caller asked for.
func webgpuCtxBeforeSlots(m *decoder.Model, kvF16, kvI8 bool) int {
	// decoder.WebGPUCtxCeiling is the single source for the per-precision ceilings; the planner (Model.Plan) calls the same
	// function, so a plan's promised ctx and this allocation cannot drift apart.
	ctxCap := decoder.WebGPUCtxCeiling(kvF16, kvI8)
	// Clamp to the model's own window first. WebGPUCtxCeiling's ceilings were proven against GQA geometries (nKV well below nH)
	// and say nothing about a model with nKV == nH, whose per-position KV cost is several times higher: reserving positions the
	// model can never serve can run the per-layer allocation out of VRAM and decline to the staged path for a reason unrelated
	// to whether the model fits. There is no live free-VRAM query on this backend (unlike CUDA's checkKVFits), so the model's
	// own window is the one geometry-aware signal; cuda/resident.go's resolveCtxCap clamps the same way.
	if mc := m.Config().MaxPositions; mc > 0 && mc < ctxCap {
		ctxCap = mc
	}
	// Options.ResidentContext (`serve -ctx`) lowers the ceiling: min(), not the request, because the caps above are proven-fit
	// ceilings and honouring a LARGER request would trade a clear checkCap refusal for an OOM.
	if req := m.ResidentContextRequest(); req > 0 && req < ctxCap {
		ctxCap = req
	}
	if minCtxCapForTest > ctxCap {
		ctxCap = minCtxCapForTest
	}
	return ctxCap
}
