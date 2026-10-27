//go:build gpu

//lint:file-ignore SA1019 Release is Close's deprecated alias, called here as before; it is deprecated for new code only.

package gpu

import (
	"context"
	"fmt"
	"github.com/townsendmerino/goinfer/gpu/internal/kernels"
	"math"
	"slices"

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
// MoE FFN and the layouts without a quantized KV cache. It returns a typed decline (decoder.DeclineResident) or nil. SB2's
// layer B-webgpu census (gpu/sb2_census_test.go) records it per fixture and option cell.
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
	// Only the generic GQA branch honours the KV precision flag: the Nemotron, Qwen3.5 and MLA branches always allocate f32
	// caches, while ctxCap below is raised by the flag and the kernel selection is model-wide. With --kv i8 the missing scale
	// buffers would surface as "device allocation failed (VRAM exhausted?)" and the model would silently run on CPU for a false
	// reason; with --kv f16 each cache would be ctxCap×kvDim×4 bytes at the RAISED cap, twice the intended footprint. Decline,
	// naming the flag, rather than invent unvalidated quantized KV for those layouts; the operator can drop the flag and get the
	// resident path back.
	kvDelivered := m.KVDeliveredFor("webgpu")
	if kvI8, kvF16 := kvDelivered == "i8", kvDelivered == "f16"; kvI8 || kvF16 {
		_, _, _, _, _, _, _, _, mlaOK := m.MLAResidentParams()
		_, _, _, _, _, _, dnetOK := m.Qwen35ResidentParams()
		if family := map[bool]string{true: "nemotron"}[nemoOK] + map[bool]string{true: "qwen3_5"}[dnetOK] +
			map[bool]string{true: "mla"}[mlaOK]; family != "" && !quantKVDeclineOffForTest {
			flag := map[bool]string{true: "--kv i8"}[kvI8] + map[bool]string{true: "--kv f16"}[kvF16]
			return decoder.DeclineResident("%s does not implement %s KV on webgpu (only the generic GQA path does); "+
				"drop the flag to use the resident path", family, flag)
		}
	}
	return nil
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
	c := b.ctx
	w := m.Weights()
	hidden, _, nH, nKV, hd, inter, vocab := m.Dims() // arch-backed (Cfg may be zero for GGUF/.giw)
	half := m.RotaryDimResident() / 2
	eps := m.NormEps()
	kvDelivered := m.KVDeliveredFor("webgpu") // the resolver's answer, not the raw request (decoder.ResolveKV)
	kvF16, kvI8 := kvDelivered == "f16", kvDelivered == "i8"

	// No-GQA models (nKV == nH, e.g. phi3-mini) are NOT declined on WebGPU. A resident-vs-CPU cosine bar does not transfer to a
	// real int4 checkpoint: the CPU forward is itself that sensitive to a ±1-ULP perturbation of its f32 norms (the int8
	// activation quantizer flips rounding decisions, and the flips compound per layer), so a divergence near that floor is not
	// evidence of a kernel bug. Do not re-add a no-GQA decline, or hold a real int4 checkpoint to a fixed 0.999 floor, on that
	// evidence. The tests that establish this: TestDecodeRunnerW4A8_geometries (the production plan at phi3-mini's shape against
	// a bit-exact oracle), TestResidentCaptureParityWebGPU (per-sublayer differencing on a real checkpoint) and
	// TestCPUQuantSensitivity (the CPU against itself). Full account:
	// docs/code-notes/gpu.md#webgpuBackend.BuildResident.nogqa and docs/completed/task-webgpu-nogqa-decode-bug.md.

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
	// An unpinned context gives way to the requested KV slots (darwin; slotsBeforeContext in gpu/kv_slots.go).
	ctxCap = slotsBeforeContext(m, ctxCap, kvF16, kvI8, avail0)
	kvDim := nKV * hd

	rd := &residentDecoder{c: c, nKV: nKV, hd: hd, ctxCap: ctxCap, finalSoftcap: m.FinalLogitSoftcapResident(), hidden: hidden, inter: inter, nH: nH, vocab: vocab}
	keepF := func(f func()) { rd.keep = append(rd.keep, f) }
	up32 := func(v []float32) (*wgpu.Buffer, error) {
		d, err := c.UploadF32(v)
		if err != nil {
			return nil, err
		}
		keepF(d.Release)
		return (*kernels.DeviceBufferBuf(d)), nil
	}
	proj := func(pw *linalg.WeightMat) (kernels.DecodeWeight, error) {
		dw, err := uploadProj(c, pw)
		if err != nil {
			return nil, err
		}
		switch t := dw.(type) {
		case *ResidentW4A8:
			keepF(t.Release)
		case *ResidentW8A8:
			keepF(t.Release)
		}
		return dw, nil
	}
	// projF32 quantizes a raw f32 [N,K] weight to int8 row-wise and uploads it (the MLA
	// projections are stored f32 in the decoder; the resident GEMVs are W8A8).
	projF32 := func(w []float32, N, K int) (kernels.DecodeWeight, error) {
		q8, scales := linalg.QuantizeRowsInt8(w, N, K)
		rm, err := c.UploadW8A8(q8, scales, N, K)
		if err != nil {
			return nil, err
		}
		keepF(rm.Release)
		return rm, nil
	}

	// projInt8 uploads an already-int8 weight matrix (per-row scales) as projF32 uploads its quantized f32.
	projInt8 := func(w *linalg.WeightMat, N, K int) (kernels.DecodeWeight, error) {
		q8, scales, _, ok := w.Int8()
		if !ok || w.Rows() != N || w.Cols() != K {
			return nil, fmt.Errorf("gpu: an int8 projection of %d×%d (int8 %v), want %d×%d", w.Rows(), w.Cols(), ok, N, K)
		}
		rm, err := c.UploadW8A8(q8, scales, N, K)
		if err != nil {
			return nil, err
		}
		keepF(rm.Release)
		return rm, nil
	}

	// Granite multiplier folding: ResidMul scales every residual-add weight (o-proj / mamba out-proj / MoE expert-down /
	// shared-down); 1/LogitScale folds into the lm_head; EmbMul applies to the embedding in decoder.embedResident. mult==1 ⇒
	// the plain proj path, byte-identical for non-granite models.
	scaleCopy := func(s []float32, m float32) []float32 {
		out := make([]float32, len(s))
		for i, v := range s {
			out[i] = v * m
		}
		return out
	}
	projMul := func(pw *linalg.WeightMat, mult float32) (kernels.DecodeWeight, error) {
		if mult == 1 {
			return proj(pw)
		}
		N, K := pw.Rows(), pw.Cols()
		var q8 []int8
		var scales []float32
		switch pw.Kind() {
		case "int8":
			q8, scales, _, _ = pw.Int8()
		case "f32":
			f32, ok := pw.F32()
			if !ok {
				return nil, fmt.Errorf("gpu: granite mult-fold f32 has no data")
			}
			q8, scales = linalg.QuantizeRowsInt8(f32, N, K)
		default:
			return nil, fmt.Errorf("gpu: granite mult-fold unsupported kind %q", pw.Kind())
		}
		rm, err := c.UploadW8A8(q8, scaleCopy(scales, mult), N, K)
		if err != nil {
			return nil, err
		}
		keepF(rm.Release)
		return rm, nil
	}
	projF32Mul := func(w []float32, N, K int, mult float32) (kernels.DecodeWeight, error) {
		q8, scales := linalg.QuantizeRowsInt8(w, N, K)
		if mult != 1 {
			scales = scaleCopy(scales, mult)
		}
		rm, err := c.UploadW8A8(q8, scales, N, K)
		if err != nil {
			return nil, err
		}
		keepF(rm.Release)
		return rm, nil
	}
	// stateBuf allocates a build-once, zeroed, in-place-updatable Mamba state buffer
	// (Storage|CopyDst so Reset can re-zero it per generation).
	stateBuf := func(n int) (*wgpu.Buffer, error) {
		b2, err := (*kernels.ContextDevice(c)).TryCreateBuffer(&wgpu.BufferDescriptor{Label: "mamba-state", Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst})
		if err != nil {
			return nil, err
		}
		(*kernels.ContextQueue(c)).TryWriteBuffer(b2, 0, wgpu.ToBytes(make([]float32, n)))
		keepF(b2.Release)
		return b2, nil
	}

	fail := func(err error) (decoder.ResidentForward, bool, error) { rd.release(); return nil, false, err }

	// Per-layer RoPE: bind each layer the global or local invFreq table. Most models share one (the cache is keyed on table
	// equality); a family with local/global RoPE has two.
	invFreq := m.RopeInvFreq()
	type ropeEntry struct {
		invf []float32
		buf  *wgpu.Buffer
	}
	var ropeInvCache []ropeEntry
	ropeInvBuf := func(layer int) (*wgpu.Buffer, error) {
		invf := m.RopeInvFreqLayerResident(layer)
		if len(invf) == 0 {
			return nil, nil
		}
		for _, ent := range ropeInvCache {
			if slices.Equal(ent.invf, invf) {
				return ent.buf, nil
			}
		}
		b, err := up32(invf)
		if err != nil {
			return nil, err
		}
		ropeInvCache = append(ropeInvCache, ropeEntry{invf: invf, buf: b})
		return b, nil
	}
	finalNorm, err := up32(w.FinalNorm)
	if err != nil {
		return fail(err)
	}
	// Granite-4.0-H, the Mamba/attention hybrid: rmul folds ResidMul into every residual-add weight; lmMult folds
	// 1/LogitScale into the head. mambaP carries the model-level SSM geometry (per-layer mixer kind from m.GraniteMambaLayer).
	gNH, gHd, gDS, gNG, gK, _, gResidMul, gLogitScale, _, granOK := m.GraniteResidentParams()
	rmul := float32(1)
	lmMult := float32(1)
	if granOK {
		rmul = gResidMul
		if gLogitScale != 0 {
			lmMult = 1 / gLogitScale
		}
		gIn := gNH * gHd
		rd.rm.Mamba = &kernels.MambaRunParams{
			NHeads: gNH, Hp: gHd, Dn: gDS, NGroups: gNG, DConv: gK,
			DInner: gIn, ConvDim: gIn + 2*gNG*gDS, ProjDim: 2*gIn + 2*gNG*gDS + gNH,
			GSize: gNG * gDS, Repeat: gNH / gNG, NormGroups: 1,
		}
	}
	// Nemotron-H (dense squared-ReLU hybrid): the same Mamba-2 geometry but the gated RMSNorm
	// is PER GROUP (normGroups = NGroups), and there are NO multipliers (rmul/lmMult stay 1).
	nNH, nHd, nDS, nNG, nK, nNormG, _, nemoOK := m.NemotronResidentParams()
	var nemoZeroInvFreq *wgpu.Buffer // NoPE: a zeroed invFreq makes the rope kernel an identity
	if nemoOK {
		nIn := nNH * nHd
		rd.rm.Mamba = &kernels.MambaRunParams{
			NHeads: nNH, Hp: nHd, Dn: nDS, NGroups: nNG, DConv: nK,
			DInner: nIn, ConvDim: nIn + 2*nNG*nDS, ProjDim: 2*nIn + 2*nNG*nDS + nNH,
			GSize: nNG * nDS, Repeat: nNH / nNG, NormGroups: nNormG,
		}
		var ze error
		if nemoZeroInvFreq, ze = up32(make([]float32, hd/2)); ze != nil {
			return fail(ze)
		}
	}
	// LM head: tied (LMHead empty → the Embed matrix is the head) or separate.
	headW := &w.LMHead
	if w.LMHead.Rows() == 0 {
		headW = &w.Embed
	}
	lmHead, err := projMul(headW, lmMult)
	if err != nil {
		return fail(err)
	}
	// ropeHalf = len(invFreq) = rotaryDim/2 drives the rope dispatch (partial RoPE rotates only the first rotaryDim dims of
	// each head).
	rd.rm.FinalNorm, rd.rm.LmHead, rd.rm.RopeHalf, rd.rm.SlidingWindow = finalNorm, lmHead, len(invFreq), m.SlidingWindowResident()

	// MoE: models route to stacked experts on-device. moeOK gates the per-layer FFN build below; the params are model-level.
	nExp, topK, moeInter, shInter, sig, normTopK, shUngated, rScale, nGroup, topkGroup, moeOK := m.MoEResidentParams()
	if moeOK {
		rd.rm.Moe = &kernels.MoeRunParams{
			NE: nExp, K: topK, Inter: moeInter, Sigmoid: sig, Norm: normTopK, Scale: float32(rScale),
			SharedInter: shInter, SharedUngated: shUngated, NGroup: nGroup, TopkGroup: topkGroup,
		}
		// gpt-oss's clamped-SwiGLU constants, uniform across every gpt-oss layer.
		if alpha, limit, isGptOss := m.GptOssActResident(); isGptOss {
			rd.rm.Moe.Gptoss = true
			rd.rm.Moe.GptossAlpha, rd.rm.Moe.GptossLimit = alpha, limit
		}
	}
	// MLA: latent attention replaces the q/k/v/o block. mlaOK gates the per-layer MLA build below; the geometry is
	// model-level. attnScale overrides the GQA 1/√HeadDim with the resolved qk_head_dim score scale.
	qLoRA, kvLoRA, qkNope, qkRope, vHead, interleave, mlaAttnScale, mlaRopeScale, mlaOK := m.MLAResidentParams()
	if mlaOK {
		rd.rm.Mla = &kernels.MlaRunParams{
			QLoRARank: qLoRA, KvLoRARank: kvLoRA, QkNope: qkNope, QkRope: qkRope,
			VHead: vHead, Interleave: interleave, RopeScale: float32(mlaRopeScale),
		}
	}
	// Gated-DeltaNet hybrid. Two things make this family its own branch rather than a flag on the generic one: most layers are
	// the recurrent delta rule instead of attention, and its FULL-attention layers carry a per-head output gate fused into a
	// double-width q_proj. Its attention weights also live off lw.QProj (the family's own forward keeps them in its own
	// struct), so the generic q/k/v/o build below cannot see them.
	dnConvK, dnHK, dnHV, dnNK, dnNV, dnAttnGate, dnetOK := m.Qwen35ResidentParams()
	var dnZeroConvB *wgpu.Buffer // DeltaNet's causal conv is bias-free; mambaConv binds a bias
	if dnetOK {
		keyDim, valueDim := dnNK*dnHK, dnNV*dnHV
		rd.rm.Dnet = &kernels.DnetRunParams{
			ConvK: dnConvK, Hk: dnHK, Hv: dnHV, Nk: dnNK, Nv: dnNV, Rep: dnNV / dnNK,
			KeyDim: keyDim, ValueDim: valueDim, ConvDim: 2*keyDim + valueDim,
			StateElems: dnNV * dnHV * dnHK, Eps: float32(eps),
		}
		var ze error
		if dnZeroConvB, ze = up32(make([]float32, rd.rm.Dnet.ConvDim)); ze != nil {
			return fail(ze)
		}
	}
	// Gemma's NormSandwich4 (FeatSandwichNorm): extra norms on each sublayer OUTPUT, applied before the residual add. Required
	// on every layer when the arch declares it (a silently missing one would DROP the norm, not error).
	sandwichOK := m.SandwichNormResident()
	g4MoE := m.HasGemma4MoEResident() // some layers' feed-forward is Gemma 4's parallel dense and MoE block, which carries its own norms
	if sandwichOK {
		for i := range w.Layers {
			lw := &w.Layers[i]
			_, moeLayer := m.Gemma4MoEResidentLayer(i)
			if len(lw.PostAttnNorm) != hidden || (!moeLayer && len(lw.PostMLPNorm) != hidden) {
				return fail(fmt.Errorf("gpu: layer %d: arch declares sandwich norms but PostAttnNorm/PostMLPNorm are not len==hidden(%d) (got %d/%d)",
					i, hidden, len(lw.PostAttnNorm), len(lw.PostMLPNorm)))
			}
		}
		// Sandwich norms + the generic MoE branch is refused: decoderunner.go's MoE FFN branch writes the expert-down output and
		// the shared-expert add straight into r.xd with no postMLPNorm step (the dense FFN branch applies it before the residual
		// add), so admitting the combination would give silently wrong logits. Gemma 4's own MoE block is a different branch with
		// its norms inside (runLayer.g4moe); refuse any other family loudly so the gap cannot ship silently.
		if moeOK && !g4MoE {
			return fail(fmt.Errorf("gpu: sandwich norms (FeatSandwichNorm) + MoE is not supported — " +
				"the MoE FFN dispatch has no postMLPNorm step (N-89, docs/audit-2026-09-10.md)"))
		}
	}
	// buildStacked packs one projection (gate/up/down) across all nE experts into a resident stacked buffer (int8 or int4; a
	// mixed set is an error) that the indexed expert GEMV reads. mult folds Granite's ResidMul into the scales.
	buildStacked := func(nE int, get func(e int) *linalg.WeightMat, mult float32) (*ResidentStackedW8A8, error) {
		w0 := get(0)
		N, K := w0.Rows(), w0.Cols()
		switch w0.Kind() {
		case "int8":
			q8 := make([][]int8, nE)
			sc := make([][]float32, nE)
			for e := range nE {
				w := get(e)
				if w.Kind() != "int8" {
					return nil, fmt.Errorf("gpu: MoE residency expert %d kind %q (mixed; want int8)", e, w.Kind())
				}
				q, s, _, _ := w.Int8()
				if mult != 1 { // Granite ResidMul fold into the expert-down residual add
					s = scaleCopy(s, mult)
				}
				q8[e], sc[e] = q, s
			}
			return c.UploadStackedExperts(q8, sc, nE, N, K)
		case "int4":
			// int4 stacked experts (W4A8) fit VRAM where int8 would spill. Fast path (K%32==0): the decoder's 2-nibble/byte int4 is
			// byte-identical to the GPU packed layout, so its bytes go straight to the Packed upload, with no per-element unpack and
			// re-pack.
			if K%kernels.W4a8GroupSize == 0 {
				q4s := make([][]byte, nE)
				scs := make([][]float32, nE)
				for e := range nE {
					w := get(e)
					if w.Kind() != "int4" {
						return nil, fmt.Errorf("gpu: MoE residency expert %d kind %q (mixed; want int4)", e, w.Kind())
					}
					q4, q4sc, group, _ := decoder.Int4F32(w)
					if group != kernels.W4a8GroupSize {
						return nil, fmt.Errorf("gpu: MoE residency int4 group %d != %d", group, kernels.W4a8GroupSize)
					}
					s := q4sc
					if mult != 1 { // Granite ResidMul fold (no-op for Mellum)
						s = scaleCopy(s, mult)
					}
					q4s[e], scs[e] = q4, s
				}
				return c.UploadStackedExpertsInt4Packed(q4s, scs, nE, N, K)
			}
			// Fallback (K not a multiple of 32): unpack 2-nibble/byte → one nibble per element.
			nib := make([][]uint8, nE)
			sc := make([][]float32, nE)
			for e := range nE {
				w := get(e)
				if w.Kind() != "int4" {
					return nil, fmt.Errorf("gpu: MoE residency expert %d kind %q (mixed; want int4)", e, w.Kind())
				}
				q4, q4s, group, _ := decoder.Int4F32(w)
				if group != kernels.W4a8GroupSize {
					return nil, fmt.Errorf("gpu: MoE residency int4 group %d != %d", group, kernels.W4a8GroupSize)
				}
				un := make([]uint8, N*K)
				for r := range N {
					row := q4[r*((K+1)/2):]
					d := un[r*K : r*K+K]
					for k := range K {
						b := row[k>>1]
						if k&1 == 0 {
							d[k] = b & 0x0F
						} else {
							d[k] = b >> 4
						}
					}
				}
				s := q4s
				if mult != 1 { // Granite ResidMul fold (no-op for Mellum)
					s = scaleCopy(s, mult)
				}
				nib[e], sc[e] = un, s
			}
			return c.UploadStackedExpertsInt4(nib, sc, nE, N, K)
		default:
			return nil, fmt.Errorf("gpu: MoE residency expert kind %q (int8/int4 only)", w0.Kind())
		}
	}

	rd.rm.PleDim = m.Gemma4PLEDimResident()
	for i := range w.Layers {
		lw := &w.Layers[i]
		lhd := m.HeadDimAtResident(i)
		lnKV := m.KVHeadsAtResident(i)
		lrot := m.RotaryDimAtResident(i)
		lrhalf := lrot / 2
		kEqV := m.VFromKResident(i)
		rl := kernels.RunLayer{
			IsLocal:     m.LayerIsLocalResident(i),     // sliding-window layer
			RopeScale:   float32(m.RopeMscaleLayer(i)), // per-layer YaRN mscale
			LayerScalar: m.Gemma4DenseLayerScalarAtResident(i),
			VNorm:       m.IsGemma4Resident() && !kernels.G4DropVNormForTest,
		}
		if kernels.G4DropLayerScalarForTest {
			rl.LayerScalar = 1
		}
		if lhd != hd || lnKV != nKV || lrhalf != half || kEqV {
			rl.Ghd = lhd
			rl.GnKV = lnKV
			rl.Ghalf = lrhalf
			rl.GKEqV = kEqV
		}
		var e error
		if nemoOK {
			// Nemotron-H single-op-per-block: ONE pre-norm (PreAttnNorm) + exactly one op +
			// residual. No FFN pairing, no multipliers. The resident layer-loop branches on
			// rl.nemoKind (mamba / NoPE-attn / relu²-MLP). int8 mamba projections (projF32);
			// attn/MLP projections take the model's native precision via proj (W8A8 or W4A8).
			if rl.AttnNorm, e = up32(lw.PreAttnNorm); e != nil {
				return fail(e)
			}
			rl.MlpNorm = rl.AttnNorm // the relu²-MLP block reuses the single pre-norm
			mp := rd.rm.Mamba
			switch m.NemotronBlockKind(i) {
			case 0: // Mamba-2 mixer
				rl.NemoKind = kernels.NemoKMamba
				rl.IsMamba = true
				inP, cW, cB, aLog, dW, dtB, nW, outP := m.GraniteMambaWeights(i)
				if inQ, outQ := m.MambaProjInt8(i); inQ != nil { // a quantized load holds them as int8: projF32's codes
					if rl.MambaInProj, e = projInt8(inQ, mp.ProjDim, hidden); e != nil {
						return fail(e)
					}
					if rl.MambaOutProj, e = projInt8(outQ, hidden, mp.DInner); e != nil {
						return fail(e)
					}
				} else {
					if rl.MambaInProj, e = projF32(inP, mp.ProjDim, hidden); e != nil {
						return fail(e)
					}
					if rl.MambaOutProj, e = projF32(outP, hidden, mp.DInner); e != nil {
						return fail(e)
					}
				}
				if rl.MambaConvW, e = up32(cW); e != nil {
					return fail(e)
				}
				if rl.MambaConvB, e = up32(cB); e != nil {
					return fail(e)
				}
				headP := make([]float32, mp.NHeads*3)
				for h := 0; h < mp.NHeads; h++ {
					headP[h*3+0] = float32(-math.Exp(float64(aLog[h])))
					headP[h*3+1] = dtB[h]
					headP[h*3+2] = dW[h]
				}
				if rl.MambaHeadP, e = up32(headP); e != nil {
					return fail(e)
				}
				if rl.MambaNormW, e = up32(nW); e != nil {
					return fail(e)
				}
				if rl.MambaWin, e = stateBuf((mp.DConv - 1) * mp.ConvDim); e != nil {
					return fail(e)
				}
				if rl.MambaSSM, e = stateBuf(mp.NHeads * mp.Hp * mp.Dn); e != nil {
					return fail(e)
				}
			case 1: // NoPE GQA attention (identity rope via the zeroed invFreq)
				rl.NemoKind = kernels.NemoKAttn
				rl.InvFreq = nemoZeroInvFreq
				kc, e1 := c.NewKVCache(nil, ctxCap*kvDim)
				vc, e2 := c.NewKVCache(nil, ctxCap*kvDim)
				if e1 != nil || e2 != nil {
					return fail(fmt.Errorf("gpu: nemotron KV alloc (layer %d): %v %v", i, e1, e2))
				}
				keepF(kc.Release)
				keepF(vc.Release)
				rl.KCache, rl.VCache = (*kernels.DeviceBufferBuf(kc)), (*kernels.DeviceBufferBuf(vc))
				if rl.Q, e = proj(&lw.QProj); e != nil {
					return fail(e)
				}
				if rl.K, e = proj(&lw.KProj); e != nil {
					return fail(e)
				}
				if rl.V, e = proj(&lw.VProj); e != nil {
					return fail(e)
				}
				if rl.O, e = proj(&lw.OProj); e != nil {
					return fail(e)
				}
			case 2: // non-gated relu² MLP
				rl.NemoKind = kernels.NemoKMLP
				if rl.Up, e = proj(&lw.UpProj); e != nil {
					return fail(e)
				}
				if rl.Down, e = proj(&lw.DownProj); e != nil {
					return fail(e)
				}
			case 3: // MoE FFN
				// rd.rm.moe is already populated MODEL-level by MoEResidentParams (it reads arch.MoE generically, which Nemotron also sets);
				// this is the PER-LAYER weight build. There is no expGate/shGate/shGateW: this family's experts, and its shared expert, have
				// only up_proj/down_proj (see decoder/forward_nemotron.go).
				rl.NemoKind = kernels.NemoKMoE
				rl.IsMoE = true
				if rl.Router, e = proj(&lw.Router); e != nil {
					return fail(e)
				}
				if len(lw.RouterBias) > 0 {
					if rl.RouterBias, e = up32(lw.RouterBias); e != nil {
						return fail(e)
					}
				}
				if rl.ExpUp, e = buildStacked(len(lw.Experts), func(x int) *linalg.WeightMat { return &lw.Experts[x].Up }, 1); e != nil {
					return fail(e)
				}
				keepF(rl.ExpUp.Release)
				if rl.ExpDown, e = buildStacked(len(lw.Experts), func(x int) *linalg.WeightMat { return &lw.Experts[x].Down }, 1); e != nil {
					return fail(e)
				}
				keepF(rl.ExpDown.Release)
				// Gated on shInter > 0, like the generic branch below: n_shared_experts == 0 is a valid config (validateNemotron in
				// decoder/config.go rejects only the opposite combination), and the CPU path in decoder/forward_nemotron.go gates the add on
				// moe.SharedIntermediateDim > 0. Projecting a zero-value SharedExpert would fail with "unsupported projection precision" and
				// fall back to CPU with the wrong stated reason.
				if shInter > 0 {
					if rl.ShUp, e = proj(&lw.SharedExpert.Up); e != nil {
						return fail(e)
					}
					if rl.ShDown, e = proj(&lw.SharedExpert.Down); e != nil {
						return fail(e)
					}
				}
			default:
				// No default would let an unhandled block kind append a layer with NIL weights, and decoderunner's gemv would
				// nil-dereference on the first token. The nemotron bypass above skips DecodeRunnerEligible, which decoder/residency.go calls
				// "the one predicate every backend's admission funnels through", so this switch is the last place the shape is checked.
				// Reachable through the exported ResidencyBackend.BuildResident.
				return fail(fmt.Errorf("gpu: nemotron block kind %d at layer %d is not "+
					"implemented by this runner", m.NemotronBlockKind(i), i))
			}
			rd.rm.Layers = append(rd.rm.Layers, rl)
			continue
		}
		if rl.AttnNorm, e = up32(lw.PreAttnNorm); e != nil {
			return fail(e)
		}
		g4b, isG4MoE := m.Gemma4MoEResidentLayer(i)
		if !isG4MoE { // a Gemma 4 MoE layer's feed-forward norms are its block's own
			if rl.MlpNorm, e = up32(lw.PreMLPNorm); e != nil {
				return fail(e)
			}
		}
		if rl.InvFreq, e = ropeInvBuf(i); e != nil {
			return fail(e)
		}
		if dnetOK && m.Qwen35LinearLayer(i) {
			// Gated-DeltaNet mixer layer. No KV cache and no q/k/v/o: the recurrence's state is the whole history, fixed-size and NOT
			// position-truncatable (which is why this family declines prefix reuse and batched ForwardN). The conv reuses the Mamba-2
			// ring window; DeltaNet's conv is bias-free, so every layer binds one shared zeroed convB rather than allocating an
			// all-zero bias per layer.
			rl.IsDeltaNet = true
			dp := rd.rm.Dnet
			qkv, z, outP, inB, inA, cW, dtB, negA, nW := m.Qwen35DeltaWeights(i)
			if rl.DnQKV, e = proj(qkv); e != nil {
				return fail(e)
			}
			if rl.DnZ, e = proj(z); e != nil {
				return fail(e)
			}
			if rl.DnOut, e = proj(outP); e != nil {
				return fail(e)
			}
			// inB/inA are f32 on the CPU by deliberate choice (they feed the write/decay gates,
			// where the recurrence is most precision-sensitive). projF32 quantizes them to W8A8
			// for the resident path — the one place this port is coarser than the reference, and
			// the reason the parity gate scores the gates as their own stage.
			if rl.DnB, e = projF32(inB, dp.Nv, hidden); e != nil {
				return fail(e)
			}
			if rl.DnA, e = projF32(inA, dp.Nv, hidden); e != nil {
				return fail(e)
			}
			rl.MambaConvB = dnZeroConvB
			if rl.MambaConvW, e = up32(cW); e != nil {
				return fail(e)
			}
			if rl.DnDtBias, e = up32(dtB); e != nil {
				return fail(e)
			}
			if rl.DnNegExpA, e = up32(negA); e != nil { // ALREADY -exp(A_log); do not re-exp
				return fail(e)
			}
			if rl.DnNormW, e = up32(nW); e != nil {
				return fail(e)
			}
			if rl.MambaWin, e = stateBuf((dp.ConvK - 1) * dp.ConvDim); e != nil {
				return fail(e)
			}
			if rl.DnState, e = stateBuf(dp.StateElems); e != nil {
				return fail(e)
			}
		} else if dnetOK {
			// Gated SOFTMAX attention layer of the same family: ordinary GQA + per-head QK-norm
			// + partial RoPE, plus the output gate. q_proj stays fused at double width because it
			// is quantized; the runner splits the activation (runLayer.qGate).
			rl.QGate = dnAttnGate
			kc, e1 := c.NewKVCache(nil, ctxCap*kvDim)
			vc, e2 := c.NewKVCache(nil, ctxCap*kvDim)
			if e1 != nil || e2 != nil {
				return fail(fmt.Errorf("gpu: qwen35 KV alloc (layer %d): %v %v", i, e1, e2))
			}
			keepF(kc.Release)
			keepF(vc.Release)
			rl.KCache, rl.VCache = (*kernels.DeviceBufferBuf(kc)), (*kernels.DeviceBufferBuf(vc))
			qP, kP, vP, oP, qN, kN := m.Qwen35AttnWeights(i)
			if rl.Q, e = proj(qP); e != nil {
				return fail(e)
			}
			if rl.K, e = proj(kP); e != nil {
				return fail(e)
			}
			if rl.V, e = proj(vP); e != nil {
				return fail(e)
			}
			if rl.O, e = proj(oP); e != nil {
				return fail(e)
			}
			if rl.QNorm, e = up32(qN); e != nil {
				return fail(e)
			}
			if rl.KNorm, e = up32(kN); e != nil {
				return fail(e)
			}
		} else if granOK && m.GraniteMambaLayer(i) {
			// Mamba-2 SSM mixer layer: in/out_proj W8A8, conv/headP/normW f32, build-once {win, ssm} state. headP interleaves the
			// per-head [Aexp=-exp(aLog), dtBias, D]. ResidMul folds into out_proj. No KV cache, no q/k/v/o (the plan branches on
			// isMamba).
			rl.IsMamba = true
			mp := rd.rm.Mamba
			inP, cW, cB, aLog, dW, dtB, nW, outP := m.GraniteMambaWeights(i)
			// int8 projections: f16 projections do not recover the quality lost to the SSM's f32 (vs f64) exp feeding the router
			// (docs/ssm-int8-quality.md).
			if rl.MambaInProj, e = projF32(inP, mp.ProjDim, hidden); e != nil {
				return fail(e)
			}
			if rl.MambaOutProj, e = projF32Mul(outP, hidden, mp.DInner, rmul); e != nil {
				return fail(e)
			}
			if rl.MambaConvW, e = up32(cW); e != nil {
				return fail(e)
			}
			if rl.MambaConvB, e = up32(cB); e != nil {
				return fail(e)
			}
			headP := make([]float32, mp.NHeads*3)
			for h := 0; h < mp.NHeads; h++ {
				headP[h*3+0] = float32(-math.Exp(float64(aLog[h])))
				headP[h*3+1] = dtB[h]
				headP[h*3+2] = dW[h]
			}
			if rl.MambaHeadP, e = up32(headP); e != nil {
				return fail(e)
			}
			if rl.MambaNormW, e = up32(nW); e != nil {
				return fail(e)
			}
			if rl.MambaWin, e = stateBuf((mp.DConv - 1) * mp.ConvDim); e != nil {
				return fail(e)
			}
			if rl.MambaSSM, e = stateBuf(mp.NHeads * mp.Hp * mp.Dn); e != nil {
				return fail(e)
			}
		} else if mlaOK {
			// MLA: a single compressed-latent cache [ctxCap·latDim] replaces the per-head
			// K/V caches; the per-head K/V are rebuilt in rank-space, never materialized.
			latDim := kvLoRA + qkRope
			lc, e1 := c.NewKVCache(nil, ctxCap*latDim)
			if e1 != nil {
				return fail(fmt.Errorf("gpu: residency latent alloc (layer %d): %v", i, e1))
			}
			keepF(lc.Release)
			rl.LatCache = (*kernels.DeviceBufferBuf(lc))
			qA, qANorm, qB, qProj, kvA, kvANorm, kvB, oProj := m.MLALayerWeights(i)
			qAQ, qBQ, qQ, kvAQ, oQ := m.MLALayerInt8(i)
			// mlaProj uploads a projection's int8 form when the load holds one (the codes projF32 makes from the f32), else the f32.
			mlaProj := func(q *linalg.WeightMat, f []float32, N, K int) (kernels.DecodeWeight, error) {
				if q.Rows() > 0 {
					return projInt8(q, N, K)
				}
				return projF32(f, N, K)
			}
			if qLoRA > 0 { // q_a → norm → q_b LoRA bottleneck
				if rl.MlaQA, e = mlaProj(qAQ, qA, qLoRA, hidden); e != nil {
					return fail(e)
				}
				if rl.MlaQB, e = mlaProj(qBQ, qB, nH*(qkNope+qkRope), qLoRA); e != nil {
					return fail(e)
				}
				if rl.MlaQANorm, e = up32(qANorm); e != nil {
					return fail(e)
				}
			} else { // direct q_proj (V2-Lite)
				if rl.MlaQ, e = mlaProj(qQ, qProj, nH*(qkNope+qkRope), hidden); e != nil {
					return fail(e)
				}
			}
			if rl.MlaKVA, e = mlaProj(kvAQ, kvA, latDim, hidden); e != nil {
				return fail(e)
			}
			if rl.MlaKVANorm, e = up32(kvANorm); e != nil {
				return fail(e)
			}
			if rl.MlaO, e = mlaProj(oQ, oProj, hidden, nH*vHead); e != nil {
				return fail(e)
			}
			// kvB is [nH*(qkNope+vHead), kvLoRA] row-major (per head: k_nope rows ‖ v rows).
			// Slice into W_UKᵀ [nH, kvLoRA, qkNope] (transposed for the absorb GEMV) and
			// W_UV [nH, vHead, kvLoRA] (the lift, used as-is).
			hRow := qkNope + vHead
			wuk := make([]float32, nH*kvLoRA*qkNope)
			wuv := make([]float32, nH*vHead*kvLoRA)
			for h := range nH {
				for d := range qkNope {
					src := kvB[(h*hRow+d)*kvLoRA : (h*hRow+d)*kvLoRA+kvLoRA]
					for cc := range kvLoRA {
						wuk[(h*kvLoRA+cc)*qkNope+d] = src[cc] // transpose: [c][d] ← kvB[d][c]
					}
				}
				for ev := range vHead {
					copy(wuv[(h*vHead+ev)*kvLoRA:(h*vHead+ev)*kvLoRA+kvLoRA], kvB[(h*hRow+qkNope+ev)*kvLoRA:(h*hRow+qkNope+ev)*kvLoRA+kvLoRA])
				}
			}
			if rl.MlaWUK, e = up32(wuk); e != nil {
				return fail(e)
			}
			if rl.MlaWUV, e = up32(wuv); e != nil {
				return fail(e)
			}
		} else {
			// Per-layer KV geometry: each layer's cache is sized by its own nKV·hd, not a model-level kvDim. A layer without a
			// geometry override (rl.ghd == 0) falls back to the model values.
			lnKV, lhd, lkvDim := nKV, hd, kvDim
			if rl.Ghd != 0 {
				lnKV, lhd, lkvDim = rl.GnKV, rl.Ghd, rl.GnKV*rl.Ghd
			}
			// A Gemma 4 E-model's tail layers attend over an earlier layer's K/V (Model.KVSrcAtResident): such a layer owns no
			// cache, binds its source's, and projects Q only.
			kvSrc := m.KVSrcAtResident(i)
			if kernels.G4KVSrcOffForTest && kvSrc != i {
				for j := kvSrc - 1; j >= 0; j-- {
					if m.LayerIsLocalResident(j) == m.LayerIsLocalResident(i) {
						kvSrc = j
						break
					}
				}
			}
			if kvSrc != i {
				if kvSrc < 0 || kvSrc >= len(rd.rm.Layers) {
					return fail(fmt.Errorf("gpu: layer %d shares the K/V of layer %d, which is not built before it", i, kvSrc))
				}
				sl := rd.rm.Layers[kvSrc]
				if sl.KCache == nil || sl.Ghd != rl.Ghd || sl.GnKV != rl.GnKV || sl.GKEqV != rl.GKEqV {
					return fail(fmt.Errorf("gpu: layer %d shares the K/V of layer %d, whose attention geometry differs", i, kvSrc))
				}
				rl.KCache, rl.VCache, rl.KScale, rl.VScale = sl.KCache, sl.VCache, sl.KScale, sl.VScale
				rl.KvShared = true
			} else {
				var kc, vc *DeviceBuffer
				var e1, e2 error
				switch {
				case kvI8:
					var ks, vs *DeviceBuffer
					kc, ks, e1 = c.NewKVCacheI8(nil, ctxCap*lkvDim, lnKV, lhd)
					vc, vs, e2 = c.NewKVCacheI8(nil, ctxCap*lkvDim, lnKV, lhd)
					if e1 == nil && e2 == nil {
						keepF(ks.Release)
						keepF(vs.Release)
						rl.KScale, rl.VScale = (*kernels.DeviceBufferBuf(ks)), (*kernels.DeviceBufferBuf(vs))
					}
				case kvF16:
					kc, e1 = c.NewKVCacheF16(nil, ctxCap*lkvDim)
					vc, e2 = c.NewKVCacheF16(nil, ctxCap*lkvDim)
				default:
					kc, e1 = c.NewKVCache(nil, ctxCap*lkvDim)
					vc, e2 = c.NewKVCache(nil, ctxCap*lkvDim)
				}
				if e1 != nil || e2 != nil {
					return fail(fmt.Errorf("gpu: residency KV alloc (layer %d): %v %v", i, e1, e2))
				}
				// Cross-file invariant guard: the decode runner indexes this cache with strides of g.kvDim = gnKV·ghd up to ctxCap
				// positions. The alloc above derives lkvDim from the same tuple, so this holds by construction; a builder that sized the
				// cache from an independent source would index off the end into garbage output, not a panic, so assert it here, loudly, at
				// plan time.
				if want := ctxCap * lkvDim; (*kernels.DeviceBufferN(kc)) < want || (*kernels.DeviceBufferN(vc)) < want {
					return fail(fmt.Errorf("gpu: layer %d KV cache too small for its geometry: have k=%d v=%d elems, need ctxCap·nKV·hd = %d·%d·%d = %d", i, (*kernels.DeviceBufferN(kc)), (*kernels.DeviceBufferN(vc)), ctxCap, lnKV, lhd, want))
				}
				keepF(kc.Release)
				keepF(vc.Release)
				rl.KCache, rl.VCache = (*kernels.DeviceBufferBuf(kc)), (*kernels.DeviceBufferBuf(vc))
			}
			if rl.Q, e = proj(&lw.QProj); e != nil {
				return fail(e)
			}
			if !rl.KvShared {
				if rl.K, e = proj(&lw.KProj); e != nil {
					return fail(e)
				}
				if !rl.GKEqV {
					if rl.V, e = proj(&lw.VProj); e != nil {
						return fail(e)
					}
				}
			}
			if rl.O, e = projMul(&lw.OProj, rmul); e != nil {
				return fail(e)
			}
		}
		if isG4MoE { // Gemma 4's parallel dense and MoE block (decoder's gemma4MoEFFN)
			g := &kernels.G4MoELayer{NE: g4b.NE, TopK: g4b.TopK, DenseInter: g4b.DenseInter, MoeInter: g4b.MoeInter}
			if g4b.NE <= 0 || g4b.TopK <= 0 || g4b.TopK > g4b.NE || len(g4b.ExpertsGateUp) != g4b.NE || len(g4b.ExpertsDown) != g4b.NE ||
				len(g4b.RouterProjScaled) != g4b.NE*hidden || len(g4b.PerExpertScale) != g4b.NE {
				return fail(fmt.Errorf("gpu: layer %d: Gemma 4 MoE block with %d experts (top %d), router %d values, %d per-expert scales, %d/%d expert projections",
					i, g4b.NE, g4b.TopK, len(g4b.RouterProjScaled), len(g4b.PerExpertScale), len(g4b.ExpertsGateUp), len(g4b.ExpertsDown)))
			}
			for _, nb := range []struct {
				dst *(*wgpu.Buffer)
				v   []float32
			}{{&g.PreFFNNorm, g4b.PreFFNNorm}, {&g.PostFFNNorm1, g4b.PostFFNNorm1}, {&g.PreFFNNorm2, g4b.PreFFNNorm2},
				{&g.PostFFNNorm2, g4b.PostFFNNorm2}, {&g.PostFFNNorm, g4b.PostFFNNorm}} {
				if len(nb.v) != hidden {
					return fail(fmt.Errorf("gpu: layer %d: a Gemma 4 MoE block norm has %d weights, want %d", i, len(nb.v), hidden))
				}
				if *nb.dst, e = up32(nb.v); e != nil {
					return fail(e)
				}
			}
			// The router's norm has no weight. The runner's RMSNorm multiplies by 1+w when the arch adds one and by w otherwise, so
			// the weight that makes it weightless is 0 or 1.
			flat := make([]float32, hidden)
			if !m.RMSAddOne() {
				for j := range flat {
					flat[j] = 1
				}
			}
			if g.RouterNorm, e = up32(flat); e != nil {
				return fail(e)
			}
			if g.Router, e = up32(g4b.RouterProjScaled); e != nil {
				return fail(e)
			}
			if g.PerExpertScale, e = up32(g4b.PerExpertScale); e != nil {
				return fail(e)
			}
			if g.ZeroBias, e = up32(make([]float32, g4b.NE)); e != nil {
				return fail(e)
			}
			if g.Gate, e = proj(g4b.MlpGate); e != nil {
				return fail(e)
			}
			if g.Up, e = proj(g4b.MlpUp); e != nil {
				return fail(e)
			}
			if g.Down, e = proj(g4b.MlpDown); e != nil {
				return fail(e)
			}
			if g.ExpGateUp, e = buildStacked(g4b.NE, func(x int) *linalg.WeightMat { return g4b.ExpertsGateUp[x] }, 1); e != nil {
				return fail(e)
			}
			keepF(g.ExpGateUp.Release)
			if g.ExpDown, e = buildStacked(g4b.NE, func(x int) *linalg.WeightMat { return g4b.ExpertsDown[x] }, 1); e != nil {
				return fail(e)
			}
			keepF(g.ExpDown.Release)
			if (*kernels.ResidentStackedW8A8Rows(g.ExpGateUp)) != 2*g4b.MoeInter || (*kernels.ResidentStackedW8A8Rows(g.ExpDown)) != hidden {
				return fail(fmt.Errorf("gpu: layer %d: Gemma 4 expert projections have %d and %d rows, want %d and %d", i, (*kernels.ResidentStackedW8A8Rows(g.ExpGateUp)), (*kernels.ResidentStackedW8A8Rows(g.ExpDown)), 2*g4b.MoeInter, hidden))
			}
			rl.G4moe, rd.rm.G4moe = g, true
			rl.LayerScalar = g4b.LayerScalar
			if kernels.G4DropLayerScalarForTest {
				rl.LayerScalar = 1
			}
		} else if moeOK && len(lw.Experts) > 0 { // Mixtral-class sparse MoE FFN
			rl.IsMoE = true
			if rl.Router, e = proj(&lw.Router); e != nil {
				return fail(e)
			}
			if len(lw.RouterBias) > 0 {
				if rl.RouterBias, e = up32(lw.RouterBias); e != nil {
					return fail(e)
				}
			}
			if rl.ExpGate, e = buildStacked(len(lw.Experts), func(x int) *linalg.WeightMat { return &lw.Experts[x].Gate }, 1); e != nil {
				return fail(e)
			}
			keepF(rl.ExpGate.Release)
			if rl.ExpUp, e = buildStacked(len(lw.Experts), func(x int) *linalg.WeightMat { return &lw.Experts[x].Up }, 1); e != nil {
				return fail(e)
			}
			keepF(rl.ExpUp.Release)
			if rl.ExpDown, e = buildStacked(len(lw.Experts), func(x int) *linalg.WeightMat { return &lw.Experts[x].Down }, rmul); e != nil {
				return fail(e)
			}
			keepF(rl.ExpDown.Release)
			// gpt-oss (FeatAttnSink): expert-major gate‖up and down bias tables.
			if rd.rm.Moe != nil && rd.rm.Moe.Gptoss {
				if rl.GateUpBias, e = up32(m.GptOssExpertBiasResident(i)); e != nil {
					return fail(e)
				}
				if rl.DownBias, e = up32(m.GptOssExpertDownBiasResident(i)); e != nil {
					return fail(e)
				}
			}
			if shInter > 0 { // always-on shared expert (qwen2_moe gated / GLM ungated)
				if rl.ShGate, e = proj(&lw.SharedExpert.Gate); e != nil {
					return fail(e)
				}
				if rl.ShUp, e = proj(&lw.SharedExpert.Up); e != nil {
					return fail(e)
				}
				if rl.ShDown, e = projMul(&lw.SharedExpert.Down, rmul); e != nil {
					return fail(e)
				}
				if !shUngated { // qwen2_moe: the [1,hidden] sigmoid gate
					if rl.ShGateW, e = proj(&lw.SharedGate); e != nil {
						return fail(e)
					}
				}
			}
		} else {
			if rl.Gate, e = proj(&lw.GateProj); e != nil {
				return fail(e)
			}
			if rl.Up, e = proj(&lw.UpProj); e != nil {
				return fail(e)
			}
			if rl.Down, e = projMul(&lw.DownProj, rmul); e != nil {
				return fail(e)
			}
			// Gemma 4 E-model: the per-layer-embedding branch's two projections and its norm.
			if rd.rm.PleDim > 0 {
				if rl.PleGate, e = proj(&lw.PLEGate); e != nil {
					return fail(e)
				}
				if rl.PleProj, e = proj(&lw.PLEProj); e != nil {
					return fail(e)
				}
				if rl.PostPLENorm, e = up32(lw.PostPLENorm); e != nil {
					return fail(e)
				}
			}
		}
		if len(lw.QBias) > 0 { // Qwen2 q/k/v bias
			if rl.QBias, e = up32(lw.QBias); e != nil {
				return fail(e)
			}
			if rl.KBias, e = up32(lw.KBias); e != nil {
				return fail(e)
			}
			if rl.VBias, e = up32(lw.VBias); e != nil {
				return fail(e)
			}
		}
		if len(lw.QNorm) > 0 { // Qwen3/GLM per-head QK-norm (applied before RoPE)
			if rl.QNorm, e = up32(lw.QNorm); e != nil {
				return fail(e)
			}
			if rl.KNorm, e = up32(lw.KNorm); e != nil {
				return fail(e)
			}
		}
		// FeatOutBias (gpt-oss o_proj bias) and FeatSandwichNorm (Gemma's post-sublayer norms): the same "len>0, else leave nil"
		// gate as QBias/QNorm above.
		if len(lw.OBias) > 0 {
			if rl.OBias, e = up32(lw.OBias); e != nil {
				return fail(e)
			}
		}
		if sandwichOK { // both present together (SandwichNormResident requires both non-empty)
			if rl.PostAttnNorm, e = up32(lw.PostAttnNorm); e != nil {
				return fail(e)
			}
			if !isG4MoE {
				if rl.PostMLPNorm, e = up32(lw.PostMLPNorm); e != nil {
					return fail(e)
				}
			}
		}
		// FeatAttnSink (gpt-oss): attnSinks is ALWAYS bound (a one-element dummy when the layer has none) because WGSL bind groups
		// cannot bind a null storage buffer; Metal does the same, CUDA uses a null-pointer sentinel. DecodeRunner's noAttnSinks is a
		// separate runner-local fallback for hand-built test fixtures that skip BuildResident. The two buffers have different owners
		// and lifetimes (resident, shared across runners, vs per-runner), so they are deliberately not unified.
		if sinks := m.GptOssSinksResident(i); len(sinks) > 0 {
			if rl.AttnSinks, e = up32(sinks); e != nil {
				return fail(e)
			}
			rl.HasSink = true
		} else {
			if rl.AttnSinks, e = up32([]float32{0}); e != nil {
				return fail(e)
			}
		}
		rd.rm.Layers = append(rd.rm.Layers, rl)
	}
	_ = vocab // logits length is lmHead.nRows()

	rd.rm.KvF16 = kvF16                         // the runner picks the f16 attn/store kernels off this
	rd.rm.KvI8 = kvI8                           // …or the int8 kernels + scale binds
	rd.rm.GatedGELU = m.GatedActResident() == 0 // FeatGatedGELU — decoder's ActKind 0=GELU-tanh (Gemma)
	scale, addOne := m.AttnScale(), m.RMSAddOne()
	if mlaOK { // MLA scores over qk_head_dim, not HeadDim — use the resolved score scale
		scale = float32(mlaAttnScale)
	}
	rd.runnerFor = func(rm kernels.RunModel) (*DecodeRunner, error) {
		return kernels.ContextNewDecodeRunner(c, rm, hidden, nH, nKV, hd, inter, 0, eps, scale, addOne)
	}
	rd.newRunner = func() (*DecodeRunner, error) { return rd.runnerFor(rd.rm) }
	rd.mropeAxis = m.MRopeAxisResident
	rd.prefillLast = func(xs [][]float32, startPos int, mrope [][3]int) ([]float32, error) {
		mw, pp, ok := kernels.RunModelToPrefill(&rd.rm, hd)
		pp.MropeAxis = rd.mropeAxis
		if !ok {
			return nil, fmt.Errorf("gpu: PrefillLast declines — model uses a feature outside the plain dense pass (MoE/MLA/SSM/sliding-window/…)")
		}
		// q/k/v bias (Qwen2) on an int8 projection is admitted on Vulkan only: the fused-epilogue tiled GEMM
		// (matmulTiledW8A8BiasKernelWGSL, see runModelToPrefill) is bit-exact there but still diverges on Metal past nKeys~15
		// (cosine ~0.998, a real gap, not float noise). Do not widen this gate to another backend without a bit-exactness check
		// on it, or a check against the real ship gate (docs/completed/task-prefill-gap.md §3.2);
		// docs/code-notes/gpu.md#webgpuBackend.BuildResident.biasgate has the measurements. An int4 projection adds its bias in
		// a separate step, as decode does, and is not behind this gate.
		if pp.Int4() && !kernels.W4MultiVerified[c.Backend()] {
			return nil, fmt.Errorf("gpu: PrefillLast declines — the multi-row int4 kernels are not yet verified bit-exact on backend %q", c.Backend())
		}
		if pp.Int8Bias(mw) && c.Backend() != "vulkan" {
			return nil, fmt.Errorf("gpu: PrefillLast declines — q/k/v bias on backend %q is not yet verified bit-exact (Vulkan only for now)", c.Backend())
		}
		positions := make([]int, len(xs))
		for i := range positions {
			positions[i] = startPos + i
		}
		return kernels.ContextPrefillLastRows(c, xs, mw, pp, mrope, hidden, nH, nKV, hd, inter, positions, 0, eps, scale, addOne)
	}
	runner, err := rd.newRunner()
	if err != nil {
		return fail(err)
	}
	rd.runner = runner
	rd.buildKVSlots(m, kvSlotsRequest(m, &rd.rm), avail0)
	return rd, true, nil
}

// ContextCap is the resident KV capacity in positions, queryable so the decode loop can clamp or refuse up front instead of
// hitting the mid-generation cap error.
func (rd *residentDecoder) ContextCap() int { return rd.ctxCap }

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
// (docs/tasks/task-multimodal-support-2026-10.md, "G-S12.4b-4a-5").
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
