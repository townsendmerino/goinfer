//go:build gpu

//lint:file-ignore SA1019 Release is Close's deprecated alias, called here as before; it is deprecated for new code only.

package webgpu

import (
	"fmt"
	"slices"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/backend/webgpu/internal/kernels"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// webgpuBuild is the state of one webgpuBackend.BuildResident: the model, the device context and the resident decoder being
// assembled. Its methods are the build's phases and upload helpers; each registers what it allocates with keep, so a failed
// build releases everything through rd.release.
type webgpuBuild struct {
	m  *decoder.Model
	c  *Context
	rd *residentDecoder
	w  *decoder.Weights

	// The geometry newBuild reads from the model: hidden width, query and KV heads, head dim, FFN width, half the rotary dim, the
	// norm epsilon, the delivered KV precision, the resident context cap and the KV row width kvDim = nKV*hd.
	hidden, nH, nKV, hd, inter, half int
	eps                              float32
	kvF16, kvI8                      bool
	ctxCap, kvDim                    int

	// What uploadModelLevel records for the per-layer phase: the Granite and Nemotron-H hybrids (rmul is Granite's ResidMul),
	// the MoE shared-expert shape, MLA's geometry, the Gated-DeltaNet flags, and the norm placements.
	granOK, nemoOK                bool
	rmul                          float32
	nemoZeroInvFreq, dnZeroConvB  *wgpu.Buffer
	moeOK, shUngated              bool
	shInter                       int
	mlaOK                         bool
	qLoRA, kvLoRA, qkNope, qkRope int
	vHead                         int
	mlaAttnScale                  float64
	dnetOK, dnAttnGate            bool
	sandwichOK, postOnlyOK        bool
	qkNormWhole                   bool

	// ropeInvCache holds the invFreq tables uploaded so far; a layer whose table is equal reuses the buffer.
	ropeInvCache []ropeEntry
}

type ropeEntry struct {
	invf []float32
	buf  *wgpu.Buffer
}

// newBuild reads m's geometry and resolves the KV precision and the resident context cap, which an unpinned build lets give way to
// the requested KV slots. It allocates no device memory. avail0 is the host memory read before the build's first upload.
func (b *webgpuBackend) newBuild(m *decoder.Model, avail0 int64) *webgpuBuild {
	c := b.ctx
	w := m.Weights()
	hidden, _, nH, nKV, hd, inter, vocab := m.Dims() // arch-backed (Cfg may be zero for GGUF/.giw)
	half := m.RotaryDimResident() / 2
	eps := m.NormEps()
	kvDelivered := m.KVDeliveredFor("webgpu") // the resolver's answer, not the raw request (decoder.ResolveKV)
	kvF16, kvI8 := kvDelivered == "f16", kvDelivered == "i8"

	ctxCap := webgpuCtxBeforeSlots(m, kvF16, kvI8)
	// An unpinned context gives way to the requested KV slots (darwin; slotsBeforeContext in gpu/kv_slots.go).
	ctxCap = slotsBeforeContext(m, ctxCap, kvF16, kvI8, avail0)
	kvDim := nKV * hd

	rd := &residentDecoder{c: c, nKV: nKV, hd: hd, ctxCap: ctxCap, finalSoftcap: m.FinalLogitSoftcapResident(), hidden: hidden, inter: inter, nH: nH, vocab: vocab}
	return &webgpuBuild{
		m: m, c: c, rd: rd, w: w,
		hidden: hidden, nH: nH, nKV: nKV, hd: hd, inter: inter, half: half, eps: eps,
		kvF16: kvF16, kvI8: kvI8, ctxCap: ctxCap, kvDim: kvDim,
	}
}

// keep registers a release for rd.Close to run.
func (bs *webgpuBuild) keep(f func()) { bs.rd.keep = append(bs.rd.keep, f) }

// up32 uploads v as a dense f32 buffer.
func (bs *webgpuBuild) up32(v []float32) (*wgpu.Buffer, error) {
	d, err := bs.c.UploadF32On(mem.LineDense, v)
	if err != nil {
		return nil, err
	}
	bs.keep(d.Release)
	return (*kernels.DeviceBufferBuf(d)), nil
}

// proj uploads one projection at its native precision (uploadProj).
func (bs *webgpuBuild) proj(pw *linalg.WeightMat) (kernels.DecodeWeight, error) {
	dw, err := uploadProj(bs.c, pw)
	if err != nil {
		return nil, err
	}
	switch t := dw.(type) {
	case *ResidentW4A8:
		bs.keep(t.Release)
	case *ResidentW8A8:
		bs.keep(t.Release)
	}
	return dw, nil
}

// projF32 quantizes a raw f32 [N,K] weight to int8 row-wise and uploads it (the MLA
// projections are stored f32 in the decoder; the resident GEMVs are W8A8).
func (bs *webgpuBuild) projF32(w []float32, N, K int) (kernels.DecodeWeight, error) {
	q8, scales := linalg.QuantizeRowsInt8(w, N, K)
	rm, err := bs.c.UploadW8A8(q8, scales, N, K)
	if err != nil {
		return nil, err
	}
	bs.keep(rm.Release)
	return rm, nil
}

// projInt8 uploads an already-int8 weight matrix (per-row scales) as projF32 uploads its quantized f32.
func (bs *webgpuBuild) projInt8(w *linalg.WeightMat, N, K int) (kernels.DecodeWeight, error) {
	q8, scales, _, ok := w.Int8()
	if !ok || w.Rows() != N || w.Cols() != K {
		return nil, fmt.Errorf("gpu: an int8 projection of %d×%d (int8 %v), want %d×%d", w.Rows(), w.Cols(), ok, N, K)
	}
	rm, err := bs.c.UploadW8A8(q8, scales, N, K)
	if err != nil {
		return nil, err
	}
	bs.keep(rm.Release)
	return rm, nil
}

// Granite multiplier folding: ResidMul scales every residual-add weight (o-proj / mamba out-proj / MoE expert-down /
// shared-down); 1/LogitScale folds into the lm_head; EmbMul applies to the embedding in decoder.embedResident. mult==1 ⇒
// the plain proj path, byte-identical for non-granite models.
func scaleCopy(s []float32, m float32) []float32 {
	out := make([]float32, len(s))
	for i, v := range s {
		out[i] = v * m
	}
	return out
}

// projMul is proj with mult folded into the row scales; mult == 1 is proj itself.
func (bs *webgpuBuild) projMul(pw *linalg.WeightMat, mult float32) (kernels.DecodeWeight, error) {
	if mult == 1 {
		return bs.proj(pw)
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
	rm, err := bs.c.UploadW8A8(q8, scaleCopy(scales, mult), N, K)
	if err != nil {
		return nil, err
	}
	bs.keep(rm.Release)
	return rm, nil
}

// projF32Mul is projF32 with mult folded into the row scales.
func (bs *webgpuBuild) projF32Mul(w []float32, N, K int, mult float32) (kernels.DecodeWeight, error) {
	q8, scales := linalg.QuantizeRowsInt8(w, N, K)
	if mult != 1 {
		scales = scaleCopy(scales, mult)
	}
	rm, err := bs.c.UploadW8A8(q8, scales, N, K)
	if err != nil {
		return nil, err
	}
	bs.keep(rm.Release)
	return rm, nil
}

// stateBuf allocates a build-once, zeroed, in-place-updatable Mamba state buffer
// (Storage|CopyDst so Reset can re-zero it per generation).
func (bs *webgpuBuild) stateBuf(n int) (*wgpu.Buffer, error) {
	b2, err := kernels.CreateBufferOn(bs.c, mem.LineState, &wgpu.BufferDescriptor{Label: "mamba-state", Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst})
	if err != nil {
		return nil, err
	}
	(*kernels.ContextQueue(bs.c)).TryWriteBuffer(b2, 0, wgpu.ToBytes(make([]float32, n)))
	bs.keep(kernels.BufferReleaser(b2))
	return b2, nil
}

// buildStacked packs one projection (gate/up/down) across all nE experts into a resident stacked buffer (int8 or int4; a
// mixed set is an error) that the indexed expert GEMV reads. mult folds Granite's ResidMul into the scales.
func (bs *webgpuBuild) buildStacked(nE int, get func(e int) *linalg.WeightMat, mult float32) (*ResidentStackedW8A8, error) {
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
		return bs.c.UploadStackedExperts(q8, sc, nE, N, K)
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
			return bs.c.UploadStackedExpertsInt4Packed(q4s, scs, nE, N, K)
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
		return bs.c.UploadStackedExpertsInt4(nib, sc, nE, N, K)
	default:
		return nil, fmt.Errorf("gpu: MoE residency expert kind %q (int8/int4 only)", w0.Kind())
	}
}

// ropeInvBuf returns the invFreq table buffer layer binds, nil when the layer has none. Most models share one table (the cache is
// keyed on table equality); a family with local/global RoPE has two. Layer 0 resolves a NoPE or NoPositionEncoding layer (SmolLM3,
// Olmo Hybrid) to the rotaryDim/2 zero table rather than an empty slice.
func (bs *webgpuBuild) ropeInvBuf(layer int) (*wgpu.Buffer, error) {
	invf := bs.m.RopeInvFreqLayerResident(layer)
	if len(invf) == 0 {
		return nil, nil
	}
	if nonzero := bs.m.RopeInvFreq(); noPEOffForTest && len(nonzero) > 0 && !slices.ContainsFunc(invf, func(v float32) bool { return v != 0 }) {
		invf = nonzero
	}
	for _, ent := range bs.ropeInvCache {
		if slices.Equal(ent.invf, invf) {
			return ent.buf, nil
		}
	}
	b, err := bs.up32(invf)
	if err != nil {
		return nil, err
	}
	bs.ropeInvCache = append(bs.ropeInvCache, ropeEntry{invf: invf, buf: b})
	return b, nil
}

// uploadModelLevel is the build's model-level phase: the final norm and LM head, the family parameters (Granite and Nemotron-H
// Mamba, MoE, MLA, Gated-DeltaNet) and the checks that refuse an unsupported norm and MoE combination. It records on bs what the
// per-layer phase reads and returns the first error; the caller releases the partly built decoder.
func (bs *webgpuBuild) uploadModelLevel() error {
	m, w, rd := bs.m, bs.w, bs.rd
	hidden, hd := bs.hidden, bs.hd
	invFreq := m.RopeInvFreqLayerResident(0)
	finalNorm, err := bs.up32(w.FinalNorm)
	if err != nil {
		return err
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
		if nemoZeroInvFreq, ze = bs.up32(make([]float32, hd/2)); ze != nil {
			return ze
		}
	}
	// LM head: tied (LMHead empty → the Embed matrix is the head) or separate.
	headW := &w.LMHead
	if w.LMHead.Rows() == 0 {
		headW = &w.Embed
	}
	lmHead, err := bs.projMul(headW, lmMult)
	if err != nil {
		return err
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
	// double-width q_proj (when dnAttnGate is true; Olmo Hybrid's full-attention layer uses the plain scheme).
	dnConvK, dnHK, dnHV, dnNK, dnNV, dnAttnGate, dnetOK := m.Qwen35ResidentParams()
	var dnZeroConvB *wgpu.Buffer // DeltaNet's causal conv is bias-free; mambaConv binds a bias
	if dnetOK {
		dnBetaScale, dnONormEps, dnErr := deltaNetOpParams(m)
		if dnErr != nil {
			return dnErr
		}
		keyDim, valueDim := dnNK*dnHK, dnNV*dnHV
		rd.rm.Dnet = &kernels.DnetRunParams{
			ConvK: dnConvK, Hk: dnHK, Hv: dnHV, Nk: dnNK, Nv: dnNV, Rep: dnNV / dnNK,
			KeyDim: keyDim, ValueDim: valueDim, ConvDim: 2*keyDim + valueDim,
			StateElems: dnNV * dnHV * dnHK, Eps: dnONormEps, BetaScale: dnBetaScale,
		}
		var ze error
		if dnZeroConvB, ze = bs.up32(make([]float32, rd.rm.Dnet.ConvDim)); ze != nil {
			return ze
		}
	}
	// Gemma's NormSandwich4 (FeatSandwichNorm) and Olmo 3 / Olmo Hybrid's NormPostOnly (FeatPostOnlyNorm): norms on each
	// sublayer OUTPUT, applied before the residual add.
	sandwichOK := m.SandwichNormResident()
	postOnlyOK := m.PostOnlyNormResident()
	qkNormWhole := m.QKNormWholeResident() && !qkNormWholeOffForTest
	rd.rm.QKNormWhole, rd.rm.PostOnlyNorm = qkNormWhole, postOnlyOK
	g4MoE := m.HasGemma4MoEResident() // some layers' feed-forward is Gemma 4's parallel dense and MoE block, which carries its own norms
	if sandwichOK {
		for i := range w.Layers {
			lw := &w.Layers[i]
			_, moeLayer := m.Gemma4MoEResidentLayer(i)
			if len(lw.PostAttnNorm) != hidden || (!moeLayer && len(lw.PostMLPNorm) != hidden) {
				return fmt.Errorf("gpu: layer %d: arch declares sandwich norms but PostAttnNorm/PostMLPNorm are not len==hidden(%d) (got %d/%d)",
					i, hidden, len(lw.PostAttnNorm), len(lw.PostMLPNorm))
			}
		}
	}
	if (sandwichOK || postOnlyOK) && moeOK && !g4MoE {
		// Sandwich / post-only norms + the generic MoE branch is refused: decoderunner.go's MoE FFN branch writes the expert-down output and
		// the shared-expert add straight into r.xd with no postMLPNorm step (the dense FFN branch applies it before the residual
		// add), so admitting the combination would give silently wrong logits. Gemma 4's own MoE block is a different branch with
		// its norms inside (runLayer.g4moe); refuse any other family loudly so the gap cannot ship silently.
		return fmt.Errorf("gpu: sandwich norms (FeatSandwichNorm) + MoE is not supported — " +
			"the MoE FFN dispatch has no postMLPNorm step (N-89, docs/audit-2026-09-10.md)")
	}
	bs.granOK, bs.rmul = granOK, rmul
	bs.nemoOK, bs.nemoZeroInvFreq = nemoOK, nemoZeroInvFreq
	bs.moeOK, bs.shInter, bs.shUngated = moeOK, shInter, shUngated
	bs.mlaOK, bs.qLoRA, bs.kvLoRA, bs.qkNope, bs.qkRope, bs.vHead, bs.mlaAttnScale = mlaOK, qLoRA, kvLoRA, qkNope, qkRope, vHead, mlaAttnScale
	bs.dnetOK, bs.dnAttnGate, bs.dnZeroConvB = dnetOK, dnAttnGate, dnZeroConvB
	bs.sandwichOK, bs.postOnlyOK, bs.qkNormWhole = sandwichOK, postOnlyOK, qkNormWhole
	return nil
}
