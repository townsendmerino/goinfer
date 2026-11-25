//go:build gpu

package kernels

import (
	"fmt"
	"math"
	"slices"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// validateDecodeModel rejects model shapes the resident decode kernels do not implement, so the caller falls back to the
// staged/CPU path instead of emitting wrong output or hitting shader limits.
func validateDecodeModel(m RunModel, hd int) error {
	// MLA is exempt from attnHeadDimSupported: its attention runs the mlaAttn kernel family with its own rank-bounded
	// accumulator, held to the per-lane rank cap MLAAttn itself checks (rank <= 1024).
	if m.Mla != nil {
		if m.Mla.KvLoRARank > 1024 {
			return fmt.Errorf("gpu: newDecodeRunner: MLA kv-LoRA rank %d exceeds the resident per-lane cap 1024; declining to CPU", m.Mla.KvLoRARank)
		}
	} else if err := attnHeadDimSupported(hd, m.Layers); err != nil {
		return err
	}
	// The mamba causal-conv kernel holds its window in a fixed `array<f32, 8>` indexed by conv_kernel-1.
	if m.Mamba != nil && (m.Mamba.DConv > 8 || m.Mamba.DConv < 1) {
		return fmt.Errorf("gpu: newDecodeRunner: mamba conv_kernel %d out of range [1,8] for the resident conv kernel; declining to CPU", m.Mamba.DConv)
	}
	if (m.LayerNorm || m.Parallel) && (m.Moe != nil || m.G4moe || m.Mla != nil || m.Mamba != nil || m.Dnet != nil || m.PleDim > 0 || m.PostOnlyNorm) {
		return fmt.Errorf("gpu: newDecodeRunner: LayerNorm and the parallel block are implemented on the dense attention path only")
	}
	// LayerNorm reaches only the norms rmsQuant builds (every pre-norm and the final norm). The unfused ones (QK-norm, V-norm,
	// the post-sublayer norms) bind RMSNorm kernels, so a LayerNorm model carrying any of them is refused.
	if m.LayerNorm && (m.QKNormWhole || slices.ContainsFunc(m.Layers, func(l RunLayer) bool {
		return l.QNorm != nil || l.KNorm != nil || l.VNorm || l.GKEqV || l.PostAttnNorm != nil || l.PostMLPNorm != nil || l.PostPLENorm != nil
	})) {
		return fmt.Errorf("gpu: newDecodeRunner: a LayerNorm model's QK-, V- or post-sublayer norms would run as RMSNorm; only its pre-norms and final norm take LayerNorm")
	}
	if m.AttnTemp != nil && m.Mla != nil {
		return fmt.Errorf("gpu: newDecodeRunner: the attention temperature on the MLA path is not implemented")
	}
	if m.ActG32 {
		switch {
		case m.Moe != nil || m.G4moe:
			return fmt.Errorf("gpu: newDecodeRunner: per-32 activations: the MoE kernels read per-vector activation scales")
		case m.LayerNorm:
			return fmt.Errorf("gpu: newDecodeRunner: per-32 activations: LayerNorm has no per-32 quantize twin")
		case m.Mla != nil || m.Mamba != nil || m.Dnet != nil || m.PleDim > 0:
			return fmt.Errorf("gpu: newDecodeRunner: per-32 activations: the MLA, SSM, DeltaNet and per-layer-embedding paths are not implemented")
		}
	}
	if m.Pairwise && m.Mla != nil {
		return fmt.Errorf("gpu: newDecodeRunner: pairwise RoPE on the MLA path is not implemented")
	}
	return nil
}

// ensureDecodePipelines compiles every pipeline m's plan will bind.
func (c *Context) ensureDecodePipelines(m RunModel, hd int) error {
	ensures := []func() error{c.ensureGEMV, c.ensureGEMVBias, c.ensureQuantize, c.ensureLayer, c.ensureAttn, c.ensureFuse, c.ensureGEMVW4, c.ensureQKNorm, c.ensureLora}
	if m.Moe != nil {
		ensures = append(ensures, c.ensureMoERoute, c.ensureMoEExpert, c.ensureMoEExpertW4)
		if m.Moe.SharedInter > 0 && !m.Moe.SharedUngated {
			ensures = append(ensures, c.ensureSharedGate)
		}
		if m.Moe.Gptoss {
			ensures = append(ensures, c.ensureRouteGptOss, c.ensureGptOssGluQuant, c.ensureMoEExpertGptOssDown, c.ensureMoEExpertGptOssDownW4)
		}
	}
	if m.G4moe {
		ensures = append(ensures, c.ensureMoERoute, c.ensureMoEExpert, c.ensureMoEExpertW4, c.ensureGemma4MoE)
	}
	if m.Mla != nil {
		ensures = append(ensures, c.ensureMLAStore, c.ensureMLAHeadMV, c.ensureMLAQRope, c.ensureMLAAttn)
	}
	if m.Mamba != nil {
		ensures = append(ensures, c.ensureMambaConv, c.ensureMambaSSM, c.ensureMambaGNorm, c.ensureRelu2)
	}
	if hd > AttnWG || slices.ContainsFunc(m.Layers, func(l RunLayer) bool { return l.Ghd > AttnWG }) {
		// Only for a plan that needs it. A device that cannot do 256-invocation workgroups errors here, which the caller turns
		// into a staged fallback.
		ensures = append(ensures, c.ensureAttnWide)
	}
	if slices.ContainsFunc(m.Layers, func(l RunLayer) bool { return l.GKEqV || l.VNorm }) {
		ensures = append(ensures, c.ensureVNorm)
	}
	if slices.ContainsFunc(m.Layers, func(l RunLayer) bool { return l.LayerScalar != 0 && l.LayerScalar != 1.0 }) {
		ensures = append(ensures, c.ensureScaleVec)
	}
	if m.Dnet != nil {
		// mambaConv is shared with the SSM engine (DeltaNet's causal conv is the same op); the
		// other four are DeltaNet's own. deltaQSplit/deltaAttnGate serve the family's SOFTMAX
		// layers, which are part of the same admission and so are compiled unconditionally with it.
		ensures = append(ensures, c.ensureMambaConv, c.ensureDeltaRule, c.ensureDeltaNorm,
			c.ensureDeltaGates, c.ensureDeltaGNorm, c.ensureDeltaQSplit, c.ensureDeltaAttnGate)
	}
	if m.LayerNorm {
		ensures = append(ensures, c.ensureLayerNorm)
	}
	if m.ActG32 {
		ensures = append(ensures, c.ensureActG32)
	}
	if m.Pairwise {
		ensures = append(ensures, c.ensurePairwise)
	}
	for _, e := range ensures {
		if err := e(); err != nil {
			return err
		}
	}
	return nil
}

// decodeRunnerBuild holds the transient builder state for Context.newDecodeRunner while it allocates scratch buffers,
// bind groups, and per-token uniforms and records the fixed dispatch sequence into r.
type decodeRunnerBuild struct {
	c *Context
	m RunModel
	r *DecodeRunner

	hidden, nH, nKV, hd, inter, start int
	eps, scale                        float32
	addOne                            bool
	half                              int
	tempOff                           bool
	captureOn                         bool

	// buildErr holds the FIRST device-allocation/bind failure: storF/uni/bind short-circuit once set and finish returns it,
	// so VRAM exhaustion is an error the caller can fall back on, never a panic in library code.
	buildErr error
	// role is the op builder or build step the scratch leases belong to (ScratchList).
	role string

	noAttnSinks *wgpu.Buffer
	moeZeroBias *wgpu.Buffer
	geomCache   map[[4]int]*attnGeom
	rp          pairwisePipes

	// Mamba-2 shared uniforms (pos-independent).
	mambaDC, mambaDS, mambaDG *wgpu.Buffer
	// DeltaNet shared uniforms (pos-independent).
	dnDC, dnDGate, dnDNorm, dnDRule, dnDGN *wgpu.Buffer
	// MLA shared uniforms (pos-dependent; registered on r.posUnis in initMLA).
	mlaStoreUni, mlaQRopeUni, mlaAttnUni *wgpu.Buffer
}

type bgEnt struct {
	b         *wgpu.Buffer
	off, size uint64
}

func (b *decodeRunnerBuild) keepBuf(buf *wgpu.Buffer) *wgpu.Buffer {
	if buf != nil {
		b.r.keep = append(b.r.keep, releaser(buf))
	}
	return buf
}

func (b *decodeRunnerBuild) keepBG(bg *wgpu.BindGroup) *wgpu.BindGroup {
	b.r.keep = append(b.r.keep, bg.Release)
	return bg
}

// inRole names the current scratch role and returns a closure that restores the enclosing one, so a builder calling
// another is not charged for the callee's leases.
func (b *decodeRunnerBuild) inRole(name string) func() {
	outer := b.role
	b.role = name
	return func() { b.role = outer }
}

func (b *decodeRunnerBuild) storF(n int) *wgpu.Buffer {
	if b.buildErr != nil {
		return nil
	}
	buf, e := b.c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if e != nil {
		b.buildErr = e
		return nil
	}
	b.r.scratch.add(b.role, int64(buf.GetSize()))
	return b.keepBuf(buf)
}

func (b *decodeRunnerBuild) uni(v []uint32) *wgpu.Buffer {
	if b.buildErr != nil {
		return nil
	}
	buf, e := b.c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(v), Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst})
	if e != nil {
		b.buildErr = e
		return nil
	}
	b.r.scratch.add(b.role, int64(buf.GetSize()))
	return b.keepBuf(buf)
}

func (b *decodeRunnerBuild) bind(layout *wgpu.BindGroupLayout, bufs ...*wgpu.Buffer) *wgpu.BindGroup {
	if b.buildErr != nil {
		return nil
	}
	es := make([]wgpu.BindGroupEntry, len(bufs))
	for i, buf := range bufs {
		if buf == nil { // an upstream storF/uni failed and short-circuited to nil
			b.buildErr = fmt.Errorf("gpu: newDecodeRunner: nil buffer for binding %d (allocation failed)", i)
			return nil
		}
		es[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: buf, Size: buf.GetSize()}
	}
	bg, e := b.c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: es})
	if e != nil {
		b.buildErr = e
		return nil
	}
	return b.keepBG(bg)
}

func (b *decodeRunnerBuild) bindOff(layout *wgpu.BindGroupLayout, es []bgEnt) *wgpu.BindGroup {
	if b.buildErr != nil {
		return nil
	}
	en := make([]wgpu.BindGroupEntry, len(es))
	for i, e := range es {
		if e.b == nil { // an upstream allocation failed and short-circuited to nil
			b.buildErr = fmt.Errorf("gpu: newDecodeRunner: nil buffer for binding %d (allocation failed)", i)
			return nil
		}
		sz := e.size
		if sz == 0 {
			sz = e.b.GetSize()
		}
		en[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: e.b, Offset: e.off, Size: sz}
	}
	bg, e := b.c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: en})
	if e != nil {
		b.buildErr = e
		return nil
	}
	return b.keepBG(bg)
}

func (b *decodeRunnerBuild) add(pl *wgpu.ComputePipeline, bg *wgpu.BindGroup, gx, gy uint32) {
	b.r.steps = append(b.r.steps, RunStep{Pl: pl, Bg: bg, Gx: gx, Gy: gy})
}

// scaleBuf is a quantizer's scale buffer for a kp-wide activation: one float, or kp/32 under ActG32.
func (b *decodeRunnerBuild) scaleBuf(kp int) *wgpu.Buffer {
	if b.m.ActG32 {
		return b.storF(kp / 32)
	}
	return b.storF(1)
}

// --- core op builders ---

// rmsQuant fuses RMSNorm→quantize: one dispatch, no xn round-trip, one fewer link on the serialized decode spine.
func (b *decodeRunnerBuild) rmsQuant(in, w *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
	defer b.inRole("rmsQuant")()
	kp := PadK32(K)
	q, s := b.storF(kp/4), b.scaleBuf(kp)
	p := b.uni([]uint32{uint32(K), F32bits(b.eps), BoolU32(b.addOne), uint32(kp)})
	pl, ly := b.c.rmsQuantPipeline, b.c.rmsQuantLayout
	if b.m.LayerNorm && !LayerNormAsRMSForTest {
		pl, ly = b.c.layerNormQuant.pl, b.c.layerNormQuant.ly
	} else if b.m.ActG32 {
		pl, ly = b.c.actG32.rmsQuant.pl, b.c.actG32.rmsQuant.ly
	}
	b.add(pl, b.bind(ly, in, w, q, s, p), 1, 1)
	return q, s
}

func (b *decodeRunnerBuild) gatedQuantPipe(gelu bool) (*wgpu.ComputePipeline, *wgpu.BindGroupLayout) {
	switch {
	case b.m.ActG32 && gelu:
		return b.c.actG32.gegluQuant.pl, b.c.actG32.gegluQuant.ly
	case b.m.ActG32:
		return b.c.actG32.swigluQuant.pl, b.c.actG32.swigluQuant.ly
	case gelu:
		return b.c.gegluQuantPipeline, b.c.gegluQuantLayout
	}
	return b.c.swigluQuantPipeline, b.c.swigluQuantLayout
}

func (b *decodeRunnerBuild) actQuantPad(gate, up *wgpu.Buffer, K, kp int) (*wgpu.Buffer, *wgpu.Buffer) {
	defer b.inRole("actQuantPad")()
	q, s := b.storF(kp/4), b.scaleBuf(kp)
	p := b.uni([]uint32{uint32(K), uint32(kp), 0, 0})
	pl, ly := b.gatedQuantPipe(b.m.GatedGELU)
	b.add(pl, b.bind(ly, gate, up, q, s, p), 1, 1)
	return q, s
}

func (b *decodeRunnerBuild) swigluQuant(gate, up *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
	defer b.inRole("swigluQuant")()
	kp := PadK32(K)
	q, s := b.storF(kp/4), b.scaleBuf(kp)
	p := b.uni([]uint32{uint32(K), uint32(kp), 0, 0})
	pl, ly := b.gatedQuantPipe(b.m.GatedGELU)
	b.add(pl, b.bind(ly, gate, up, q, s, p), 1, 1)
	return q, s
}

// relu2Quant fuses Nemotron-H's non-gated relu²(up)→int8 (the squared-ReLU MLP).
func (b *decodeRunnerBuild) relu2Quant(up *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
	defer b.inRole("relu2Quant")()
	kp := PadK32(K)
	q, s := b.storF(kp/4), b.storF(1)
	p := b.uni([]uint32{uint32(K), uint32(kp), 0, 0})
	b.add(b.c.relu2Pipeline, b.bind(b.c.relu2Layout, up, q, s, p), 1, 1)
	return q, s
}

func (b *decodeRunnerBuild) quant(in *wgpu.Buffer, K int) (*wgpu.Buffer, *wgpu.Buffer) {
	defer b.inRole("quant")()
	kp := PadK32(K)
	q, s := b.storF(kp/4), b.scaleBuf(kp)
	p := b.uni([]uint32{1, uint32(K), uint32(kp), 0})
	pl, ly := b.c.quantizePipeline, b.c.quantizeLayout
	if b.m.ActG32 {
		pl, ly = b.c.actG32.quantize.pl, b.c.actG32.quantize.ly
	}
	b.add(pl, b.bind(ly, in, q, s, p), 1, 1)
	return q, s
}

func (b *decodeRunnerBuild) gemvPipe(w DecodeWeight, bias bool) (*wgpu.ComputePipeline, *wgpu.BindGroupLayout) {
	if !b.m.ActG32 {
		if bias {
			return b.c.gemvBiasPipeline, b.c.gemvBiasLayout
		}
		return w.gPipe(b.c), w.gLayout(b.c)
	}
	if _, ok := w.(*ResidentW8A8); !ok || w.kPad()%32 != 0 {
		if b.buildErr == nil {
			b.buildErr = fmt.Errorf("gpu: newDecodeRunner: per-32 activations need int8 weights over whole 32-wide groups (a %d-wide %T)", w.kPad(), w)
		}
		return nil, nil
	}
	if bias {
		return b.c.actG32.gemvBias.pl, b.c.actG32.gemvBias.ly
	}
	return b.c.actG32.gemv.pl, b.c.actG32.gemv.ly
}

func (b *decodeRunnerBuild) gemv(aq, as *wgpu.Buffer, w DecodeWeight) *wgpu.Buffer {
	defer b.inRole("gemv")()
	out := b.storF(w.nRows())
	p := b.uni([]uint32{1, uint32(w.kPad()), uint32(w.nRows()), 0})
	gx, gy := GemvGrid(w.nRows())
	pl, ly := b.gemvPipe(w, false)
	b.add(pl, b.bind(ly, aq, w.wbuf(), as, w.sbuf(), out, p), gx, gy)
	return out
}

func (b *decodeRunnerBuild) gemvAdd(aq, as *wgpu.Buffer, w DecodeWeight, dst *wgpu.Buffer) {
	defer b.inRole("gemvAdd")()
	p := b.uni([]uint32{1, uint32(w.kPad()), uint32(w.nRows()), 1})
	gx, gy := GemvGrid(w.nRows())
	pl, ly := b.gemvPipe(w, false)
	b.add(pl, b.bind(ly, aq, w.wbuf(), as, w.sbuf(), dst, p), gx, gy)
}

func (b *decodeRunnerBuild) gemvBias(aq, as *wgpu.Buffer, w DecodeWeight, bias *wgpu.Buffer) *wgpu.Buffer {
	defer b.inRole("gemvBias")()
	out := b.storF(w.nRows())
	p := b.uni([]uint32{1, uint32(w.kPad()), uint32(w.nRows()), 0})
	gx, gy := GemvGrid(w.nRows())
	pl, ly := b.gemvPipe(w, true)
	b.add(pl, b.bind(ly, aq, w.wbuf(), as, w.sbuf(), out, p, bias), gx, gy)
	return out
}

func (b *decodeRunnerBuild) qSplit(qg *wgpu.Buffer, n, headDim int) (*wgpu.Buffer, *wgpu.Buffer) {
	defer b.inRole("qSplit")()
	q, gate := b.storF(n), b.storF(n)
	p := b.uni([]uint32{uint32(n), uint32(headDim), 0, 0})
	b.add(b.c.deltaQSplitPipeline, b.bind(b.c.deltaQSplitLayout, qg, q, gate, p), uint32(n+63)/64, 1)
	return q, gate
}

func (b *decodeRunnerBuild) attnGate(ctxv, gate *wgpu.Buffer, n int) {
	defer b.inRole("attnGate")()
	p := b.uni([]uint32{uint32(n), 0, 0, 0})
	b.add(b.c.deltaAttnGatePipeline, b.bind(b.c.deltaAttnGateLayout, ctxv, gate, p), uint32(n+63)/64, 1)
}

func (b *decodeRunnerBuild) rmsnormF32(in, weight *wgpu.Buffer, n int) *wgpu.Buffer {
	defer b.inRole("rmsnormF32")()
	out := b.storF(n)
	p := b.uni([]uint32{uint32(n), F32bits(b.eps), BoolU32(b.addOne), 0})
	b.add(b.c.rmsnormPipeline, b.bind(b.c.rmsnormLayout, in, weight, out, p), 1, 1)
	return out
}

func (b *decodeRunnerBuild) biasAdd(vec, bias *wgpu.Buffer, n int) {
	defer b.inRole("biasAdd")()
	p := b.uni([]uint32{uint32(n), 0, 0, 0})
	b.add(b.c.residualPipeline, b.bind(b.c.residualLayout, vec, bias, p), uint32(n+63)/64, 1)
}

func (b *decodeRunnerBuild) qkNorm(vec, weight *wgpu.Buffer, heads, lhd int) {
	defer b.inRole("qkNorm")()
	p := b.uni([]uint32{uint32(heads), uint32(lhd), F32bits(b.eps), BoolU32(b.addOne)})
	b.add(b.c.qkNormPipeline, b.bind(b.c.qkNormLayout, vec, weight, p), uint32(heads), 1)
}

func (b *decodeRunnerBuild) vNorm(src, dst *wgpu.Buffer, heads, lhd int) {
	defer b.inRole("vNorm")()
	p := b.uni([]uint32{uint32(heads), uint32(lhd), F32bits(b.eps), 0})
	b.add(b.c.vNormPipeline, b.bind(b.c.vNormLayout, src, dst, p), uint32(heads), 1)
}

func (b *decodeRunnerBuild) scaleVec(vec *wgpu.Buffer, scale float32, n int) {
	defer b.inRole("scaleVec")()
	p := b.uni([]uint32{uint32(n), F32bits(scale), 0, 0})
	b.add(b.c.scaleVecPipeline, b.bind(b.c.scaleVecLayout, vec, p), uint32(n+255)/256, 1)
}

func (b *decodeRunnerBuild) capStep(src *wgpu.Buffer, n int) *wgpu.Buffer {
	defer b.inRole("capStep")()
	dst := b.storF(n)
	p := b.uni([]uint32{uint32(n), 0, 0, 0})
	b.add(b.c.kvStorePipeline, b.bind(b.c.kvStoreLayout, src, dst, p), uint32(n+63)/64, 1)
	return dst
}

// --- mixer setup & per-geometry RoPE/KV uniforms ---

func (b *decodeRunnerBuild) initMamba() {
	if b.m.Mamba == nil {
		return
	}
	mp := b.m.Mamba
	b.role = "mambaSetup"
	b.mambaDC = b.uni([]uint32{uint32(mp.ConvDim), uint32(mp.DConv), uint32(mp.DInner), 0})
	b.mambaDS = b.uni([]uint32{uint32(mp.NHeads), uint32(mp.Hp), uint32(mp.Dn), uint32(mp.NGroups), uint32(mp.Repeat), uint32(mp.GSize), uint32(mp.DInner), uint32(mp.DInner + mp.ConvDim)})
	b.mambaDG = b.uni([]uint32{uint32(mp.DInner), uint32(mp.NormGroups), uint32(mp.DInner / mp.NormGroups), F32bits(b.eps), 0, 0, 0, 0})
}

func (b *decodeRunnerBuild) initDeltaNet() {
	if b.m.Dnet == nil {
		return
	}
	dp := b.m.Dnet
	betaScale := dp.BetaScale
	if betaScale == 0 {
		betaScale = 1
	}
	b.role = "deltaSetup"
	b.dnDC = b.uni([]uint32{uint32(dp.ConvDim), uint32(dp.ConvK), 0, 0})
	b.dnDGate = b.uni([]uint32{uint32(dp.Nv), F32bits(betaScale), 0, 0})
	b.dnDNorm = b.uni([]uint32{uint32(dp.Nk), uint32(dp.Hk), uint32(dp.KeyDim), 0, F32bits(float32(1 / math.Sqrt(float64(dp.Hk)))), 0, 0, 0})
	b.dnDRule = b.uni([]uint32{uint32(dp.Nv), uint32(dp.Nk), uint32(dp.Hk), uint32(dp.Hv), uint32(dp.Rep), uint32(2 * dp.KeyDim), 0, 0})
	b.dnDGN = b.uni([]uint32{uint32(dp.Nv), uint32(dp.Hv), 0, 0, F32bits(dp.Eps), 0, 0, 0})
}

func (b *decodeRunnerBuild) initRoPE() {
	c := b.c
	b.rp = pairwisePipes{
		rope: pairwisePipe{c.ropePipeline, c.ropeLayout}, ropeStore: pairwisePipe{c.ropeStorePipeline, c.ropeStoreLayout},
		qkvFin: pairwisePipe{c.qkvFinPipeline, c.qkvFinLayout}, ropeStoreF16: pairwisePipe{c.ropeStoreF16Pipeline, c.ropeStoreF16Layout},
		ropeStoreI8: pairwisePipe{c.ropeStoreI8Pipeline, c.ropeStoreI8Layout},
	}
	if b.m.Pairwise && !PairwiseNeoXForTest {
		neox := b.rp
		b.rp = *c.pairwise
		if PairwiseNeoXStoresForTest {
			b.rp.ropeStore, b.rp.ropeStoreF16, b.rp.ropeStoreI8 = neox.ropeStore, neox.ropeStoreF16, neox.ropeStoreI8
		}
	}
}

func (b *decodeRunnerBuild) initMLA() {
	if b.m.Mla == nil {
		return
	}
	mp := b.m.Mla
	qkHead := mp.QkNope + mp.QkRope
	latDim := mp.KvLoRARank + mp.QkRope
	rank := mp.KvLoRARank
	b.role = "mlaSetup"
	b.mlaStoreUni = b.uni([]uint32{uint32(rank), uint32(mp.QkRope), 0, F32bits(b.eps), 0, F32bits(mp.RopeScale), BoolU32(mp.Interleave), 0})
	b.r.posUnis = append(b.r.posUnis, posUni{buf: b.mlaStoreUni, fill: func(dst *[16]uint32, pos, ropePos int) int {
		dst[0] = uint32(rank)
		dst[1] = uint32(mp.QkRope)
		dst[2] = uint32(pos)
		dst[3] = F32bits(b.eps)
		dst[4] = uint32(pos * latDim)
		dst[5] = F32bits(mp.RopeScale)
		dst[6] = BoolU32(mp.Interleave)
		dst[7] = 0
		return 8
	}})
	b.mlaQRopeUni = b.uni([]uint32{uint32(b.nH), uint32(qkHead), uint32(mp.QkNope), uint32(mp.QkRope), uint32(rank), uint32(latDim), 0, BoolU32(mp.Interleave), F32bits(mp.RopeScale), 0, 0, 0})
	b.r.posUnis = append(b.r.posUnis, posUni{buf: b.mlaQRopeUni, fill: func(dst *[16]uint32, pos, ropePos int) int {
		dst[0] = uint32(b.nH)
		dst[1] = uint32(qkHead)
		dst[2] = uint32(mp.QkNope)
		dst[3] = uint32(mp.QkRope)
		dst[4] = uint32(rank)
		dst[5] = uint32(latDim)
		dst[6] = uint32(pos)
		dst[7] = BoolU32(mp.Interleave)
		dst[8] = F32bits(mp.RopeScale)
		dst[9] = 0
		dst[10] = 0
		dst[11] = 0
		return 12
	}})
	b.mlaAttnUni = b.uni([]uint32{uint32(b.nH), uint32(latDim), uint32(rank), 0, F32bits(b.scale), 0, 0, 0})
	b.r.posUnis = append(b.r.posUnis, posUni{buf: b.mlaAttnUni, fill: func(dst *[16]uint32, pos, ropePos int) int {
		dst[0] = uint32(b.nH)
		dst[1] = uint32(latDim)
		dst[2] = uint32(rank)
		dst[3] = uint32(pos + 1)
		dst[4] = F32bits(b.scale)
		dst[5] = 0
		dst[6] = 0
		dst[7] = 0
		return 8
	}})
}

func (b *decodeRunnerBuild) allocResidualAndPLE() {
	if b.buildErr == nil {
		buf, e := b.c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Size: uint64(b.hidden * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc})
		if e != nil {
			b.buildErr = e
		} else {
			b.r.scratch.add("residual", int64(buf.GetSize()))
			b.r.xd = b.keepBuf(buf)
		}
	}
	if b.m.PleDim > 0 {
		b.r.pleIn = make([]*wgpu.Buffer, len(b.m.Layers))
		for i := range b.r.pleIn {
			if b.buildErr != nil {
				break
			}
			buf, e := b.c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Size: uint64(b.m.PleDim * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc})
			if e != nil {
				b.buildErr = e
				break
			}
			b.r.scratch.add("pleInput", int64(buf.GetSize()))
			b.r.pleIn[i] = b.keepBuf(buf)
		}
	}
}

func (b *decodeRunnerBuild) scoreScale(pos int) float32 {
	if b.m.AttnTemp == nil || b.tempOff {
		return b.scale
	}
	return b.scale * b.m.AttnTemp(pos)
}

func (b *decodeRunnerBuild) geomFor(ghd, gnKV, ghalf int, kEqV bool) *attnGeom {
	defer b.inRole("geomFor")()
	kb := 0
	if kEqV {
		kb = 1
	}
	key := [4]int{ghd, gnKV, ghalf, kb}
	if g, ok := b.geomCache[key]; ok {
		return g
	}
	g := &attnGeom{
		hd: ghd, nKV: gnKV, half: ghalf, kvDim: gnKV * ghd, kEqV: kEqV,
		ropeQUnis:  map[float32]*wgpu.Buffer{},
		ropeKUnis:  map[float32]*wgpu.Buffer{},
		qkvFinUnis: map[float32]*wgpu.Buffer{},
	}
	g.vStoreUni = b.uni([]uint32{uint32(g.kvDim), 0, 0, 0})
	b.r.posUnis = append(b.r.posUnis, posUni{buf: g.vStoreUni, fill: func(dst *[16]uint32, pos, ropePos int) int {
		dst[0] = uint32(g.kvDim)
		dst[1] = uint32(pos * g.kvDim)
		dst[2] = 0
		dst[3] = 0
		return 4
	}})
	if b.m.KvI8 {
		g.vStoreI8Uni = b.uni([]uint32{uint32(g.nKV), uint32(g.hd), 0, 0, uint32(g.nKV), 0, 0, 0})
		b.r.posUnis = append(b.r.posUnis, posUni{buf: g.vStoreI8Uni, fill: func(dst *[16]uint32, pos, ropePos int) int {
			dst[0] = uint32(g.nKV)
			dst[1] = uint32(g.hd)
			dst[2] = uint32(pos * g.kvDim)
			dst[3] = uint32(pos)
			dst[4] = uint32(g.nKV)
			dst[5] = 0
			dst[6] = 0
			dst[7] = 0
			return 8
		}})
	}
	nH, start, scale := b.nH, b.start, b.scale
	g.attnUni = b.uni([]uint32{uint32(nH), uint32(g.nKV), uint32(g.hd), 0, uint32(start), uint32(nH / g.nKV), F32bits(scale), 0})
	b.r.posUnis = append(b.r.posUnis, posUni{buf: g.attnUni, fill: func(dst *[16]uint32, pos, ropePos int) int {
		dst[0] = uint32(nH)
		dst[1] = uint32(g.nKV)
		dst[2] = uint32(g.hd)
		dst[3] = uint32(pos + 1)
		dst[4] = uint32(start)
		dst[5] = uint32(nH / g.nKV)
		dst[6] = F32bits(b.scoreScale(pos))
		dst[7] = 0
		return 8
	}})
	g.attnUniLocal = g.attnUni
	if b.m.SlidingWindow > 0 {
		w := b.m.SlidingWindow
		g.attnUniLocal = b.uni([]uint32{uint32(nH), uint32(g.nKV), uint32(g.hd), 0, uint32(start), uint32(nH / g.nKV), F32bits(scale), 0})
		b.r.posUnis = append(b.r.posUnis, posUni{buf: g.attnUniLocal, fill: func(dst *[16]uint32, pos, ropePos int) int {
			ws := start
			if lo := pos + 1 - w; lo > ws {
				ws = lo
			}
			dst[0] = uint32(nH)
			dst[1] = uint32(g.nKV)
			dst[2] = uint32(g.hd)
			dst[3] = uint32(pos + 1)
			dst[4] = uint32(ws)
			dst[5] = uint32(nH / g.nKV)
			dst[6] = F32bits(b.scoreScale(pos))
			dst[7] = 0
			return 8
		}})
	}
	b.geomCache[key] = g
	return g
}

func (b *decodeRunnerBuild) ropeQUniFor(g *attnGeom, rs float32) *wgpu.Buffer {
	defer b.inRole("ropeQUniFor")()
	if buf, ok := g.ropeQUnis[rs]; ok {
		return buf
	}
	nH := b.nH
	buf := b.uni([]uint32{uint32(nH), uint32(g.hd), uint32(g.half), 0, F32bits(rs), 0, 0, 0})
	b.r.posUnis = append(b.r.posUnis, posUni{buf: buf, fill: func(dst *[16]uint32, pos, ropePos int) int {
		dst[0] = uint32(nH)
		dst[1] = uint32(g.hd)
		dst[2] = uint32(g.half)
		dst[3] = uint32(ropePos)
		dst[4] = F32bits(rs)
		dst[5] = 0
		dst[6] = 0
		dst[7] = 0
		return 8
	}})
	g.ropeQUnis[rs] = buf
	return buf
}

func (b *decodeRunnerBuild) ropeKUniFor(g *attnGeom, rs float32) *wgpu.Buffer {
	defer b.inRole("ropeKUniFor")()
	if buf, ok := g.ropeKUnis[rs]; ok {
		return buf
	}
	buf := b.uni([]uint32{uint32(g.nKV), uint32(g.hd), uint32(g.half), 0, F32bits(rs), 0, uint32(g.nKV), 0})
	b.r.posUnis = append(b.r.posUnis, posUni{buf: buf, fill: func(dst *[16]uint32, pos, ropePos int) int {
		dst[0] = uint32(g.nKV)
		dst[1] = uint32(g.hd)
		dst[2] = uint32(g.half)
		dst[3] = uint32(ropePos)
		dst[4] = F32bits(rs)
		dst[5] = uint32(pos * g.kvDim)
		dst[6] = uint32(g.nKV)
		dst[7] = uint32(pos)
		return 8
	}})
	g.ropeKUnis[rs] = buf
	return buf
}

func (b *decodeRunnerBuild) qkvFinUniFor(g *attnGeom, rs float32) *wgpu.Buffer {
	defer b.inRole("qkvFinUniFor")()
	if buf, ok := g.qkvFinUnis[rs]; ok {
		return buf
	}
	nH := b.nH
	buf := b.uni([]uint32{uint32(nH), uint32(g.nKV), uint32(g.hd), uint32(g.half), 0, 0, F32bits(rs), uint32(g.kvDim)})
	b.r.posUnis = append(b.r.posUnis, posUni{buf: buf, fill: func(dst *[16]uint32, pos, ropePos int) int {
		dst[0] = uint32(nH)
		dst[1] = uint32(g.nKV)
		dst[2] = uint32(g.hd)
		dst[3] = uint32(g.half)
		dst[4] = uint32(ropePos)
		dst[5] = uint32(pos * g.kvDim)
		dst[6] = F32bits(rs)
		dst[7] = uint32(g.kvDim)
		return 8
	}})
	g.qkvFinUnis[rs] = buf
	return buf
}

func (b *decodeRunnerBuild) rope(g *attnGeom, vec, invFreq *wgpu.Buffer, ropeScale float32) {
	if ropeScale == 0 {
		ropeScale = 1
	}
	b.add(b.rp.rope.pl, b.bind(b.rp.rope.ly, vec, invFreq, b.ropeQUniFor(g, ropeScale)), uint32(b.nH*g.half+63)/64, 1)
}

func (b *decodeRunnerBuild) ropeStore(g *attnGeom, src, invFreq, cache, scale *wgpu.Buffer, ropeScale float32) {
	if ropeScale == 0 {
		ropeScale = 1
	}
	ku := b.ropeKUniFor(g, ropeScale)
	if b.m.KvI8 {
		b.add(b.rp.ropeStoreI8.pl, b.bind(b.rp.ropeStoreI8.ly, src, invFreq, cache, scale, ku), uint32(g.nKV+63)/64, 1)
		return
	}
	if b.m.KvF16 {
		b.add(b.rp.ropeStoreF16.pl, b.bind(b.rp.ropeStoreF16.ly, src, invFreq, cache, ku), uint32(g.nKV*g.hd/2+63)/64, 1)
	} else {
		b.add(b.rp.ropeStore.pl, b.bind(b.rp.ropeStore.ly, src, invFreq, cache, ku), uint32(g.nKV*(g.hd-g.half)+63)/64, 1)
	}
}

func (b *decodeRunnerBuild) vStore(g *attnGeom, src, cache, scale *wgpu.Buffer) {
	if b.m.KvI8 {
		b.add(b.c.kvStoreI8Pipeline, b.bind(b.c.kvStoreI8Layout, src, cache, scale, g.vStoreI8Uni), uint32(g.nKV+63)/64, 1)
		return
	}
	if b.m.KvF16 {
		words := g.kvDim / 2
		b.add(b.c.kvStoreF16Pipeline, b.bind(b.c.kvStoreF16Layout, src, cache, g.vStoreUni), uint32(words+63)/64, 1)
		return
	}
	b.add(b.c.kvStorePipeline, b.bind(b.c.kvStoreLayout, src, cache, g.vStoreUni), uint32(g.kvDim+63)/64, 1)
}

func (b *decodeRunnerBuild) qkvFinalize(g *attnGeom, q, k, v, invFreq, kCache, vCache *wgpu.Buffer, ropeScale float32) {
	if ropeScale == 0 {
		ropeScale = 1
	}
	n := max(g.kvDim, b.nH*g.half)
	b.add(b.rp.qkvFin.pl, b.bind(b.rp.qkvFin.ly, q, k, v, invFreq, kCache, vCache, b.qkvFinUniFor(g, ropeScale)), uint32(n+63)/64, 1)
}

// --- MoE op helpers ---

func (b *decodeRunnerBuild) moeRoute(logits, bias, idx, wgt *wgpu.Buffer, hasBias bool) {
	defer b.inRole("moeRoute")()
	mp := b.m.Moe
	p := b.uni([]uint32{uint32(mp.NE), uint32(mp.K), BoolU32(mp.Sigmoid), BoolU32(mp.Norm), F32bits(mp.Scale), BoolU32(hasBias), uint32(mp.NGroup), uint32(mp.TopkGroup)})
	b.add(b.c.moeRoutePipeline, b.bind(b.c.moeRouteLayout, logits, bias, idx, wgt, p), 1, 1)
}

func (b *decodeRunnerBuild) moeExpert(aq, as *wgpu.Buffer, s *ResidentStackedW8A8, idx, wgt, dst *wgpu.Buffer, slot, mode int) {
	defer b.inRole("moeExpert")()
	d := b.uni([]uint32{uint32(s.kp), uint32(s.rows), uint32(slot), uint32(mode)})
	gx, gy := GemvGrid(s.rows)
	if s.w4 {
		b.add(b.c.moeExpertW4Pipeline, b.bind(b.c.moeExpertW4Layout, aq, s.bq, as, s.bScales, dst, idx, wgt, d), gx, gy)
		return
	}
	b.add(b.c.moeExpertPipeline, b.bind(b.c.moeExpertLayout, aq, s.bq, as, s.bScales, dst, idx, wgt, d), gx, gy)
}

func (b *decodeRunnerBuild) routeGptOss(logits, bias, idx, wgt *wgpu.Buffer, hasBias bool) {
	defer b.inRole("routeGptOss")()
	mp := b.m.Moe
	p := b.uni([]uint32{uint32(mp.NE), uint32(mp.K), BoolU32(hasBias), 0})
	b.add(b.c.routeGptOssPipeline, b.bind(b.c.routeGptOssLayout, logits, bias, idx, wgt, p), 1, 1)
}

func (b *decodeRunnerBuild) gptossActQuant(gate, up, biasGU, idx *wgpu.Buffer, slot, K int) (*wgpu.Buffer, *wgpu.Buffer) {
	defer b.inRole("gptossActQuant")()
	mp := b.m.Moe
	kp := PadK32(K)
	q, s := b.storF(kp/4), b.storF(1)
	p := b.uni([]uint32{uint32(K), uint32(kp), uint32(slot), 1, F32bits(mp.GptossAlpha), F32bits(mp.GptossLimit), 0, 0})
	b.add(b.c.gptossGluQuantPipeline, b.bind(b.c.gptossGluQuantLayout, gate, up, biasGU, idx, q, s, p), 1, 1)
	return q, s
}

func (b *decodeRunnerBuild) moeExpertGptOssDown(aq, as *wgpu.Buffer, s *ResidentStackedW8A8, idx, wgt, dbias, dst *wgpu.Buffer, slot int) {
	defer b.inRole("moeExpertGptOssDown")()
	d := b.uni([]uint32{uint32(s.kp), uint32(s.rows), uint32(slot), 0})
	gx, gy := GemvGrid(s.rows)
	pl, ly := b.c.gptOssDownPipelineFor(s)
	b.add(pl, b.bind(ly, aq, s.bq, as, s.bScales, dst, idx, wgt, dbias, d), gx, gy)
}

func (b *decodeRunnerBuild) sharedGatedCombine(dst, src, gl *wgpu.Buffer, n int) {
	defer b.inRole("sharedGatedCombine")()
	p := b.uni([]uint32{uint32(n), 0, 0, 0})
	b.add(b.c.sharedGatePipeline, b.bind(b.c.sharedGateLayout, dst, src, gl, p), uint32(n+63)/64, 1)
}

// --- per-layer block emitters ---

func (b *decodeRunnerBuild) emitLayerTail(lw *RunLayer) {
	if lw.LayerScalar != 0 && lw.LayerScalar != 1.0 {
		b.scaleVec(b.r.xd, lw.LayerScalar, b.hidden)
	}
	if b.captureOn {
		b.r.capMLP = append(b.r.capMLP, b.capStep(b.r.xd, b.hidden))
	}
}

func (b *decodeRunnerBuild) emitLayer(i int, lw *RunLayer) {
	if lw.NemoKind == NemoKMLP {
		b.emitNemotronMLP(lw)
		return
	}
	if lw.NemoKind == NemoKMoE {
		b.emitNemotronMoE(lw)
		return
	}
	var parAq, parAs *wgpu.Buffer
	switch {
	case lw.IsMamba:
		b.emitMambaMixer(lw)
	case lw.IsDeltaNet:
		b.emitDeltaNetMixer(lw)
	case b.m.Mla != nil:
		b.emitMLAMixer(lw)
	default:
		parAq, parAs = b.emitAttnMixer(i, lw)
	}
	if lw.NemoKind == NemoKMamba || lw.NemoKind == NemoKAttn {
		return // Nemotron single-op-per-block: the mixer IS the layer — no FFN sub-block
	}
	if lw.G4moe != nil {
		b.emitGemma4MoE(lw)
		b.emitLayerTail(lw)
		return
	}
	var mq, ms *wgpu.Buffer
	if b.m.Parallel && ParallelFromResidualForTest {
		mq, ms = b.rmsQuant(b.r.xd, lw.AttnNorm, b.hidden)
	} else if b.m.Parallel {
		mq, ms = parAq, parAs
	} else if lw.MlpNorm == nil {
		mq, ms = b.quant(b.r.xd, b.hidden)
	} else {
		mq, ms = b.rmsQuant(b.r.xd, lw.MlpNorm, b.hidden)
	}
	if lw.IsMoE {
		b.emitSparseMoE(lw, mq, ms)
	} else {
		b.emitDenseFFN(i, lw, mq, ms)
	}
	b.emitLayerTail(lw)
}

func (b *decodeRunnerBuild) emitNemotronMLP(lw *RunLayer) {
	mq, ms := b.rmsQuant(b.r.xd, lw.MlpNorm, b.hidden)
	up := b.gemv(mq, ms, lw.Up)
	rq, rs := b.relu2Quant(up, lw.Up.nRows())
	b.gemvAdd(rq, rs, lw.Down, b.r.xd)
}

func (b *decodeRunnerBuild) emitNemotronMoE(lw *RunLayer) {
	mq, ms := b.rmsQuant(b.r.xd, lw.MlpNorm, b.hidden)
	mp := b.m.Moe
	logits := b.gemv(mq, ms, lw.Router)
	b.role = "nemoMoE"
	idx, wgt := b.storF(mp.K), b.storF(mp.K)
	bias, hasBias := b.moeZeroBias, false
	if lw.RouterBias != nil {
		bias, hasBias = lw.RouterBias, true
	}
	b.moeRoute(logits, bias, idx, wgt, hasBias)
	for j := 0; j < mp.K; j++ {
		upOut := b.storF(mp.Inter)
		b.moeExpert(mq, ms, lw.ExpUp, idx, wgt, upOut, j, 0)
		rq, rs := b.relu2Quant(upOut, mp.Inter)
		b.moeExpert(rq, rs, lw.ExpDown, idx, wgt, b.r.xd, j, 1)
	}
	if lw.ShUp != nil {
		su := b.gemv(mq, ms, lw.ShUp)
		sq, ss := b.relu2Quant(su, mp.SharedInter)
		b.gemvAdd(sq, ss, lw.ShDown, b.r.xd)
	}
}

func (b *decodeRunnerBuild) emitMambaMixer(lw *RunLayer) {
	mp := b.m.Mamba
	aq, as := b.rmsQuant(b.r.xd, lw.AttnNorm, b.hidden)
	proj := b.gemv(aq, as, lw.MambaInProj)
	b.role = "mambaLayer"
	conv := b.storF(mp.ConvDim)
	b.add(b.c.mambaConvPipeline, b.bindOff(b.c.mambaConvLayout, []bgEnt{
		{proj, 0, 0}, {lw.MambaConvW, 0, 0}, {lw.MambaConvB, 0, 0}, {lw.MambaWin, 0, 0}, {conv, 0, 0}, {b.mambaDC, 0, 0},
	}), uint32(mp.ConvDim+63)/64, 1)
	y := b.storF(mp.DInner)
	b.add(b.c.mambaSSMPipeline, b.bindOff(b.c.mambaSSMLayout, []bgEnt{
		{conv, 0, 0}, {proj, 0, 0}, {lw.MambaHeadP, 0, 0}, {lw.MambaSSM, 0, 0}, {y, 0, 0}, {b.mambaDS, 0, 0},
	}), uint32(mp.NHeads*mp.Hp+63)/64, 1)
	gated := b.storF(mp.DInner)
	b.add(b.c.mambaGNormPipeline, b.bindOff(b.c.mambaGNormLayout, []bgEnt{
		{y, 0, 0}, {proj, 0, 0}, {lw.MambaNormW, 0, 0}, {gated, 0, 0}, {b.mambaDG, 0, 0},
	}), uint32(mp.NormGroups), 1)
	if b.r.mcapProj == nil {
		b.r.mcapProj, b.r.mcapConv, b.r.mcapY, b.r.mcapGated = proj, conv, y, gated
	}
	gq, gs := b.quant(gated, mp.DInner)
	b.gemvAdd(gq, gs, lw.MambaOutProj, b.r.xd)
}

func (b *decodeRunnerBuild) emitDeltaNetMixer(lw *RunLayer) {
	dp := b.m.Dnet
	aq, as := b.rmsQuant(b.r.xd, lw.AttnNorm, b.hidden)
	mixed := b.gemv(aq, as, lw.DnQKV)
	b.role = "deltaLayer"
	conv := b.storF(dp.ConvDim)
	b.add(b.c.mambaConvPipeline, b.bind(b.c.mambaConvLayout, mixed, lw.MambaConvW, lw.MambaConvB, lw.MambaWin, conv, b.dnDC), uint32(dp.ConvDim+63)/64, 1)
	qn, kn := b.storF(dp.KeyDim), b.storF(dp.KeyDim)
	b.add(b.c.deltaNormPipeline, b.bind(b.c.deltaNormLayout, conv, qn, kn, b.dnDNorm), uint32(dp.Nk+31)/32, 1)
	headP := b.storF(dp.Nv * 2)
	b.add(b.c.deltaGatesPipeline, b.bind(b.c.deltaGatesLayout, b.gemv(aq, as, lw.DnB), b.gemv(aq, as, lw.DnA), lw.DnDtBias, lw.DnNegExpA, headP, b.dnDGate), uint32(dp.Nv+63)/64, 1)
	core := b.storF(dp.ValueDim)
	b.add(b.c.deltaRulePipeline, b.bind(b.c.deltaRuleLayout, qn, kn, conv, headP, lw.DnState, core, b.dnDRule), uint32(dp.ValueDim+63)/64, 1)
	gated := b.storF(dp.ValueDim)
	b.add(b.c.deltaGNormPipeline, b.bind(b.c.deltaGNormLayout, core, b.gemv(aq, as, lw.DnZ), lw.DnNormW, gated, b.dnDGN), uint32(dp.Nv+63)/64, 1)
	gq, gs := b.quant(gated, dp.ValueDim)
	b.gemvAdd(gq, gs, lw.DnOut, b.r.xd)
}

func (b *decodeRunnerBuild) emitMLAMixer(lw *RunLayer) {
	mp := b.m.Mla
	qkHead := mp.QkNope + mp.QkRope
	latDim := mp.KvLoRARank + mp.QkRope
	rank := mp.KvLoRARank
	rhalf := mp.QkRope / 2

	aq, as := b.rmsQuant(b.r.xd, lw.AttnNorm, b.hidden)
	var qf *wgpu.Buffer
	if mp.QLoRARank > 0 {
		qa := b.gemv(aq, as, lw.MlaQA)
		qaq, qas := b.rmsQuant(qa, lw.MlaQANorm, mp.QLoRARank)
		qf = b.gemv(qaq, qas, lw.MlaQB)
	} else {
		qf = b.gemv(aq, as, lw.MlaQ)
	}
	kvDown := b.gemv(aq, as, lw.MlaKVA)
	b.add(b.c.mlaStorePipeline, b.bind(b.c.mlaStoreLayout, kvDown, lw.MlaKVANorm, lw.InvFreq, lw.LatCache, b.mlaStoreUni), 1, 1)
	b.role = "mlaLayer"
	qAbs := b.storF(b.nH * latDim)
	// W_UK absorb: qNopeAbs_h = W_UKᵀ_h·q_nope_h, written strided into qAbs[h·latDim..+rank].
	func() {
		defer b.inRole("mlaAbsorb")()
		p := b.uni([]uint32{uint32(b.nH), uint32(rank), uint32(mp.QkNope), uint32(qkHead), uint32(latDim), 0, 0, 0})
		gx, gy := GemvGrid(b.nH * rank)
		b.add(b.c.mlaHeadMVPipeline, b.bind(b.c.mlaHeadMVLayout, qf, lw.MlaWUK, qAbs, p), gx, gy)
	}()
	b.add(b.c.mlaQRopePipeline, b.bind(b.c.mlaQRopeLayout, qf, lw.InvFreq, qAbs, b.mlaQRopeUni), uint32(b.nH*rhalf+63)/64, 1)
	wsum := b.storF(b.nH * rank)
	b.add(b.c.mlaAttnPipeline, b.bind(b.c.mlaAttnLayout, qAbs, lw.LatCache, wsum, b.mlaAttnUni), uint32(b.nH), 1)
	ctxv := b.storF(b.nH * mp.VHead)
	// W_UV lift: ctx_h = W_UV_h·wsum_h ([vHead] per head), contiguous output.
	func() {
		defer b.inRole("mlaLift")()
		p := b.uni([]uint32{uint32(b.nH), uint32(mp.VHead), uint32(rank), uint32(rank), uint32(mp.VHead), 0, 0, 0})
		gx, gy := GemvGrid(b.nH * mp.VHead)
		b.add(b.c.mlaHeadMVPipeline, b.bind(b.c.mlaHeadMVLayout, wsum, lw.MlaWUV, ctxv, p), gx, gy)
	}()
	cq, cs := b.quant(ctxv, b.nH*mp.VHead)
	b.gemvAdd(cq, cs, lw.MlaO, b.r.xd)
}

func (b *decodeRunnerBuild) emitAttnMixer(i int, lw *RunLayer) (parAq, parAs *wgpu.Buffer) {
	var aq, as *wgpu.Buffer
	if lw.AttnNorm == nil {
		aq, as = b.quant(b.r.xd, b.hidden)
	} else {
		aq, as = b.rmsQuant(b.r.xd, lw.AttnNorm, b.hidden)
	}
	parAq, parAs = aq, as

	ghd, gnKV, ghalf := b.hd, b.nKV, b.half
	if lw.Ghd != 0 {
		ghd, gnKV, ghalf = lw.Ghd, lw.GnKV, lw.Ghalf
	}
	g := b.geomFor(ghd, gnKV, ghalf, lw.GKEqV)
	var q, k, v, aGate *wgpu.Buffer
	_, w8 := lw.Q.(*ResidentW8A8)
	if lw.KvShared {
		q = b.gemv(aq, as, lw.Q)
		b.r.loraHooks = append(b.r.loraHooks, loraHook{afterIdx: len(b.r.steps) - 1, layer: i, kind: loraQ, aq: aq, ascale: as, dst: q, k: b.hidden})
		if lw.QNorm != nil {
			if b.m.QKNormWhole {
				b.qkNorm(q, lw.QNorm, 1, b.nH*g.hd)
			} else {
				b.qkNorm(q, lw.QNorm, b.nH, g.hd)
			}
		}
		b.rope(g, q, lw.InvFreq, lw.RopeScale)
	} else if lw.QBias != nil && w8 {
		q = b.gemvBias(aq, as, lw.Q, lw.QBias)
		k = b.gemvBias(aq, as, lw.K, lw.KBias)
		if !lw.GKEqV {
			v = b.gemvBias(aq, as, lw.V, lw.VBias)
		}
	} else {
		q, k = b.gemv(aq, as, lw.Q), b.gemv(aq, as, lw.K)
		if !lw.GKEqV {
			v = b.gemv(aq, as, lw.V)
		}
		if lw.QBias != nil {
			b.biasAdd(q, lw.QBias, b.nH*g.hd)
			b.biasAdd(k, lw.KBias, g.kvDim)
			if !lw.GKEqV {
				b.biasAdd(v, lw.VBias, g.kvDim)
			}
		}
	}
	if lw.GKEqV && !lw.KvShared {
		b.role = "attnLayer"
		v = b.storF(g.kvDim)
		b.vNorm(k, v, g.nKV, g.hd)
	}
	after := len(b.r.steps) - 1
	if !lw.KvShared {
		b.r.loraHooks = append(b.r.loraHooks,
			loraHook{afterIdx: after, layer: i, kind: loraQ, aq: aq, ascale: as, dst: q, k: b.hidden},
			loraHook{afterIdx: after, layer: i, kind: loraK, aq: aq, ascale: as, dst: k, k: b.hidden},
		)
		if !lw.GKEqV {
			b.r.loraHooks = append(b.r.loraHooks,
				loraHook{afterIdx: after, layer: i, kind: loraV, aq: aq, ascale: as, dst: v, k: b.hidden},
			)
		}
	}
	if lw.VNorm && !lw.GKEqV && !lw.KvShared {
		b.role = "attnLayer"
		vn := b.storF(g.kvDim)
		b.vNorm(v, vn, g.nKV, g.hd)
		v = vn
	}
	if lw.QGate {
		q, aGate = b.qSplit(q, b.nH*g.hd, g.hd)
	}
	if lw.QNorm != nil && !lw.KvShared {
		if b.m.QKNormWhole {
			b.qkNorm(q, lw.QNorm, 1, b.nH*g.hd)
			b.qkNorm(k, lw.KNorm, 1, g.nKV*g.hd)
		} else {
			b.qkNorm(q, lw.QNorm, b.nH, g.hd)
			b.qkNorm(k, lw.KNorm, g.nKV, g.hd)
		}
	}
	if lw.KvShared {
		// Q is already normed and rotated above; nothing is stored.
	} else if b.m.KvF16 || b.m.KvI8 {
		b.rope(g, q, lw.InvFreq, lw.RopeScale)
		b.ropeStore(g, k, lw.InvFreq, lw.KCache, lw.KScale, lw.RopeScale)
		b.vStore(g, v, lw.VCache, lw.VScale)
	} else {
		b.qkvFinalize(g, q, k, v, lw.InvFreq, lw.KCache, lw.VCache, lw.RopeScale)
	}
	b.role = "attnLayer"
	ctxv := b.storF(b.nH * g.hd)
	aUni := g.attnUni
	if lw.IsLocal {
		aUni = g.attnUniLocal
	}
	wide := g.hd > AttnWG
	b.role = "attnLayer"
	hasSinkUni := b.uni([]uint32{BoolU32(lw.HasSink), 0, 0, 0})
	attnSinks := lw.AttnSinks
	if attnSinks == nil {
		attnSinks = b.noAttnSinks
	}
	if b.m.KvI8 {
		pl, ly := b.c.attnI8Pipeline, b.c.attnI8Layout
		if wide {
			pl, ly = b.c.attnI8WidePipeline, b.c.attnI8WideLayout
		}
		b.add(pl, b.bind(ly, q, lw.KCache, lw.VCache, lw.KScale, lw.VScale, ctxv, attnSinks, aUni, hasSinkUni), uint32(b.nH), 1)
	} else {
		attnPl, attnLy := b.c.attnKernel(g.hd, g.kvDim, b.m.KvF16)
		b.add(attnPl, b.bind(attnLy, q, lw.KCache, lw.VCache, ctxv, attnSinks, aUni, hasSinkUni), uint32(b.nH), 1)
	}
	if aGate != nil {
		b.attnGate(ctxv, aGate, b.nH*g.hd)
	}
	if b.captureOn {
		b.r.capCtx = append(b.r.capCtx, b.capStep(ctxv, b.nH*g.hd))
		b.r.capCtxW = append(b.r.capCtxW, b.nH*g.hd)
	}
	cq, cs := b.quant(ctxv, b.nH*g.hd)
	if lw.OBias != nil || lw.PostAttnNorm != nil {
		attnOut := b.gemv(cq, cs, lw.O)
		b.r.loraHooks = append(b.r.loraHooks,
			loraHook{afterIdx: len(b.r.steps) - 1, layer: i, kind: loraO, aq: cq, ascale: cs, dst: attnOut, k: b.nH * g.hd})
		if lw.OBias != nil {
			b.biasAdd(attnOut, lw.OBias, b.hidden)
		}
		if lw.PostAttnNorm != nil {
			attnOut = b.rmsnormF32(attnOut, lw.PostAttnNorm, b.hidden)
		}
		b.biasAdd(b.r.xd, attnOut, b.hidden)
	} else {
		b.gemvAdd(cq, cs, lw.O, b.r.xd)
		b.r.loraHooks = append(b.r.loraHooks,
			loraHook{afterIdx: len(b.r.steps) - 1, layer: i, kind: loraO, aq: cq, ascale: cs, dst: b.r.xd, k: b.nH * g.hd})
	}
	if b.captureOn {
		b.r.capAttn = append(b.r.capAttn, b.capStep(b.r.xd, b.hidden))
	}
	return parAq, parAs
}

func (b *decodeRunnerBuild) emitGemma4MoE(lw *RunLayer) {
	g := lw.G4moe
	d := G4MoEDefectForTest
	rnW := g.RouterNorm
	if d == 7 {
		rnW = g.PreFFNNorm
	}
	rn := b.rmsnormF32(b.r.xd, rnW, b.hidden)
	b.role = "gemma4MoE"
	logits := b.storF(g.NE)
	b.add(b.c.pipeline, b.bind(b.c.layout, rn, g.Router, logits, b.uni([]uint32{1, uint32(b.hidden), uint32(g.NE), 0})), 1, uint32(g.NE+15)/16)
	idx, wgt := b.storF(g.TopK), b.storF(g.TopK)
	b.add(b.c.moeRoutePipeline, b.bind(b.c.moeRouteLayout, logits, g.ZeroBias, idx, wgt,
		b.uni([]uint32{uint32(g.NE), uint32(g.TopK), 0, 1, F32bits(1), 0, 1, 0})), 1, 1)
	if d != 1 {
		b.add(b.c.g4RouteScalePipeline, b.bind(b.c.g4RouteScaleLayout, idx, g.PerExpertScale, wgt, b.uni([]uint32{uint32(g.TopK), 0, 0, 0})), 1, 1)
	}
	mq, ms := b.rmsQuant(b.r.xd, g.PreFFNNorm, b.hidden)
	gate, up := b.gemv(mq, ms, g.Gate), b.gemv(mq, ms, g.Up)
	dq, ds := b.actQuantPad(gate, up, g.DenseInter, PadK32(g.DenseInter))
	x1 := b.gemv(dq, ds, g.Down)
	if d != 2 {
		x1 = b.rmsnormF32(x1, g.PostFFNNorm1, b.hidden)
	}
	eq, es := mq, ms
	if d != 3 {
		eq, es = b.rmsQuant(b.r.xd, g.PreFFNNorm2, b.hidden)
	}
	x2 := b.storF(b.hidden)
	if d != 5 {
		b.scaleVec(x2, 0, b.hidden)
	}
	gu, upHalf := b.storF(2*g.MoeInter), b.storF(g.MoeInter)
	for j := 0; j < g.TopK; j++ {
		b.moeExpert(eq, es, g.ExpGateUp, idx, wgt, gu, j, 0)
		b.add(b.c.g4TailCopyPipeline, b.bind(b.c.g4TailCopyLayout, gu, upHalf, b.uni([]uint32{uint32(g.MoeInter), uint32(g.MoeInter), 0, 0})), uint32(g.MoeInter+63)/64, 1)
		ga, ub := gu, upHalf
		if d == 6 {
			ga, ub = upHalf, gu
		}
		xq, xs := b.actQuantPad(ga, ub, g.MoeInter, PadK32(g.MoeInter))
		b.moeExpert(xq, xs, g.ExpDown, idx, wgt, x2, j, 1)
	}
	x2n := b.rmsnormF32(x2, g.PostFFNNorm2, b.hidden)
	b.biasAdd(x1, x2n, b.hidden)
	comb := x1
	if d != 4 {
		comb = b.rmsnormF32(x1, g.PostFFNNorm, b.hidden)
	}
	b.biasAdd(b.r.xd, comb, b.hidden)
}

func (b *decodeRunnerBuild) emitSparseMoE(lw *RunLayer, mq, ms *wgpu.Buffer) {
	mp := b.m.Moe
	logits := b.gemv(mq, ms, lw.Router)
	b.role = "moeFFN"
	idx, wgt := b.storF(mp.K), b.storF(mp.K)
	bias, hasBias := b.moeZeroBias, false
	if lw.RouterBias != nil {
		bias, hasBias = lw.RouterBias, true
	}
	if mp.Gptoss {
		b.routeGptOss(logits, bias, idx, wgt, hasBias)
		gateOut, upOut := b.storF(mp.Inter), b.storF(mp.Inter)
		for j := 0; j < mp.K; j++ {
			b.moeExpert(mq, ms, lw.ExpGate, idx, wgt, gateOut, j, 0)
			b.moeExpert(mq, ms, lw.ExpUp, idx, wgt, upOut, j, 0)
			dq, ds := b.gptossActQuant(gateOut, upOut, lw.GateUpBias, idx, j, mp.Inter)
			b.moeExpertGptOssDown(dq, ds, lw.ExpDown, idx, wgt, lw.DownBias, b.r.xd, j)
		}
	} else {
		b.moeRoute(logits, bias, idx, wgt, hasBias)
		gateOut, upOut := b.storF(mp.Inter), b.storF(mp.Inter)
		for j := 0; j < mp.K; j++ {
			b.moeExpert(mq, ms, lw.ExpGate, idx, wgt, gateOut, j, 0)
			b.moeExpert(mq, ms, lw.ExpUp, idx, wgt, upOut, j, 0)
			dq, ds := b.swigluQuant(gateOut, upOut, mp.Inter)
			b.moeExpert(dq, ds, lw.ExpDown, idx, wgt, b.r.xd, j, 1)
		}
	}
	if lw.ShGate != nil {
		sg, su := b.gemv(mq, ms, lw.ShGate), b.gemv(mq, ms, lw.ShUp)
		sdq, sds := b.swigluQuant(sg, su, mp.SharedInter)
		if lw.ShGateW != nil {
			sdown := b.gemv(sdq, sds, lw.ShDown)
			gl := b.gemv(mq, ms, lw.ShGateW)
			b.sharedGatedCombine(b.r.xd, sdown, gl, b.hidden)
		} else {
			b.gemvAdd(sdq, sds, lw.ShDown, b.r.xd)
		}
	}
}

func (b *decodeRunnerBuild) emitDenseFFN(i int, lw *RunLayer, mq, ms *wgpu.Buffer) {
	gate, up := b.gemv(mq, ms, lw.Gate), b.gemv(mq, ms, lw.Up)
	b.r.loraHooks = append(b.r.loraHooks,
		loraHook{afterIdx: len(b.r.steps) - 1, layer: i, kind: loraGate, aq: mq, ascale: ms, dst: gate, k: b.hidden},
		loraHook{afterIdx: len(b.r.steps) - 1, layer: i, kind: loraUp, aq: mq, ascale: ms, dst: up, k: b.hidden},
	)
	ffn := lw.Gate.nRows()
	ffnAct := ffn
	if G4OneFFNWidthForTest && b.inter < ffn {
		ffnAct = b.inter
	}
	dq, ds := b.actQuantPad(gate, up, ffnAct, PadK32(ffn))
	if lw.PostMLPNorm != nil {
		downOut := b.gemv(dq, ds, lw.Down)
		b.r.loraHooks = append(b.r.loraHooks,
			loraHook{afterIdx: len(b.r.steps) - 1, layer: i, kind: loraDown, aq: dq, ascale: ds, dst: downOut, k: ffn})
		downOut = b.rmsnormF32(downOut, lw.PostMLPNorm, b.hidden)
		b.biasAdd(b.r.xd, downOut, b.hidden)
	} else {
		b.gemvAdd(dq, ds, lw.Down, b.r.xd)
		b.r.loraHooks = append(b.r.loraHooks,
			loraHook{afterIdx: len(b.r.steps) - 1, layer: i, kind: loraDown, aq: dq, ascale: ds, dst: b.r.xd, k: ffn})
	}
	if lw.PleGate != nil && !G4SkipPLEForTest {
		pl := b.r.pleIn[i]
		if G4PLEOffByOneForTest {
			pl = b.r.pleIn[(i+1)%len(b.r.pleIn)]
		}
		hq, hs := b.quant(b.r.xd, b.hidden)
		px := b.gemv(hq, hs, lw.PleGate)
		pq, ps := b.actQuantPad(px, pl, b.m.PleDim, PadK32(b.m.PleDim))
		sub := b.gemv(pq, ps, lw.PleProj)
		sub = b.rmsnormF32(sub, lw.PostPLENorm, b.hidden)
		b.biasAdd(b.r.xd, sub, b.hidden)
	}
}

func (b *decodeRunnerBuild) finish() (*DecodeRunner, error) {
	b.r.geomVariants = len(b.geomCache)
	fq, fs := b.rmsQuant(b.r.xd, b.m.FinalNorm, b.hidden)
	b.r.lastLogits = b.gemv(fq, fs, b.m.LmHead)
	if b.buildErr == nil {
		stag, e := b.c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Size: uint64(b.r.vocab * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		if e != nil {
			b.buildErr = e
		} else {
			b.r.scratch.add("logitsStaging", int64(stag.GetSize()))
			b.r.stag = b.keepBuf(stag)
		}
	}
	// A device-allocation or bind failure during construction (VRAM exhaustion is the common one) is returned so the caller
	// falls back to the staged/CPU path, never a panic.
	if b.buildErr != nil {
		b.r.release()
		return nil, fmt.Errorf("gpu: newDecodeRunner: %w (VRAM exhausted?): %w", mem.DeviceAllocError{}, b.buildErr)
	}
	// Snapshot the pristine (no-adapter) plan: SetAdapter rebuilds r.steps from it plus r.loraHooks rather than mutating it.
	b.r.baseSteps = b.r.steps
	return b.r, nil
}
