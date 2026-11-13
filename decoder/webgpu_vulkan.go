package decoder

import (
	"runtime"
	"slices"

	"github.com/townsendmerino/aikit/linalg"
)

// WebGPU's device memory as Vulkan allocates it (M1′, docs/measurements/sb2-vk-2026-10-31/README.md): wgpu-native's gpu-allocator
// places each buffer into 128 MiB, then 256 MiB, device-local blocks, first fit, newest block first, and gives a buffer larger than
// the current block size a block of its own; it keeps one 64 MiB host-visible block for the upload staging; and the device itself
// holds 144 MiB. The planner prices a webgpu load as that, not as the sum of its buffers, and the WebGPU slot census predicts
// newKVSlot over the same list (WebGPUDeviceBuffersForTest). One card and driver so far (nobara's
// RTX 2070 SUPER on 595.112.07): the constants are re-measured, not carried, on another.
const (
	webgpuVkInternal  = 512<<10 + 256 + 256 // wgpu-core's own per-device buffers
	webgpuVkAlign     = 256                 // the driver's buffer alignment, assumed (M1's source reading)
	webgpuVkDevFirst  = 128 << 20
	webgpuVkDevMax    = 256 << 20
	webgpuVkHostBlock = 64 << 20  // the host-visible block gpu-allocator keeps
	webgpuVkDevice    = 144 << 20 // the device's own memory, fitted on G-VK1's six loads and judged on five held out
	webgpuVkScratch   = 8 << 20   // the runner's scratch line (3 to 7 MiB in G-VK1's ledgers)
)

// webgpuVulkan reports whether this build's WebGPU runs on Vulkan, whose allocation M1′ models; on darwin wgpu runs on Metal, whose
// allocator is not measured, so the planner prices a darwin webgpu load as the sum of its buffers.
const webgpuVulkan = runtime.GOOS != "darwin"

// webgpuVkFixed is what a WebGPU load holds beyond its blocks: the device and the kept host block.
const webgpuVkFixed = webgpuVkDevice + webgpuVkHostBlock

// webgpuVkBlocks is the device-local bytes gpu-allocator 0.28.0 opens for sizes allocated in order, wgpu-core's own buffers first.
func webgpuVkBlocks(sizes []int64) int64 {
	var free []int64
	var total int64
	place := func(r int64) {
		r = (max(r, 4) + 3) / 4 * 4
		r = (r + webgpuVkAlign - 1) / webgpuVkAlign * webgpuVkAlign
		mb := min(int64(webgpuVkDevFirst)<<min(len(free), 7), webgpuVkDevMax)
		if r > mb {
			total += r
			return
		}
		for i := len(free) - 1; i >= 0; i-- {
			if free[i] >= r {
				free[i] -= r
				return
			}
		}
		free = append(free, mb-r)
		total += mb
	}
	place(webgpuVkInternal)
	for _, n := range sizes {
		place(n)
	}
	return total
}

// wmBuffers is a weight matrix's device buffers: each payload it holds and its scales, as WebGPU uploads them, sized by the
// accessors wmBytes sums.
func wmBuffers(w *linalg.WeightMat) []int64 {
	var out []int64
	add := func(ns ...int64) {
		for _, n := range ns {
			if n > 0 {
				out = append(out, n)
			}
		}
	}
	if q8, sc, _, ok := w.Int8(); ok {
		add(int64(len(q8)), 4*int64(len(sc)))
	}
	if q4, sc, _, ok := w.Int4F16(); ok {
		add(int64(len(q4)), 2*int64(len(sc)))
	}
	if p4, sc, ok := w.Int4Row4F16(); ok {
		add(int64(len(p4)), 2*int64(len(sc)))
	}
	if f, ok := w.F32(); ok {
		add(4 * int64(len(f)))
	}
	if raw, ok := w.Q4K(); ok {
		add(int64(len(raw)))
	}
	if u, _, ok := w.Half16(); ok {
		add(2 * int64(len(u)))
	}
	add(int64(w.SplitHalfBytes()))
	return out
}

// webgpuProjBuffers is a projection's device buffers as WebGPU uploads it (uploadProj): an int4 matrix as packed codes, its rows
// padded to 32 where K is not a multiple of 32, and f16 scale pairs; an int8 or f32 matrix as W8A8 codes, rows padded to 16, and
// per-row f32 scales. Any other kind is sized as the CPU holds it (wmBuffers).
func webgpuProjBuffers(w *linalg.WeightMat) []int64 {
	return webgpuStackedBuffers(1, w)
}

// webgpuStackedBuffers is nE experts shaped as w stacked into one codes buffer and one scales buffer, as WebGPU's MoE upload
// (buildStacked) makes them; nE = 1 is a single projection.
func webgpuStackedBuffers(nE int, w *linalg.WeightMat) []int64 {
	n, k := int64(w.Rows()), int64(w.Cols())
	if n == 0 || k == 0 {
		return nil
	}
	e := int64(nE)
	switch w.Kind() {
	case "int4":
		kp := k
		if k%32 != 0 {
			kp = (k + 31) &^ 31
		}
		return []int64{e * n * kp / 2, 4 * ((e*n*(kp/32) + 1) / 2)}
	case "int8", "f32":
		return []int64{e * n * ((k + 15) &^ 15), 4 * e * n}
	}
	if nE != 1 {
		return nil
	}
	return wmBuffers(w)
}

// webgpuLayerCaches is layer i's per-sequence buffers on a WebGPU build at ctx, as the build allocates them: K and V on an
// attention layer that owns its cache (a Gemma 4 KV-shared layer aliases its source's), the latent cache on an MLA layer, and the
// conv window and recurrent state on a DeltaNet or Mamba-2 layer. kv is false for the recurrent state, which is not context and which
// ResidentKVBytes does not price.
func (m *Model) webgpuLayerCaches(i, ctx int, kvF16, kvI8 bool) (bufs []int64, kv bool) {
	perElem := 4.0
	switch {
	case kvI8:
		perElem = 1.125
	case kvF16:
		perElem = 2
	}
	mixer, _ := m.ResidentLayerOps(i)
	switch mixer {
	case MixerAttention:
		if m.w.arch != nil && m.w.arch.gemma4KVSrcAt(i) != i {
			return nil, true
		}
		var kvDim int
		if m.w.arch != nil {
			kvDim = m.w.arch.kvDimAt(i)
		} else {
			kvDim = m.KVHeadsAtResident(i) * m.HeadDimAtResident(i)
		}
		if kvDim == 0 || ctx <= 0 {
			return nil, true
		}
		c := int64(float64(kvDim) * perElem * float64(ctx))
		return []int64{c, c}, true
	case MixerMLA:
		// The latent row (kv_lora_rank + the rope key) is f32 at any precision. kvDimAt is 0 on a layer with no latent, such as a
		// Kimi Delta Attention layer that ResidentLayerOps reports as MLA.
		if m.w.arch == nil || ctx <= 0 {
			return nil, true
		}
		if d := m.w.arch.kvDimAt(i); d > 0 {
			return []int64{int64(ctx) * int64(d) * 4}, true
		}
		return nil, true
	case MixerDeltaNet:
		convK, hk, hv, nk, nv, _, ok := m.Qwen35ResidentParams()
		if !ok {
			return nil, false
		}
		return []int64{int64(convK-1) * int64(2*nk*hk+nv*hv) * 4, int64(nv*hv*hk) * 4}, false
	case MixerMamba:
		nH, hd, dS, nG, dConv, _, _, _, _, ok := m.GraniteResidentParams()
		if !ok {
			var nok bool
			nH, hd, dS, nG, dConv, _, _, nok = m.NemotronResidentParams()
			if !nok {
				return nil, false
			}
		}
		return []int64{int64(dConv-1) * int64(nH*hd+2*nG*dS) * 4, int64(nH*hd*dS) * 4}, false
	}
	return nil, false
}

// webgpuDeviceBuffers is the buffer list a WebGPU resident build of m makes at ctx, in its creation order: the LM head (the token
// table when tied; an untied table, the position table and Gemma 4's per-layer tables stay on the host), then each layer's caches
// (webgpuLayerCaches) and matrices, then the scratch. slotKV is the caches one further slot adds; weights is the dense matrices'
// bytes as WebGPU uploads them (the routed experts, stacked, are listed but not counted in it).
func (m *Model) webgpuDeviceBuffers(ctx int, kvF16, kvI8 bool) (slot0, slotKV []int64, weights int64) {
	if m == nil || m.w == nil {
		return nil, nil, 0
	}
	w := m.w
	mats := func(ws ...*linalg.WeightMat) {
		for _, x := range ws {
			for _, n := range webgpuProjBuffers(x) {
				slot0 = append(slot0, n)
				weights += n
			}
		}
	}
	f32s := func(ss ...[]float32) {
		for _, s := range ss {
			if n := f32SliceBytes(s); n > 0 {
				slot0 = append(slot0, n)
				weights += n
			}
		}
	}
	if w.LMHead.Rows() != 0 {
		mats(&w.LMHead)
	} else {
		mats(&w.Embed)
	}
	stacked := func(nE int, w0 *linalg.WeightMat) {
		slot0 = append(slot0, webgpuStackedBuffers(nE, w0)...)
	}
	// w8a8 is a projection WebGPU uploads as int8 codes and per-row f32 scales (UploadW8A8): the codes' rows padded to 16.
	w8a8 := func(rows, cols int) {
		if rows <= 0 || cols <= 0 {
			return
		}
		for _, n := range []int64{int64(rows) * int64((cols+15)&^15), 4 * int64(rows)} {
			slot0 = append(slot0, n)
			weights += n
		}
	}
	hidden, nLayers, _, _, _, _, _ := m.Dims()
	for i := range w.Layers {
		var bufs []int64
		var kv bool
		if i < nLayers {
			bufs, kv = m.webgpuLayerCaches(i, ctx, kvF16, kvI8)
		}
		caches := func() {
			slot0 = append(slot0, bufs...)
			if kv {
				slotKV = append(slotKV, bufs...)
			}
		}
		l := &w.Layers[i]
		// A recurrent mixer's weights come first and its state after them, as the build uploads them; the layer's other matrices
		// follow. Any other layer's caches come first.
		switch d, mb := l.delta, l.mamba; {
		case d != nil:
			mats(&d.inProjQKV, &d.inProjZ, &d.outProj)
			w8a8(len(d.inProjB)/max(hidden, 1), hidden)
			w8a8(len(d.inProjA)/max(hidden, 1), hidden)
			f32s(d.convW, d.dtBias, d.negExpA, d.normW)
			caches()
		case mb != nil:
			if mb.inQ.Rows() != 0 {
				w8a8(mb.inQ.Rows(), mb.inQ.Cols())
				w8a8(mb.outQ.Rows(), mb.outQ.Cols())
			} else {
				w8a8(len(mb.inProj)/max(hidden, 1), hidden)
				w8a8(hidden, len(mb.outProj)/max(hidden, 1))
			}
			f32s(mb.convW, mb.convB, make([]float32, 3*len(mb.aLog)), mb.normW)
			caches()
		default:
			caches()
		}
		// The mixer's matrices, then the FFN's: the dense MLP, or the router, each projection's experts stacked into one buffer, the
		// shared expert and its gate.
		mats(&l.QProj, &l.KProj, &l.VProj, &l.OProj, &l.GProj)
		if q := l.qattn; q != nil {
			mats(&q.qProj, &q.kProj, &q.vProj, &q.oProj)
		}
		if a := l.mla; a != nil {
			// Each projection W8A8, from its int8 form or its f32 (mlaProj), then kv_b split into W_UKᵀ and W_UV, f32.
			qLoRA, kvLoRA, qkNope, qkRope, vHead, _, _, _, _ := m.MLAResidentParams()
			_, _, nH, _, _, _, _ := m.Dims()
			proj := func(q *linalg.WeightMat, rows, cols int) {
				if q.Rows() != 0 {
					rows, cols = q.Rows(), q.Cols()
				}
				w8a8(rows, cols)
			}
			if qLoRA > 0 {
				proj(&a.qAQ, qLoRA, hidden)
				proj(&a.qBQ, nH*(qkNope+qkRope), qLoRA)
			} else {
				proj(&a.qQ, nH*(qkNope+qkRope), hidden)
			}
			proj(&a.kvAQ, kvLoRA+qkRope, hidden)
			proj(&a.oQ, hidden, nH*vHead)
			for _, n := range []int64{4 * int64(nH*kvLoRA*qkNope), 4 * int64(nH*vHead*kvLoRA)} {
				slot0 = append(slot0, n)
				weights += n
			}
		}
		if k := l.kda; k != nil {
			mats(&k.qProj, &k.kProj, &k.vProj, &k.fProj, &k.gProj, &k.oProj)
			f32s(k.qConvW, k.kConvW, k.vConvW, k.bProj, k.dtBias, k.aLog, k.oNormW)
		}
		if sc := l.shortConv; sc != nil {
			mats(&sc.inProj, &sc.outProj)
			f32s(sc.convW)
		}
		mo := l.gemma4moe
		if mo != nil { // Gemma 4's parallel block: its router, f32, ahead of the dense MLP
			if n := 4 * int64(mo.routerProj.Rows()) * int64(mo.routerProj.Cols()); n > 0 {
				slot0 = append(slot0, n)
				weights += n
			}
		}
		mats(&l.GateProj, &l.UpProj, &l.DownProj, &l.PLEGate, &l.PLEProj)
		if nE := len(l.Experts); nE > 0 {
			mats(&l.Router)
			stacked(nE, &l.Experts[0].Gate)
			stacked(nE, &l.Experts[0].Up)
			stacked(nE, &l.Experts[0].Down)
		} else {
			mats(&l.Router)
		}
		mats(&l.SharedExpert.Gate, &l.SharedExpert.Up, &l.SharedExpert.Down, &l.SharedGate)
		if mo != nil && len(mo.expertsGateUp) > 0 {
			stacked(len(mo.expertsGateUp), &mo.expertsGateUp[0])
			stacked(len(mo.expertsDown), &mo.expertsDown[0])
		}
	}
	slot0 = append(slot0, webgpuVkScratch)
	return slot0, slotKV, weights
}

// webgpuBlockRounding is what a WebGPU load holds at ctx beyond what the plan already prices on its own lines (the dense matrices, the
// routed experts, the context, and one slot's recurrent state): gpu-allocator's blocks over webgpuDeviceBuffers, less those. The plan's
// total is therefore M1′'s blocks plus webgpuVkFixed and what sits beside the blocks (the vectors, biases and derived tables).
func (m *Model) webgpuBlockRounding(ctx int, kvF16, kvI8 bool) int64 {
	slot0, _, dense := m.webgpuDeviceBuffers(ctx, kvF16, kvI8)
	return webgpuVkBlocks(slot0) - dense - m.webgpuExpertLine() - m.ResidentKVBytes("webgpu", ctx, kvF16, kvI8) - m.stateBytes("webgpu")
}

// webgpuDenseLine is the webgpu plan's dense line: the dense matrices as WebGPU uploads them (webgpuDeviceBuffers), and beside them
// the vectors, the routed experts' biases and the derived tables.
func (m *Model) webgpuDenseLine() int64 {
	_, _, dense := m.webgpuDeviceBuffers(0, false, false)
	return dense + m.denseVectorBytes() + m.expertBiasBytes() + m.webgpuDerivedTableBytes()
}

// webgpuExpertLine is the plan's routed-expert line (deviceLoad's ExpertBytesFull): 0 for a dense model.
func (m *Model) webgpuExpertLine() int64 {
	if _, _, isMoE := m.moeGeometry(); !isMoE {
		return 0
	}
	return m.ResidentWeightBytesPaged(0) - m.ResidentDenseWeightBytes() - m.expertBiasBytes()
}

// webgpuDerivedTableBytes is what a WebGPU build uploads beside the weights that no weight line holds: one RoPE inverse-frequency
// table per distinct table (the build caches them by value), a one-element attention sink on each layer without learned sinks
// (a WGSL bind group binds no null buffer), on a DeltaNet model one zeroed conv bias of the conv width, which every DeltaNet
// layer binds, and on each of Gemma 4's MoE blocks the router norm's unit weight and the router's zero bias, less the router scale
// the build folds into the router. The plan prices it on the dense line, beside the blocks, as it prices the vectors.
func (m *Model) webgpuDerivedTableBytes() int64 {
	if m == nil || m.w == nil || m.w.arch == nil {
		return 0
	}
	var tabs [][]float32
	var n int64
	for i := range m.w.arch.NumLayers {
		if inv := m.RopeInvFreqLayerResident(i); len(inv) > 0 && !slices.ContainsFunc(tabs, func(t []float32) bool { return slices.Equal(t, inv) }) {
			tabs = append(tabs, inv)
			n += 4 * int64(len(inv))
		}
		if len(m.GptOssSinksResident(i)) == 0 {
			n += 4
		}
		if g4 := m.w.Layers[i].gemma4moe; g4 != nil {
			n += 4 * (int64(m.w.arch.HiddenDim) + int64(g4.nE) - int64(len(g4.routerScale)))
		}
	}
	if _, hk, hv, nk, nv, _, ok := m.Qwen35ResidentParams(); ok {
		n += 4 * int64(2*nk*hk+nv*hv)
	}
	return n
}
