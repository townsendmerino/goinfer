//go:build goinfer_testhooks

package decoder

import "github.com/townsendmerino/aikit/linalg"

// WebGPU's device memory as Vulkan allocates it (M1′, docs/measurements/sb2-vk-2026-10-31/README.md), for the WebGPU slot census
// (gpu's slotCensusLine) through WebGPUDeviceBuffersForTest; the planner does not price it yet (docs/tasks/task-subsystems-2026-10.md,
// "SB2-vk"): wgpu-native's gpu-allocator
// places each buffer into 128 MiB, then 256 MiB, device-local blocks, first fit, newest block first, and gives a buffer larger than
// the current block size a block of its own; it keeps one 64 MiB host-visible block for the upload staging; and the device itself
// holds 144 MiB. One card and driver so far (nobara's
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

// webgpuLayerCaches is layer i's per-sequence buffers on a WebGPU build at ctx, as the build allocates them: K and V on an
// attention layer that owns its cache (a Gemma 4 KV-shared layer aliases its source's), the latent cache on an MLA layer, and the
// conv window and recurrent state on a DeltaNet or Mamba-2 layer. kv is false for the recurrent state, which is not context.
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
		_, kvLoRA, _, qkRope, _, _, _, _, _ := m.MLAResidentParams()
		return []int64{int64(ctx) * int64(kvLoRA+qkRope) * 4}, true
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

// webgpuKVBytes is the context a WebGPU build allocates at ctx: every layer's K, V or latent cache that webgpuLayerCaches gives.
func (m *Model) webgpuKVBytes(ctx int, kvF16, kvI8 bool) int64 {
	if m == nil || m.w == nil {
		return 0
	}
	_, nLayers, _, _, _, _, _ := m.Dims()
	var n int64
	for i := range min(nLayers, len(m.w.Layers)) {
		if bufs, kv := m.webgpuLayerCaches(i, ctx, kvF16, kvI8); kv {
			for _, b := range bufs {
				n += b
			}
		}
	}
	return n
}

// webgpuDeviceBuffers is the buffer list a WebGPU resident build of m makes at ctx, in its creation order: the LM head (the token
// table when tied; an untied table, the position table and Gemma 4's per-layer tables stay on the host), then each layer's caches
// (webgpuLayerCaches) and matrices, then the scratch. slotKV is the caches one further slot adds; weights is the matrices' bytes,
// the dense bytes less the host-side tables plus the routed experts.
func (m *Model) webgpuDeviceBuffers(ctx int, kvF16, kvI8 bool) (slot0, slotKV []int64, weights int64) {
	if m == nil || m.w == nil {
		return nil, nil, 0
	}
	w := m.w
	mats := func(ws ...*linalg.WeightMat) {
		for _, x := range ws {
			for _, n := range wmBuffers(x) {
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
	_, nLayers, _, _, _, _, _ := m.Dims()
	for i := range w.Layers {
		if i < nLayers {
			bufs, kv := m.webgpuLayerCaches(i, ctx, kvF16, kvI8)
			slot0 = append(slot0, bufs...)
			if kv {
				slotKV = append(slotKV, bufs...)
			}
		}
		l := &w.Layers[i]
		mats(&l.QProj, &l.KProj, &l.VProj, &l.OProj, &l.GProj, &l.GateProj, &l.UpProj, &l.DownProj, &l.Router, &l.SharedGate, &l.PLEGate, &l.PLEProj)
		if d := l.delta; d != nil {
			mats(&d.inProjQKV, &d.inProjZ, &d.outProj)
		}
		if q := l.qattn; q != nil {
			mats(&q.qProj, &q.kProj, &q.vProj, &q.oProj)
		}
		if a := l.mla; a != nil {
			f32s(a.qAProj, a.qBProj, a.qProj, a.kvAProj, a.kvBProj, a.oProj, a.gProj)
			mats(&a.qAQ, &a.qBQ, &a.qQ, &a.kvAQ, &a.oQ)
		}
		if k := l.kda; k != nil {
			mats(&k.qProj, &k.kProj, &k.vProj, &k.fProj, &k.gProj, &k.oProj)
			f32s(k.qConvW, k.kConvW, k.vConvW, k.bProj, k.dtBias, k.aLog, k.oNormW)
		}
		if mb := l.mamba; mb != nil {
			f32s(mb.inProj, mb.convW, mb.outProj)
			mats(&mb.inQ, &mb.outQ)
		}
		if sc := l.shortConv; sc != nil {
			mats(&sc.inProj, &sc.outProj)
			f32s(sc.convW)
		}
		for j := range l.Experts {
			mats(&l.Experts[j].Gate, &l.Experts[j].Up, &l.Experts[j].Down)
		}
		mats(&l.SharedExpert.Gate, &l.SharedExpert.Up, &l.SharedExpert.Down)
		if mo := l.gemma4moe; mo != nil {
			mats(&mo.routerProj)
			for e := range mo.expertsGateUp {
				mats(&mo.expertsGateUp[e], &mo.expertsDown[e])
			}
		}
	}
	slot0 = append(slot0, webgpuVkScratch)
	return slot0, slotKV, weights
}
