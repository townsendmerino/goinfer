package memprice

import (
	"runtime"
	"slices"

	"github.com/townsendmerino/aikit/linalg"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

const (
	WebGPUVkInternal  = 512<<10 + 256 + 256
	WebGPUVkAlign     = 256
	WebGPUVkDevFirst  = 128 << 20
	WebGPUVkDevMax    = 256 << 20
	WebGPUVkHostBlock = 64 << 20
	WebGPUVkDevice    = 144 << 20
	WebGPUVkScratch   = 8 << 20

	WebGPUVulkan  = runtime.GOOS != "darwin"
	WebGPUVkFixed = WebGPUVkDevice + WebGPUVkHostBlock
)

// WebGPUCtxCeilingFor returns WebGPU's fixed per-precision KV capacity ceiling.
func WebGPUCtxCeilingFor(kvF16, kvI8 bool) int {
	switch {
	case kvI8:
		return 65536
	case kvF16:
		return 32768
	default:
		return 16384
	}
}

// WebGPUVkBlocks returns the device-local bytes gpu-allocator 0.28.0 opens for sizes allocated in order.
func WebGPUVkBlocks(sizes []int64) int64 {
	var free []int64
	var total int64
	place := func(r int64) {
		r = (max(r, 4) + 3) / 4 * 4
		r = (r + WebGPUVkAlign - 1) / WebGPUVkAlign * WebGPUVkAlign
		mb := min(int64(WebGPUVkDevFirst)<<min(len(free), 7), WebGPUVkDevMax)
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
	place(WebGPUVkInternal)
	for _, n := range sizes {
		place(n)
	}
	return total
}

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

func webgpuProjBuffers(w *linalg.WeightMat) []int64 {
	return webgpuStackedBuffers(1, w)
}

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

// WebGPUKVRowBytes returns one position's row of b in a WebGPU cache and any separate f32 scale bytes.
func WebGPUKVRowBytes(b program.Buf, f16, i8 bool) (row, scale int64) {
	switch {
	case b.Prec != program.PrecKV:
		return 4 * int64(b.Elems), 0
	case i8:
		return int64(b.Elems), 4 * int64(b.Heads)
	case f16:
		return 2 * int64(b.Elems), 0
	}
	return 4 * int64(b.Elems), 0
}

// WebGPUKVBuffersOf returns one layer's WebGPU cache buffer sizes at ctx from its declared buffers.
func WebGPUKVBuffersOf(declared []program.Buf, ctx int, kvF16, kvI8 bool) []int64 {
	if ctx <= 0 {
		return nil
	}
	var bufs, scales []int64
	for _, b := range declared {
		if b.Unit != program.PerPosition {
			continue
		}
		row, scale := WebGPUKVRowBytes(b, kvF16, kvI8)
		bufs = append(bufs, row*int64(ctx))
		if scale > 0 {
			scales = append(scales, scale*int64(ctx))
		}
	}
	return append(bufs, scales...)
}

// WebGPUKVBytesPerPos returns the per-position KV byte rate across all layers on WebGPU.
func WebGPUKVBytesPerPos(nLayers int, bufsOf func(int) []program.Buf, f16, i8 bool) int64 {
	var perPos int64
	for l := range nLayers {
		for _, b := range bufsOf(l) {
			if b.Unit != program.PerPosition {
				continue
			}
			row, scale := WebGPUKVRowBytes(b, f16, i8)
			perPos += row + scale
		}
	}
	return perPos
}

// WebGPULayerCaches returns layer i's per-sequence buffers on WebGPU at ctx, and whether they are positional KV.
func WebGPULayerCaches(w *darch.Weights, bufsOf func(int) []program.Buf, i, ctx int, kvF16, kvI8 bool) (bufs []int64, kv bool) {
	if w == nil || w.Arch == nil {
		return nil, false
	}
	a := w.Arch
	mixer, _ := LayerOps(a, i)
	switch mixer {
	case program.MixerAttention, program.MixerMLA:
		return WebGPUKVBuffersOf(bufsOf(i), ctx, kvF16, kvI8), true
	case program.MixerDeltaNet:
		q := a.Qwen35
		if q == nil {
			return nil, false
		}
		convK, hk, hv, nk, nv := q.ConvKernel, q.KeyHeadDim, q.ValueHeadDim, q.NumKeyHeads, q.NumValueHeads
		return []int64{int64(convK-1) * int64(2*nk*hk+nv*hv) * 4, int64(nv*hv*hk) * 4}, false
	case program.MixerMamba:
		var nH, hd, dS, nG, dConv int
		if g := a.Granite; g != nil {
			nH, hd, dS, nG, dConv = g.NHeads, g.HeadDim, g.DState, g.NGroups, g.DConv
		} else if np := a.Nemotron; np != nil {
			nH, hd, dS, nG, dConv = np.NHeads, np.HeadDim, np.DState, np.NGroups, np.DConv
		} else {
			return nil, false
		}
		return []int64{int64(dConv-1) * int64(nH*hd+2*nG*dS) * 4, int64(nH*hd*dS) * 4}, false
	}
	return nil, false
}

// WebGPUDeviceBuffers returns the buffer list a WebGPU resident build allocates at ctx in creation order.
func WebGPUDeviceBuffers(w *darch.Weights, bufsOf func(int) []program.Buf, ctx int, kvF16, kvI8 bool) (slot0, slotKV []int64, weights int64) {
	if w == nil || w.Arch == nil {
		return nil, nil, 0
	}
	a := w.Arch
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
			if n := F32SliceBytes(s); n > 0 {
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
	w8a8 := func(rows, cols int) {
		if rows <= 0 || cols <= 0 {
			return
		}
		for _, n := range []int64{int64(rows) * int64((cols+15)&^15), 4 * int64(rows)} {
			slot0 = append(slot0, n)
			weights += n
		}
	}
	hidden, nLayers, nH := a.HiddenDim, a.NumLayers, a.NumHeads
	for i := range w.Layers {
		var bufs []int64
		var kv bool
		if i < nLayers {
			bufs, kv = WebGPULayerCaches(w, bufsOf, i, ctx, kvF16, kvI8)
		}
		caches := func() {
			slot0 = append(slot0, bufs...)
			if kv {
				slotKV = append(slotKV, bufs...)
			}
		}
		l := &w.Layers[i]
		switch d, mb := l.Delta, l.Mamba; {
		case d != nil:
			mats(&d.InProjQKV, &d.InProjZ, &d.OutProj)
			w8a8(len(d.InProjB)/max(hidden, 1), hidden)
			w8a8(len(d.InProjA)/max(hidden, 1), hidden)
			f32s(d.ConvW, d.DtBias, d.NegExpA, d.NormW)
			caches()
		case mb != nil:
			if mb.InQ.Rows() != 0 {
				w8a8(mb.InQ.Rows(), mb.InQ.Cols())
				w8a8(mb.OutQ.Rows(), mb.OutQ.Cols())
			} else {
				w8a8(len(mb.InProj)/max(hidden, 1), hidden)
				w8a8(hidden, len(mb.OutProj)/max(hidden, 1))
			}
			f32s(mb.ConvW, mb.ConvB, make([]float32, 3*len(mb.ALog)), mb.NormW)
			caches()
		default:
			caches()
		}
		mats(&l.QProj, &l.KProj, &l.VProj, &l.OProj, &l.GProj)
		if q := l.QAttn; q != nil {
			mats(&q.QProj, &q.KProj, &q.VProj, &q.OProj)
		}
		if mlaW := l.MLA; mlaW != nil && a.MLA != nil {
			mla := a.MLA
			qLoRA, kvLoRA, qkNope, qkRope, vHead := mla.QLoRARank, mla.KVLoRARank, mla.QKNopeHeadDim, mla.QKRopeHeadDim, mla.VHeadDim
			proj := func(q *linalg.WeightMat, rows, cols int) {
				if q.Rows() != 0 {
					rows, cols = q.Rows(), q.Cols()
				}
				w8a8(rows, cols)
			}
			if qLoRA > 0 {
				proj(&mlaW.QAQ, qLoRA, hidden)
				proj(&mlaW.QBQ, nH*(qkNope+qkRope), qLoRA)
			} else {
				proj(&mlaW.QQ, nH*(qkNope+qkRope), hidden)
			}
			proj(&mlaW.KVAQ, kvLoRA+qkRope, hidden)
			proj(&mlaW.OQ, hidden, nH*vHead)
			for _, n := range []int64{4 * int64(nH*kvLoRA*qkNope), 4 * int64(nH*vHead*kvLoRA)} {
				slot0 = append(slot0, n)
				weights += n
			}
		}
		if k := l.KDA; k != nil {
			mats(&k.QProj, &k.KProj, &k.VProj, &k.FProj, &k.GProj, &k.OProj)
			f32s(k.QConvW, k.KConvW, k.VConvW, k.BProj, k.DtBias, k.ALog, k.ONormW)
		}
		if sc := l.ShortConv; sc != nil {
			mats(&sc.InProj, &sc.OutProj)
			f32s(sc.ConvW)
		}
		mo := l.Gemma4MoE
		if mo != nil {
			if n := 4 * int64(mo.RouterProj.Rows()) * int64(mo.RouterProj.Cols()); n > 0 {
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
		if mo != nil && len(mo.ExpertsGateUp) > 0 {
			stacked(len(mo.ExpertsGateUp), &mo.ExpertsGateUp[0])
			stacked(len(mo.ExpertsDown), &mo.ExpertsDown[0])
		}
	}
	slot0 = append(slot0, WebGPUVkScratch)
	return slot0, slotKV, weights
}

// WebGPUDerivedTableBytes returns the RoPE, attention-sink, DeltaNet conv-bias, and Gemma 4 router tables WebGPU derives beside its weights.
func WebGPUDerivedTableBytes(w *darch.Weights) int64 {
	if w == nil || w.Arch == nil {
		return 0
	}
	a := w.Arch
	var tabs [][]float32
	var n int64
	for i := range a.NumLayers {
		if inv := LayerRopeInvFreq(a, i); len(inv) > 0 && !slices.ContainsFunc(tabs, func(t []float32) bool { return slices.Equal(t, inv) }) {
			tabs = append(tabs, inv)
			n += 4 * int64(len(inv))
		}
		if !a.HasAttnSinks() || i >= len(w.Layers) || len(w.Layers[i].AttnSinks) == 0 {
			n += 4
		}
		if i < len(w.Layers) {
			if g4 := w.Layers[i].Gemma4MoE; g4 != nil {
				n += 4 * (int64(a.HiddenDim) + int64(g4.NE) - int64(len(g4.RouterScale)))
			}
		}
	}
	if q := a.Qwen35; q != nil {
		n += 4 * int64(2*q.NumKeyHeads*q.KeyHeadDim+q.NumValueHeads*q.ValueHeadDim)
	}
	return n
}

// WebGPUDenseLine returns WebGPU's planned dense weight line (uploaded dense matrices plus vectors, expert biases, and derived tables).
func WebGPUDenseLine(w *darch.Weights, bufsOf func(int) []program.Buf) int64 {
	_, _, dense := WebGPUDeviceBuffers(w, bufsOf, 0, false, false)
	return dense + DenseVectorBytes(w) + ExpertBiasBytes(w) + WebGPUDerivedTableBytes(w)
}
