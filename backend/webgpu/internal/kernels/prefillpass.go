//go:build gpu

package kernels

import (
	"fmt"
	"time"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
)

// prefillPass is one prefillLastRows call: its command encoder, the buffers and bind groups it releases at the end, the
// per-layer buffer pool, and the first device-allocation or bind failure. Its op builders work on PACKED [M, width]
// row-major buffers: one dispatch across all M rows, not M separate dispatches (docs/code-notes/gpu.md#ensurePrefillBatched
// has the measured before and after).
type prefillPass struct {
	c                          *Context
	m                          ModelW
	pp                         prefillProjs
	M                          int
	hidden, nH, nKV, hd, inter int
	positions                  []int
	eps, scale                 float32
	addOne                     bool

	enc  *wgpu.CommandEncoder
	keep []func()
	// buildErr accumulates the FIRST device-allocation/bind failure (as DecodeTokenFusedBatched
	// does): short-circuit once set, return it before Submit, so
	// VRAM exhaustion is an error the caller falls back on rather than a panic or a
	// downstream nil-buffer deref.
	buildErr error
	// A layer's buffers are dead once the layer is encoded (the residual
	// stream is updated in place, K/V are copied into the cache), so each layer recycles the one before it: storF hands
	// out a free buffer of the same size before it creates one. A recycled buffer was written before, so wgpu does not
	// zero-fill it again, and the pass holds one layer's intermediates instead of every layer's. The queue orders the
	// dispatches, so a later layer's writes follow the earlier layer's reads. prefillBufFresh (tests only) restores a
	// fresh buffer per call; prefillBufPoison (tests only) overwrites every recycled buffer with NaN bytes first.
	free      map[int][]*wgpu.Buffer
	layerBufs []*wgpu.Buffer
	poison    *wgpu.Buffer
	dispCount int

	// mposBuf is each row's three rotary positions and axisFor a head half's axis table, for an m-RoPE pass.
	mposBuf  *wgpu.Buffer
	axisBufs map[int]*wgpu.Buffer
	// xd is the packed [M, hidden] residual stream; pleIn a Gemma 4 E-model's per-layer inputs, pleStride floats a layer.
	xd        *wgpu.Buffer
	pleIn     *wgpu.Buffer
	pleStride int
	// pt is the open profiling category's start (time.Time{} with profiling off); sinks0 and hs0 the single-row attention
	// kernel's unused sink bindings.
	pt          time.Time
	sinks0, hs0 *wgpu.Buffer
}

// prefillLayer is one layer's geometry and cache buffers as the pass reads them.
type prefillLayer struct {
	i              int
	lw             *LayerW
	lp             *prefillLayerProj
	lhd, lnKV      int
	lkvDim, qRow   int
	kCache, vCache *wgpu.Buffer
}

// release runs at the end of prefillLastRows: the open encoder, then every buffer and bind group the pass kept.
func (pa *prefillPass) release() {
	pa.enc.Release()
	for _, f := range pa.keep {
		f()
	}
}

func (pa *prefillPass) keepBuf(b *wgpu.Buffer) {
	if PrefillBufSeenForTest != nil {
		PrefillBufSeenForTest(int64(b.GetSize()))
	}
	pa.keep = append(pa.keep, releaser(b))
}

func (pa *prefillPass) keepBG(b *wgpu.BindGroup) { pa.keep = append(pa.keep, b.Release) }

func (pa *prefillPass) recycle() {
	for _, b := range pa.layerBufs {
		n := int(b.GetSize() / 4)
		pa.free[n] = append(pa.free[n], b)
	}
	pa.layerBufs = pa.layerBufs[:0]
}

func (pa *prefillPass) storF(n int) *wgpu.Buffer {
	c := pa.c
	if pa.buildErr != nil {
		return nil
	}
	if l := pa.free[n]; !PrefillBufFresh && len(l) > 0 {
		b := l[len(l)-1]
		pa.free[n] = l[:len(l)-1]
		if prefillBufPoison {
			if pa.poison == nil || pa.poison.GetSize() < uint64(n*4) {
				fill := make([]uint32, n)
				for i := range fill {
					fill[i] = 0xFFFFFFFF
				}
				p, e := c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(fill), Usage: wgpu.BufferUsageCopySrc})
				if e != nil {
					pa.buildErr = e
					return nil
				}
				pa.keepBuf(p)
				pa.poison = p
			}
			if e := pa.enc.TryCopyBufferToBuffer(pa.poison, 0, b, 0, uint64(n*4)); e != nil {
				pa.buildErr = e
				return nil
			}
		}
		pa.layerBufs = append(pa.layerBufs, b)
		prefillBufReused.Add(1)
		return b
	}
	b, e := c.tryCreateBuffer(mem.LineScratch, &wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst})
	if e != nil {
		pa.buildErr = e
		return nil
	}
	pa.keepBuf(b)
	pa.layerBufs = append(pa.layerBufs, b)
	return b
}

func (pa *prefillPass) uni(v []uint32) *wgpu.Buffer {
	if pa.buildErr != nil {
		return nil
	}
	b, e := pa.c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(v), Usage: wgpu.BufferUsageUniform})
	if e != nil {
		pa.buildErr = e
		return nil
	}
	pa.keepBuf(b)
	return b
}

func (pa *prefillPass) bind(layout *wgpu.BindGroupLayout, bufs ...*wgpu.Buffer) *wgpu.BindGroup {
	if pa.buildErr != nil {
		return nil
	}
	es := make([]wgpu.BindGroupEntry, len(bufs))
	for i, b := range bufs {
		if b == nil {
			pa.buildErr = fmt.Errorf("gpu: PrefillLastW8A8: nil buffer for binding %d (allocation failed)", i)
			return nil
		}
		es[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.GetSize()}
	}
	bg, e := pa.c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: es})
	if e != nil {
		pa.buildErr = e
		return nil
	}
	pa.keepBG(bg)
	return bg
}

// flushPasses: same Metal-uncommitted-command-buffer-cap workaround as
// DecodeTokenFusedBatched (see its doc comment) — a real constraint here too, and
// worse: real prefill M runs to the hundreds/thousands vs speculative verify's ≤16.
func (pa *prefillPass) flushPasses() {
	if pa.buildErr != nil {
		return
	}
	cmd, e := pa.enc.TryFinish(nil)
	if e != nil {
		pa.buildErr = e
		return
	}
	pa.c.queue.Submit(cmd)
	cmd.Release()
	pa.enc.Release()
	pa.enc, e = pa.c.device.TryCreateCommandEncoder(nil)
	if e != nil {
		pa.buildErr = e
	}
}

const prefillPassesPerFlush = 32

func (pa *prefillPass) disp(pl *wgpu.ComputePipeline, bg *wgpu.BindGroup, gx, gy uint32) {
	if pa.buildErr != nil || bg == nil {
		return
	}
	pass := pa.enc.BeginComputePass(nil)
	pass.SetPipeline(pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(gx, gy, 1)
	if err := pass.TryEnd(); err != nil {
		pa.buildErr = err
	}
	pass.Release()
	pa.dispCount++
	if pa.dispCount >= prefillPassesPerFlush {
		pa.flushPasses()
		pa.dispCount = 0
	}
}

func (pa *prefillPass) cpy(src *wgpu.Buffer, so uint64, dst *wgpu.Buffer, do, sz uint64) {
	if pa.buildErr != nil || src == nil || dst == nil {
		return
	}
	if err := pa.enc.TryCopyBufferToBuffer(src, so, dst, do, sz); err != nil {
		pa.buildErr = err
	}
}

// prefillBindEntry / bindOff: like bind(), but each argument may be a byte-range VIEW into
// a larger buffer rather than the whole thing — needed by dispFlat's chunking below.
// Every offset dispFlat passes is a multiple of maxChunkElems*4 bytes, and
// maxChunkElems is itself a multiple of 64 (the workgroup size), so every offset is
// automatically a multiple of 256 — satisfying GPUBindGroupEntry.offset's
// minStorageBufferOffsetAlignment requirement (the same constraint that ruled out
// per-row offset views for the attention step, where nH*hd*4 was not
// guaranteed 256-aligned; a chunk boundary chosen as a multiple of the workgroup
// size always is).
type prefillBindEntry struct {
	buf    *wgpu.Buffer
	off, n uint32 // n in ELEMENTS (f32); off in elements too
}

func whole(b *wgpu.Buffer) prefillBindEntry { return prefillBindEntry{buf: b} }

func (pa *prefillPass) bindOff(layout *wgpu.BindGroupLayout, entries ...prefillBindEntry) *wgpu.BindGroup {
	if pa.buildErr != nil {
		return nil
	}
	es := make([]wgpu.BindGroupEntry, len(entries))
	for i, e := range entries {
		if e.buf == nil {
			pa.buildErr = fmt.Errorf("gpu: PrefillLastW8A8: nil buffer for binding %d (allocation failed)", i)
			return nil
		}
		sz := e.buf.GetSize() - uint64(e.off)*4
		if e.n != 0 {
			sz = uint64(e.n) * 4
		}
		es[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: e.buf, Offset: uint64(e.off) * 4, Size: sz}
	}
	bg, e := pa.c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: es})
	if e != nil {
		pa.buildErr = e
		return nil
	}
	pa.keepBG(bg)
	return bg
}

// maxChunkElems: WebGPU limits a dispatch to 65535 workgroups per dimension
// (maxComputeWorkgroupsPerDimension) and each workgroup here covers 64 elements, so one 1-D
// dispatch tops out at 65535*64 elements; larger elementwise work is chunked. Exceeding the
// limit is a validation error that PrefillLast declines into, not a wrong-output risk.
const maxChunkElems = 65535 * 64

func (pa *prefillPass) dispFlat(pl *wgpu.ComputePipeline, ly *wgpu.BindGroupLayout, total int, bufs ...*wgpu.Buffer) {
	off := 0
	for off < total {
		n := min(total-off, maxChunkElems)
		p := pa.uni([]uint32{uint32(n), 0, 0, 0})
		entries := make([]prefillBindEntry, 0, len(bufs)+1)
		for _, b := range bufs {
			entries = append(entries, prefillBindEntry{buf: b, off: uint32(off), n: uint32(n)})
		}
		entries = append(entries, whole(p))
		pa.disp(pl, pa.bindOff(ly, entries...), uint32(n+63)/64, 1)
		off += n
	}
}

// rmsQuantB is decode's fused RMSNorm→int8 for all M rows of a packed [M, hidden] buffer, one workgroup a row,
// into the [M, kp/4] packed-int8 / [M] scales layout a projection reads.
func (pa *prefillPass) rmsQuantB(inPacked, w *wgpu.Buffer, kp int) (aqC, asC *wgpu.Buffer) {
	M := pa.M
	aqC, asC = pa.storF(M*kp/4), pa.storF(M)
	p := pa.uni([]uint32{uint32(pa.hidden), F32bits(pa.eps), BoolU32(pa.addOne), uint32(kp)})
	f := pa.c.fusedRows[fusedRowsRMS]
	pa.disp(f.Pl, pa.bind(f.Layout, inPacked, w, aqC, asC, p), uint32(M), 1)
	return aqC, asC
}

// qkNormB normalizes ALL M rows of a packed [M, heads*hd] buffer in place with
// weight [hd], before RoPE (Qwen3/GLM). Each workgroup processes one head across
// the batch.
func (pa *prefillPass) qkNormB(vecPacked, weight *wgpu.Buffer, heads, lhd int) {
	c := pa.c
	totalHeads := uint32(pa.M * heads)
	p := pa.uni([]uint32{totalHeads, uint32(lhd), F32bits(pa.eps), BoolU32(pa.addOne)})
	gx := totalHeads
	gy := uint32(1)
	if gx > 65535 {
		gx = 65535
		gy = (totalHeads + 65534) / 65535
	}
	pa.disp(c.qkNormPipeline, pa.bind(c.qkNormLayout, vecPacked, weight, p), gx, gy)
}

func (pa *prefillPass) axisFor(half int) *wgpu.Buffer {
	if b, ok := pa.axisBufs[half]; ok || pa.buildErr != nil {
		return b
	}
	ax := pa.pp.MropeAxis(half)
	if len(ax) != half {
		pa.buildErr = fmt.Errorf("gpu: the batched pass has no m-RoPE axis table for a head half of %d", half)
		return nil
	}
	tab := make([]uint32, half)
	for d := range tab {
		tab[d] = uint32(ax[d])
		switch PrefillDefectForTest {
		case 12:
			tab[d] = uint32(ax[(d+1)%half])
		case 14:
			tab[d] = []uint32{0, 2, 1}[ax[d]]
		}
	}
	b, e := pa.c.tryCreateBufferInit(mem.LineScratch, &wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(tab), Usage: wgpu.BufferUsageStorage})
	if e != nil {
		pa.buildErr = e
		return nil
	}
	pa.keepBuf(b)
	pa.axisBufs[half] = b
	return b
}

// ropeB rotates ALL M rows of a packed [M, heads*hd] buffer in place, in one
// dispatch. positions are always start, start+1, …, start+M-1 within one
// PrefillLastW8A8 call (residency.go), so passing start (not a per-row array) is
// enough — ropeBatchedShaderWGSL recovers row r's position as start+r.
func (pa *prefillPass) ropeB(vecPacked, invFreq *wgpu.Buffer, heads, lhd int) {
	c, M := pa.c, pa.M
	half := lhd / 2
	rpos := pa.positions[0]
	if PrefillDefectForTest == 1 {
		rpos++
	}
	p := pa.uni([]uint32{uint32(heads), uint32(lhd), uint32(half), uint32(rpos), F32bits(1), 0, 0, 0})
	if pa.mposBuf != nil {
		pa.disp(c.ropeMRoPERows.Pl, pa.bind(c.ropeMRoPERows.Layout, vecPacked, invFreq, p, pa.mposBuf, pa.axisFor(half)), uint32(heads*half+63)/64, uint32(M))
		return
	}
	pa.disp(c.ropeBatchedPipeline, pa.bind(c.ropeBatchedLayout, vecPacked, invFreq, p), uint32(heads*half+63)/64, uint32(M))
}

// residualB dispatches its elementwise add as ONE call over the full M*width flattened range instead of M separate
// width-sized dispatches: the kernel indexes by a flat global_invocation_id with no cross-row state
// (residualShaderWGSL, layer.go), so [M, width] IS [M*width] to it.
func (pa *prefillPass) residualB(xPacked, yPacked *wgpu.Buffer) *wgpu.Buffer { // x += y, in place; returns x
	pa.dispFlat(pa.c.residualPipeline, pa.c.residualLayout, pa.M*pa.hidden, xPacked, yPacked)
	return xPacked
}

// gatedQuantB is decode's fused gated product→int8 for all M rows, one workgroup a row, over n activations a row:
// the [M, n] product is never stored. up is a range of a buffer, so one layer's per-layer inputs bind out of the
// model's.
func (pa *prefillPass) gatedQuantB(gatePacked *wgpu.Buffer, up prefillBindEntry, n, kp int) (aqC, asC *wgpu.Buffer) {
	M := pa.M
	aqC, asC = pa.storF(M*kp/4), pa.storF(M)
	p := pa.uni([]uint32{uint32(n), uint32(kp), 0, 0})
	f := pa.c.fusedRows[fusedRowsSwiGLU]
	if pa.m.GatedGELU {
		f = pa.c.fusedRows[fusedRowsGeGLU]
	}
	pa.disp(f.Pl, pa.bindOff(f.Layout, whole(gatePacked), up, whole(aqC), whole(asC), whole(p)), uint32(M), 1)
	return aqC, asC
}

// rmsF32B is decode's float RMSNorm (rmsnormShaderWGSL) for all M rows of a packed [M, hidden] buffer.
func (pa *prefillPass) rmsF32B(inPacked, w *wgpu.Buffer) *wgpu.Buffer {
	out := pa.storF(pa.M * pa.hidden)
	p := pa.uni([]uint32{uint32(pa.hidden), F32bits(pa.eps), BoolU32(pa.addOne), 0})
	pa.disp(pa.c.rmsnormBatchedPipeline, pa.bind(pa.c.rmsnormBatchedLayout, inPacked, w, out, p), 1, uint32(pa.M))
	return out
}

// vNormB is decode's value norm (vNormWGSL, one workgroup a head) over every row's heads at once.
func (pa *prefillPass) vNormB(src *wgpu.Buffer, heads, lhd int) *wgpu.Buffer {
	M := pa.M
	dst := pa.storF(M * heads * lhd)
	if M*heads > 65535 && pa.buildErr == nil {
		pa.buildErr = fmt.Errorf("gpu: the batched pass declines %d rows of %d value heads in one dispatch", M, heads)
	}
	p := pa.uni([]uint32{uint32(M * heads), uint32(lhd), F32bits(pa.eps), 0})
	pa.disp(pa.c.vNormPipeline, pa.bind(pa.c.vNormLayout, src, dst, p), uint32(M*heads), 1)
	return dst
}

// scaleB multiplies every row of a packed [M, hidden] buffer by a layer's output scalar, in place.
func (pa *prefillPass) scaleB(vec *wgpu.Buffer, scalar float32) {
	n := pa.M * pa.hidden
	p := pa.uni([]uint32{uint32(n), F32bits(scalar), 0, 0})
	pa.disp(pa.c.scaleVecPipeline, pa.bind(pa.c.scaleVecLayout, vec, p), uint32(n+255)/256, 1)
}

// quantPackedM quantizes all M rows of a packed [M, K] buffer in ONE dispatch, straight into the [M, kp/4]
// packed-int8 / [M] scales layout a projection reads (quantizeShaderWGSL, device.go, takes M rows natively via
// QDims.m). It is decode's quantizer for an activation that no fused kernel produces: the attention context.
func (pa *prefillPass) quantPackedM(inPacked *wgpu.Buffer, K, kp int) (aqC, asC *wgpu.Buffer) {
	M := pa.M
	aqC, asC = pa.storF(M*kp/4), pa.storF(M)
	p := pa.uni([]uint32{uint32(M), uint32(K), uint32(kp), 0})
	pa.disp(pa.c.quantizePipeline, pa.bind(pa.c.quantizeLayout, inPacked, aqC, asC, p), uint32(M), 1)
	return aqC, asC
}

// sharedQ returns the projections of one quantized input through rms. The
// input is quantized once when every projection has the same K and reads it at the same padding, which q/k/v and
// gate/up of one precision always do. Quantization is deterministic, so each projection reads the bytes it would
// have read from its own copy; prefillQuantPerProj (tests only) restores a quantization each.
func (pa *prefillPass) sharedQ(quantize func(kp int) (aqC, asC *wgpu.Buffer), rms []prefillProj, biases []*wgpu.Buffer) []*wgpu.Buffer {
	out := make([]*wgpu.Buffer, len(rms))
	same := !PrefillQuantPerProj
	for _, rm := range rms[1:] {
		same = same && rm.cols() == rms[0].cols() && rm.kp() == rms[0].kp()
	}
	var aqC, asC *wgpu.Buffer
	if same {
		aqC, asC = quantize(rms[0].kp())
		PrefillSharedQuants.Add(1)
	}
	for i, rm := range rms {
		if PrefillDefectForTest == 3 && i == 0 && len(rms) == 3 && rm.w4 != nil && rms[1].w4 != nil {
			wrong := *rm.w4
			wrong.bScales = rms[1].w4.bScales
			rm = prefillProj{w4: &wrong}
		}
		if same {
			out[i] = pa.projQB(aqC, asC, pa.M, rm, biases[i])
		} else {
			a, as := quantize(rm.kp())
			out[i] = pa.projQB(a, as, pa.M, rm, biases[i])
		}
	}
	return out
}

// projQB runs one projection over rowsM quantized rows and returns the packed [rowsM, N] output: an int8 matrix
// through ONE unbounded-M tiled GEMM (weight streamed once, DP4A when available), an int4 one through a multi-row
// int4 kernel.
//
// bias (nil for every projection except Qwen2's q/k/v) selects, for an int8 matrix, the bias-epilogue tiled kernel
// instead of a separate add: the add has to be fused into the GEMM's expression to stay bit-exact with production's
// gemvBias (docs/code-notes/gpu.md#runModelToModelW). An int4 matrix adds it in a separate dispatch, as decode does.
func (pa *prefillPass) projQB(aqC, asC *wgpu.Buffer, rowsM int, pj prefillProj, bias *wgpu.Buffer) *wgpu.Buffer {
	c := pa.c
	N := pj.rows()
	dstC := pa.storF(rowsM * N)
	if pj.w4 != nil {
		if pa.buildErr != nil {
			return dstC
		}
		kind := c.w4MultiFor(rowsM)
		encodeW4Multi(kind, c.w4Multi[kind], pj.w4, rowsM, aqC, asC, dstC, pa.uni, pa.bind, pa.disp)
		if bias != nil {
			total := uint32(rowsM * N)
			gx := (total + 63) / 64
			gy := uint32(1)
			if gx > 65535 {
				gy = (gx + 65534) / 65535
				gx = 65535
			}
			p := pa.uni([]uint32{total, uint32(N), gx * 64, 0})
			pa.disp(c.biasRowsPipeline, pa.bind(c.biasRowsLayout, dstC, bias, p), gx, gy)
		}
		return dstC
	}
	rm := pj.w8
	p := pa.uni([]uint32{uint32(rowsM), uint32(rm.kp), uint32(N), 0})
	gx, gy := c.gemmGrid(N, rowsM)
	if bias != nil {
		pa.disp(c.tiledBiasPipeline, pa.bind(c.tiledBiasLayout, aqC, rm.bq, asC, rm.bScales, dstC, p, bias), gx, gy)
	} else {
		pa.disp(c.tiledPipeline, pa.bind(c.tiledLayout, aqC, rm.bq, asC, rm.bScales, dstC, p), gx, gy)
	}
	return dstC
}

// The prefill decomposition (gpu/prefill_prof.go): pt is a no-op (time.Time{}) whenever c.prefillProf is
// nil, so profiling off costs one nil-check per call. Category boundaries are syncs (as in cudaResident's
// profTic/profToc), so the category sum runs a bit over pipelined wall time; numbers in
// docs/measurements/webgpu-prefill-decomp-2026-09-22.md. residualB and swigluB (elementwise) fold into
// normsRope, the same "glue" role cuda/prefill.go's glueCat plays.
// profBoundary closes one category. It FLUSHES the open encoder first: Poll only waits for submitted work,
// so dispatches still in the encoder are invisible to it and the split would follow the 32-dispatch flush
// cadence, not the categories. The extra flushes happen only when profiling is on.
func (pa *prefillPass) profBoundary(cat prefillProfCat) {
	if pa.c.prefillProf == nil {
		return
	}
	pa.flushPasses()
	pa.dispCount = 0
	pa.c.prefillProfToc(cat, pa.pt)
	pa.pt = pa.c.prefillProfTic()
}
