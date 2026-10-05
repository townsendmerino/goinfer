//go:build darwin

package metal

import (
	"fmt"
	"runtime"
	"slices"
	"time"
)

// g4LayerMajorOn routes a paged Gemma 4 MoE's prompt through prefillG4Paged (docs/tasks/task-m26-mac-2026-10.md, 4b)
// instead of the sequential loop. ON since 2026-10-04 (owner's bar, ship at >= 1.02x for a bit-identical change; a smoke
// on M26 read 1.42x); the night grade at M = 128 and 512 turns it off if it reads below 1.00x.
var g4LayerMajorOn = true

// g4LayerMajorChunk is the most prompt rows one pass holds: each row keeps its own residual, seam state and uniforms
// (about 25 KB on M26), and every row's dispatches for a layer go in one command buffer.
const g4LayerMajorChunk = 256

// g4Row is one prompt row's own state in the layer-major pass: what crosses the router seam of a paged Gemma 4 MoE
// layer (the residual x, the dense branch g4x1, the expert input mq/mSc, the route rIdx/rWgt, the pool rows slotIdx)
// and the row's position uniforms. Everything else an encoder binds is scratch written and read within one row's
// dispatches, which a serial compute encoder keeps apart.
type g4Row struct {
	pos                                 int
	x, x1, mq, mSc, rIdx, rWgt, slotIdx Buffer
	u                                   posUniforms
}

// prefillG4Paged runs a prompt through a paged Gemma 4 MoE layer by layer: per layer, one command buffer runs every
// row's attention and phase 1 (dense branch, router, expert input); the host reads the routes, stages the union of the
// routed experts into the pool once, and runs every row's phase 2 and join from the pool, as many rows to a command
// buffer as the pool holds their experts at once, in row order. Each row's dispatches are exactly decode's
// (forwardLogitsPaged's) at its position on its own buffers, so the K/V and the last row's logits are the sequential
// loop's bit for bit. M26's attention (head dim 256 and 512 by layer, K = V, a window) takes no batched pass; this
// keeps decode's kernels and removes the per-token submit-and-wait trips and the per-token staging. Returns the last
// row's logits, or nil without the head when head is false.
func (r *resident) prefillG4Paged(embs [][]float32, startPos int, head bool) (logits []float32) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer func() { // the paged forward's own discipline (forwardLogitsPaged): a staging panic becomes execErr
		if p := recover(); p != nil {
			r.recordExecErr(fmt.Errorf("metal: layer-major paged prefill aborted: %v", p))
			logits = nil
		}
	}()
	r.g4LayerMajorRuns++
	for c := 0; c < len(embs); c += g4LayerMajorChunk {
		end := min(c+g4LayerMajorChunk, len(embs))
		logits = r.g4LayerMajorRows(embs[c:end], startPos+c, head && end == len(embs))
		if r.execErr != nil {
			return nil
		}
	}
	return logits
}

func (r *resident) g4LayerMajorRows(embs [][]float32, start int, head bool) []float32 {
	g, d := r.g4moe, r.d
	var made []Buffer
	nb := func(b Buffer) Buffer { made = append(made, b); return b }
	defer func() {
		for _, b := range made {
			d.ReleaseBuf(b)
		}
	}()
	rows := make([]g4Row, len(embs))
	for i := range rows {
		rw := &rows[i]
		rw.pos = start + i
		rw.x, rw.x1 = nb(NewBufferFloats(d, embs[i])), nb(d.NewBufferLen(r.H))
		rw.mq, rw.mSc = nb(byteBuf(d, r.H)), nb(d.NewBufferLen(1))
		rw.rIdx, rw.rWgt = nb(NewBufferUint32s(d, make([]uint32, g.topK))), nb(d.NewBufferLen(g.topK))
		rw.slotIdx = nb(NewBufferUint32s(d, make([]uint32, g.topK)))
		rw.u = posUniforms{nb(NewBufferU32(d, 0)), nb(NewBufferU32(d, 0)), nb(NewBufferU32(d, 0)),
			nb(NewBufferFloats(d, []float32{1})), nb(NewBufferU32(d, 0))}
		r.writePosUniforms(rw.u, rw.pos, rw.pos)
	}
	// Each row's encodes bind its own buffers: the encoders read these fields at encode time.
	sx, smq, smSc, sx1, sx2, sIdx, sWgt, sSlot := r.x, r.mq, r.mSc, g.g4x1, g.g4x2, g.rIdx, g.rWgt, g.slotIdx
	defer func() {
		r.x, r.mq, r.mSc, g.g4x1, g.g4x2, g.rIdx, g.rWgt, g.slotIdx = sx, smq, smSc, sx1, sx2, sIdx, sWgt, sSlot
		r.encNKeys, r.encFANSplit = 0, Buffer{}
	}()
	use := func(rw *g4Row) {
		r.x, r.mq, r.mSc, g.g4x1, g.rIdx, g.rWgt, g.slotIdx = rw.x, rw.mq, rw.mSc, rw.x1, rw.rIdx, rw.rWgt, rw.slotIdx
		r.encNKeys, r.encFANSplit = rw.pos+1, rw.u.uFANSplit
	}
	submit := func(e *Encoder, w0 time.Time, gpu, wall *int64) {
		e.End()
		r.recordExecErr(e.Err())
		lmProf(e, w0, gpu, wall)
	}
	// D-P01: a slot group's phase 2 through the batched expert kernels (prefill_g4batch.go), when the shapes admit it
	var batch g4Batch
	group := func(L *residLayer, pool *expertPool) func(e *Encoder, i, j int) {
		if !r.g4ExpertBatchOK() {
			return nil
		}
		return func(e2 *Encoder, i, j int) {
			grp := make([]*g4Row, 0, j-i)
			for k := i; k < j; k++ {
				grp = append(grp, &rows[k])
			}
			r.encodeG4Phase2Batch(e2, pool, &batch, grp, func(e *Encoder, rw *g4Row, x2 Buffer) {
				use(rw)
				g.g4x2 = x2
				r.encodeG4Join(e, L)
			}, nb)
			g.g4x2 = sx2
		}
	}
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		if L.g4moe == nil || L.g4moe.pool == nil {
			w0 := time.Now()
			e := r.q.Begin()
			for i := range rows {
				use(&rows[i])
				r.encodeLayerWith(e, l, rows[i].u.uPos, rows[i].u.uNKeys, rows[i].u.uQTempScale, rows[i].u.uRopePos)
			}
			submit(e, w0, &r.prof.denseGpuNanos, &r.prof.denseWallNanos)
			continue
		}
		w0 := time.Now()
		e := r.q.Begin() // phase 1, every row: attention, dense branch, router, expert input
		for i := range rows {
			use(&rows[i])
			r.encodeAttentionWith(e, l, rows[i].u.uPos, rows[i].u.uNKeys, rows[i].u.uQTempScale, rows[i].u.uRopePos)
			r.encodeG4Phase1(e, L)
		}
		submit(e, w0, &r.prof.p1GpuNanos, &r.prof.p1WallNanos)
		if r.execErr != nil {
			return nil
		}
		pool := L.g4moe.pool
		ok := r.layerMajorExpertGroups(len(rows), pool, g.topK,
			func(i int) []uint32 { return rows[i].rIdx.U32s()[:g.topK] },
			func(i int) []uint32 { return rows[i].slotIdx.U32s() },
			func(e2 *Encoder, i int) {
				use(&rows[i])
				r.encodeG4Phase2Paged(e2, pool)
				r.encodeG4Join(e2, L)
			}, group(L, pool))
		if !ok {
			return nil
		}
	}
	if !head {
		return nil
	}
	use(&rows[len(rows)-1])
	w0 := time.Now()
	e := r.q.Begin()
	e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, r.x, r.finalNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
	e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
	submit(e, w0, &r.prof.denseGpuNanos, &r.prof.denseWallNanos) // the head counts as dense
	r.finalizeLogits()
	return r.logitsHost
}

// lmProf adds a waited command buffer's GPU-busy time, and the wall time since w0, to two of r.prof's counters: the
// layer-major prefills keep the paged decode's profile (PagedProfile), phase 1, staging, phase 2 and dense alike.
func lmProf(e *Encoder, w0 time.Time, gpu, wall *int64) {
	*gpu += int64((e.GPUEnd() - e.GPUStart()) * 1e9)
	*wall += time.Since(w0).Nanoseconds()
}

// layerMajorExpertGroups runs one layer's phase 2 for n prompt rows of a layer-major paged prefill: rows in order, in the
// longest runs whose routed experts fit the pool at once (ensureResidentBatch must never evict an expert another row of
// the same command buffer reads), each run's experts staged once, then one command buffer of the run's phase 2s. ids(i)
// is row i's route, slots(i) its slot table to fill, encode(e, i) its phase-2 dispatches. False when a buffer aborted.
func (r *resident) layerMajorExpertGroups(n int, pool *expertPool, topK int, ids, slots func(i int) []uint32, encode func(e *Encoder, i int),
	group func(e *Encoder, i, j int)) bool {
	nSlots := len(pool.slotExpert)
	for i := 0; i < n; {
		var union []int
		in := map[int]bool{}
		j := i
		for ; j < n; j++ {
			var add []int
			for _, id := range ids(j)[:topK] {
				if !in[int(id)] && !slices.Contains(add, int(id)) {
					add = append(add, int(id))
				}
			}
			if len(union)+len(add) > nSlots {
				break
			}
			for _, id := range add {
				in[id] = true
			}
			union = append(union, add...)
		}
		ws := time.Now()
		staged := pool.ensureResidentBatch(union)
		r.prof.stageWallNanos += time.Since(ws).Nanoseconds()
		slotOf := make(map[int]uint32, len(union))
		for k, id := range union {
			slotOf[id] = uint32(staged[k].slot)
		}
		w2 := time.Now()
		e2 := r.q.Begin()
		if r.residency != (ResidencySet{}) {
			e2.UseResidencySet(r.residency)
		}
		for k := i; k < j; k++ {
			si := slots(k)
			for jj, id := range ids(k)[:topK] {
				si[jj] = slotOf[int(id)]
			}
			if group == nil {
				encode(e2, k)
			}
		}
		if group != nil {
			group(e2, i, j)
		}
		e2.End()
		r.recordExecErr(e2.Err())
		lmProf(e2, w2, &r.prof.p2GpuNanos, &r.prof.p2WallNanos)
		if r.execErr != nil {
			return false
		}
		i = j
	}
	return true
}

// moeLayerMajorOn routes a paged generic MoE's prompt (Qwen MoE, Mixtral, the Qwen3.5/3.6 DeltaNet hybrids, M35) through
// prefillMoEPaged, the generic twin of prefillG4Paged: bit-identical to the sequential loop by construction. A DeltaNet
// layer's recurrent state is per layer, and its rows run in order, as the loop runs them.
var moeLayerMajorOn = true

// prefillMoEPaged is prefillG4Paged for the generic MoE pager (forwardLogitsMoEPaged's layers): per layer, one command
// buffer of every row's mixer (attention, or the DeltaNet mixer) and router; the routes read; phase 2 (the routed experts
// from the pool and the shared expert, into the row's residual) in runs that fit the pool. A layer with no pool runs every
// row's whole decode layer in one buffer.
func (r *resident) prefillMoEPaged(embs [][]float32, startPos int, head bool) (logits []float32) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer func() {
		if p := recover(); p != nil {
			r.recordExecErr(fmt.Errorf("metal: layer-major paged MoE prefill aborted: %v", p))
			logits = nil
		}
	}()
	r.moeLayerMajorRuns++
	for c := 0; c < len(embs); c += g4LayerMajorChunk {
		end := min(c+g4LayerMajorChunk, len(embs))
		logits = r.moeLayerMajorRows(embs[c:end], startPos+c, head && end == len(embs))
		if r.execErr != nil {
			return nil
		}
	}
	return logits
}

func (r *resident) moeLayerMajorRows(embs [][]float32, start int, head bool) []float32 {
	mo, d := r.moe, r.d
	var made []Buffer
	nb := func(b Buffer) Buffer { made = append(made, b); return b }
	defer func() {
		for _, b := range made {
			d.ReleaseBuf(b)
		}
	}()
	type row struct {
		pos                          int
		x, mq, mSc, rIdx, rWgt, slot Buffer
		u                            posUniforms
	}
	rows := make([]row, len(embs))
	for i := range rows {
		rw := &rows[i]
		rw.pos = start + i
		rw.x, rw.mq, rw.mSc = nb(NewBufferFloats(d, embs[i])), nb(byteBuf(d, r.H)), nb(d.NewBufferLen(1))
		rw.rIdx, rw.rWgt = nb(NewBufferUint32s(d, make([]uint32, mo.k))), nb(d.NewBufferLen(mo.k))
		rw.slot = nb(NewBufferUint32s(d, make([]uint32, mo.k)))
		rw.u = posUniforms{nb(NewBufferU32(d, 0)), nb(NewBufferU32(d, 0)), nb(NewBufferU32(d, 0)),
			nb(NewBufferFloats(d, []float32{1})), nb(NewBufferU32(d, 0))}
		r.writePosUniforms(rw.u, rw.pos, rw.pos)
	}
	sx, smq, smSc, sIdx, sWgt, sSlot := r.x, r.mq, r.mSc, mo.rIdx, mo.rWgt, mo.slotIdx
	defer func() {
		r.x, r.mq, r.mSc, mo.rIdx, mo.rWgt, mo.slotIdx = sx, smq, smSc, sIdx, sWgt, sSlot
		r.encNKeys, r.encFANSplit = 0, Buffer{}
	}()
	use := func(rw *row) {
		r.x, r.mq, r.mSc, mo.rIdx, mo.rWgt, mo.slotIdx = rw.x, rw.mq, rw.mSc, rw.rIdx, rw.rWgt, rw.slot
		r.encNKeys, r.encFANSplit = rw.pos+1, rw.u.uFANSplit
	}
	submit := func(e *Encoder, w0 time.Time, gpu, wall *int64) {
		e.End()
		r.recordExecErr(e.Err())
		lmProf(e, w0, gpu, wall)
	}
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		if L.moe == nil || L.moe.pool == nil {
			w0 := time.Now()
			e := r.q.Begin()
			for i := range rows {
				use(&rows[i])
				r.encodeLayerWith(e, l, rows[i].u.uPos, rows[i].u.uNKeys, rows[i].u.uQTempScale, rows[i].u.uRopePos)
			}
			submit(e, w0, &r.prof.denseGpuNanos, &r.prof.denseWallNanos)
			if r.execErr != nil {
				return nil
			}
			continue
		}
		w0 := time.Now()
		e := r.q.Begin() // phase 1, every row: the mixer and the router
		for i := range rows {
			use(&rows[i])
			if L.delta != nil {
				r.encodeDeltaNetMixer(e, L)
			} else {
				r.encodeAttentionWith(e, l, rows[i].u.uPos, rows[i].u.uNKeys, rows[i].u.uQTempScale, rows[i].u.uRopePos)
			}
			r.encodeMoERouter(e, L)
		}
		submit(e, w0, &r.prof.p1GpuNanos, &r.prof.p1WallNanos)
		if r.execErr != nil {
			return nil
		}
		pool := L.moe.pool
		ok := r.layerMajorExpertGroups(len(rows), pool, mo.k,
			func(i int) []uint32 { return rows[i].rIdx.U32s()[:mo.k] },
			func(i int) []uint32 { return rows[i].slot.U32s() },
			func(e2 *Encoder, i int) {
				use(&rows[i])
				r.encodeMoEExpertsPaged(e2, L, pool)
			}, nil)
		if !ok {
			return nil
		}
	}
	if !head {
		return nil
	}
	use(&rows[len(rows)-1])
	w0 := time.Now()
	e := r.q.Begin()
	e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, r.x, r.finalNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
	e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
	submit(e, w0, &r.prof.denseGpuNanos, &r.prof.denseWallNanos) // the head counts as dense
	r.finalizeLogits()
	return r.logitsHost
}
