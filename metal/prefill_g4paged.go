//go:build darwin

package metal

import (
	"fmt"
	"runtime"
	"slices"
)

// g4LayerMajorOn routes a paged Gemma 4 MoE's prompt through prefillG4Paged (docs/tasks/task-m26-mac-2026-10.md, 4b)
// instead of the sequential loop. OFF until its grade on M26.
var g4LayerMajorOn = false

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
	sx, smq, smSc, sx1, sIdx, sWgt, sSlot := r.x, r.mq, r.mSc, g.g4x1, g.rIdx, g.rWgt, g.slotIdx
	defer func() {
		r.x, r.mq, r.mSc, g.g4x1, g.rIdx, g.rWgt, g.slotIdx = sx, smq, smSc, sx1, sIdx, sWgt, sSlot
		r.encNKeys, r.encFANSplit = 0, Buffer{}
	}()
	use := func(rw *g4Row) {
		r.x, r.mq, r.mSc, g.g4x1, g.rIdx, g.rWgt, g.slotIdx = rw.x, rw.mq, rw.mSc, rw.x1, rw.rIdx, rw.rWgt, rw.slotIdx
		r.encNKeys, r.encFANSplit = rw.pos+1, rw.u.uFANSplit
	}
	submit := func(e *Encoder) {
		e.End()
		r.recordExecErr(e.Err())
	}
	for l := 0; l < r.nL; l++ {
		L := &r.layers[l]
		if L.g4moe == nil || L.g4moe.pool == nil {
			e := r.q.Begin()
			for i := range rows {
				use(&rows[i])
				r.encodeLayerWith(e, l, rows[i].u.uPos, rows[i].u.uNKeys, rows[i].u.uQTempScale, rows[i].u.uRopePos)
			}
			submit(e)
			continue
		}
		e := r.q.Begin() // phase 1, every row: attention, dense branch, router, expert input
		for i := range rows {
			use(&rows[i])
			r.encodeAttentionWith(e, l, rows[i].u.uPos, rows[i].u.uNKeys, rows[i].u.uQTempScale, rows[i].u.uRopePos)
			r.encodeG4Phase1(e, L)
		}
		submit(e)
		if r.execErr != nil {
			return nil
		}
		pool := L.g4moe.pool
		nSlots := len(pool.slotExpert)
		for i := 0; i < len(rows); {
			// The longest run of rows from i whose routed experts fit the pool at once: ensureResidentBatch must never
			// evict an expert another row of the same command buffer reads.
			var union []int
			in := map[int]bool{}
			j := i
			for ; j < len(rows); j++ {
				ids := rows[j].rIdx.U32s()[:g.topK]
				var add []int
				for _, id := range ids {
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
			slots := pool.ensureResidentBatch(union)
			slotOf := make(map[int]uint32, len(union))
			for k, id := range union {
				slotOf[id] = uint32(slots[k].slot)
			}
			e2 := r.q.Begin() // phase 2 and join, rows i..j-1 in order
			if r.residency != (ResidencySet{}) {
				e2.UseResidencySet(r.residency)
			}
			for k := i; k < j; k++ {
				ids, si := rows[k].rIdx.U32s()[:g.topK], rows[k].slotIdx.U32s()
				for jj, id := range ids {
					si[jj] = slotOf[int(id)]
				}
				use(&rows[k])
				r.encodeG4Phase2Paged(e2, pool)
				r.encodeG4Join(e2, L)
			}
			submit(e2)
			if r.execErr != nil {
				return nil
			}
			i = j
		}
	}
	if !head {
		return nil
	}
	use(&rows[len(rows)-1])
	e := r.q.Begin()
	e.Dispatch(r.pRms, tgReduceNorm, tgReduceNorm, r.x, r.finalNorm, r.aq, r.aSc, r.uH, r.uEps, r.uAddOne)
	e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH)
	submit(e)
	r.finalizeLogits()
	return r.logitsHost
}
