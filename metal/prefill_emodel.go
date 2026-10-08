//go:build darwin

package metal

import (
	"fmt"
	"runtime"
)

// emodelLayerMajorOn routes a Gemma 4 E-model's prompt (E2B, E4B: per-layer embeddings, KV-shared layers) through
// prefillEModel instead of declining PrefillLast (S9 of docs/tasks/task-multimodal-support-2026-10.md). The f16-MMA pass
// declines every dense Gemma 4 (head size varies by layer), so without this a text prompt prefilled one token at a
// time on the GPU and an image turn prefilled on the CPU. The pass is bit-identical to the sequential loop by
// construction (G-S9a); on by default, as the paged 26B's layer-major pass is, with a night speed grade to turn it off
// below 1.00x.
var emodelLayerMajorOn = true

// emodelChunk is the most prompt rows one layer's command buffer holds: each row keeps its residual (H floats), its
// per-layer inputs (L·P floats, about 35 KB on E2B) and its position uniforms.
const emodelChunk = 256

// emRow is one prompt row's own state in the E-model layer-major pass: the residual, the PLE inputs encodePLE reads,
// and the row's position uniforms. Every other buffer a layer's encode binds is scratch written and read within the
// row's own dispatches, which a serial compute encoder keeps apart (the paged 26B pass relies on the same).
type emRow struct {
	pos    int
	x, ple Buffer
	u      posUniforms
}

// emodelBatchedOn routes a Gemma 4 E-model's prompt through the f16 batched pass (S9 step 2,
// docs/tasks/task-multimodal-support-2026-10.md) instead of the layer-major one. OFF until G-S9c and G-S9d pass and the
// night speed grade clears 1.02x; with it off the layer-major pass keeps the route. Tests set it.
var emodelBatchedOn = false

// emodelBatchDefect is G-S9c's planted defects in the batched pass (0 in production): 1 a shared layer attends over an
// empty cache, 2 the PLE block skipped, 3 each row's PLE inputs from the next row, 4 the layer scalar dropped, 5 v_norm
// dropped, 6 every layer's FFN at the first layer's width.
var emodelBatchDefect int

// emodelBatched reports whether this resident's E-model prompts take the batched pass: the switch, and a model the pass
// implements (attention geometry on every layer, no K=V layer, f16 KV, int4 projections).
func (a *metalResident) emodelBatched() bool {
	r := a.r
	if !emodelBatchedOn || !a.emodelLayerMajor() || r.kvI8 || r.w8 || r.w8Attn || r.q4kLane || r.dnet != nil {
		return false
	}
	for l := range r.layers {
		if g := r.layers[l].geom; g == nil || g.kEqV {
			return false
		}
	}
	return true
}

// emodelLayerMajor reports whether this resident takes its prompts through prefillEModel: a dense Gemma 4 E-model (no
// paged MoE, no learned positions).
func (a *metalResident) emodelLayerMajor() bool {
	return emodelLayerMajorOn && a.r.pleP > 0 && a.r.g4moe == nil && a.r.moe == nil && !a.r.learnedPos
}

// prefillEModel runs a prompt through a Gemma 4 E-model layer by layer: per layer, one command buffer encodes decode's
// own layer (encodeLayerWith: attention with a KV-shared layer reading its source's cache, the per-layer FFN width, the
// PLE block) once per row, each row on its own residual and PLE inputs at its own position. A row's keys run up to its
// own position, so the K/V and the last row's logits are the sequential loop's bit for bit; what goes away is one
// submit-and-wait per token. embs are resident rows, [h ‖ L·P] each (metalResident.embLen). Returns the last row's
// logits, or nil when head is false.
func (r *resident) prefillEModel(embs [][]float32, startPos int, head bool) (logits []float32) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer func() {
		if p := recover(); p != nil {
			r.recordExecErr(fmt.Errorf("metal: E-model layer-major prefill aborted: %v", p))
			logits = nil
		}
	}()
	r.emodelLayerMajorRuns++
	for c := 0; c < len(embs); c += emodelChunk {
		end := min(c+emodelChunk, len(embs))
		logits = r.emodelRows(embs[c:end], startPos+c, head && end == len(embs))
		if r.execErr != nil {
			return nil
		}
	}
	return logits
}

func (r *resident) emodelRows(embs [][]float32, start int, head bool) []float32 {
	d := r.d
	var made []Buffer
	nb := func(b Buffer) Buffer { made = append(made, b); return b }
	defer func() {
		for _, b := range made {
			d.ReleaseBuf(b)
		}
	}()
	rows := make([]emRow, len(embs))
	for i := range rows {
		rw := &rows[i]
		rw.pos = start + i
		rw.x, rw.ple = nb(NewBufferFloats(d, embs[i][:r.H])), nb(NewBufferFloats(d, embs[i][r.H:]))
		rw.u = posUniforms{nb(NewBufferU32(d, 0)), nb(NewBufferU32(d, 0)), nb(NewBufferU32(d, 0)),
			nb(NewBufferFloats(d, []float32{1})), nb(NewBufferU32(d, 0))}
		if emodelPlant.posShift {
			rw.pos++ // G-S9b (3)
		}
		r.writePosUniforms(rw.u, rw.pos, rw.pos)
	}
	sx, sple := r.x, r.pleIn
	defer func() {
		r.x, r.pleIn = sx, sple
		r.encNKeys, r.encFANSplit = 0, Buffer{}
	}()
	use := func(rw *emRow) {
		r.x, r.pleIn = rw.x, rw.ple
		r.encNKeys, r.encFANSplit = rw.pos+1, rw.u.uFANSplit
	}
	emodelPlantDefect(rows)              // G-S9b's planted defects; a no-op in production
	order := r.emodelRowOrder(len(rows)) // G-S9b's row-order seam; identity in production
	for l := 0; l < r.nL; l++ {
		e := r.q.Begin()
		for _, i := range order {
			use(&rows[i])
			r.encodeLayerWith(e, l, rows[i].u.uPos, rows[i].u.uNKeys, rows[i].u.uQTempScale, rows[i].u.uRopePos)
		}
		e.End()
		r.recordExecErr(e.Err())
		if r.execErr != nil {
			return nil
		}
	}
	if !head {
		return nil
	}
	use(&rows[len(rows)-1])
	e := r.q.Begin()
	r.encodeNorm(e, r.x, r.finalNorm, r.finalNormBias, r.aq, r.aSc)
	e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH) // decode's head (forwardLogits)
	e.End()
	r.recordExecErr(e.Err())
	if r.execErr != nil {
		return nil
	}
	r.finalizeLogits()
	return r.logitsHost
}

// emodelRowOrder is the order rows are encoded within a layer: 0..n-1, and it must be. Within a layer each row writes
// its own K/V and later rows attend to it, so a row encoded before an earlier one reads keys not yet written. A test
// seam: G-S9b reverses it as a planted defect.
var emodelRowOrderHook func(n int) []int

func (r *resident) emodelRowOrder(n int) []int {
	if emodelRowOrderHook != nil {
		return emodelRowOrderHook(n)
	}
	o := make([]int, n)
	for i := range o {
		o[i] = i
	}
	return o
}

// emodelPlant is G-S9b's planted defects (zero in production): pleShift binds each row's PLE inputs to the next row's,
// pleZero drops the PLE term (a zero input makes the block add nothing), posShift runs every row one position late.
var emodelPlant struct{ pleShift, pleZero, posShift bool }

func emodelPlantDefect(rows []emRow) {
	switch {
	case emodelPlant.pleShift:
		for i := 0; i+1 < len(rows); i++ {
			rows[i].ple = rows[i+1].ple
		}
	case emodelPlant.pleZero:
		for i := range rows {
			clear(rows[i].ple.Floats())
		}
	}
}
