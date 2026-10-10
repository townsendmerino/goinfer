//go:build darwin

package metal

import (
	"math"
	"runtime"

	"github.com/townsendmerino/goinfer/decoder"
)

var _ decoder.ResidentSample = (*resident)(nil)

// gumbelBlocks is the stage-1 dispatch size for a vocabulary of v entries (GB_THREADS threads x
// 4 vocabulary entries each — gumbel.go's gumbel_stage1).
func gumbelBlocks(v int) int { return (v + 4*256 - 1) / (4 * 256) }

// SampleAvailable reports whether this resident can draw a temperature-only sample on-device: only when r.logits is the raw LM-head
// output the host sampler would see (a final-logit softcap or non-identity logit scale is applied host-side in finalizeLogits,
// after the device row; gpu.residentDecoder.SampleAvailable uses the same condition), and not on a paged MoE, whose forward is a
// multi-command-buffer submit/wait/stage sequence (forwardLogitsPaged / forwardLogitsMoEPaged) that ForwardSample does not
// implement. Declining routes those models to the always-correct host draw.
func (r *resident) SampleAvailable() bool {
	if r.finalSoftcap != 0 || (r.logitScale != 0 && r.logitScale != 1) {
		return false
	}
	if r.g4moe != nil && r.g4moe.paged {
		return false
	}
	if r.moe != nil && r.moe.paged {
		return false
	}
	return true
}

// ForwardSample runs one token's forward and draws the NEXT token on-device by Gumbel-max (decoder.ResidentSample), returning just
// the id: no logits readback, no host normalisation. (seed, draw) are the sampler's own (decoder.Sampler.NextDraw), so this is the
// draw the host would have made. It mirrors ForwardEmb/forwardLogits's preamble (LockOSThread, copy into r.x, addLearnedPos,
// setPos), then extends the same command buffer with the two gumbel dispatches instead of a separate Begin/End round trip.
func (r *resident) ForwardSample(embedding []float32, pos int, temperature float64, seed, draw uint64) (int, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	r.loadEmb(embedding)
	r.addLearnedPos(pos)
	invT := float32(1 / temperature)
	if math.IsInf(float64(invT), 0) { // absurdly small temperature: greedy, exactly as the host does
		logits := r.forwardLogits(pos)
		return argmaxF32(logits), nil
	}
	r.setPos(pos)
	r.uGumbelInvT.Floats()[0] = invT
	r.uGumbelK0.SetU32(uint32(seed))
	r.uGumbelK1.SetU32(uint32(seed >> 32))
	r.uGumbelD0.SetU32(uint32(draw))
	r.uGumbelD1.SetU32(uint32(draw >> 32))
	e := r.q.Begin()
	r.encodeTrunkInto(e)                                                           // layers → final norm → r.aq/r.aSc
	e.Dispatch(r.pGemvW8, (r.V)*32, 32, r.aq, r.aSc, r.lmW, r.lmS, r.logits, r.uH) // full lm head, same as forwardLogits
	const gbThreads = 256
	const gbShmBytes = gbThreads * 2 * 4 // GB_THREADS (key,idx) pairs, 2 floats each
	e.DispatchTG(r.pGumbel1, r.gumbelNB*gbThreads, gbThreads, gbShmBytes,
		r.logits, r.uGumbelV, r.uGumbelInvT, r.uGumbelK0, r.uGumbelK1, r.uGumbelD0, r.uGumbelD1,
		r.gumbelBKey, r.gumbelBIdx)
	e.DispatchTG(r.pGumbel2, gbThreads, gbThreads, gbShmBytes,
		r.gumbelBKey, r.gumbelBIdx, r.uGumbelNB, r.gumbelOut)
	e.End()
	r.recordExecErr(e.Err())
	id := int32(r.gumbelOut.U32())
	if id >= 0 {
		return int(id), nil
	}
	// Nothing comparable (an all -inf row): the host draws the argmax; so does the device.
	// r.logits is already the row the dispatch above just wrote — read it directly (Metal's
	// buffers are host-visible unified memory, no download needed).
	return argmaxF32(r.logits.Floats()[:r.V]), nil
}

// GumbelForTest runs the on-device Gumbel-max draw over caller-supplied logits (any length) for an explicit (seed, draw): the seam
// TestGumbelDeviceAgreesWithHost uses to compare the kernel with decoder's reference, including vocab sizes both smaller and larger
// than the loaded model's. It allocates its own logits and partial buffers sized to len(logits); reusing the model-sized
// r.logits/r.gumbelBKey/r.gumbelBIdx would be a silent out-of-bounds write for a larger test size.
func (r *resident) GumbelForTest(logits []float32, temperature float64, seed, draw uint64) (int, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	v := len(logits)
	nb := gumbelBlocks(v)
	logitsBuf := NewBufferFloats(r.d, logits)
	bkey, bidx := r.d.NewBufferLen(nb), r.d.NewBufferLen(nb)
	uV := NewBufferU32(r.d, uint32(v))
	uNB := NewBufferU32(r.d, uint32(nb))
	invT := float32(1 / temperature)
	r.uGumbelInvT.Floats()[0] = invT
	r.uGumbelK0.SetU32(uint32(seed))
	r.uGumbelK1.SetU32(uint32(seed >> 32))
	r.uGumbelD0.SetU32(uint32(draw))
	r.uGumbelD1.SetU32(uint32(draw >> 32))
	const gbThreads = 256
	const gbShmBytes = gbThreads * 2 * 4
	e := r.q.Begin()
	e.DispatchTG(r.pGumbel1, nb*gbThreads, gbThreads, gbShmBytes,
		logitsBuf, uV, r.uGumbelInvT, r.uGumbelK0, r.uGumbelK1, r.uGumbelD0, r.uGumbelD1,
		bkey, bidx)
	e.DispatchTG(r.pGumbel2, gbThreads, gbThreads, gbShmBytes,
		bkey, bidx, uNB, r.gumbelOut)
	e.End()
	if err := e.Err(); err != nil {
		return 0, err
	}
	id := int32(r.gumbelOut.U32())
	if id >= 0 {
		return int(id), nil
	}
	return argmaxF32(logits), nil
}
