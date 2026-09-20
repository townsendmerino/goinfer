//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestAttentionFA_pipelinedEncodeRace is R2's own suggested next step
// (r2-attn-fa-2026-09-19.md): a minimal, isolated repro chaining several
// command buffers under the REAL production pipelining pattern (encode
// buffer N+1 while buffer N is still executing — metal/model.go's execLoop,
// Commit()/FinishEncoding()/WaitDone(), NOT the synchronous Begin()/End()
// attn_fa_test.go's gate (1) uses) instead of a full 28-layer real model.
//
// Production's shared uniform buffers (r.uAttnFAG/r.uAttnFANSplit) are
// SetU32'd — a raw CPU write to shared memory, not a tracked Metal command —
// while encoding buffer N+1, which happens WHILE buffer N is still
// executing on the GPU. In production this is harmless because G/nSplit
// never change between layers or decode steps (the debug print already
// proved this). This test uses the SAME shared-buffer-SetU32-during-encode
// pattern but varies the value every iteration specifically so a real race
// becomes OBSERVABLE — if buffer N's dispatch reads iteration N+1's value
// instead of its own, that is the mechanism, confirmed in isolation rather
// than inferred from a 28-layer model's stably-wrong logits.
func TestAttentionFA_pipelinedEncodeRace(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pFA, err := d.NewComputePipeline(lib, "attention_fa")
	if err != nil {
		t.Fatalf("pipeline attention_fa: %v", err)
	}
	pCombine, err := d.NewComputePipeline(lib, "attention_fa_combine")
	if err != nil {
		t.Fatalf("pipeline attention_fa_combine: %v", err)
	}

	const nH, nKV, hd = 12, 2, 128 // qwen2.5-1.5b shape, R2's registered decision cell
	G := nH / nKV
	const nKeys = 3900 // the registered decision depth -- enough GPU work per dispatch to open a real race window
	const nSplit = 14
	const iters = 8 // past R2's own "3rd command buffer" observation with margin

	rng := rand.New(rand.NewSource(0xA77A))
	scale := float32(1 / math.Sqrt(float64(hd)))
	kvDim := nKV * hd

	// One fixed K/V/Q per iteration (same shape every time; only the SHARED
	// uniform buffers' values vary across iterations, mimicking production's
	// per-layer SetU32 call against per-decode-step-constant production
	// values, except here deliberately varying so a race is visible).
	q := make([]float32, nH*hd)
	for i := range q {
		q[i] = float32(rng.NormFloat64()) * 0.5
	}
	kf := make([]float32, nKeys*kvDim)
	vf := make([]float32, nKeys*kvDim)
	for i := range kf {
		kf[i] = float32(rng.NormFloat64()) * 0.5
		vf[i] = float32(rng.NormFloat64()) * 0.5
	}
	hot := nKeys / 2
	for h := range nKV {
		for dd := range hd {
			kf[hot*kvDim+h*hd+dd] = q[(h*(nH/nKV))*hd+dd] * 3
		}
	}
	kh, vh := make([]uint16, len(kf)), make([]uint16, len(vf))
	for i := range kf {
		kh[i], vh[i] = f32ToF16(kf[i]), f32ToF16(vf[i])
	}
	for i := range kf {
		kf[i], vf[i] = f16ToF32(kh[i]), f16ToF32(vh[i])
	}
	ref := cpuAttention(q, kf, vf, nH, nKV, hd, nKeys, 0, scale)

	qB := NewBufferFloats(d, q)
	kc, vc := NewBufferU16s(d, kh), NewBufferU16s(d, vh)
	pStride := G * (hd + 2)
	partial := d.NewBufferLen(nKV * nSplit * pStride)
	uNKV := NewBufferU32(d, uint32(nKV))
	uNKeys := NewBufferU32(d, uint32(nKeys))
	uScale := NewBufferFloats(d, []float32{scale})
	uWin := NewBufferU32(d, 0)
	uHd := NewBufferU32(d, uint32(hd))

	// The SHARED, reused-every-dispatch uniforms -- the production
	// r.uAttnFAG/r.uAttnFANSplit analogue. ONE buffer each, SetU32'd fresh
	// per iteration during that iteration's encode, exactly like
	// metal/model.go:2362-2363.
	uG := NewBufferU32(d, uint32(G))
	uNSplit := NewBufferU32(d, uint32(nSplit))

	shmBytes := 128 * 6 * G * 4

	cq := d.NewCommandQueue()
	pool := NewARPool()
	defer pool.Drain()

	// out[i] holds iteration i's result; wantG[i]/wantNSplit[i] record what
	// G/nSplit SHOULD have been read for that iteration (its OWN value, at
	// the time it was encoded) -- not what the shared buffer holds NOW.
	outs := make([]Buffer, iters)
	wantG := make([]uint32, iters)
	wantNSplit := make([]uint32, iters)

	encodeIter := func(i int) *Encoder {
		// Vary G and nSplit per iteration -- both must still divide the
		// dispatch geometry validly, so cycle through a small set of
		// legitimate (G, nSplit) pairs rather than arbitrary values.
		gVariants := []int{G, G, G} // G is architecture-fixed at 6 for this shape; kept constant, only nSplit varies below
		splitVariants := []int{14, 8, 4, 28, 14, 8, 4, 28}
		thisG := gVariants[i%len(gVariants)]
		thisSplit := splitVariants[i%len(splitVariants)]
		wantG[i] = uint32(thisG)
		wantNSplit[i] = uint32(thisSplit)

		uG.SetU32(uint32(thisG))
		uNSplit.SetU32(uint32(thisSplit))

		out := d.NewBufferLen(nH * hd)
		outs[i] = out

		enc := cq.BeginNP()
		enc.DispatchTG(pFA, nKV*thisSplit*128, 128, shmBytes, qB, kc, vc, partial, uNKV, uG, uNKeys, uScale, uWin, uNSplit)
		enc.Dispatch(pCombine, nH*hd, hd, partial, out, uG, uHd, uNSplit)
		enc.FinishEncoding()
		return enc
	}

	// The REAL production pattern: commit i, encode i+1 (which SetU32's the
	// SHARED uniforms with i+1's values) WHILE i is still executing on the
	// GPU, THEN wait for i. If SetU32-during-encode races i's still-running
	// GPU read, i's result will reflect i+1's nSplit, not its own.
	cur := encodeIter(0)
	cur.Commit()
	for i := 1; i < iters; i++ {
		next := encodeIter(i)
		cur.WaitDone()
		if err := cur.Err(); err != nil {
			t.Fatalf("iter %d command buffer aborted: %v", i-1, err)
		}
		next.Commit()
		cur = next
	}
	cur.WaitDone()
	if err := cur.Err(); err != nil {
		t.Fatalf("final command buffer aborted: %v", err)
	}

	for i := range iters {
		got := outs[i].Floats()
		var dot, na, nb float64
		var maxabs float64
		for j := range ref {
			dot += float64(got[j]) * float64(ref[j])
			na += float64(got[j]) * float64(got[j])
			nb += float64(ref[j]) * float64(ref[j])
			if dd := math.Abs(float64(got[j] - ref[j])); dd > maxabs {
				maxabs = dd
			}
		}
		cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
		t.Logf("iter %d: wantG=%d wantNSplit=%d cosine=%.7f maxAbs=%.4f", i, wantG[i], wantNSplit[i], cos, maxabs)
		if cos < 0.9999 || maxabs > 1e-2 {
			t.Errorf("iter %d: encode-ahead race reproduced -- expected nSplit=%d, cosine=%.7f maxAbs=%.4f (correct output requires this iteration's OWN nSplit, not a later iteration's)",
				i, wantNSplit[i], cos, maxabs)
		}
	}
}
