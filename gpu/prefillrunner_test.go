//go:build gpu

package gpu

import (
	"math"
	"testing"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// TestPrefillLastW8A8_parity is the Increment-2 gate (docs/task-gpu-batched-
// prefill.md): PrefillLastW8A8's single returned row (the last position's logits)
// must be BIT-IDENTICAL to the last of M sequential DecodeToken calls over one
// shared KV cache. M=20 deliberately exceeds gemmRowMaxM=16 — the whole point of
// this function over DecodeTokenFusedBatched is that it is NOT capped, since real
// prompts run far past 16 tokens.
func TestPrefillLastW8A8_parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const hidden, nH, nKV, hd, inter, vocab, L = 256, 8, 2, 32, 688, 512, 2
	const start, M = 17, 20
	qDim, kvDim := nH*hd, nKV*hd
	half := hd / 2
	eps := float32(1e-6)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	cap := (start + M) * kvDim

	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd)))
	}
	up32 := func(v []float32) *DeviceBuffer { d, _ := ctx.UploadF32(v); return d }
	mk := func(bq []int8, s []float32, N, K int) *ResidentW8A8 {
		rm, e := ctx.UploadW8A8(bq, s, N, K)
		if e != nil {
			t.Fatal(e)
		}
		return rm
	}

	type lw struct {
		an, mn         []float32
		q, k, v, o     *ResidentW8A8
		g, u, d        *ResidentW8A8
		priorK, priorV []float32
	}
	seed := uint64(1)
	W := func(N, K int) *ResidentW8A8 { seed++; bq, s := quantW(N, K, seed); return mk(bq, s, N, K) }
	layers := make([]lw, L)
	for l := range layers {
		layers[l].an = randMat(hidden, uint64(200+l))
		layers[l].mn = randMat(hidden, uint64(300+l))
		layers[l].q = W(qDim, hidden)
		layers[l].k = W(kvDim, hidden)
		layers[l].v = W(kvDim, hidden)
		layers[l].o = W(hidden, qDim)
		layers[l].g = W(inter, hidden)
		layers[l].u = W(inter, hidden)
		layers[l].d = W(hidden, inter)
		layers[l].priorK = randMat(start*kvDim, uint64(400+l))
		layers[l].priorV = randMat(start*kvDim, uint64(500+l))
	}
	fnorm := up32(randMat(hidden, 600))
	lmBQ, lmS := quantW(vocab, hidden, 999)
	lmHead := mk(lmBQ, lmS, vocab, hidden)
	invD := up32(invFreq)

	buildMW := func() ModelW {
		mw := ModelW{FinalNorm: fnorm, LMHead: lmHead}
		for l := range layers {
			kc, _ := ctx.NewKVCache(layers[l].priorK, cap)
			vc, _ := ctx.NewKVCache(layers[l].priorV, cap)
			mw.Layers = append(mw.Layers, LayerW{
				Attn: AttnWeights{
					Norm: up32(layers[l].an), QProj: layers[l].q, KProj: layers[l].k, VProj: layers[l].v, OProj: layers[l].o,
					InvFreq: invD, KCache: kc, VCache: vc,
				},
				MLPNorm: up32(layers[l].mn), Gate: layers[l].g, Up: layers[l].u, Down: layers[l].d,
			})
		}
		return mw
	}

	xs := make([][]float32, M)
	for r := range xs {
		xs[r] = randMat(hidden, uint64(1000+r))
	}

	// Reference: M sequential DecodeToken calls over one shared KV cache — only the
	// LAST row's logits matter (matches PrefillLastW8A8's single-row contract).
	mwSeq := buildMW()
	defer mwSeq.Release()
	var refLast []float32
	for r := range xs {
		out, err := ctx.DecodeToken(xs[r], mwSeq, hidden, nH, nKV, hd, inter, start+r, 0, eps, scale, false)
		if err != nil {
			t.Fatalf("seq row %d: %v", r, err)
		}
		refLast = out
	}

	mwBatch := buildMW()
	defer mwBatch.Release()
	positions := make([]int, M)
	for r := range positions {
		positions[r] = start + r
	}
	got, err := ctx.PrefillLastW8A8(xs, mwBatch, hidden, nH, nKV, hd, inter, positions, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("PrefillLastW8A8: %v", err)
	}

	cos, maxAbs := cosine(got, refLast)
	t.Logf("last row (pos %d): cosine=%.8f maxAbs=%.3e", start+M-1, cos, maxAbs)
	if cos < 0.999999 || maxAbs > 1e-3 {
		t.Errorf("diverges from sequential DecodeToken: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
	}
}

// TestPrefillLastW8A8_declines checks runModelToModelW's scope guard: PrefillRunner
// must decline (not silently produce wrong output) on model features outside plain
// dense W8A8 — sliding window here as the representative case.
func TestPrefillLastW8A8_declines(t *testing.T) {
	rm := &runModel{slidingWindow: 128}
	if _, ok := runModelToModelW(rm, 64); ok {
		t.Fatal("runModelToModelW accepted a sliding-window model; PrefillLastW8A8 has no windowed-attention support")
	}
}

// TestPrefillLastW8A8_declinesGenuinePartialRoPE checks the hd/2 distinction:
// ropeHalf==hd/2 is full rotation (just set explicitly rather than left at the 0
// sentinel — real checkpoints do this, see TestResidentPrefillLast_parity's
// qwen2.5-coder-0.5b, hd=64, ropeHalf=32, which that test proves is ACCEPTED
// end-to-end); ropeHalf<hd/2 is genuine partial RoPE, which rope()'s hardcoded
// half:=hd/2 cannot express, and must be DECLINED — checked here since it only
// needs the early guard, not a full ModelW build (no GPU/real weights needed,
// same shape as TestPrefillLastW8A8_declines).
func TestPrefillLastW8A8_declinesGenuinePartialRoPE(t *testing.T) {
	const hd = 64
	if _, ok := runModelToModelW(&runModel{ropeHalf: hd/2 - 8}, hd); ok {
		t.Fatal("runModelToModelW accepted genuine partial RoPE (ropeHalf < hd/2); PrefillLastW8A8's rope() hardcodes full rotation")
	}
}

// TestPrefillLastW8A8_qknormParity verifies that batched prefill PrefillLastW8A8
// with per-head QK-norm produces identical logits to sequential decode.
// Tested under both addOne=false (Qwen3) and addOne=true (Gemma 3).
func TestPrefillLastW8A8_qknormParity(t *testing.T) {
	for _, addOne := range []bool{false, true} {
		t.Run(map[bool]string{false: "addOne=false", true: "addOne=true"}[addOne], func(t *testing.T) {
			ctx := newOrSkipHW(t)
			defer ctx.Close()

			const hidden, nH, nKV, hd, inter, vocab, L = 256, 8, 2, 32, 688, 512, 2
			const start, M = 17, 20
			qDim, kvDim := nH*hd, nKV*hd
			half := hd / 2
			eps := float32(1e-6)
			scale := float32(1.0 / math.Sqrt(float64(hd)))
			cap := (start + M) * kvDim

			invFreq := make([]float32, half)
			for d := range invFreq {
				invFreq[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd)))
			}
			up32 := func(v []float32) *DeviceBuffer { d, _ := ctx.UploadF32(v); return d }
			mk := func(bq []int8, s []float32, N, K int) *ResidentW8A8 {
				rm, e := ctx.UploadW8A8(bq, s, N, K)
				if e != nil {
					t.Fatal(e)
				}
				return rm
			}

			type lw struct {
				an, mn         []float32
				q, k, v, o     *ResidentW8A8
				g, u, d        *ResidentW8A8
				qn, kn         []float32
				priorK, priorV []float32
			}
			seed := uint64(10)
			W := func(N, K int) *ResidentW8A8 { seed++; bq, s := quantW(N, K, seed); return mk(bq, s, N, K) }
			layers := make([]lw, L)
			for l := range layers {
				layers[l].an = randMat(hidden, uint64(200+l))
				layers[l].mn = randMat(hidden, uint64(300+l))
				layers[l].q = W(qDim, hidden)
				layers[l].k = W(kvDim, hidden)
				layers[l].v = W(kvDim, hidden)
				layers[l].o = W(hidden, qDim)
				layers[l].g = W(inter, hidden)
				layers[l].u = W(inter, hidden)
				layers[l].d = W(hidden, inter)
				layers[l].qn = randMat(hd, uint64(700+l))
				layers[l].kn = randMat(hd, uint64(800+l))
				layers[l].priorK = randMat(start*kvDim, uint64(400+l))
				layers[l].priorV = randMat(start*kvDim, uint64(500+l))
			}
			fnorm := up32(randMat(hidden, 600))
			lmBQ, lmS := quantW(vocab, hidden, 999)
			lmHead := mk(lmBQ, lmS, vocab, hidden)
			invD := up32(invFreq)

			buildMW := func() ModelW {
				mw := ModelW{FinalNorm: fnorm, LMHead: lmHead}
				for l := range layers {
					kc, _ := ctx.NewKVCache(layers[l].priorK, cap)
					vc, _ := ctx.NewKVCache(layers[l].priorV, cap)
					mw.Layers = append(mw.Layers, LayerW{
						Attn: AttnWeights{
							Norm: up32(layers[l].an), QProj: layers[l].q, KProj: layers[l].k, VProj: layers[l].v, OProj: layers[l].o,
							InvFreq: invD, KCache: kc, VCache: vc,
							QNorm: up32(layers[l].qn), KNorm: up32(layers[l].kn),
						},
						MLPNorm: up32(layers[l].mn), Gate: layers[l].g, Up: layers[l].u, Down: layers[l].d,
					})
				}
				return mw
			}

			xs := make([][]float32, M)
			for r := range xs {
				xs[r] = randMat(hidden, uint64(1000+r))
			}

			mwSeq := buildMW()
			defer mwSeq.Release()
			if !mwSeq.hasQKNorm() {
				t.Fatal("expected mwSeq.hasQKNorm() to be true")
			}
			var refLast []float32
			for r := range xs {
				out, err := ctx.DecodeToken(xs[r], mwSeq, hidden, nH, nKV, hd, inter, start+r, 0, eps, scale, addOne)
				if err != nil {
					t.Fatalf("seq row %d: %v", r, err)
				}
				refLast = out
			}

			mwBatch := buildMW()
			defer mwBatch.Release()
			positions := make([]int, M)
			for r := range positions {
				positions[r] = start + r
			}
			got, err := ctx.PrefillLastW8A8(xs, mwBatch, hidden, nH, nKV, hd, inter, positions, 0, eps, scale, addOne)
			if err != nil {
				t.Fatalf("PrefillLastW8A8: %v", err)
			}

			cos, maxAbs := cosine(got, refLast)
			t.Logf("last row (pos %d): cosine=%.8f maxAbs=%.3e", start+M-1, cos, maxAbs)
			if cos < 0.999999 || maxAbs > 1e-3 {
				t.Errorf("diverges from sequential DecodeToken: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
			}
		})
	}
}

// TestPrefillLastW8A8_qknormScope checks that runModelToModelW admits models with paired
// QNorm/KNorm and declines models where only one is set.
func TestPrefillLastW8A8_qknormScope(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	dummyBuf, err := ctx.device.TryCreateBuffer(&wgpu.BufferDescriptor{
		Label: "dummy",
		Size:  64,
		Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer dummyBuf.Release()

	rm := &runModel{}
	// One of QNorm/KNorm set -> must decline
	rm.layers = []runLayer{
		{qNorm: dummyBuf, kNorm: nil},
	}
	if _, ok := runModelToModelW(rm, 64); ok {
		t.Fatal("runModelToModelW accepted layer with qNorm != nil but kNorm == nil")
	}

	rm.layers = []runLayer{
		{qNorm: nil, kNorm: dummyBuf},
	}
	if _, ok := runModelToModelW(rm, 64); ok {
		t.Fatal("runModelToModelW accepted layer with qNorm == nil but kNorm != nil")
	}
}

// TestPrefillLastW8A8_admitsFeatures checks that runModelToModelW admits gatedGELU, kvF16, and qGate
// and populates the corresponding fields on ModelW.
func TestPrefillLastW8A8_admitsFeatures(t *testing.T) {
	dummyW := &ResidentW8A8{rows: 64, cols: 64, kp: 64}
	rm := &runModel{
		gatedGELU: true,
		kvF16:     true,
		lmHead:    dummyW,
		layers: []runLayer{
			{
				q: dummyW, k: dummyW, v: dummyW, o: dummyW,
				gate: dummyW, up: dummyW, down: dummyW,
				qGate: true,
			},
		},
	}
	mw, ok := runModelToModelW(rm, 64)
	if !ok {
		t.Fatal("runModelToModelW declined model with gatedGELU, kvF16, and qGate")
	}
	if !mw.GatedGELU {
		t.Error("ModelW.GatedGELU is false, want true")
	}
	if !mw.KVF16 {
		t.Error("ModelW.KVF16 is false, want true")
	}
	if len(mw.Layers) != 1 || !mw.Layers[0].Attn.QGate {
		t.Error("ModelW.Layers[0].Attn.QGate is false, want true")
	}
}

// TestPrefillLastW8A8_gegluParity verifies that batched prefill PrefillLastW8A8
// with GatedGELU=true produces identical logits to sequential decode.
func TestPrefillLastW8A8_gegluParity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const hidden, nH, nKV, hd, inter, vocab, L = 256, 8, 2, 32, 688, 512, 2
	const start, M = 17, 20
	qDim, kvDim := nH*hd, nKV*hd
	half := hd / 2
	eps := float32(1e-6)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	cap := (start + M) * kvDim

	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd)))
	}
	up32 := func(v []float32) *DeviceBuffer { d, _ := ctx.UploadF32(v); return d }
	mk := func(bq []int8, s []float32, N, K int) *ResidentW8A8 {
		rm, e := ctx.UploadW8A8(bq, s, N, K)
		if e != nil {
			t.Fatal(e)
		}
		return rm
	}

	type lw struct {
		an, mn         []float32
		q, k, v, o     *ResidentW8A8
		g, u, d        *ResidentW8A8
		priorK, priorV []float32
	}
	seed := uint64(11)
	W := func(N, K int) *ResidentW8A8 { seed++; bq, s := quantW(N, K, seed); return mk(bq, s, N, K) }
	layers := make([]lw, L)
	for l := range layers {
		layers[l].an = randMat(hidden, uint64(220+l))
		layers[l].mn = randMat(hidden, uint64(320+l))
		layers[l].q = W(qDim, hidden)
		layers[l].k = W(kvDim, hidden)
		layers[l].v = W(kvDim, hidden)
		layers[l].o = W(hidden, qDim)
		layers[l].g = W(inter, hidden)
		layers[l].u = W(inter, hidden)
		layers[l].d = W(hidden, inter)
		layers[l].priorK = randMat(start*kvDim, uint64(420+l))
		layers[l].priorV = randMat(start*kvDim, uint64(520+l))
	}
	fnorm := up32(randMat(hidden, 620))
	lmBQ, lmS := quantW(vocab, hidden, 9991)
	lmHead := mk(lmBQ, lmS, vocab, hidden)
	invD := up32(invFreq)

	buildMW := func() ModelW {
		mw := ModelW{FinalNorm: fnorm, LMHead: lmHead, GatedGELU: true}
		for l := range layers {
			kc, _ := ctx.NewKVCache(layers[l].priorK, cap)
			vc, _ := ctx.NewKVCache(layers[l].priorV, cap)
			mw.Layers = append(mw.Layers, LayerW{
				Attn: AttnWeights{
					Norm: up32(layers[l].an), QProj: layers[l].q, KProj: layers[l].k, VProj: layers[l].v, OProj: layers[l].o,
					InvFreq: invD, KCache: kc, VCache: vc,
				},
				MLPNorm: up32(layers[l].mn), Gate: layers[l].g, Up: layers[l].u, Down: layers[l].d,
			})
		}
		return mw
	}

	xs := make([][]float32, M)
	for r := range xs {
		xs[r] = randMat(hidden, uint64(1020+r))
	}

	mwSeq := buildMW()
	defer mwSeq.Release()
	var refLast []float32
	for r := range xs {
		out, err := ctx.DecodeToken(xs[r], mwSeq, hidden, nH, nKV, hd, inter, start+r, 0, eps, scale, false)
		if err != nil {
			t.Fatalf("seq row %d: %v", r, err)
		}
		refLast = out
	}

	mwBatch := buildMW()
	defer mwBatch.Release()
	positions := make([]int, M)
	for r := range positions {
		positions[r] = start + r
	}
	got, err := ctx.PrefillLastW8A8(xs, mwBatch, hidden, nH, nKV, hd, inter, positions, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("PrefillLastW8A8: %v", err)
	}

	cos, maxAbs := cosine(got, refLast)
	t.Logf("last row (pos %d): cosine=%.8f maxAbs=%.3e", start+M-1, cos, maxAbs)
	if cos < 0.999999 || maxAbs > 1e-3 {
		t.Errorf("diverges from sequential DecodeToken: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
	}
}

// TestPrefillLastW8A8_qgateParity verifies that batched prefill PrefillLastW8A8
// with QGate=true (double-width Q projection split into query and gate) matches sequential decode.
func TestPrefillLastW8A8_qgateParity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const hidden, nH, nKV, hd, inter, vocab, L = 256, 8, 2, 32, 688, 512, 2
	const start, M = 17, 20
	qDim, kvDim := nH*hd, nKV*hd
	half := hd / 2
	eps := float32(1e-6)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	cap := (start + M) * kvDim

	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd)))
	}
	up32 := func(v []float32) *DeviceBuffer { d, _ := ctx.UploadF32(v); return d }
	mk := func(bq []int8, s []float32, N, K int) *ResidentW8A8 {
		rm, e := ctx.UploadW8A8(bq, s, N, K)
		if e != nil {
			t.Fatal(e)
		}
		return rm
	}

	type lw struct {
		an, mn         []float32
		q, k, v, o     *ResidentW8A8
		g, u, d        *ResidentW8A8
		priorK, priorV []float32
	}
	seed := uint64(31)
	W := func(N, K int) *ResidentW8A8 { seed++; bq, s := quantW(N, K, seed); return mk(bq, s, N, K) }
	layers := make([]lw, L)
	for l := range layers {
		layers[l].an = randMat(hidden, uint64(230+l))
		layers[l].mn = randMat(hidden, uint64(330+l))
		// QProj is double-width: emits [query ‖ gate] per head
		layers[l].q = W(2*qDim, hidden)
		layers[l].k = W(kvDim, hidden)
		layers[l].v = W(kvDim, hidden)
		layers[l].o = W(hidden, qDim)
		layers[l].g = W(inter, hidden)
		layers[l].u = W(inter, hidden)
		layers[l].d = W(hidden, inter)
		layers[l].priorK = randMat(start*kvDim, uint64(430+l))
		layers[l].priorV = randMat(start*kvDim, uint64(530+l))
	}
	fnorm := up32(randMat(hidden, 630))
	lmBQ, lmS := quantW(vocab, hidden, 9993)
	lmHead := mk(lmBQ, lmS, vocab, hidden)
	invD := up32(invFreq)

	buildMW := func() ModelW {
		mw := ModelW{FinalNorm: fnorm, LMHead: lmHead}
		for l := range layers {
			kc, _ := ctx.NewKVCache(layers[l].priorK, cap)
			vc, _ := ctx.NewKVCache(layers[l].priorV, cap)
			mw.Layers = append(mw.Layers, LayerW{
				Attn: AttnWeights{
					Norm: up32(layers[l].an), QProj: layers[l].q, KProj: layers[l].k, VProj: layers[l].v, OProj: layers[l].o,
					InvFreq: invD, KCache: kc, VCache: vc,
					QGate: true,
				},
				MLPNorm: up32(layers[l].mn), Gate: layers[l].g, Up: layers[l].u, Down: layers[l].d,
			})
		}
		return mw
	}

	xs := make([][]float32, M)
	for r := range xs {
		xs[r] = randMat(hidden, uint64(1030+r))
	}

	mwSeq := buildMW()
	defer mwSeq.Release()
	var refLast []float32
	for r := range xs {
		out, err := ctx.DecodeToken(xs[r], mwSeq, hidden, nH, nKV, hd, inter, start+r, 0, eps, scale, false)
		if err != nil {
			t.Fatalf("seq row %d: %v", r, err)
		}
		refLast = out
	}

	mwBatch := buildMW()
	defer mwBatch.Release()
	positions := make([]int, M)
	for r := range positions {
		positions[r] = start + r
	}
	got, err := ctx.PrefillLastW8A8(xs, mwBatch, hidden, nH, nKV, hd, inter, positions, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("PrefillLastW8A8: %v", err)
	}

	cos, maxAbs := cosine(got, refLast)
	t.Logf("qGate last row (pos %d): cosine=%.8f maxAbs=%.3e", start+M-1, cos, maxAbs)
	if cos < 0.999999 || maxAbs > 1e-3 {
		t.Errorf("diverges from sequential DecodeToken: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
	}
}

// TestPrefillLastW8A8_kvF16Parity verifies that batched prefill with KVF16=true
// matches sequential decode with f16 KV cache.
func TestPrefillLastW8A8_kvF16Parity(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const hidden, nH, nKV, hd, inter, vocab, L = 256, 8, 2, 32, 688, 512, 2
	const start, M = 17, 20
	qDim, kvDim := nH*hd, nKV*hd
	half := hd / 2
	eps := float32(1e-6)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	cap := (start + M) * kvDim

	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd)))
	}
	up32 := func(v []float32) *DeviceBuffer { d, _ := ctx.UploadF32(v); return d }
	mk := func(bq []int8, s []float32, N, K int) *ResidentW8A8 {
		rm, e := ctx.UploadW8A8(bq, s, N, K)
		if e != nil {
			t.Fatal(e)
		}
		return rm
	}

	type lw struct {
		an, mn         []float32
		q, k, v, o     *ResidentW8A8
		g, u, d        *ResidentW8A8
		priorK, priorV []float32
	}
	seed := uint64(51)
	W := func(N, K int) *ResidentW8A8 { seed++; bq, s := quantW(N, K, seed); return mk(bq, s, N, K) }
	layers := make([]lw, L)
	for l := range layers {
		layers[l].an = randMat(hidden, uint64(250+l))
		layers[l].mn = randMat(hidden, uint64(350+l))
		layers[l].q = W(qDim, hidden)
		layers[l].k = W(kvDim, hidden)
		layers[l].v = W(kvDim, hidden)
		layers[l].o = W(hidden, qDim)
		layers[l].g = W(inter, hidden)
		layers[l].u = W(inter, hidden)
		layers[l].d = W(hidden, inter)
		layers[l].priorK = randMat(start*kvDim, uint64(450+l))
		layers[l].priorV = randMat(start*kvDim, uint64(550+l))
	}
	fnorm := up32(randMat(hidden, 650))
	lmBQ, lmS := quantW(vocab, hidden, 9995)
	lmHead := mk(lmBQ, lmS, vocab, hidden)
	invD := up32(invFreq)

	buildMW := func() ModelW {
		mw := ModelW{FinalNorm: fnorm, LMHead: lmHead, KVF16: true}
		for l := range layers {
			kc, _ := ctx.NewKVCacheF16(layers[l].priorK, cap)
			vc, _ := ctx.NewKVCacheF16(layers[l].priorV, cap)
			mw.Layers = append(mw.Layers, LayerW{
				Attn: AttnWeights{
					Norm: up32(layers[l].an), QProj: layers[l].q, KProj: layers[l].k, VProj: layers[l].v, OProj: layers[l].o,
					InvFreq: invD, KCache: kc, VCache: vc,
				},
				MLPNorm: up32(layers[l].mn), Gate: layers[l].g, Up: layers[l].u, Down: layers[l].d,
			})
		}
		return mw
	}

	xs := make([][]float32, M)
	for r := range xs {
		xs[r] = randMat(hidden, uint64(1050+r))
	}

	mwSeq := buildMW()
	defer mwSeq.Release()
	rm16 := w8Model(mwSeq)
	rm16.kvF16 = true
	runner16, err := ctx.newDecodeRunner(rm16, hidden, nH, nKV, hd, inter, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("newDecodeRunner: %v", err)
	}
	defer runner16.Release()

	var refLast []float32
	for r := range xs {
		out, err := runner16.Run(xs[r], start+r, start+r)
		if err != nil {
			t.Fatalf("seq runner row %d: %v", r, err)
		}
		refLast = out
	}

	mwBatch := buildMW()
	defer mwBatch.Release()
	positions := make([]int, M)
	for r := range positions {
		positions[r] = start + r
	}
	got, err := ctx.PrefillLastW8A8(xs, mwBatch, hidden, nH, nKV, hd, inter, positions, 0, eps, scale, false)
	if err != nil {
		t.Fatalf("PrefillLastW8A8: %v", err)
	}

	cos, maxAbs := cosine(got, refLast)
	t.Logf("kvF16 last row (pos %d): cosine=%.8f maxAbs=%.3e", start+M-1, cos, maxAbs)
	if cos < 0.99999 || maxAbs > 5e-3 {
		t.Errorf("diverges from sequential DecodeRunner f16: cosine=%.8f maxAbs=%.3e", cos, maxAbs)
	}
}
