package decoder

import (
	"math"
	"math/rand"
	"testing"
)

func benchFloatSlice(n int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(r.NormFloat64())
	}
	return s
}

func BenchmarkRMSNorm_Dim896(b *testing.B) {
	const dim = 896
	x := benchFloatSlice(dim, 1)
	weight := benchFloatSlice(dim, 2)
	b.SetBytes(dim * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rmsNorm(x, weight, 1, dim, 1e-6, false)
	}
}

func BenchmarkRMSNorm_Dim2048(b *testing.B) {
	const dim = 2048
	x := benchFloatSlice(dim, 1)
	weight := benchFloatSlice(dim, 2)
	b.SetBytes(dim * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rmsNorm(x, weight, 1, dim, 1e-6, false)
	}
}

func BenchmarkRMSNorm_Dim4096(b *testing.B) {
	const dim = 4096
	x := benchFloatSlice(dim, 1)
	weight := benchFloatSlice(dim, 2)
	b.SetBytes(dim * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rmsNorm(x, weight, 1, dim, 1e-6, false)
	}
}

func BenchmarkLayerNorm_Dim896(b *testing.B) {
	const dim = 896
	x := benchFloatSlice(dim, 1)
	weight := benchFloatSlice(dim, 2)
	bias := benchFloatSlice(dim, 3)
	b.SetBytes(dim * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layerNorm(x, weight, bias, 1, dim, 1e-5)
	}
}

func BenchmarkLayerNorm_Dim2048(b *testing.B) {
	const dim = 2048
	x := benchFloatSlice(dim, 1)
	weight := benchFloatSlice(dim, 2)
	bias := benchFloatSlice(dim, 3)
	b.SetBytes(dim * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layerNorm(x, weight, bias, 1, dim, 1e-5)
	}
}

func BenchmarkRoPE_Heads14_Dim64(b *testing.B) {
	const heads = 14
	const headDim = 64
	const half = headDim / 2
	vec := benchFloatSlice(heads*headDim, 1)
	invFreq := make([]float64, half)
	for i := range invFreq {
		invFreq[i] = 1.0 / float64(10000*(i+1))
	}
	b.SetBytes(heads * headDim * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applyRoPE(vec, heads, headDim, 42, invFreq, 1.0)
	}
}

func BenchmarkRoPE_Heads32_Dim128(b *testing.B) {
	const heads = 32
	const headDim = 128
	const half = headDim / 2
	vec := benchFloatSlice(heads*headDim, 1)
	invFreq := make([]float64, half)
	for i := range invFreq {
		invFreq[i] = 1.0 / float64(10000*(i+1))
	}
	b.SetBytes(heads * headDim * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applyRoPE(vec, heads, headDim, 42, invFreq, 1.0)
	}
}

func BenchmarkSwiGLU_Inter4864(b *testing.B) {
	const dim = 4864
	gate := benchFloatSlice(dim, 1)
	up := benchFloatSlice(dim, 2)
	b.SetBytes(dim * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		swiglu(gate, up)
	}
}

func BenchmarkSwiGLU_Inter8960(b *testing.B) {
	const dim = 8960
	gate := benchFloatSlice(dim, 1)
	up := benchFloatSlice(dim, 2)
	b.SetBytes(dim * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		swiglu(gate, up)
	}
}

func BenchmarkAddBias_Dim896(b *testing.B) {
	const dim = 896
	x := benchFloatSlice(dim, 1)
	bias := benchFloatSlice(dim, 2)
	b.SetBytes(dim * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		addBias(x, bias)
	}
}

func BenchmarkAddBias_Dim4864(b *testing.B) {
	const dim = 4864
	x := benchFloatSlice(dim, 1)
	bias := benchFloatSlice(dim, 2)
	b.SetBytes(dim * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		addBias(x, bias)
	}
}

func BenchmarkResidualAdd_Dim896(b *testing.B) {
	const dim = 896
	h := benchFloatSlice(dim, 1)
	sub := benchFloatSlice(dim, 2)
	b.SetBytes(dim * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		addResidual(h, sub)
	}
}

func BenchmarkResidualAdd_Dim2048(b *testing.B) {
	const dim = 2048
	h := benchFloatSlice(dim, 1)
	sub := benchFloatSlice(dim, 2)
	b.SetBytes(dim * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		addResidual(h, sub)
	}
}

func BenchmarkRing_BatchReadLocal_W1024(b *testing.B) {
	const nKV, hd, W = 4, 128, 1024
	kvDim := nKV * hd
	cache := NewKVCache(1, nKV, hd, W, W*2, nil)
	cache.enableRings(W, func(int) bool { return false })
	kRow := make([]float32, kvDim)
	vRow := make([]float32, kvDim)
	for p := range W {
		cache.rings[0].write(p, kRow, vRow)
	}
	cache.rings[0].count = W
	newK := make([]float32, kvDim)
	newV := make([]float32, kvDim)
	dstK := make([]float32, W*kvDim)
	dstV := make([]float32, W*kvDim)
	startPos := W + 500 // wrapped around
	b.SetBytes(int64(W * kvDim * 4 * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.batchReadLocal(0, startPos, 1, newK, newV, dstK, dstV)
	}
}

func BenchmarkRing_CommitBatch_K64_W1024(b *testing.B) {
	const nKV, hd, W, K = 4, 128, 1024, 64
	kvDim := nKV * hd
	cache := NewKVCache(1, nKV, hd, W, W*2, nil)
	cache.enableRings(W, func(int) bool { return false })
	kRow := make([]float32, kvDim)
	vRow := make([]float32, kvDim)
	cache.rings[0].write(0, kRow, vRow)
	newK := make([]float32, K*kvDim)
	newV := make([]float32, K*kvDim)
	startPos := 500
	b.SetBytes(int64(K * kvDim * 4 * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.commitBatch(0, startPos, K, newK, newV)
	}
}

func BenchmarkKV_DequantGlobalLayer_N512(b *testing.B) {
	const nKV, hd, N = 4, 128, 512
	kvDim := nKV * hd
	cache := NewKVCache(1, nKV, hd, 0, N, nil)
	cache.setQuant(kvI8, N)
	kRow := make([]float32, kvDim)
	vRow := make([]float32, kvDim)
	for range N {
		cache.Append(0, kRow, vRow)
	}
	dstK := make([]float32, N*kvDim)
	dstV := make([]float32, N*kvDim)
	b.SetBytes(int64(N * kvDim * 4 * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.dequantGlobalLayer(0, kvDim, dstK, dstV)
	}
}

func BenchmarkNgram_Draft_Ctx1024(b *testing.B) {
	const ctxLen = 1024
	ctx := make([]int, ctxLen)
	for i := range ctx {
		ctx[i] = i % 32
	}
	d := &NgramDrafter{MinMatch: 2, MaxMatch: 16}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		d.Draft(ctx, 4)
	}
}

func BenchmarkArgmax_Vocab32k(b *testing.B) {
	const vocab = 32000
	logits := benchFloatSlice(vocab, 1)
	b.SetBytes(vocab * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = argmax(logits)
	}
}

func BenchmarkArgmax_Vocab128k(b *testing.B) {
	const vocab = 128000
	logits := benchFloatSlice(vocab, 1)
	b.SetBytes(vocab * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = argmax(logits)
	}
}

func BenchmarkSampler_Observe_Greedy(b *testing.B) {
	s := NewSampler(SamplingParams{Temperature: 0})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Observe(i % 1000)
	}
}

func BenchmarkMoE_Combine_Dim2048(b *testing.B) {
	const dim = 2048
	out := benchFloatSlice(dim, 1)
	expOut := benchFloatSlice(dim, 2)
	b.SetBytes(dim * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		addScaled(out, expOut, 0.75)
	}
}

func BenchmarkNormCopy_Dim2048(b *testing.B) {
	const dim = 2048
	src := benchFloatSlice(dim, 1)
	dst := benchFloatSlice(dim, 2)
	weight := benchFloatSlice(dim, 3)
	b.SetBytes(dim * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rmsNormInto(dst, src, weight, 1, dim, 1e-6, false)
	}
}

func BenchmarkRoPE_Interleaved_Heads32_Dim128(b *testing.B) {
	const heads = 32
	const headDim = 128
	const half = headDim / 2
	vec := benchFloatSlice(heads*headDim, 1)
	invFreq := make([]float64, half)
	for i := range invFreq {
		invFreq[i] = 1.0 / float64(10000*(i+1))
	}
	b.SetBytes(heads * headDim * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applyRoPEInterleaved(vec, heads, headDim, 42, invFreq, 1.0)
	}
}

func BenchmarkForwardN_NormPrefill_K512_Dim2048(b *testing.B) {
	const K = 512
	const dim = 2048
	h := benchFloatSlice(K*dim, 1)
	norm := make([]float32, K*dim)
	weight := benchFloatSlice(dim, 2)
	b.SetBytes(K * dim * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for r := 0; r < K; r++ {
			rmsNormInto(norm[r*dim:(r+1)*dim], h[r*dim:(r+1)*dim], weight, 1, dim, 1e-6, false)
		}
	}
}

func BenchmarkAttendQuery_Heads32_K512_Dim128(b *testing.B) {
	const nH = 32
	const nKV = 8
	const hd = 128
	const nKeys = 512
	arch := &Architecture{
		NumHeads:   nH,
		NumKVHeads: nKV,
		HeadDim:    hd,
		AttnScale:  1.0 / math.Sqrt(float64(hd)),
	}
	cache := NewKVCache(1, nKV, hd, 0, nKeys, nil)
	k := benchFloatSlice(nKV*hd, 2)
	v := benchFloatSlice(nKV*hd, 3)
	for s := 0; s < nKeys; s++ {
		cache.Append(0, k, v)
	}
	q := benchFloatSlice(nH*hd, 1)
	ctx := make([]float32, nH*hd)
	scores := make([]float32, nKeys)

	b.SetBytes(int64(nKeys * hd * 4 * nH))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		attendQuery(q, ctx, scores, cache, 0, nKeys, true, arch)
	}
}

func BenchmarkSoftmaxF32_E64(b *testing.B) {
	const E = 64
	logits := benchFloatSlice(E, 1)
	b.SetBytes(E * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = softmaxF32(logits)
	}
}

func BenchmarkTopFilter_Vocab32k(b *testing.B) {
	const V = 32000
	logits := benchFloatSlice(V, 1)
	vocabScratch := make([]float64, V)
	candScratch := make([]int, V)
	ipsScratch := make([]indexedProb, V)

	b.SetBytes(V * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = topFilterLogits(logits, 0.8, 50, 0.9, 0.05, vocabScratch, candScratch, ipsScratch)
	}
}

func BenchmarkDeltaNet_Recurrence_Heads16_Dim128(b *testing.B) {
	const nv = 16
	const nk = 4
	const hk = 128
	const hv = 128
	keyDim := hk * nk
	valueDim := hv * nv
	convDim := 2*keyDim + valueDim

	conv := benchFloatSlice(convDim, 1)
	at := benchFloatSlice(nv, 2)
	bt := benchFloatSlice(nv, 3)
	w := &deltaNetWeights{
		dtBias:  benchFloatSlice(nv, 4),
		negExpA: benchFloatSlice(nv, 5),
	}
	p := qwen35Params{
		NumKeyHeads:   nk,
		NumValueHeads: nv,
		KeyHeadDim:    hk,
		ValueHeadDim:  hv,
	}
	st := &deltaState{
		s: make([]float32, nv*hk*hv),
	}
	sOrig := benchFloatSlice(nv*hk*hv, 6)
	core := make([]float32, valueDim)

	b.SetBytes(int64(nv * hk * hv * 4 * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(st.s, sOrig)
		deltaNetRecurrence(core, conv, at, bt, w, p, st)
	}
}

func BenchmarkKDA_Recurrence_Heads16_Dim128(b *testing.B) {
	const H = 16
	const D = 128
	projSize := H * D

	q := benchFloatSlice(projSize, 1)
	k := benchFloatSlice(projSize, 2)
	v := benchFloatSlice(projSize, 3)
	gDecayRaw := benchFloatSlice(projSize, 4)
	betaLogits := benchFloatSlice(H, 5)
	w := &kdaWeights{
		dtBias: benchFloatSlice(projSize, 6),
		aLog:   benchFloatSlice(H, 7),
	}
	p := kdaParams{
		NumHeads:   H,
		HeadDim:    D,
		LowerBound: -5.0,
	}
	st := &kdaState{
		s: make([]float32, H*D*D),
	}
	sOrig := benchFloatSlice(H*D*D, 8)
	core := make([]float32, projSize)

	b.SetBytes(int64(H * D * D * 4 * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(st.s, sOrig)
		kdaRecurrence(core, q, k, v, gDecayRaw, betaLogits, w, p, st)
	}
}
