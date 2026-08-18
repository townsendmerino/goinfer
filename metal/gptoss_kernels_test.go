//go:build darwin

// gpt-oss's three resident departures, checked against the CPU reference (decoder/forward_gptoss.go):
// the attention sink (promoted in-place into kernels.go's shared `attention`, mirroring the
// rope mscale precedent), and the clamped-SwiGLU expert + custom router (promoted into
// moe.go's moeKernels as swiglu_quant_gptoss/route_gptoss). All three tests here compile the
// REAL shared library (allKernels[+moeKernels]), not an isolated copy, so they gate what
// actually ships. FeatAttnSink is still not declared for Metal — the kernels/promotion are done,
// but nothing is wired into model.go's per-layer resident dispatch yet (own-forward bridging,
// docs/queue-correctness.md G7/G10).
package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestGptOssAttnSink_metal gates the sink term against decoder/forward_gptoss.go's
// gptOssAttention: a per-head learned logit with no key and no value that joins the softmax MAX
// and the DENOMINATOR only —
//
//	m     = max(max_s score_s, sink_h)
//	denom = Σ_s exp(score_s − m) + exp(sink_h − m)
//
// It cannot be folded in as a post-hoc denominator correction (that would need every exponent
// re-based on a new max), so the dominates-the-max case is the one that actually exercises the
// ordering — most random inputs would pass a wrong post-hoc version too.
//
// Compiles allKernels (kernels.go's REAL, shipped `attention`) rather than an isolated copy —
// that kernel is the one every family's decode path actually dispatches (f16 KV; gpt-oss uses
// this exact path, not attention_f32, which stays Gemma-sandwich-only and disabled), so this
// gates what will actually run, not a copy that could silently drift from it.
func TestGptOssAttnSink_metal(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "attention")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const nH, nKV, hd, nKeys = 4, 2, 8, 6
	const tgs = 128
	scale := float32(0.35)
	qDim, kvDim := nH*hd, nKV*hd

	q := make([]float32, qDim)
	kc := make([]float32, nKeys*kvDim)
	vc := make([]float32, nKeys*kvDim)
	for i := range q {
		q[i] = float32(math.Sin(float64(i)*0.7)) * 0.8
	}
	for i := range kc {
		kc[i] = float32(math.Cos(float64(i)*0.31)) * 0.9
		vc[i] = float32(math.Sin(float64(i)*0.13)) * 1.1
	}
	kcH := make([]uint16, len(kc))
	vcH := make([]uint16, len(vc))
	for i := range kc {
		kcH[i], vcH[i] = f32ToF16(kc[i]), f32ToF16(vc[i])
	}

	// CPU reference — the same math forward_gptoss.go implements. Runs on the f16-rounded KV
	// (kcH/vcH dequantized back), so it compares against what the kernel ACTUALLY reads, not the
	// pre-rounding f32 source — the same convention every other f16-KV test in this package uses.
	ref := func(sinks []float32) []float32 {
		out := make([]float32, qDim)
		group := nH / nKV
		for h := range nH {
			kvh := h / group
			sc := make([]float64, nKeys)
			mx := math.Inf(-1)
			for s := range nKeys {
				var dot float64
				for dd := range hd {
					dot += float64(q[h*hd+dd]) * float64(f16ToF32(kcH[s*kvDim+kvh*hd+dd]))
				}
				sc[s] = dot * float64(scale)
				mx = math.Max(mx, sc[s])
			}
			if sinks != nil {
				mx = math.Max(mx, float64(sinks[h]))
			}
			denom := 0.0
			for s := range nKeys {
				sc[s] = math.Exp(sc[s] - mx)
				denom += sc[s]
			}
			if sinks != nil {
				denom += math.Exp(float64(sinks[h]) - mx)
			}
			for dd := range hd {
				acc := 0.0
				for s := range nKeys {
					acc += sc[s] * float64(f16ToF32(vcH[s*kvDim+kvh*hd+dd]))
				}
				out[h*hd+dd] = float32(acc / denom)
			}
		}
		return out
	}

	q_ := d.NewCommandQueue()
	dQ := d.NewBufferFloats(q)
	dK := d.NewBufferU16s(kcH)
	dV := d.NewBufferU16s(vcH)
	uNH, uNKV, uHd, uNKeys := d.NewBufferU32(uint32(nH)), d.NewBufferU32(uint32(nKV)), d.NewBufferU32(uint32(hd)), d.NewBufferU32(uint32(nKeys))
	uScale, uWindow0 := d.NewBufferFloats([]float32{scale}), d.NewBufferU32(0)

	run := func(sinks []float32) []float32 {
		dOut := d.NewBufferLen(qDim)
		var dSink Buffer
		var hasSink uint32
		if sinks != nil {
			dSink = d.NewBufferFloats(sinks)
			hasSink = 1
		} else {
			dSink = d.NewBufferLen(nH) // unused placeholder — hasSink=0 means the kernel never reads it
		}
		q_.Run1D(pipe, nH*tgs, tgs, dQ, dK, dV, dOut, uNH, uNKV, uHd, uNKeys, uScale, uWindow0, dSink, d.NewBufferU32(hasSink))
		return dOut.Floats()
	}

	cmp := func(name string, got, want []float32, tol float64) {
		worst, at := 0.0, -1
		for i := range want {
			diff := math.Abs(float64(got[i] - want[i]))
			if diff > worst {
				worst, at = diff, i
			}
		}
		t.Logf("%s: max|diff| = %.3e", name, worst)
		if worst > tol {
			t.Errorf("%s: max|diff| %.3e > %.1e at %d (got %v want %v)", name, worst, tol, at, got[at], want[at])
		}
	}

	// 1. NULL sink must be exactly the pre-existing behaviour.
	cmp("no sink", run(nil), ref(nil), 1e-5)

	// 2. A modest sink, below the score max.
	small := []float32{-1.0, -0.5, 0.25, -2.0}
	cmp("small sink", run(small), ref(small), 1e-5)

	// 3. THE SINK DOMINATES the max — the case a post-hoc denominator patch gets wrong.
	big := []float32{20, 25, 30, 18}
	gotBig := run(big)
	cmp("sink dominates", gotBig, ref(big), 1e-5)
	var sumAbs float64
	for _, v := range gotBig {
		sumAbs += math.Abs(float64(v))
	}
	if sumAbs > 1e-3 {
		t.Errorf("with a dominating sink the context should collapse toward zero, got Σ|out| = %g", sumAbs)
	}

	// 4. A very negative sink must be indistinguishable from no sink at all.
	tiny := []float32{-60, -60, -60, -60}
	cmp("negligible sink ≈ no sink", run(tiny), ref(nil), 1e-5)
}

// TestGptOssSwigluQuant_metal gates gpt-oss's clamped interleaved-SwiGLU expert against
// decoder/forward_gptoss.go's gptOssExpert:
//
//	gate = clamp(Gate·h + gateBias, max = limit)      // UPPER only
//	up   = clamp(Up·h   + upBias,   [-limit, limit])  // BOTH
//	glu  = gate · sigmoid(alpha · gate)
//	d    = (up + 1) · glu
//
// Inputs deliberately saturate both branches — a clamp that is symmetric on the gate (the
// tidier-looking mistake) agrees everywhere the activation does not saturate, so a test that
// never crosses ±limit cannot tell the two apart. Per-expert biases are looked up via
// idx[slot]*2*I, matching the on-device addressing gpt-oss's real MoE dispatch needs (the
// router decides which expert runs; the launch geometry stays static).
// Compiles allKernels (the REAL, shipped swiglu_quant_gptoss) rather than an
// isolated copy — same rationale as TestGptOssAttnSink_metal above.
func TestGptOssSwigluQuant_metal(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "swiglu_quant_gptoss")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const I = 256
	const alpha, limit = float32(1.702), float32(7.0)

	g := make([]float32, I)
	u := make([]float32, I)
	gb := make([]float32, I)
	ub := make([]float32, I)
	for i := range g {
		g[i] = float32(math.Sin(float64(i)*0.37)) * 12 // |g| up to 12 > limit 7
		u[i] = float32(math.Cos(float64(i)*0.29)) * 11
		gb[i] = float32(math.Sin(float64(i)*1.1)) * 0.5
		ub[i] = float32(math.Cos(float64(i)*0.9)) * 0.5
	}
	clampedHi, clampedLo := 0, 0
	for i := range g {
		if g[i]+gb[i] > limit {
			clampedHi++
		}
		if u[i]+ub[i] < -limit {
			clampedLo++
		}
	}
	if clampedHi == 0 || clampedLo == 0 {
		t.Fatalf("inputs do not exercise both clamps (%d over, %d under) — the test would pass "+
			"against a symmetric clamp", clampedHi, clampedLo)
	}
	t.Logf("clamp coverage: %d elements over +limit, %d under -limit", clampedHi, clampedLo)

	// Bias table for TWO experts; idx selects expert 1. Expert 0 is a decoy that would be
	// visibly wrong if the on-device biasOff arithmetic picked the wrong row.
	const nExp, slot = 2, 0
	biasGU := make([]float32, nExp*2*I)
	for k := range I {
		biasGU[k] = 99        // expert 0 gate bias: decoy
		biasGU[I+k] = -99     // expert 0 up bias: decoy
		biasGU[2*I+k] = gb[k] // expert 1 gate bias (the one idx selects)
		biasGU[3*I+k] = ub[k] // expert 1 up bias
	}

	q_ := d.NewCommandQueue()
	dQ := d.NewBufferBytes(I)
	dS := d.NewBufferLen(1)
	q_.Run1D(pipe, 256, 256,
		d.NewBufferFloats(g), d.NewBufferFloats(u), dQ, dS, d.NewBufferU32(uint32(I)),
		d.NewBufferFloats(biasGU), d.NewBufferUint32s([]uint32{1}), d.NewBufferU32(uint32(slot)),
		d.NewBufferU32(1), d.NewBufferFloats([]float32{alpha}), d.NewBufferFloats([]float32{limit}))
	qs := dQ.Int8s()
	sc := dS.Floats()[0]

	// CPU reference — decoder/forward_gptoss.go's gptOssExpert, activation only.
	want := make([]float32, I)
	for i := range want {
		gx := g[i] + gb[i]
		ux := u[i] + ub[i]
		if gx > limit {
			gx = limit
		}
		if ux > limit {
			ux = limit
		} else if ux < -limit {
			ux = -limit
		}
		glu := gx * float32(1.0/(1.0+math.Exp(-float64(alpha*gx))))
		want[i] = (ux + 1) * glu
	}

	step := float64(sc)
	worst, at := 0.0, -1
	for i := range want {
		got := float64(qs[i]) * step
		diff := math.Abs(got - float64(want[i]))
		if diff > worst {
			worst, at = diff, i
		}
	}
	t.Logf("clamped-SwiGLU: scale %.6g, max|diff| %.3e (one int8 step = %.3e)", step, worst, step)
	if worst > 1.5*step {
		t.Errorf("max|diff| %.3e at %d exceeds 1.5 quantization steps (%.3e) — a real disagreement, "+
			"not rounding", worst, at, 1.5*step)
	}
}

// TestGptOssRoute_metal gates gpt-oss's MoE router against decoder's gptOssMoE arithmetic.
// Metal's existing moe_route (DeepSeek/GLM/Qwen-MoE shape) computes sel = score + bias,
// top-k on sel, weight = score[best] — the bias steers SELECTION only. gpt-oss's router folds
// the bias into the logits BEFORE top-k and softmaxes over the SELECTED biased logits, so
// running it through moe_route would emit well-formed weights that are simply not this
// model's — invisible to any check that only asks whether the output is finite or normalized.
// The test asserts the WEIGHTS, and includes a bias that changes WHICH experts win, since a
// router ignoring the bias would still look like a valid softmax otherwise.
// Compiles allKernels (the REAL, shipped route_gptoss) rather than an isolated
// copy — same rationale as TestGptOssAttnSink_metal above.
func TestGptOssRoute_metal(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "route_gptoss")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const nE, k = 32, 4
	logits := make([]float32, nE)
	bias := make([]float32, nE)
	for i := range logits {
		logits[i] = float32(math.Sin(float64(i)*0.9)) * 2
	}
	// Bias REORDERS the winners: experts 30/31 have low logits and large biases.
	bias[30], bias[31] = 5, 4.5

	q_ := d.NewCommandQueue()
	dIdx := d.NewBufferLen(k) // uint32-sized scratch reused as a uint buffer (4 bytes/elt)
	dW := d.NewBufferLen(k)
	q_.Run1D(pipe, 1, 1,
		d.NewBufferFloats(logits), d.NewBufferFloats(bias), dIdx, dW,
		d.NewBufferU32(uint32(nE)), d.NewBufferU32(uint32(k)))
	gotIdx := dIdx.U32s()
	gotW := dW.Floats()

	// Reference: bias into the logits, top-k (ties → lowest index), softmax over the selected.
	ref := make([]float64, nE)
	for i := range ref {
		ref[i] = float64(logits[i]) + float64(bias[i])
	}
	taken := make([]bool, nE)
	wantIdx := make([]int, k)
	sel := make([]float64, k)
	for j := range k {
		best, bv := -1, math.Inf(-1)
		for i := range nE {
			if !taken[i] && ref[i] > bv {
				bv, best = ref[i], i
			}
		}
		taken[best] = true
		wantIdx[j], sel[j] = best, bv
	}
	mx := sel[0]
	for _, v := range sel {
		mx = math.Max(mx, v)
	}
	sum := 0.0
	for j := range sel {
		sel[j] = math.Exp(sel[j] - mx)
		sum += sel[j]
	}

	if wantIdx[0] != 30 && wantIdx[1] != 30 {
		t.Fatalf("test setup no longer exercises bias-driven selection: winners %v", wantIdx)
	}
	t.Logf("route_gptoss: idx %v (bias promoted expert 30/31), weights %v", gotIdx, gotW)

	for j := range k {
		if int(gotIdx[j]) != wantIdx[j] {
			t.Fatalf("idx[%d] = %d, want %d (full got %v want %v)", j, gotIdx[j], wantIdx[j], gotIdx, wantIdx)
		}
		want := sel[j] / sum
		if diff := math.Abs(float64(gotW[j]) - want); diff > 1e-6 {
			t.Errorf("weight[%d] = %v, want %v (|diff| %.2e) — softmax must be over the SELECTED "+
				"top-k biased logits, not the unbiased scores", j, gotW[j], want, diff)
		}
	}
	tot := 0.0
	for _, w := range gotW {
		tot += float64(w)
	}
	if math.Abs(tot-1) > 1e-5 {
		t.Errorf("weights sum to %v, want 1", tot)
	}
}

// TestGptOssMoEDownBias_metal gates gemv_w4a8_moe_wacc_bias — gpt-oss's MoE down-projection
// combine. Unlike GPT-2's o-proj/down-proj bias (a plain += bias, no router weight involved),
// gpt-oss's expert bias is added INSIDE the expert before the router weight scales the result
// (decoder/forward_gptoss.go: dst = Down·h + downBias; gptOssMoE combines out += w·dst), so the
// kernel must compute out[row] += wgt[slot]*(acc*asc[0] + bias[row]), not wgt[slot]*acc*asc[0] +
// bias[row]. Two-expert setup (like TestSAGemvBiasResid) so the test also exercises the
// idx[slot]*rowsPerExpert addressing, not just the math.
func TestGptOssMoEDownBias_metal(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "gemv_w4a8_moe_wacc_bias")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const nExp, N, K = 2, 64, 512 // N = rowsPerExpert (down-proj output = hidden)
	const slot = 0
	rng := rand.New(rand.NewSource(53))

	a := make([]float32, K)
	for i := range a {
		a[i] = rng.Float32()*2 - 1
	}
	amx := float32(0)
	for _, v := range a {
		if x := float32(math.Abs(float64(v))); x > amx {
			amx = x
		}
	}
	aSc := amx / 127
	aq := make([]int8, K)
	for i, v := range a {
		aq[i] = int8(math.Max(-127, math.Min(127, math.Round(float64(v/aSc)))))
	}

	wgt := []float32{0.37} // this slot's router weight
	resid0 := make([]float32, N)
	bias := make([]float32, nExp*N) // per-expert down bias, decoy on expert 0
	for i := range resid0 {
		resid0[i] = rng.Float32()*4 - 2
	}
	for i := range N {
		bias[i] = 999                   // expert 0: decoy, would be visibly wrong if the addressing picked it
		bias[N+i] = rng.Float32()*2 - 1 // expert 1: the one idx selects
	}

	words := make([]uint32, nExp*N*(K/8))
	scalesH := make([]uint16, nExp*N*(K/32))
	ref := make([]float32, N)
	for row := range N {
		wrow := 1*N + row // expert 1 (idx selects 1), matching idx[slot]*rowsPerExpert+row
		rowVals := make([]float32, K)
		for k := range rowVals {
			rowVals[k] = rng.Float32()*2 - 1
		}
		w, s := packW4A8Row(rowVals)
		copy(words[wrow*(K/8):(wrow+1)*(K/8)], w)
		var acc float64
		for g := range K / 32 {
			scalesH[wrow*(K/32)+g] = f32ToF16(s[g])
			sc := float64(f16ToF32(scalesH[wrow*(K/32)+g]))
			for e := range 32 {
				k := g*32 + e
				nib := int((w[k/8]>>(4*uint(k%8)))&0xF) - 8
				acc += float64(nib) * float64(aq[k]) * sc
			}
		}
		ref[row] = resid0[row] + wgt[slot]*(float32(acc)*aSc+bias[wrow])
	}
	// Fill expert 0's (unselected) weight rows too, so a wrong idx would read something real
	// rather than zeros — zeros could accidentally look "close enough" on a small K.
	for row := range N {
		rowVals := make([]float32, K)
		for k := range rowVals {
			rowVals[k] = rng.Float32()*2 - 1
		}
		w, s := packW4A8Row(rowVals)
		copy(words[row*(K/8):(row+1)*(K/8)], w)
		for g := range K / 32 {
			scalesH[row*(K/32)+g] = f32ToF16(s[g])
		}
	}

	q := d.NewCommandQueue()
	out := d.NewBufferFloats(resid0)
	q.Run1DTG(pipe, N*32, 256, K*2,
		d.NewBufferUint32s(words), d.NewBufferU16s(scalesH), d.NewBufferInt8(aq),
		d.NewBufferFloats([]float32{aSc}), out,
		d.NewBufferU32(uint32(K)), d.NewBufferUint32s([]uint32{1}), d.NewBufferFloats(wgt),
		d.NewBufferU32(uint32(slot)), d.NewBufferU32(uint32(N)), d.NewBufferFloats(bias))
	got := out.Floats()

	var dot, na, nb, maxRel float64
	for n := range N {
		dot += float64(got[n]) * float64(ref[n])
		na += float64(got[n]) * float64(got[n])
		nb += float64(ref[n]) * float64(ref[n])
		if dd := math.Abs(float64(got[n] - ref[n])); dd > 1e-3 {
			if rel := dd / (math.Abs(float64(ref[n])) + 1e-3); rel > maxRel {
				maxRel = rel
			}
		}
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	mustFinite(t, "gpt-oss MoE down-bias cosine", cos)
	if cos < 0.9999 || maxRel > 5e-3 {
		t.Fatalf("gemv_w4a8_moe_wacc_bias parity FAIL: cos=%.7f maxRel=%.4f", cos, maxRel)
	}
	t.Logf("gemv_w4a8_moe_wacc_bias N=%d K=%d vs CPU: cos=%.7f maxRel=%.4f — PARITY ✓", N, K, cos, maxRel)
}
