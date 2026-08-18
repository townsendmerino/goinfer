//go:build darwin

// gpt-oss's three resident departures, ported to Metal and gated standalone — the same phase
// CUDA's own gpt-oss residency work is currently at (cuda/gptoss_act.cu, cuda/attn_sink_test.go,
// cuda/gptoss_act_test.go): kernels exist and are checked against the CPU reference
// (decoder/forward_gptoss.go), but nothing here is wired into model.go's resident dispatch, and
// FeatAttnSink is NOT declared for Metal. Two more capabilities gpt-oss needs (FeatOutBias for
// the o_proj bias, FeatRopeMscale for YaRN) are still missing on BOTH CUDA and Metal — CUDA's own
// FeatAttnSink declaration was tried and reverted for exactly this reason (2224441): kernel-level
// parity is not end-to-end parity. See docs/task-mxfp4-gptoss.md.
//
// Each kernel compiles its own isolated MSL source (the swiglu_test.go pattern), not
// allKernels/moeKernels — so this touches zero shipped call sites and carries zero regression
// risk to any resident family already running on Metal.
package metal

import (
	"math"
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
func TestGptOssAttnSink_metal(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	const src = `
#include <metal_stdlib>
using namespace metal;
kernel void attention_f32_sink(device const float* q[[buffer(0)]], device const float* kc[[buffer(1)]],
    device const float* vc[[buffer(2)]], device float* out[[buffer(3)]], constant uint& nH[[buffer(4)]],
    constant uint& nKV[[buffer(5)]], constant uint& hd[[buffer(6)]], constant uint& nKeys[[buffer(7)]],
    constant float& scale[[buffer(8)]], device const float* sinks[[buffer(9)]], constant uint& hasSink[[buffer(10)]],
    uint qh[[threadgroup_position_in_grid]],
    uint tid[[thread_index_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    uint kvDim = nKV*hd; uint kvh = qh/(nH/nKV);
    device const float* qr = q + qh*hd;
    device const float* kb = kc + kvh*hd;
    device const float* vb = vc + kvh*hd;
    threadgroup float sc[4096];
    threadgroup float red[128];
    for (uint s=tid; s<nKeys; s+=tgs) {
        float a=0; device const float* k=kb+s*kvDim;
        for (uint d=0u; d<hd; d++) a += qr[d]*k[d];
        sc[s]=a*scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m=-INFINITY; for (uint s=tid;s<nKeys;s+=tgs) m=max(m,sc[s]);
    red[tid]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st=tgs/2; st>0; st>>=1){ if(tid<st) red[tid]=max(red[tid],red[tid+st]); threadgroup_barrier(mem_flags::mem_threadgroup); }
    float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
    float sink=0.0f; bool hasS = hasSink != 0u;
    if (hasS) { sink = sinks[qh]; mx = max(mx, sink); }
    float ls=0; for (uint s=tid;s<nKeys;s+=tgs){ float p=exp(sc[s]-mx); sc[s]=p; ls+=p; }
    red[tid]=ls; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st=tgs/2; st>0; st>>=1){ if(tid<st) red[tid]+=red[tid+st]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float sum=red[0];
    if (hasS) sum += exp(sink-mx);
    for (uint d=tid; d<hd; d+=tgs){ float a=0; for(uint s=0u;s<nKeys;s++) a += sc[s]*vb[s*kvDim+d]; out[qh*hd+d]=a/sum; }
}`
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "attention_f32_sink")
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

	// CPU reference — the same math forward_gptoss.go implements.
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
					dot += float64(q[h*hd+dd]) * float64(kc[s*kvDim+kvh*hd+dd])
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
					acc += sc[s] * float64(vc[s*kvDim+kvh*hd+dd])
				}
				out[h*hd+dd] = float32(acc / denom)
			}
		}
		return out
	}

	q_ := d.NewCommandQueue()
	dQ := d.NewBufferFloats(q)
	dK := d.NewBufferFloats(kc)
	dV := d.NewBufferFloats(vc)
	uNH, uNKV, uHd, uNKeys := d.NewBufferU32(uint32(nH)), d.NewBufferU32(uint32(nKV)), d.NewBufferU32(uint32(hd)), d.NewBufferU32(uint32(nKeys))
	uScale := d.NewBufferFloats([]float32{scale})

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
		q_.Run1D(pipe, nH*tgs, tgs, dQ, dK, dV, dOut, uNH, uNKV, uHd, uNKeys, uScale, dSink, d.NewBufferU32(hasSink))
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
func TestGptOssSwigluQuant_metal(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	const src = `
#include <metal_stdlib>
using namespace metal;
inline float gptoss_glu(float gx, float ux, float alpha, float limit) {
    gx = min(gx, limit);
    ux = clamp(ux, -limit, limit);
    float glu = gx/(1.0f+exp(-alpha*gx));
    return (ux+1.0f)*glu;
}
kernel void swiglu_quant_gptoss(device const float* g[[buffer(0)]], device const float* u[[buffer(1)]],
    device char* dq[[buffer(2)]], device float* ds[[buffer(3)]], constant uint& I[[buffer(4)]],
    device const float* biasGU[[buffer(5)]], device const uint* idx[[buffer(6)]], constant uint& slot[[buffer(7)]],
    constant uint& hasBias[[buffer(8)]], constant float& alpha[[buffer(9)]], constant float& limit[[buffer(10)]],
    uint tid[[thread_position_in_threadgroup]], uint tgs[[threads_per_threadgroup]]) {
    threadgroup float red[256];
    uint biasOff = hasBias != 0u ? idx[slot]*2u*I : 0u;
    float mx=0;
    for (uint i=tid;i<I;i+=tgs) {
        float gx=g[i], ux=u[i];
        if (hasBias!=0u) { gx += biasGU[biasOff+i]; ux += biasGU[biasOff+I+i]; }
        float dd = gptoss_glu(gx, ux, alpha, limit);
        mx = max(mx, fabs(dd));
    }
    red[tid]=mx; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=tgs/2;s>0;s>>=1){ if(tid<s) red[tid]=max(red[tid],red[tid+s]); threadgroup_barrier(mem_flags::mem_threadgroup);}
    float sc=red[0]/127.0f; if(sc==0)sc=1; if(tid==0)ds[0]=sc; float inv=1/sc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i=tid;i<I;i+=tgs) {
        float gx=g[i], ux=u[i];
        if (hasBias!=0u) { gx += biasGU[biasOff+i]; ux += biasGU[biasOff+I+i]; }
        float dd = gptoss_glu(gx, ux, alpha, limit);
        dq[i]=char(clamp(int(round(dd*inv)),-127,127));
    }
}`
	lib, err := d.CompileLibrary(src, MSL3_1)
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
func TestGptOssRoute_metal(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	const src = `
#include <metal_stdlib>
using namespace metal;
kernel void route_gptoss(device const float* logits[[buffer(0)]], device const float* bias[[buffer(1)]],
    device uint* outIdx[[buffer(2)]], device float* outWgt[[buffer(3)]], constant uint& nE[[buffer(4)]],
    constant uint& k[[buffer(5)]],
    uint tid[[thread_position_in_grid]]) {
    if (tid != 0u) return;
    float sc[256];
    for (uint i=0u;i<nE;i++) sc[i] = logits[i] + bias[i];
    bool taken[256];
    for (uint i=0u;i<nE;i++) taken[i]=false;
    float chosen[256];
    for (uint j=0u;j<k;j++) {
        int best=-1; float bv=-INFINITY;
        for (uint i=0u;i<nE;i++) if (!taken[i] && sc[i]>bv) { bv=sc[i]; best=int(i); }
        taken[uint(best)]=true;
        outIdx[j]=uint(best);
        chosen[j]=bv;
    }
    float mx=chosen[0];
    for (uint j=1u;j<k;j++) mx=max(mx,chosen[j]);
    float sum=0.0f;
    for (uint j=0u;j<k;j++) { chosen[j]=exp(chosen[j]-mx); sum+=chosen[j]; }
    float inv=1.0f/sum;
    for (uint j=0u;j<k;j++) outWgt[j]=chosen[j]*inv;
}`
	lib, err := d.CompileLibrary(src, MSL3_1)
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
