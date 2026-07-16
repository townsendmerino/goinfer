//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestLayerB_attentionParity — Layer B: the non-trivial kernel. GQA causal attention with
// ONLINE (numerically-stable) softmax over the resident KV cache: for query head qh
// (kv head kvh = qh/(nH/nKV)), score_s = scale·(q·k_s), streamed softmax over s∈[0,nKeys),
// output = Σ softmax_s · v_s. One thread per query head (correct-first; the per-head
// threadgroup parallelism is the tuning step). Validated vs a plain CPU softmax.
func TestLayerB_attentionParity(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	const src = `
#include <metal_stdlib>
using namespace metal;
kernel void attention(device const float* q      [[buffer(0)]],  // [nH*hd] (post-RoPE)
                      device const float* kc     [[buffer(1)]],  // [nKeys*kvDim]
                      device const float* vc     [[buffer(2)]],  // [nKeys*kvDim]
                      device float*       out    [[buffer(3)]],  // [nH*hd]
                      constant uint&      nH     [[buffer(4)]],
                      constant uint&      nKV    [[buffer(5)]],
                      constant uint&      hd     [[buffer(6)]],
                      constant uint&      nKeys  [[buffer(7)]],
                      constant float&     scale  [[buffer(8)]],
                      uint qh [[thread_position_in_grid]]) {
    if (qh >= nH) return;
    uint kvDim = nKV*hd;
    uint kvh   = qh / (nH/nKV);
    uint qbase = qh*hd;
    uint kvbase= kvh*hd;
    float acc[128];
    for (uint dd=0; dd<hd; dd++) acc[dd]=0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint s=0; s<nKeys; s++) {
        float score = 0.0f;
        for (uint dd=0; dd<hd; dd++) score += q[qbase+dd]*kc[s*kvDim + kvbase + dd];
        score *= scale;
        float mnew = max(m, score);
        float alpha = exp(m - mnew);
        float p = exp(score - mnew);
        l = l*alpha + p;
        for (uint dd=0; dd<hd; dd++) acc[dd] = acc[dd]*alpha + p*vc[s*kvDim + kvbase + dd];
        m = mnew;
    }
    for (uint dd=0; dd<hd; dd++) out[qbase+dd] = acc[dd]/l;
}`
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "attention")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const nH, nKV, hd, nKeys = 12, 2, 128, 96 // GQA 6:1, 96 cached positions
	kvDim := nKV * hd
	scale := float32(1 / math.Sqrt(float64(hd)))
	rng := rand.New(rand.NewSource(9))
	q := make([]float32, nH*hd)
	for i := range q {
		q[i] = rng.Float32()*2 - 1
	}
	kc := make([]float32, nKeys*kvDim)
	vc := make([]float32, nKeys*kvDim)
	for i := range kc {
		kc[i] = rng.Float32()*2 - 1
		vc[i] = rng.Float32()*2 - 1
	}

	// CPU reference — plain (max-subtracted) softmax.
	ref := make([]float32, nH*hd)
	for qh := 0; qh < nH; qh++ {
		kvh := qh / (nH / nKV)
		sc := make([]float64, nKeys)
		mx := math.Inf(-1)
		for s := 0; s < nKeys; s++ {
			var dot float64
			for dd := 0; dd < hd; dd++ {
				dot += float64(q[qh*hd+dd]) * float64(kc[s*kvDim+kvh*hd+dd])
			}
			sc[s] = dot * float64(scale)
			if sc[s] > mx {
				mx = sc[s]
			}
		}
		var sum float64
		for s := range sc {
			sc[s] = math.Exp(sc[s] - mx)
			sum += sc[s]
		}
		for dd := 0; dd < hd; dd++ {
			var acc float64
			for s := 0; s < nKeys; s++ {
				acc += sc[s] * float64(vc[s*kvDim+kvh*hd+dd])
			}
			ref[qh*hd+dd] = float32(acc / sum)
		}
	}

	cq := d.NewCommandQueue()
	out := d.NewBufferLen(nH * hd)
	cq.Run1D(pipe, nH, 32,
		d.NewBufferFloats(q), d.NewBufferFloats(kc), d.NewBufferFloats(vc), out,
		d.NewBufferU32(nH), d.NewBufferU32(nKV), d.NewBufferU32(hd), d.NewBufferU32(nKeys),
		d.NewBufferFloats([]float32{scale}),
	)
	got := out.Floats()

	var dot, na, nb, maxabs float64
	for i := range got {
		dot += float64(got[i]) * float64(ref[i])
		na += float64(got[i]) * float64(got[i])
		nb += float64(ref[i]) * float64(ref[i])
		if dd := math.Abs(float64(got[i] - ref[i])); dd > maxabs {
			maxabs = dd
		}
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	if cos < 0.999999 || maxabs > 1e-4 {
		t.Fatalf("attention parity FAIL: cosine=%.7f maxAbs=%.2e", cos, maxabs)
	}
	t.Logf("GQA online-softmax attention nH=%d nKV=%d hd=%d nKeys=%d on Metal GPU (cgo-free) vs CPU: cosine=%.7f maxAbs=%.2e — PARITY",
		nH, nKV, hd, nKeys, cos, maxabs)
}
