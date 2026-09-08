//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"math/rand"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
)

// TestLayerNormQuant is the CUDA twin of metal/gpt2_kernels_test.go's TestLayerNormQuant — this
// backend's layernorm_quant (cuda/glue.cu) is a BRAND NEW kernel (G5, docs/task-gpu-paths-2026-09.md,
// the last row: Cohere/Command-R + Cohere2/Command-R7B), so it gets the same isolated,
// exact-CPU-reference proof Metal's kernel already had before any family was declared resident on
// the strength of it.
//
// Bias-free ONLY, unlike Metal's kernel: Cohere's LayerNorm carries no learned bias term, and this
// backend's layernorm_quant has no hasBias parameter at all — a future bias-bearing LayerNorm
// family reaching CUDA would need its own kernel, not a flag on this one (see the kernel's own
// comment). Deliberately not zero-mean input, so the mean-subtraction actually matters — a kernel
// that silently skipped it would still pass a zero-mean-input test.
//
// COSINE, not exact equality: rsqrtf is not IEEE-exact (the same reason quant_vec_exact_test.go's
// comment gives for why rmsnorm_quant has no exact-equality gate of its own), so this compares
// against a float64 CPU reference at a 0.9999 cosine floor, matching Metal's own bar exactly.
func TestLayerNormQuant(t *testing.T) {
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	bg := context.Background()
	ctx := dev.Context()
	mod, err := ctx.LoadModule(gluePTX)
	if err != nil {
		t.Fatalf("LoadModule(gluePTX): %v", err)
	}
	fn, err := mod.Function("layernorm_quant")
	if err != nil {
		t.Fatalf("layernorm_quant: %v", err)
	}
	stream := mustStream(t, ctx)

	const H = 768 // GPT-2/Cohere-scale hidden dim, matching Metal's own TestLayerNormQuant
	const eps = 1e-5
	rng := rand.New(rand.NewSource(41))
	x := make([]float32, H)
	w := make([]float32, H)
	for i := range x {
		x[i] = rng.Float32()*4 - 2 // deliberately not zero-mean
		w[i] = rng.Float32()*0.5 + 0.75
	}

	// Matches decoder/rmsnorm.go's layerNorm (bias-free branch) exactly.
	var mean float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(H)
	var variance float64
	for _, v := range x {
		d := float64(v) - mean
		variance += d * d
	}
	variance /= float64(H)
	inv := 1.0 / math.Sqrt(variance+eps)
	want := make([]float32, H)
	for i, v := range x {
		want[i] = float32((float64(v)-mean)*inv) * w[i]
	}

	dx := mustAlloc[float32](t, ctx, H)
	dw := mustAlloc[float32](t, ctx, H)
	dq := mustAlloc[int32](t, ctx, H/4)
	dsc := mustAlloc[float32](t, ctx, 1)
	if err := gc.CopyHtoD(bg, dx, x); err != nil {
		t.Fatalf("CopyHtoD x: %v", err)
	}
	if err := gc.CopyHtoD(bg, dw, w); err != nil {
		t.Fatalf("CopyHtoD w: %v", err)
	}
	cfg := gc.LaunchConfig{GridX: 1, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((H + 256) * 4)}
	if err := fn.LaunchOn(bg, stream, cfg,
		gc.Arg(dx), gc.Arg(dw), gc.ArgValue(int32(H)), gc.ArgValue(float32(eps)), gc.Arg(dq), gc.Arg(dsc)); err != nil {
		t.Fatalf("launch: %v", err)
	}
	if err := stream.Synchronize(bg); err != nil {
		t.Fatalf("sync: %v", err)
	}
	packed := make([]int32, H/4)
	if err := gc.CopyDtoH(bg, packed, dq); err != nil {
		t.Fatalf("CopyDtoH aq: %v", err)
	}
	scale := make([]float32, 1)
	if err := gc.CopyDtoH(bg, scale, dsc); err != nil {
		t.Fatalf("CopyDtoH aScale: %v", err)
	}
	sc := scale[0]
	got := make([]float32, H)
	for j, word := range packed {
		for b := 0; b < 4; b++ {
			got[j*4+b] = float32(int8(byte(word>>(8*b)))) * sc
		}
	}

	var dot, na, nb float64
	for i := range want {
		dot += float64(got[i]) * float64(want[i])
		na += float64(got[i]) * float64(got[i])
		nb += float64(want[i]) * float64(want[i])
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	if math.IsNaN(cos) || math.IsInf(cos, 0) {
		t.Fatalf("layernorm_quant: cosine is %v, not finite", cos)
	}
	if cos < 0.9999 {
		t.Errorf("layernorm_quant: cosine %.7f < 0.9999", cos)
	}
	t.Logf("layernorm_quant: cosine %.7f", cos)
}
