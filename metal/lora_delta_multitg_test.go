//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestLoRADelta_multiThreadgroupMatchesReference pins the widened lora_delta grid (one threadgroup per 256-row block of Out,
// each recomputing t[R]) against a direct scalar reference of the LoRA formula, y[o] += scale * sum_r B[o,r] * (sum_k A[r,k] *
// dequant(aq,asc)[k]), computed without the kernel. Out=600 forces ceil(600/256)=3 threadgroups (tgs=256, total=768, the last
// one's 232-row tail masked by the kernel's own `row < Out`). The whole-model parity test (lora_resident_parity_test.go) cannot
// see this: its widest projection is 128, one threadgroup either way, so it passes whether the widened grid is wired or not
// (audit M-08, docs/audit-metal-2026-09-12.md).
func TestLoRADelta_multiThreadgroupMatchesReference(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	p, err := d.NewComputePipeline(lib, "lora_delta")
	if err != nil {
		t.Fatalf("pipeline lora_delta: %v", err)
	}

	const K, R, Out = 64, 12, 600 // Out > 256: forces 3 threadgroups, last one ragged
	rng := rand.New(rand.NewSource(2026))

	aq := make([]int8, K)
	for i := range aq {
		aq[i] = int8(rng.Intn(255) - 127)
	}
	const asc = 0.037
	A := make([]float32, R*K)
	B := make([]float32, Out*R)
	for i := range A {
		A[i] = rng.Float32()*2 - 1
	}
	for i := range B {
		B[i] = rng.Float32()*2 - 1
	}
	const scale = 0.5

	// Kernel run: A/B as f16.
	aHalf, bHalf := make([]uint16, len(A)), make([]uint16, len(B))
	parallelF32ToF16(aHalf, A)
	parallelF32ToF16(bHalf, B)

	// Reference: plain scalar Go over the same f16-rounded A/B the kernel reads, so it isolates the grid-assignment question (a
	// wrong, missed or duplicated row at a threadgroup boundary) from f16's accepted precision trade.
	want := make([]float32, Out)
	for i := range want {
		want[i] = float32(i) * 0.001 // pre-existing base-projection content the delta adds onto
	}
	x := make([]float32, K)
	for k := range x {
		x[k] = float32(aq[k]) * asc
	}
	tRef := make([]float32, R)
	for r := range R {
		var s float32
		for k := range K {
			s += f16ToF32(aHalf[r*K+k]) * x[k]
		}
		tRef[r] = s
	}
	for o := range Out {
		var acc float32
		for r := range R {
			acc += f16ToF32(bHalf[o*R+r]) * tRef[r]
		}
		want[o] += scale * acc
	}
	out := make([]float32, Out)
	for i := range out {
		out[i] = float32(i) * 0.001
	}

	dAq := NewBufferInt8(d, aq)
	dAsc := NewBufferFloats(d, []float32{asc})
	dA := NewBufferU16s(d, aHalf)
	dB := NewBufferU16s(d, bHalf)
	dOut := NewBufferFloats(d, out)
	uK, uR, uOut := NewBufferU32(d, uint32(K)), NewBufferU32(d, uint32(R)), NewBufferU32(d, uint32(Out))
	uScale := NewBufferFloats(d, []float32{scale})

	q := d.NewCommandQueue()
	e := q.Begin()
	total := (Out + tgReduceNorm - 1) / tgReduceNorm * tgReduceNorm
	e.Dispatch(p, total, tgReduceNorm, dAq, dAsc, dA, dB, dOut, uK, uR, uOut, uScale)
	e.End()
	if err := e.Err(); err != nil {
		t.Fatalf("command buffer: %v", err)
	}

	got := dOut.Floats()
	var maxAbs float64
	for i := range want {
		d := math.Abs(float64(got[i]) - float64(want[i]))
		if d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("Out=%d (3 threadgroups, last ragged): maxAbs=%.6f", Out, maxAbs)
	// The reference reads the same f16-rounded A/B, so the residual is f32 accumulation-order noise (SIMD tree reduction vs a
	// linear sum). A wrong, missed or duplicated row at a threadgroup boundary would be orders larger, hence the tight bound.
	if maxAbs > 1e-4 {
		t.Fatalf("multi-threadgroup lora_delta diverges from the scalar reference: maxAbs=%.6f (want <= 1e-4) "+
			"— a wrong row assignment or missed/duplicated row would show as a large, structured error here", maxAbs)
	}
}
