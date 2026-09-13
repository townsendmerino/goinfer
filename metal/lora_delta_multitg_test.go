//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestLoRADelta_multiThreadgroupMatchesReference gates M-08 (audit-metal-2026-09-12.md): the
// widened lora_delta grid — one threadgroup per 256-row block of Out, each independently
// recomputing t[R] — against a direct scalar reference of the LoRA formula itself
// (y[o] += scale * sum_r B[o,r] * (sum_k A[r,k] * dequant(aq,asc)[k])), computed without going
// through the kernel at all. Out=600 forces ceil(600/256)=3 threadgroups (tgs=256, total=768,
// with the last threadgroup's tail 232 rows masked by the kernel's own `row < Out` check) — the
// existing whole-model resident-vs-CPU LoRA parity test (lora_resident_parity_test.go) never
// exercises this: its fixture's widest projection is 128, one threadgroup either way, so it
// would pass identically whether this widened grid were wired correctly or not at all.
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

	// Kernel run: A/B as f16 (M-08).
	aHalf, bHalf := make([]uint16, len(A)), make([]uint16, len(B))
	parallelF32ToF16(aHalf, A)
	parallelF32ToF16(bHalf, B)

	// Reference: plain scalar Go, no kernel involved — but through the SAME f16-rounded A/B the
	// kernel actually reads, so this isolates the grid-assignment question (M-08's real risk: a
	// wrong/missed/duplicated row at a threadgroup boundary) from f16's accepted precision trade.
	want := make([]float32, Out)
	for i := range want {
		want[i] = float32(i) * 0.001 // pre-existing base-projection content the delta adds onto
	}
	x := make([]float32, K)
	for k := range x {
		x[k] = float32(aq[k]) * asc
	}
	tRef := make([]float32, R)
	for r := 0; r < R; r++ {
		var s float32
		for k := 0; k < K; k++ {
			s += f16ToF32(aHalf[r*K+k]) * x[k]
		}
		tRef[r] = s
	}
	for o := 0; o < Out; o++ {
		var acc float32
		for r := 0; r < R; r++ {
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
	// The reference reads the SAME f16-rounded A/B the kernel does (see above), so this is pure
	// f32 accumulation-order noise (SIMD tree reduction vs the reference's linear sum) — a tight
	// bound is the right check: a wrong/missed/duplicated row at a threadgroup boundary would
	// show up as an error many orders larger than float rounding, not a marginal miss.
	if maxAbs > 1e-4 {
		t.Fatalf("multi-threadgroup lora_delta diverges from the scalar reference: maxAbs=%.6f (want <= 1e-4) "+
			"— a wrong row assignment or missed/duplicated row would show as a large, structured error here", maxAbs)
	}
}
