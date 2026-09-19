//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestNib2HalfAllValues is gate (1)'s first half: the nib2half exponent-bias dequant trick
// (kernels.go, gemv_w4f16_* comment) must recover nibble-8 EXACTLY for all 16 nibble values —
// it is meant to be a bit-exact integer reconstruction, not an approximation, so this is a
// direct equality check, not a tolerance one. Runs the trick as its own tiny kernel so a bug in
// the bit manipulation is isolated from the GEMV's reduction/accumulation entirely.
func TestNib2HalfAllValues(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	// Reuses gemv_w4f16_sa itself at K=32 (one group), weight word carrying all 8 nibbles of
	// interest in .x with the other three words zeroed (nibble 8 = value 0, a neutral filler),
	// activation = 1.0 at every position, scale = 1.0 — so out[row] is exactly the sum of
	// nib2half(nibble)-8 for the 8 nibbles packed into that row's .x word. Isolate ONE nibble
	// per row by putting it in nibble slot 0 and 8 (the zero/neutral value) everywhere else.
	pipe, err := d.NewComputePipeline(lib, "gemv_w4f16_sa")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	const K = 32
	N := 16 // one row per nibble value 0..15
	words := make([]uint32, N*(K/8))
	for n := 0; n < N; n++ {
		base := n * (K / 8)
		// nibble slot 0 = n; slots 1..7 of THIS word, and every nibble of every other word in
		// the row, = 8 (neutral -> nib2half(8)-8 = 0) so only the one nibble under test
		// contributes to the row's sum.
		words[base+0] = 0x88888880 | uint32(n)
		for w := 1; w < K/8; w++ {
			words[base+w] = 0x88888888
		}
	}
	scales := make([]uint16, N*(K/32))
	for i := range scales {
		scales[i] = f32ToF16(1.0)
	}
	act := make([]uint16, K)
	for i := range act {
		act[i] = f32ToF16(1.0)
	}
	dWq := NewBufferUint32s(d, words)
	dSct := NewBufferU16s(d, scales)
	dAx := NewBufferU16s(d, act)
	dOut := d.NewBufferLen(N)
	uK := NewBufferU32(d, uint32(K))

	q_ := d.NewCommandQueue()
	e := q_.Begin()
	e.DispatchTG(pipe, N*32, 256, K*2, dWq, dSct, dAx, dOut, uK)
	e.End()

	got := dOut.Floats()
	for n := 0; n < N; n++ {
		want := float32(n - 8)
		if got[n] != want {
			t.Errorf("nib2half(%d): got %v want %v (raw nibble-8 not exact)", n, got[n], want)
		}
	}
}

// TestGemvW4F16_vsScalarReference is gate (1)'s second half: gemv_w4f16_sa against an
// INDEPENDENT scalar Go reference (not reusing any Metal-side code), at K=1536 (Qwen2.5-1.5B's
// QKV/gate-up width) and K=8960 (its intermediate width, the gate/up->down shape), N=64 output
// rows. The weight packing is packW4A8Row — the SAME packer gemv_w4a8_sa already uses and
// TestLayerB_gemvW4A8Parity already proved bit-exact — so this test isolates the ACTIVATION-side
// change (half, unquantized) and the dequant trick, not the packing.
func TestGemvW4F16_vsScalarReference(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "gemv_w4f16_sa")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	for _, K := range []int{1536, 8960} {
		K := K
		t.Run(shapeName(K), func(t *testing.T) {
			const N = 64
			rng := rand.New(rand.NewSource(int64(K) + 7))

			rows := make([][]float32, N)
			words := make([]uint32, 0, N*(K/8))
			scales := make([]float32, 0, N*(K/32))
			for n := 0; n < N; n++ {
				row := make([]float32, K)
				for k := range row {
					row[k] = (rng.Float32()*2 - 1) * 0.3 // weight-scale magnitude, mirrors real checkpoints
				}
				rows[n] = row
				w, s := packW4A8Row(row)
				words = append(words, w...)
				scales = append(scales, s...)
			}
			act := make([]float32, K)
			for k := range act {
				act[k] = rng.Float32()*4 - 2 // activation-scale magnitude
			}
			// Round-trip the activation through f16 explicitly BEFORE building the reference, so
			// the reference is scored against what the kernel actually receives (the f16-truncated
			// activation), not the f32 originals — the whole point of this lane is "no further
			// quantization past f16", not "identical to f32".
			actHalf := make([]uint16, K)
			actF16Truth := make([]float32, K)
			for k := range act {
				actHalf[k] = f32ToF16(act[k])
				actF16Truth[k] = f16ToF32(actHalf[k])
			}
			sctHalf := make([]uint16, len(scales))
			scF16Truth := make([]float32, len(scales))
			for i, s := range scales {
				sctHalf[i] = f32ToF16(s)
				scF16Truth[i] = f16ToF32(sctHalf[i])
			}

			// Independent scalar reference: unpack each nibble directly from the SAME packed
			// words (not from `rows`), so this checks the kernel's actual dequant against the
			// actual packed bits, exactly like UNP8V does on-device — just in plain Go.
			want := make([]float32, N)
			for n := 0; n < N; n++ {
				var acc float64 // f64 accumulation in the reference for a tighter ground truth
				wbase := n * (K / 8)
				for g := 0; g < K/32; g++ {
					sc := float64(scF16Truth[n*(K/32)+g])
					for kk := 0; kk < 32; kk++ {
						k := g*32 + kk
						word := words[wbase+k/8]
						nibble := (word >> (4 * uint(k%8))) & 0xF
						dq := float64(int(nibble) - 8)
						acc += dq * sc * float64(actF16Truth[k])
					}
				}
				want[n] = float32(acc)
			}

			dWq := NewBufferUint32s(d, words)
			dSct := NewBufferU16s(d, sctHalf)
			dAx := NewBufferU16s(d, actHalf)
			dOut := d.NewBufferLen(N)
			uK := NewBufferU32(d, uint32(K))

			q_ := d.NewCommandQueue()
			run := func() []float32 {
				e := q_.Begin()
				e.DispatchTG(pipe, N*32, 256, K*2, dWq, dSct, dAx, dOut, uK)
				e.End()
				return append([]float32(nil), dOut.Floats()...)
			}
			got := run()

			var maxAbs float64
			var dot, magA, magB float64
			for n := 0; n < N; n++ {
				d := math.Abs(float64(got[n] - want[n]))
				if d > maxAbs {
					maxAbs = d
				}
				dot += float64(got[n]) * float64(want[n])
				magA += float64(got[n]) * float64(got[n])
				magB += float64(want[n]) * float64(want[n])
			}
			cos := dot / (math.Sqrt(magA)*math.Sqrt(magB) + 1e-30)
			t.Logf("K=%d N=%d: maxAbs=%.6g cosine=%.8f", K, N, maxAbs, cos)
			// f32-accumulated dot of ~K terms each O(1) magnitude: expect agreement within a few
			// ULPs of f32, not a fidelity-gate-sized tolerance (this is a scalar-reference
			// correctness check, not the decode fidelity gate — that runs teacher-forced against
			// a CPU f32 model and is a separate, later gate).
			if cos < 0.999999 {
				t.Errorf("cosine %.8f below 0.999999 — dequant or accumulation bug suspected", cos)
			}
			if maxAbs > 1e-2 {
				t.Errorf("maxAbs %.6g too large for a K=%d, N=%d f32-accumulated dot", maxAbs, K, N)
			}

			// Fixed-input determinism (gate 1's third requirement): two runs, same input, must
			// match exactly — Metal's simd_sum reduction order is fixed per dispatch shape, so
			// this should be bit-identical, not just close.
			got2 := run()
			for n := 0; n < N; n++ {
				if got[n] != got2[n] {
					t.Errorf("non-deterministic at row %d: %v vs %v", n, got[n], got2[n])
				}
			}
		})
	}
}

// TestGemvW4F16_epilogues checks gemv_w4f16_sa_bias and gemv_w4f16_sa_resid against the same
// scalar-reference dot product TestGemvW4F16_vsScalarReference already validated for the base
// kernel, plus their own epilogue (+bias[row], or accumulate into an existing out buffer) —
// SA_F16_BODY is shared, so this isolates ONLY the epilogue line, not the dequant/reduction.
func TestGemvW4F16_epilogues(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	const K, N = 1536, 32
	rng := rand.New(rand.NewSource(97))

	words := make([]uint32, 0, N*(K/8))
	scales := make([]float32, 0, N*(K/32))
	for n := 0; n < N; n++ {
		row := make([]float32, K)
		for k := range row {
			row[k] = (rng.Float32()*2 - 1) * 0.3
		}
		w, s := packW4A8Row(row)
		words = append(words, w...)
		scales = append(scales, s...)
	}
	act := make([]float32, K)
	for k := range act {
		act[k] = rng.Float32()*4 - 2
	}
	actHalf := make([]uint16, K)
	actF16Truth := make([]float32, K)
	for k := range act {
		actHalf[k] = f32ToF16(act[k])
		actF16Truth[k] = f16ToF32(actHalf[k])
	}
	sctHalf := make([]uint16, len(scales))
	scF16Truth := make([]float32, len(scales))
	for i, s := range scales {
		sctHalf[i] = f32ToF16(s)
		scF16Truth[i] = f16ToF32(sctHalf[i])
	}
	dot := make([]float32, N)
	for n := 0; n < N; n++ {
		var acc float64
		wbase := n * (K / 8)
		for g := 0; g < K/32; g++ {
			sc := float64(scF16Truth[n*(K/32)+g])
			for kk := 0; kk < 32; kk++ {
				k := g*32 + kk
				word := words[wbase+k/8]
				nibble := (word >> (4 * uint(k%8))) & 0xF
				acc += float64(int(nibble)-8) * sc * float64(actF16Truth[k])
			}
		}
		dot[n] = float32(acc)
	}

	dWq := NewBufferUint32s(d, words)
	dSct := NewBufferU16s(d, sctHalf)
	dAx := NewBufferU16s(d, actHalf)
	uK := NewBufferU32(d, uint32(K))
	q_ := d.NewCommandQueue()

	check := func(name string, got, want []float32) {
		t.Helper()
		var maxAbs float64
		for n := range got {
			if d := math.Abs(float64(got[n] - want[n])); d > maxAbs {
				maxAbs = d
			}
		}
		t.Logf("%s: maxAbs=%.6g", name, maxAbs)
		if maxAbs > 1e-2 {
			t.Errorf("%s: maxAbs %.6g too large", name, maxAbs)
		}
	}

	t.Run("bias", func(t *testing.T) {
		pipe, err := d.NewComputePipeline(lib, "gemv_w4f16_sa_bias")
		if err != nil {
			t.Fatalf("pipeline: %v", err)
		}
		bias := make([]float32, N)
		for i := range bias {
			bias[i] = rng.Float32()*2 - 1
		}
		dBias := NewBufferFloats(d, bias)
		dOut := d.NewBufferLen(N)
		e := q_.Begin()
		e.DispatchTG(pipe, N*32, 256, K*2, dWq, dSct, dAx, dBias, dOut, uK)
		e.End()
		got := dOut.Floats()
		want := make([]float32, N)
		for n := range want {
			want[n] = dot[n] + bias[n]
		}
		check("bias", got, want)
	})

	t.Run("resid", func(t *testing.T) {
		pipe, err := d.NewComputePipeline(lib, "gemv_w4f16_sa_resid")
		if err != nil {
			t.Fatalf("pipeline: %v", err)
		}
		base := make([]float32, N)
		for i := range base {
			base[i] = rng.Float32()*2 - 1
		}
		dOut := NewBufferFloats(d, append([]float32(nil), base...))
		e := q_.Begin()
		e.DispatchTG(pipe, N*32, 256, K*2, dWq, dSct, dAx, dOut, uK)
		e.End()
		got := dOut.Floats()
		want := make([]float32, N)
		for n := range want {
			want[n] = base[n] + dot[n]
		}
		check("resid", got, want)
	})
}

func shapeName(k int) string {
	switch k {
	case 1536:
		return "K1536_qkv_gateup"
	case 8960:
		return "K8960_intermediate"
	default:
		return "K"
	}
}
