package decoder

import (
	"math"
	"testing"
)

// TestDequantHeads_bitIdenticalToAikit is the gate that licensed deleting dequantHeads' arithmetic
// in favour of linalg.DequantizeRowsInt8Into (aikit docs/task-goinfer-kernel-moves.md, M5).
//
// It stays after the swap on purpose. dequantHeads now *is* the aikit call, so today this compares
// aikit against a frozen copy of the body goinfer used to own — which is exactly what makes it a
// regression gate rather than a tautology: if a future aikit bump changes DequantizeRowsInt8Into's
// bits, this fails here, in goinfer, where the KV cache contract lives, instead of surfacing as a
// moved golden somewhere downstream.
//
// RAW BITS, never a tolerance: the difference this class of change produces is signed zero and
// 1-ULP, both of which a tolerance test cannot see (measured in M5 — fakequant's sym branch differs
// from aikit's int4 round trip in exactly 7/256 elements, every one of them -0 vs +0).
func TestDequantHeads_bitIdenticalToAikit(t *testing.T) {
	// The frozen reference: goinfer's body as it stood before the swap.
	ref := func(q []int8, scales []float32, nKV, headDim int, dst []float32) {
		for h := range nKV {
			o, s := h*headDim, scales[h]
			for c := range headDim {
				dst[o+c] = float32(q[o+c]) * s
			}
		}
	}
	// Adversarial scales: max/min normal, denormal, signed zero, Inf, NaN — every one reachable by
	// a caller, since the scale is data-derived (a row of all-zero KV yields 0).
	scaleVals := []float32{
		1, 0.5, -0.5, 3.4028235e38, 1.1754944e-38, 1e-45,
		float32(math.Inf(1)), float32(math.NaN()), 0, -0, 7.0e-3, -2.7182817,
	}
	// Real head dims (64/128/256) plus tails that exercise the odd-length paths.
	shapes := [][2]int{{1, 1}, {2, 64}, {2, 128}, {4, 256}, {8, 7}, {3, 33}, {5, 1}}
	for _, sh := range shapes {
		nKV, headDim := sh[0], sh[1]
		n := nKV * headDim
		q := make([]int8, n)
		for i := range q {
			q[i] = int8(((i * 37) % 256) - 128) // full int8 range, including -128
		}
		for si := range scaleVals {
			scales := make([]float32, nKV)
			for h := range scales {
				scales[h] = scaleVals[(si+h)%len(scaleVals)]
			}
			want := make([]float32, n)
			got := make([]float32, n)
			ref(q, scales, nKV, headDim, want)
			dequantHeads(q, scales, nKV, headDim, got)
			for i := range want {
				if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
					t.Fatalf("%dx%d si=%d idx=%d: frozen ref %08x != aikit %08x",
						nKV, headDim, si, i, math.Float32bits(want[i]), math.Float32bits(got[i]))
				}
			}
		}
	}
}
