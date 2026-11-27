package decoder

import "testing"

// TestSnapshot_refusesNonUniformKVWidth_C05 gates C-05's residual hole: a cache whose global
// (append-forever) layers have differing KV widths — gemma-4's per-layer geometry — must be
// refused by Snapshot, not round-tripped. The format records only the uniform kvDim for global
// layers, so a restored session would set every stride to kvDim and mis-slice the odd-width layers
// on the first TruncateTo. Refusal (nil) makes the caller cold-prefill instead.
func TestSnapshot_refusesNonUniformKVWidth_C05(t *testing.T) {
	// Uniform-width cache: snapshot succeeds (control).
	uni := NewKVCache(2, 4, 16, 0, 4, nil) // kvDim = 64
	uni.c.Stride[0], uni.c.Stride[1] = 64, 64
	uni.c.PosN = 1
	uni.c.KeysBuf[0] = make([]float32, 64)
	uni.c.ValsBuf[0] = make([]float32, 64)
	uni.c.KeysBuf[1] = make([]float32, 64)
	uni.c.ValsBuf[1] = make([]float32, 64)
	if b := (&Session{cache: uni, tokens: []int{7}}).Snapshot("s"); b == nil {
		t.Fatal("uniform-width cache should snapshot, got refusal")
	}

	// Non-uniform: layer 1 has a wider KV stride (gemma-4 style) → must refuse.
	non := NewKVCache(2, 4, 16, 0, 4, nil) // kvDim = 64
	non.c.Stride[0], non.c.Stride[1] = 64, 128
	non.c.PosN = 1
	if b := (&Session{cache: non, tokens: []int{7}}).Snapshot("s"); b != nil {
		t.Error("non-uniform per-layer KV width (gemma-4) must be refused by Snapshot (C-05), got a blob")
	}
}
