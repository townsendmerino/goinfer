//go:build cuda

package cuda

import (
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestPackWeightStack is the gate for stacked-expert packing.
//
// MoE selects an expert by INDEXING a row range in one shared buffer
// (gemv_w4a8_moe: wrow = idx[slot]*rowsPerExpert + row), so the packed bytes of expert e must
// sit at EXACTLY e*rowsPerExpert and be byte-for-byte what the dense packer would have
// produced for that weight alone. If the stacked layout drifts from the dense one — a
// different nibble permutation, a different group-scale rounding, an off-by-one row stride —
// the GEMV cannot tell: it reads whatever is at that offset and returns a plausible wrong
// number. There is no crash and no obviously-bad output to notice.
//
// So this asserts identity against packWeight itself rather than re-deriving the expected
// bytes, which would just be a second copy of the layout that could agree while both are wrong.
func TestPackWeightStack(t *testing.T) {
	const nE, N, K, group = 4, 8, 64, 32

	mk := func(seed uint32) linalg.WeightMat {
		f := make([]float32, N*K)
		s := seed
		for i := range f {
			s = s*1664525 + 1013904223
			f[i] = float32(int32(s>>8)%2000-1000) / 1000
		}
		return linalg.QuantizeInt4(f, N, K, group)
	}
	mats := make([]linalg.WeightMat, nE)
	ptrs := make([]*linalg.WeightMat, nE)
	for e := range nE {
		mats[e] = mk(uint32(1000 + e*7))
		ptrs[e] = &mats[e]
	}

	stacked, err := packWeightStack(ptrs...)
	if err != nil {
		t.Fatalf("packWeightStack: %v", err)
	}
	if stacked.N != nE*N {
		t.Fatalf("stacked N = %d, want %d (nE*N) — the row count the kernel strides by is wrong", stacked.N, nE*N)
	}
	if stacked.K != K {
		t.Fatalf("stacked K = %d, want %d", stacked.K, K)
	}

	kw, kg := K/8, K/group
	for e := range nE {
		alone, err := packWeight(ptrs[e])
		if err != nil {
			t.Fatalf("packWeight[%d]: %v", e, err)
		}
		// Expert e's packed words must start exactly where the kernel will look for them.
		off := e * N * kw
		for i, w := range alone.wpk {
			if stacked.wpk[off+i] != w {
				t.Fatalf("expert %d word %d: stacked has %#08x at offset %d, packed-alone has %#08x — the "+
					"stacked layout DIFFERS from the dense packer, so gemv_w4a8_moe's indexed read "+
					"(idx*rowsPerExpert + row) lands on the wrong bytes and returns a plausible wrong "+
					"number with no error", e, i, stacked.wpk[off+i], off+i, w)
			}
		}
		soff := e * N * kg
		for i, s := range alone.ws16 {
			if stacked.ws16[soff+i] != s {
				t.Fatalf("expert %d group-scale %d: stacked %#04x != packed-alone %#04x at offset %d",
					e, i, stacked.ws16[soff+i], s, soff+i)
			}
		}
	}
	t.Logf("%d experts row-stacked: %d words + %d f16 scales; every expert byte-identical to the "+
		"dense packer at offset e*%d", nE, len(stacked.wpk), len(stacked.ws16), N*kw)
}

// TestPackWeightStack_rejectsRagged: the kernel strides by ONE Kwords/Kgroups, so a stack whose
// members disagree on K would read across row boundaries — silently, since every offset is
// still inside the allocation. Reject it at pack time, where the error can name the cause.
func TestPackWeightStack_rejectsRagged(t *testing.T) {
	f1 := make([]float32, 8*64)
	f2 := make([]float32, 8*32)
	a := linalg.QuantizeInt4(f1, 8, 64, 32)
	b := linalg.QuantizeInt4(f2, 8, 32, 32)
	if _, err := packWeightStack(&a, &b); err == nil {
		t.Error("packWeightStack accepted a ragged stack (K=64 and K=32) — the kernel would stride " +
			"one row's Kwords across another's data and return wrong numbers with no error")
	}
	if _, err := packWeightStack(); err == nil {
		t.Error("packWeightStack accepted an empty stack")
	}
}
