package decoder

import (
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// BenchmarkAttnDistinctBytes is R13 (docs/tasks/red-october.md) step 0(iii): "the CPU's version of
// the Metal collapse probe and CUDA's ncu traffic ratio" — it asks whether the real GQA layout
// (nKV distinct KV heads, each read by nH/nKV query heads) is faster than an otherwise-identical
// MHA-EXPANDED layout (nH distinct KV heads, one per query head, same per-head QKᵀ/scores·V work)
// PURELY because of the byte-count difference, or whether the CPU's cache already dedups the
// group's repeated reads so the two run at the same speed. Neither arm computes anything
// meaningful (synthetic random data, no softmax, no correctness claim) — this is a memory-access-
// pattern probe, not a model or a golden.
//
// Reading the result: GQA MARKEDLY faster than expanded ⇒ the hardware is not deduping, every
// query head pays real DRAM/cache traffic for its group's reads, and an explicit K/V-staging
// kernel (reading each KV head's row once, sharing it across its group in registers/threadgroup
// memory) recovers real bandwidth, not just µops — "the top of the band is live" in the brief's own
// words. GQA and expanded running the SAME speed ⇒ the cache already dedups the repeated reads in
// hardware (they're hot from the immediately-prior query head's pass), and a grouped kernel's whole
// gain is the µop sharing R13's own Build section already banks on — no additional bandwidth win to
// expect.
//
// Real head shapes (NumHeads/NumKVHeads/HeadDim) come from loadBenchModel()'s own config so the
// group size G = NumHeads/NumKVHeads matches what the served kernels actually see; the K/V data
// itself is synthetic (rand, seeded) since only the ACCESS PATTERN is under test.
func BenchmarkAttnDistinctBytes(b *testing.B) {
	m, err := loadBenchModel()
	if err != nil {
		b.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	arch := m.w.arch
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	if nH == 0 || nKV == 0 || hd == 0 {
		b.Skipf("model reports zero-valued head shape (NumHeads=%d NumKVHeads=%d HeadDim=%d)", nH, nKV, hd)
	}
	group := nH / nKV

	for _, nKeys := range []int{128, 512, 2048, 3900} {
		b.Run(depthLabel(nKeys), func(b *testing.B) {
			b.Run("gqa_real", func(b *testing.B) {
				benchAttnDistinctBytesArm(b, nH, nKV, hd, group, nKeys, true)
			})
			b.Run("mha_expanded", func(b *testing.B) {
				benchAttnDistinctBytesArm(b, nH, nKV, hd, group, nKeys, false)
			})
		})
	}
}

// benchAttnDistinctBytesArm times nH heads' worth of QKᵀ + scores·V (b.N times) against a
// keys/vals buffer laid out either the real GQA way (row width nKV*hd, group heads share bytes) or
// MHA-expanded (row width nH*hd, every head reads its own distinct bytes) — same nH calls, same
// per-call (M,K,N) shape, same hd, either way; only the addressing (bOff/rowStride) and the buffer
// width differ.
func benchAttnDistinctBytesArm(b *testing.B, nH, nKV, hd, group, nKeys int, gqa bool) {
	q := randF32(nH*hd, 1)
	scores := make([]float32, nKeys)
	ctx := make([]float32, nH*hd)
	acc := make([]float64, hd)

	rowWidth := nKV * hd
	if !gqa {
		rowWidth = nH * hd
	}
	keys := randF32(nKeys*rowWidth, 2)
	vals := randF32(nKeys*rowWidth, 3)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for head := range nH {
			var off int
			if gqa {
				off = (head / group) * hd // real: heads sharing a KV head read the SAME bytes
			} else {
				off = head * hd // expanded: every head reads its OWN distinct bytes
			}
			qh := q[head*hd : (head+1)*hd]
			ch := ctx[head*hd : (head+1)*hd]
			linalg.MatmulQKAcc64(qh, keys, scores, 1, hd, nKeys, off, rowWidth)
			linalg.MatmulAVAcc64(scores, vals, ch, acc, 1, nKeys, hd, off, rowWidth)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(b.N*nH)/b.Elapsed().Seconds(), "heads/s")
}

func depthLabel(nKeys int) string {
	switch nKeys {
	case 128:
		return "K128"
	case 512:
		return "K512"
	case 2048:
		return "K2048"
	case 3900:
		return "K3900"
	default:
		return "K"
	}
}
