package decoder

import (
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// BenchmarkAttnDistinctBytes asks whether the real GQA layout (nKV distinct KV heads, each read by nH/nKV query
// heads) is faster than an otherwise-identical MHA-EXPANDED layout (nH distinct KV heads, one per query head, same
// per-head QKᵀ/scores·V work) purely because of the byte-count difference, or whether the CPU's cache already
// dedups the group's repeated reads. It is a memory-access-pattern probe: synthetic seeded data, no softmax, no
// correctness claim, not a model or a golden.
//
// Reading: GQA markedly faster than expanded means the hardware does not dedup, and an explicit K/V-staging kernel
// recovers real bandwidth. The same speed means the cache already dedups, and a grouped kernel's gain is the µop
// sharing alone (docs/tasks/red-october.md, R13). Head shapes come from loadBenchModel()'s config so the group
// size G = NumHeads/NumKVHeads matches what the served kernels see.
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
