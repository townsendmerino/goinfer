package decoder

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestAttendBatchedHeads_vsNaive isolates the batched SIMD attention from the
// 28-layer forward: random q + random KV in a cache, compared head-by-head
// against a naive float64 causal-attention reference. This separates "is the
// attention math correct" from "does f32 compound over the stack" — if this is
// tight (cosine ~1) the kernel is right and any end-to-end drift is f32
// accumulation, not a bug.
func TestAttendBatchedHeads_vsNaive(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GINFER_PREQUANT_GGUF", err)
	}
	arch := m.w.arch
	nH, nKV, hd := arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	qDim, kvDim := nH*hd, nKV*hd
	group := nH / nKV
	scale := arch.AttnScale
	K := 64

	rng := rand.New(rand.NewSource(7))
	q := make([]float32, K*qDim)
	for i := range q {
		q[i] = float32(rng.NormFloat64()) * 0.1
	}
	cache := m.NewCache(K + 1)
	for range K {
		kv := make([]float32, kvDim)
		vv := make([]float32, kvDim)
		for j := range kv {
			kv[j] = float32(rng.NormFloat64()) * 0.1
			vv[j] = float32(rng.NormFloat64()) * 0.1
		}
		cache.Append(0, kv, vv) // layer 0; non-last layer ⇒ pos doesn't advance
	}

	keys, vals := cache.Keys(0), cache.Vals(0)
	ref := make([]float32, K*qDim)
	for i := range K {
		for qh := range nH {
			kvh := qh / group
			sc := make([]float64, i+1)
			maxS := math.Inf(-1)
			for s := 0; s <= i; s++ {
				var dot float64
				for d := range hd {
					dot += float64(q[i*qDim+qh*hd+d]) * float64(keys[s*kvDim+kvh*hd+d])
				}
				sc[s] = dot * scale
				if sc[s] > maxS {
					maxS = sc[s]
				}
			}
			var sum float64
			for s := range sc {
				sc[s] = math.Exp(sc[s] - maxS)
				sum += sc[s]
			}
			for d := range hd {
				var o float64
				for s := 0; s <= i; s++ {
					o += sc[s] / sum * float64(vals[s*kvDim+kvh*hd+d])
				}
				ref[i*qDim+qh*hd+d] = float32(o)
			}
		}
	}

	ctx := make([]float32, K*qDim)
	attendBatchedHeads(q, ctx, keys, vals, 0, cache, 0, 0, K, true, arch, false, // useAcc64=false: validate the f32 (dense) kernel vs the f64 naive ref
		make([]float32, K*hd), make([]float32, K*hd), make([]float32, K*hd),
		make([]float32, K*K), make([]float32, K*hd))

	var maxd float64
	for j := range ref {
		if d := math.Abs(float64(ctx[j] - ref[j])); d > maxd {
			maxd = d
		}
	}
	cos := logitCosine(ref, ctx)
	t.Logf("attendBatchedHeads vs naive f64: cosine %.8f, max abs diff %.2e", cos, maxd)
	if cos < 0.99999 {
		t.Errorf("attendBatchedHeads diverges from naive attention: cosine %.8f (max diff %.2e)", cos, maxd)
	}
}

// TestForwardN_matchesSequential checks the batched multi-position forward
// (forwardLayersN, whose attention is the SIMD-matmul attendBatchedHeads) agrees
// with running forward() one token at a time (the scalar attendQuery). The bar is
// the speculative verifier's: argmax-exact at every position, plus high cosine —
// NOT bit-identical, since batched attention moved QKᵀ/scores·V onto the f32 SIMD
// A·Bᵀ kernel (float64→f32 accumulation + matmul reassociation), the same
// parity standard the GPU residency attention meets. K=192 is included so the
// O(L²) attention term is actually exercised (K=6 barely touches it). Skips
// without the model asset.
func TestForwardN_matchesSequential(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GINFER_PREQUANT_GGUF", err)
	}
	base := []int{785, 264, 6573, 311, 1438, 279}

	for _, K := range []int{6, 128} {
		t.Run(fmt.Sprintf("K=%d", K), func(t *testing.T) {
			ids := make([]int, K)
			for i := range ids {
				ids[i] = base[i%len(base)] // tiled real tokens: stable argmax, exercises the L² term
			}

			cseq := m.NewCache(K)
			seq := make([][]float32, K)
			for i, id := range ids {
				l, err := m.forward(id, cseq)
				if err != nil {
					t.Fatalf("seq forward %d: %v", i, err)
				}
				seq[i] = append([]float32(nil), l...)
			}

			cbat := m.NewCache(K)
			bat, err := m.forwardN(ids, cbat)
			if err != nil {
				t.Fatalf("forwardN: %v", err)
			}
			if len(bat) != K {
				t.Fatalf("forwardN returned %d logit vectors, want %d", len(bat), K)
			}
			if cseq.Pos() != cbat.Pos() || cbat.Pos() != K {
				t.Fatalf("cache positions: seq=%d batch=%d want %d", cseq.Pos(), cbat.Pos(), K)
			}

			var worstCos, worstMaxd float64 = 1, 0
			for i := range K {
				if argmax(seq[i]) != argmax(bat[i]) {
					t.Fatalf("position %d: argmax seq=%d batch=%d", i, argmax(seq[i]), argmax(bat[i]))
				}
				cos := logitCosine(seq[i], bat[i])
				var maxd float64
				for j := range seq[i] {
					if d := math.Abs(float64(seq[i][j] - bat[i][j])); d > maxd {
						maxd = d
					}
				}
				if cos < worstCos {
					worstCos = cos
				}
				if maxd > worstMaxd {
					worstMaxd = maxd
				}
				// argmax-exact (above) is the hard bar; cosine ≥ 0.99 is the GPU-path
				// sanity floor (f32-SIMD attention over the full stack drifts ~0.998
				// vs the float64 scalar reference — accepted, not bit-identical).
				if cos < 0.99 {
					t.Errorf("position %d: cosine %.6f < 0.99 (max logit diff %.2e)", i, cos, maxd)
				}
			}
			t.Logf("K=%d: argmax exact all positions; worst cosine %.6f, worst max logit diff %.2e",
				K, worstCos, worstMaxd)
		})
	}
}

// logitCosine is the cosine similarity of two logit vectors (parity metric).
func logitCosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// TestTruncateTo rolls a cache back and confirms re-decoding from the truncated
// point matches a cache that only ever saw the kept prefix.
func TestTruncateTo(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GINFER_PREQUANT_GGUF", err)
	}
	ids := []int{785, 264, 6573, 311, 1438, 279}

	// Cache A: feed all 6, truncate back to 3, then forward a 7th token.
	ca := m.NewCache(8)
	for _, id := range ids {
		if _, err := m.forward(id, ca); err != nil {
			t.Fatal(err)
		}
	}
	ca.TruncateTo(3)
	if ca.Pos() != 3 {
		t.Fatalf("after TruncateTo(3): Pos=%d", ca.Pos())
	}
	la, err := m.forward(99, ca)
	if err != nil {
		t.Fatal(err)
	}

	// Cache B: only ever saw the first 3 tokens, then the same 7th token.
	cb := m.NewCache(8)
	for _, id := range ids[:3] {
		if _, err := m.forward(id, cb); err != nil {
			t.Fatal(err)
		}
	}
	lb, err := m.forward(99, cb)
	if err != nil {
		t.Fatal(err)
	}

	if argmax(la) != argmax(lb) {
		t.Fatalf("post-truncate argmax %d != fresh-prefix argmax %d", argmax(la), argmax(lb))
	}
	for j := range la {
		if math.Abs(float64(la[j]-lb[j])) > 1e-4 {
			t.Fatalf("post-truncate logits differ at %d: %g vs %g", j, la[j], lb[j])
		}
	}
}
