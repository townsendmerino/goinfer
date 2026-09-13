//go:build darwin

package metal

import (
	"context"
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillLast_startPosGreaterThanZero gates G-08 (audit-metal-2026-09-12.md): the §3.2 pooled
// gate (prefill_gate_ref_test.go) only ever calls PrefillLast(embs, 0) — every resident-prefix-
// reuse turn (decoder/model.go's residentPrefillSeed, `from` — an agent loop continuing from an
// already-resident prefix, the peer matrix's own headline workload) calls it with startPos > 0,
// and the fused kernel's startPos/uMReal masking (attention_prefill_fused's nKeysMax computation)
// has no coverage at that shape outside one synthetic hd=64 unit case.
//
// A focused correctness check on the tiny synthetic fixture, not a change to the pooled gate's
// own carefully pre-registered statistics (decisionKs/confirmKs, the critA/B/C formulas) — G-08's
// own confidence is "plausible, coverage gap, no defect shown", and this closes the gap without
// risking the established methodology those formulas represent. Builds the SAME shared KV prefix
// [0,from) on two residents via Forward (bit-identical by construction — same sequential path),
// then diverges: one continues the reference way (Forward, one token at a time) through [from,K);
// the other takes the SAME suffix through PrefillLast(embs[from:], from) — the exact code path
// G-08 flags as uncovered. Compares the two residents' final logits at position K-1.
func TestPrefillLast_startPosGreaterThanZero(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	w := genTinyWeights(rand.New(rand.NewSource(808)))
	dir := t.TempDir()
	writeDense(t, dir, w)

	const K, from = 48, 24 // from > 0, and from < K so PrefillLast actually has work to do
	embs := make([][]float32, K)
	for i := range embs {
		embs[i] = make([]float32, tmHidden)
		for j := range embs[i] {
			embs[i][j] = float32(i*7+j) * 0.01
		}
	}

	load := func() *metalResident {
		m, err := decoder.Load(dir, decoder.Options{Quant: "int8int8"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		r, err := buildResident(m)
		if err != nil {
			t.Fatalf("build resident: %v", err)
		}
		return &metalResident{r: r, hidden: r.H}
	}

	// Reference: sequential Forward for the whole prompt [0,K).
	ref := load()
	var refLogits []float32
	for i, e := range embs {
		l, err := ref.Forward(e, i)
		if err != nil {
			t.Fatalf("ref Forward(%d): %v", i, err)
		}
		refLogits = l
	}

	// Fast arm: build the SAME [0,from) prefix via Forward (shared with the reference by
	// construction), then continue via PrefillLast(embs[from:], from) — startPos > 0.
	fast := load()
	for i := 0; i < from; i++ {
		if _, err := fast.Forward(embs[i], i); err != nil {
			t.Fatalf("fast prefix Forward(%d): %v", i, err)
		}
	}
	t.Setenv("GOINFER_METAL_FAST_PREFILL_FLOOR", "0") // K-from=24 is far below any real floor
	fastLogits, err := fast.PrefillLast(context.Background(), embs[from:], from)
	if err != nil {
		t.Fatalf("PrefillLast(startPos=%d): %v", from, err)
	}

	refArg, fastArg := argmaxF(refLogits), argmaxF(fastLogits)
	cos := cosF(refLogits, fastLogits)
	t.Logf("startPos=%d (K=%d): argmax ref=%d fast=%d, cosine=%.6f", from, K, refArg, fastArg, cos)
	if math.IsNaN(float64(cos)) || math.IsInf(float64(cos), 0) {
		t.Fatalf("PrefillLast(startPos=%d) FAIL: cosine is %v — degenerate (NaN/Inf) logits", from, cos)
	}
	if refArg != fastArg {
		t.Fatalf("PrefillLast(startPos=%d) FAIL: argmax %d != sequential-reference %d (cosine %.4f)", from, fastArg, refArg, cos)
	}
	if cos < 0.95 {
		t.Fatalf("PrefillLast(startPos=%d) FAIL: cosine %.4f too low (bug?)", from, cos)
	}
}
