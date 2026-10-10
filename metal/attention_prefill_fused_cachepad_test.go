//go:build darwin

package metal

import (
	"context"
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestAttentionPrefillFused_ctxCapNotMultipleOf8 pins that the resident kc/vc are allocated rounded up to a multiple of 8 rows
// while r.ctxCap (the checked, user-visible capacity) stays as requested: attention_prefill_fused's key loop reads whole 8-row
// simdgroup tiles and masks the ragged remainder after the load, so with a ctxCap that is not a multiple of 8 the last tile reads
// past the allocation when nKeysMax reaches the cap (buildResident, metal/model.go).
//
// It asserts the allocated byte length (Buffer.Len(), the logical size passed to the allocator), not the out-of-bounds read's
// effect on output: Metal page-rounds the physical backing (16 KB on Apple silicon), so a 37-row and a 40-row buffer this small
// share one allocation and the read is in bounds either way; a version that compared PrefillLast logits at the boundary against
// sequential Forward passed with the fix reverted. The PrefillLast run below is a smoke check (finite logits), not a gate.
func TestAttentionPrefillFused_ctxCapNotMultipleOf8(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	w := genTinyWeights(rand.New(rand.NewSource(37)))
	dir := t.TempDir()
	writeDense(t, dir, w)

	const ctxCap = 37                                 // NOT a multiple of 8
	t.Setenv("GOINFER_METAL_FAST_PREFILL_FLOOR", "0") // ctxCap=37 is far below any real floor; read at Load
	m, err := decoder.Load(dir, decoder.Options{Quant: "int8int8", ResidentContext: ctxCap})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("build resident: %v", err)
	}
	if r.ctxCap != ctxCap {
		t.Fatalf("ctxCap = %d, want %d — ResidentContext request was not honored", r.ctxCap, ctxCap)
	}

	wantRows := (ctxCap + 7) / 8 * 8 // 40
	kvDim := tmKVHeads * tmHeadDim
	wantBytes := wantRows * kvDim * 2 // f16 KV: 2 bytes/elem
	if got := r.kc[0].Len(); got != wantBytes {
		t.Fatalf("kc[0] allocated %d bytes, want %d (ctxCap=%d rounded up to %d rows * kvDim=%d * 2 "+
			"bytes) — the C-01 padding did not take, so the fused kernel's ragged last tile can read "+
			"past this buffer's end", got, wantBytes, ctxCap, wantRows, kvDim)
	}
	if got := r.vc[0].Len(); got != wantBytes {
		t.Fatalf("vc[0] allocated %d bytes, want %d (same padding as kc[0])", got, wantBytes)
	}

	// Functional smoke test alongside the structural one above: PrefillLast right up to the
	// ctxCap boundary must still run cleanly (not a gate on its own — see the doc comment).
	rf := &metalResident{r: r, hidden: r.H}
	embs := make([][]float32, ctxCap)
	for i := range embs {
		embs[i] = make([]float32, tmHidden)
		for j := range embs[i] {
			embs[i][j] = float32(i*7+j) * 0.01
		}
	}
	pre, err := rf.PrefillLast(context.Background(), embs, 0)
	if err != nil {
		t.Fatalf("PrefillLast at the ctxCap boundary: %v", err)
	}
	for _, v := range pre {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("PrefillLast produced non-finite logits at the ctxCap=%d boundary", ctxCap)
		}
	}
}
