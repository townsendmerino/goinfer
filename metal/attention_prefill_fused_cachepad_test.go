//go:build darwin

package metal

import (
	"context"
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestAttentionPrefillFused_ctxCapNotMultipleOf8 gates C-01 (audit-metal-2026-09-12.md):
// attention_prefill_fused's key loop reads whole 8-row simdgroup tiles and masks the ragged
// remainder AFTER the load, so a resident cache built for a ctxCap that is NOT a multiple of 8
// can have its last, ragged tile read past kc/vc's actual allocated end when nKeysMax reaches
// that cap exactly. The fix (metal/model.go buildResident) rounds the kc/vc ALLOCATION up to a
// multiple of 8 rows while leaving r.ctxCap itself (the checked, user-visible capacity) alone.
//
// This asserts the allocated BYTE LENGTH directly (Buffer.Len(), the logical size passed to the
// allocator) rather than trying to observe the OOB read's effect on output: a first attempt at
// this test drove PrefillLast right up to the ctxCap=37 boundary and compared logits against the
// sequential-Forward reference, and it passed identically with the fix reverted — Metal's actual
// buffer backing is page-rounded (16 KB on Apple silicon) regardless of the requested length, so
// a 37-row and a 40-row request for a buffer this small land on the exact same physical
// allocation and the "OOB" read is silently in-bounds either way. Buffer.Len() reports the
// LOGICAL length the caller asked for, not the physical rounding, so it is the only reliable way
// to see this fix take effect — the same "prove the gate can go red" discipline this repo's other
// gates are held to (CLAUDE.md).
func TestAttentionPrefillFused_ctxCapNotMultipleOf8(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	w := genTinyWeights(rand.New(rand.NewSource(37)))
	dir := t.TempDir()
	writeDense(t, dir, w)

	const ctxCap = 37                                 // NOT a multiple of 8 — the exact shape C-01 describes
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
