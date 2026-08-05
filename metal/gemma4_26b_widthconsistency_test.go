//go:build darwin && goinfer_testhooks

package metal

import (
	"os"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemma4_26B_pagingWidthConsistency is Test 3: paged-Metal at N=32 vs N=64 on the 26B (nE=128).
// The router runs on-GPU BEFORE staging, so its selection is independent of N; if the paging mechanism
// is byte-transparent at real width, the two slot budgets must produce BYTE-IDENTICAL logits — the
// nE=128 analogue of the paged≡non-paged gate that was only vetted at nE=4. A DIFFERENCE would mean an
// eviction/staging bug at width (the serious outcome): the paging byte-identity gate would not hold at
// width, and the collapse would be staging, not routing sensitivity. Sequential builds (each fits; two
// N=64 pools at once would not). Heavy-gated.
func TestGemma4_26B_pagingWidthConsistency(t *testing.T) {
	requireHeavyModel(t)
	const giw = "/Users/francistownsend-merino/models/gemma4-26b-int4.giw"
	if _, err := os.Stat(giw); err != nil {
		t.Skipf("no .giw")
	}
	prompt := []int{1, 7, 42, 100}

	runAt := func(slots int) [][]float32 {
		t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
		t.Setenv("GOINFER_METAL_MOE_SLOTS", strconv.Itoa(slots))
		m, err := decoder.Load(giw, decoder.Options{Quant: "int4"})
		if err != nil {
			t.Fatalf("load N=%d: %v", slots, err)
		}
		defer m.Close()
		r, err := buildResident(m)
		if err != nil {
			t.Fatalf("BuildResident N=%d: %v", slots, err)
		}
		defer r.Close()
		if r.g4moe == nil || r.g4moe.slots != slots {
			t.Fatalf("expected paged N=%d, got slots=%v", slots, r.g4moe)
		}
		out := make([][]float32, len(prompt))
		for i, tok := range prompt {
			out[i] = append([]float32(nil), r.ForwardEmb(m.EmbedResidentForTest(tok), i)...)
		}
		return out
	}

	n32 := runAt(32)
	n64 := runAt(64)

	mism := 0
	for i := range prompt {
		for j := range n32[i] {
			if n32[i][j] != n64[i][j] {
				mism++
			}
		}
	}
	t.Logf("paged N=32 vs N=64 on the 26B (nE=128, %d positions): %d logit mismatches", len(prompt), mism)
	if mism != 0 {
		t.Errorf("N=32 and N=64 differ (%d mismatches) — paging is NOT byte-transparent at nE=128: an eviction/"+
			"staging bug at width, and the byte-identity gate does not hold beyond nE=4 (the serious outcome)", mism)
	} else {
		t.Logf("VERDICT (Test 3): paging is byte-transparent at nE=128 (N-independent) — the collapse is NOT staging; " +
			"it is routing sensitivity (Tests 1+2). The primary byte-identity gate holds at real width.")
	}
}
