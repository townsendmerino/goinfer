//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"math/rand"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestResidentPrefillSeed_metalKVOnly_byteIdentical is decoder.residentPrefillSeed's own gate for
// M-01 (audit-metal-2026-09-12.md), mirroring cuda/kvonly_prefill_test.go's
// TestKVOnlyPrefill_byteIdentical_tiny: greedy Generate through the real production seam (not
// ForwardNoLogits called directly, as in TestForwardNoLogits_byteIdenticalKV) must emit the exact
// same token ids whether the KV-only skip is enabled or forced off via
// GOINFER_NO_KVONLY_PREFILL — proving decoder.Model actually reaches metalResident.ForwardNoLogits
// now that it implements decoder.ResidentPrefillKV, not just that the method is correct in
// isolation.
func TestResidentPrefillSeed_metalKVOnly_byteIdentical(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	w := genTinyWeights(rand.New(rand.NewSource(13)))
	dir := t.TempDir()
	writeDense(t, dir, w)

	prompt := make([]int, 16) // >= 8 so residentPrefillSeed's KV-only branch is reachable
	for i := range prompt {
		prompt[i] = (i*11 + 3) % tmVocab
	}

	run := func(kvOnly bool) []int {
		if kvOnly {
			t.Setenv("GOINFER_NO_KVONLY_PREFILL", "")
		} else {
			t.Setenv("GOINFER_NO_KVONLY_PREFILL", "1")
		}
		m, err := decoder.Load(dir, decoder.Options{Backend: "metal", Quant: "int8int8"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer m.Close()
		if _, ok := m.ResidentForwardForTest().(decoder.ResidentPrefillKV); !ok {
			t.Fatal("metalResident does not implement decoder.ResidentPrefillKV")
		}
		out, _ := m.Generate(context.Background(), prompt, 16, decoder.SamplingParams{})
		var ids []int
		for id := range out {
			ids = append(ids, id)
		}
		return ids
	}
	on := run(true)
	off := run(false)
	if len(on) == 0 {
		t.Fatal("no tokens generated")
	}
	if len(on) != len(off) {
		t.Fatalf("length differs: kv-only %d vs full-logits %d ids", len(on), len(off))
	}
	for i := range on {
		if on[i] != off[i] {
			t.Fatalf("KV-only prefill NOT byte-identical at token %d: %d != %d — the KV-only path "+
				"wrote different K/V than full-logits Forward", i, on[i], off[i])
		}
	}
	t.Logf("KV-only prefill == full-logits prefill: %d tokens byte-identical", len(on))
}
