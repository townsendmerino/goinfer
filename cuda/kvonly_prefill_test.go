//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKVOnlyPrefill_byteIdentical_tiny is the gate for the no-logits (KV-only) prefill: greedy Generate
// with the optimization ON must emit the EXACT same token ids as with it OFF (GOINFER_NO_KVONLY_PREFILL).
// KV-only prefill skips the LM head on prompt[:-1]; if the K/V it writes differed in any way from the
// full-logits Forward, the last prompt token's logits — and thus the whole continuation — would drift.
// Runs on the committed tiny resident fixture (always, on a GPU).
func TestKVOnlyPrefill_byteIdentical_tiny(t *testing.T) {
	dir := "../testdata/gemma4-moe-tiny"
	prompt := make([]int, 24) // multi-token prompt so several tokens take the KV-only path
	for i := range prompt {
		prompt[i] = (i*11 + 3) % 200 // deterministic, inside the tiny vocab
	}
	run := func(kvOnly bool) []int {
		if kvOnly {
			t.Setenv("GOINFER_NO_KVONLY_PREFILL", "")
		} else {
			t.Setenv("GOINFER_NO_KVONLY_PREFILL", "1")
		}
		mc, rf := loadG4MoECache(t, dir, false)
		defer mc.Close()
		if _, ok := rf.(decoder.ResidentPrefillKV); !ok {
			t.Fatal("cudaResident does not implement ResidentPrefillKV")
		}
		out, _ := mc.Generate(context.Background(), prompt, 24, decoder.SamplingParams{})
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
			t.Fatalf("KV-only prefill NOT byte-identical at token %d: %d != %d — the KV-only path wrote different K/V than full-logits Forward", i, on[i], off[i])
		}
	}
	t.Logf("KV-only prefill == full-logits prefill: %d tokens byte-identical", len(on))
}
