//go:build gpu && goinfer_testhooks

package gpu_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGenerate_batchedPrefillMatchesSequential is the integration gate docs/completed/task-gpu-batched-prefill.md's
// Increment 3 calls for ("TestDecodeParity-class greedy continuation unchanged") but through the
// REAL production entry point, not an isolated PrefillLast call.
//
// Increment 3 ("wire into decoder.Generate") reads as not-yet-started in this doc's own Definition
// of Done, but it already is: decoder/model.go's residentPrefillSeed (shared by every resident
// generation path) has a backend-agnostic `if pf, ok := m.resident.(Prefiller); ok` check that
// fires automatically the moment a resident type satisfies decoder.Prefiller — and
// gpu.residentDecoder has satisfied it since 813be4e7 (Increment 3's own commit, predating this
// branch's dispatch-count fix). So the moment PrefillLastW8A8 works AND is fast, every WebGPU
// resident Generate() call on an eligible prompt (>=8 tokens, no adapter) already takes the batched
// path — nobody had confirmed that through Generate() itself, only through PrefillLast in
// isolation (TestResidentPrefillLast_parity/_TTFT). This test closes that gap: same real prompt,
// same greedy sampling, GOINFER_BATCHED_PREFILL=0 (forced sequential, the trusted baseline) vs the
// default (batched prefill on — automatically hits PrefillLastW8A8 for this checkpoint's dense
// W8A8 Qwen2 shape) must produce the IDENTICAL token stream.
//
//	GOINFER_RESIDENT_GGUF=~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
//	  GOINFER_HEAVY_TESTS=1 go test -tags 'gpu goinfer_testhooks' ./gpu/ -run TestGenerate_batchedPrefillMatchesSequential -v
func TestGenerate_batchedPrefillMatchesSequential(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a multi-GB model from ~/models)")
	}
	home, _ := os.UserHomeDir()
	path := os.Getenv("GOINFER_RESIDENT_GGUF")
	if path == "" {
		path = filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no resident model at %s: %v", path, err)
	}

	// The real prompt decode_parity_test.go pins against the CPU path for this same 0.5B
	// checkpoint ("a function to print the code:") — 8 tokens, exactly at
	// residentPrefillSeed's len(suffix)>=8 batched-prefill threshold.
	prompt := []int{785, 264, 6573, 311, 1438, 279, 2038, 25}
	const n = 24

	run := func(t *testing.T, batchedOff string) []int {
		t.Helper()
		t.Setenv("GOINFER_BATCHED_PREFILL", batchedOff)
		m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer m.Close()
		if !m.ResidentActive() {
			t.Skip("model not GPU-resident (ineligible arch / no residency)")
		}
		out, gen := m.Generate(context.Background(), prompt, n, decoder.SamplingParams{Temperature: 0})
		var got []int
		for tok := range out {
			got = append(got, tok)
		}
		if err := gen.Err(); err != nil {
			t.Fatalf("generate (GOINFER_BATCHED_PREFILL=%s): %v", batchedOff, err)
		}
		return got
	}

	sequential := run(t, "0")
	batched := run(t, "") // default: on — hits residentPrefillSeed's Prefiller branch for this checkpoint
	t.Logf("sequential (%d tok): %v", len(sequential), sequential)
	t.Logf("batched    (%d tok): %v", len(batched), batched)
	if len(sequential) != len(batched) {
		t.Fatalf("token count differs: sequential=%d batched=%d\nsequential: %v\nbatched:    %v",
			len(sequential), len(batched), sequential, batched)
	}
	for i := range sequential {
		if sequential[i] != batched[i] {
			t.Fatalf("token %d differs: sequential=%d batched=%d\nsequential: %v\nbatched:    %v",
				i, sequential[i], batched[i], sequential, batched)
		}
	}
}
