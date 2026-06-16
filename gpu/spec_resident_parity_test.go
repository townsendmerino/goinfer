//go:build gpu

package gpu_test

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/gpu"
)

// TestSpeculativeResident_parity is THE gate for resident GPU speculative decoding
// (Lever 2, Stage A wiring): with the TARGET on the webgpu resident path (verify via
// the batched ForwardN, rollback via the no-op TruncateTo) and a CPU draft, the greedy
// speculative output must be TOKEN-IDENTICAL to plain resident greedy Generate — a pure
// speedup, not an approximation. Target = 1.5B (resident), draft = 0.5B (CPU); both
// dense Qwen2 (eligible) and vocab-matched.
//
//	GOINFER_SPEC_TARGET=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
//	GOINFER_SPEC_DRAFT=~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf \
//	  go test -tags gpu ./gpu/ -run TestSpeculativeResident_parity -v -timeout 30m
func TestSpeculativeResident_parity(t *testing.T) {
	if _, err := gpu.New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	home, _ := os.UserHomeDir()
	tpath := os.Getenv("GOINFER_SPEC_TARGET")
	if tpath == "" {
		tpath = filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	dpath := os.Getenv("GOINFER_SPEC_DRAFT")
	if dpath == "" {
		dpath = filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	}
	for _, p := range []string{tpath, dpath} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("missing model %s: %v", p, err)
		}
	}

	target, err := decoder.Load(tpath, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load target: %v", err)
	}
	defer target.Close()
	if !target.ResidentActive() {
		t.Skip("target not GPU-resident (ineligible / no residency)")
	}
	draft, err := decoder.Load(dpath, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load draft: %v", err)
	}
	defer draft.Close()

	collect := func(ch <-chan int) []int {
		var v []int
		for id := range ch {
			v = append(v, id)
		}
		return v
	}
	ctx := context.Background()
	greedy := decoder.SamplingParams{Temperature: 0}
	const n = 32
	prompts := [][]int{
		{785, 6722, 315, 9625, 374},      // "The capital of France is"
		{1654, 264, 729, 311, 912, 1378}, // a code-ish fragment
	}
	for pi, prompt := range prompts {
		refCh, _ := target.Generate(ctx, prompt, n, greedy)
		ref := collect(refCh)
		for _, K := range []int{4, 8} {
			ch, g, err := target.GenerateSpeculative(ctx, prompt, n, draft, K, greedy)
			if err != nil {
				t.Fatalf("prompt %d K=%d: %v", pi, K, err)
			}
			got := collect(ch)
			if g.Err() != nil {
				t.Fatalf("prompt %d K=%d: stream err %v", pi, K, g.Err())
			}
			if !slices.Equal(got, ref) {
				t.Fatalf("prompt %d K=%d: resident speculative != resident greedy\n got %v\n ref %v", pi, K, got, ref)
			}
			if g.Spec != nil {
				t.Logf("prompt %d K=%d: %d tokens, acceptance %.3f, %.2f tokens/round", pi, K, len(got), g.Spec.AcceptanceRate(), g.Spec.TokensPerRound())
			}
		}
	}
}
