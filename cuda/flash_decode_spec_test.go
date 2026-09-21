//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestFlashDecodeSpeculativeScope is option A of the spec-decode story (docs/measurements/spec-decode-lane-2026-09-21.md): with
// the flash-decode lane ON, speculative generations no longer refuse; they hold an exact-attention scope so decode and verify
// share ONE tree. It asserts, on the real 1.5B with a copy-heavy prompt (so drafts are accepted and verify rounds really run):
//
//  1. the guard reports no conflict with the lane on (it used to refuse);
//
//  2. a plain lane generation launches the lane (the scope is not stuck on);
//
//  3. an n-gram speculative generation launches the lane ZERO times (the scope is held) and its tokens equal a plain generation on
//     the EXACT tree token for token (lossless on that tree);
//
//  4. the scope is released when the speculative generation ends: the next plain generation launches the lane again;
//
//  5. the same holds when the speculative channel is abandoned mid-stream by cancelling the context.
//
//     GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestFlashDecodeSpeculativeScope -v
func TestFlashDecodeSpeculativeScope(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	t.Setenv("GOINFER_CUDA_FLASH_DECODE", "16")
	t.Setenv("GOINFER_CUDA_FLASH_DECODE_MIN_KEYS", "0")
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 4096})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok || rf.faSplit != 16 {
		t.Skip("resident path declined or lane not loaded")
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	// A repeated block of code: the n-gram drafter finds long matches, so verify rounds accept most drafts.
	block := "func add(a, b int) int {\n\treturn a + b\n}\n\nfunc sub(a, b int) int {\n\treturn a - b\n}\n\n"
	prompt, err := tk.Encode(strings.Repeat(block, 12)+"func add(a, b int) int {\n", false)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	const nNew = 96
	greedy := decoder.SamplingParams{}

	// 1. no conflict with the lane on.
	if e := m.SpecDecodeConflict(); e != nil {
		t.Fatalf("SpecDecodeConflict with the lane on = %v, want nil (speculation holds the exact-attention scope instead)", e)
	}
	plain := func() []int {
		ch, g := m.Generate(context.Background(), prompt, nNew, greedy)
		var out []int
		for id := range ch {
			out = append(out, id)
		}
		if e := g.Err(); e != nil {
			t.Fatalf("plain generate: %v", e)
		}
		return out
	}
	// The EXACT-tree reference: lane off, plain greedy.
	rf.faSplit = 0
	exact := plain()
	rf.faSplit = 16

	// 2. a plain generation uses the lane.
	rf.faLaunches = 0
	_ = plain()
	if rf.faLaunches == 0 {
		t.Fatal("a plain generation with the lane on never launched the lane — nothing below would mean anything")
	}

	// 3. speculation: lane bypassed, output == exact plain.
	rf.faLaunches = 0
	ch, g, err := m.GenerateNgramSpeculative(context.Background(), prompt, nNew, &decoder.NgramDrafter{}, 8, greedy)
	if err != nil {
		t.Fatalf("GenerateNgramSpeculative refused with the lane on: %v", err)
	}
	var spec []int
	for id := range ch {
		spec = append(spec, id)
	}
	if e := g.Err(); e != nil {
		t.Fatalf("speculative generate: %v", e)
	}
	if rf.faLaunches != 0 {
		t.Fatalf("the lane launched %d times inside a speculative generation: the exact-attention scope is not held", rf.faLaunches)
	}
	if g.Spec == nil || g.Spec.Rounds == 0 {
		t.Fatalf("no speculative verify rounds ran (stats %+v) — the test would prove nothing about verify", g.Spec)
	}
	if len(spec) != len(exact) {
		t.Fatalf("speculative produced %d tokens, plain exact %d", len(spec), len(exact))
	}
	for i := range exact {
		if spec[i] != exact[i] {
			t.Fatalf("speculative diverged from plain EXACT greedy at token %d: %d vs %d", i, spec[i], exact[i])
		}
	}
	t.Logf("speculative == plain exact over %d tokens; %d verify rounds, lane launches inside = 0", len(spec), g.Spec.Rounds)

	// 4. scope released.
	rf.faLaunches = 0
	_ = plain()
	if rf.faLaunches == 0 {
		t.Fatal("the lane did not resume after the speculative generation ended: the scope leaked")
	}
	if rf.faExactScope.Load() != 0 {
		t.Fatalf("scope count = %d after a completed speculative generation, want 0", rf.faExactScope.Load())
	}

	// 5. abandoned mid-stream.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch, _, err = m.GenerateNgramSpeculative(ctx, prompt, 1000, &decoder.NgramDrafter{}, 8, greedy)
	if err != nil {
		t.Fatalf("second speculative generate: %v", err)
	}
	n := 0
	for range ch {
		if n++; n == 5 {
			cancel()
			break
		}
	}
	for range ch { // drain so the goroutine exits and runs its deferred leave
	}
	if rf.faExactScope.Load() != 0 {
		t.Fatalf("scope count = %d after a cancelled speculative generation, want 0", rf.faExactScope.Load())
	}
}
