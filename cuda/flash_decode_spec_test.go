//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
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
	t.Setenv("GOINFER_CUDA_FLASH_DECODE_VERIFY", "0") // option A: the exact-attention scope (option B, the multi-row lane, is TestFlashDecodeSpecVerifyLane)
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

// TestFlashDecodeSpecVerifyLane is gate G2 of attn-decode-fa-verify-PREREGISTERED.md (option B): with the lane on and NO exact-attention
// scope, speculative verify rows run on the multi-row lane and speculative output equals plain LANE greedy token for token. Two floors:
// 0 (every row is a lane row) and one that the generation CROSSES mid-stream (the batches that straddle it must split into an exact prefix
// and a lane suffix exactly as M=1 decode would). It asserts the multi-row kernel really ran inside verify rounds (launch count) and that
// the scope was not held. GOINFER_CUDA_FLASH_DECODE_VERIFY=0 must fall back to option A (lane bypassed, output == exact).
func TestFlashDecodeSpecVerifyLane(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	path := modelPath("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	block := "func add(a, b int) int {\n\treturn a + b\n}\n\nfunc sub(a, b int) int {\n\treturn a - b\n}\n\n"
	prompt, err := tk.Encode(strings.Repeat(block, 12)+"func add(a, b int) int {\n", false)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	const nNew = 112
	greedy := decoder.SamplingParams{}
	run := func(t *testing.T, floor int, verify string) {
		t.Setenv("GOINFER_CUDA_FLASH_DECODE", "16")
		t.Setenv("GOINFER_CUDA_FLASH_DECODE_MIN_KEYS", fmt.Sprint(floor))
		t.Setenv("GOINFER_CUDA_FLASH_DECODE_VERIFY", verify)
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentContext: 4096})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer m.Close()
		rf, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok || rf.faSplit != 16 {
			t.Skip("resident path declined or lane not loaded")
		}
		wantVerifyLane := verify != "0"
		if rf.faVerify != wantVerifyLane {
			t.Fatalf("faVerify = %v, want %v (multi-row kernels loaded: %v)", rf.faVerify, wantVerifyLane, rf.faCombineRows != (Pipeline{}))
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
		// references: plain LANE greedy and plain EXACT greedy.
		laneRef := plain()
		rf.faSplit = 0
		exactRef := plain()
		rf.faSplit = 16
		rf.faLaunches, rf.faRowLaunches = 0, 0
		ch, g, err := m.GenerateNgramSpeculative(context.Background(), prompt, nNew, &decoder.NgramDrafter{}, 8, greedy)
		if err != nil {
			t.Fatalf("GenerateNgramSpeculative: %v", err)
		}
		var spec []int
		for id := range ch {
			spec = append(spec, id)
		}
		if e := g.Err(); e != nil {
			t.Fatalf("speculative: %v", e)
		}
		if g.Spec == nil || g.Spec.Rounds == 0 {
			t.Fatalf("no verify rounds ran (%+v)", g.Spec)
		}
		want, wantName := laneRef, "plain LANE"
		if !wantVerifyLane {
			want, wantName = exactRef, "plain EXACT (option A)"
			if rf.faLaunches != 0 {
				t.Fatalf("option A: the lane launched %d times inside speculation", rf.faLaunches)
			}
		} else if rf.faRowLaunches == 0 {
			t.Fatal("VACUOUS: no fa_partial_rows launch inside the speculative generation — verify never used the multi-row lane")
		}
		if len(spec) != len(want) {
			t.Fatalf("speculative %d tokens vs %s %d", len(spec), wantName, len(want))
		}
		for i := range want {
			if spec[i] != want[i] {
				t.Fatalf("speculative diverged from %s at token %d: %d vs %d (floor %d)", wantName, i, spec[i], want[i], floor)
			}
		}
		t.Logf("floor %d verify=%q: speculative == %s over %d tokens; %d verify rounds, %d multi-row launches, %d lane launches total; plain lane vs exact differ: %v",
			floor, verify, wantName, len(spec), g.Spec.Rounds, rf.faRowLaunches, rf.faLaunches, fmt.Sprint(len(laneRef) == len(exactRef)) != "" && !equalInts(laneRef, exactRef))
	}
	crossFloor := len(prompt) + 40 // the generation crosses it after ~40 tokens, mid-stream
	t.Run("floor0", func(t *testing.T) { run(t, 0, "") })
	t.Run("crossesFloor", func(t *testing.T) { run(t, crossFloor, "") })
	t.Run("verifyOff-optionA", func(t *testing.T) { run(t, 0, "0") })
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestFlashDecodeForwardNBitIdentical is the direct basis of speculative losslessness on the lane's tree (G2, sharper than token equality,
// which cannot tell the lane from the exact path when their greedy tokens happen to agree): the logits of a batched verify pass
// (ForwardN over M rows, multi-row lane attention) equal, BIT FOR BIT, the logits of M sequential single-row lane decode steps at the
// same positions, for M in {1,2,5,9,16,20} (20 > faMaxRows exercises chunking), at several floors including ones the batch straddles.
// It also requires that lane logits DIFFER from exact-path logits somewhere, so the equality is not the exact path compared with itself.
func TestFlashDecodeForwardNBitIdentical(t *testing.T) {
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
	if !ok || !rf.faVerify {
		t.Skip("resident path declined or multi-row lane not loaded")
	}
	_, _, _, _, _, _, vocab := m.Dims()
	const P = 2048
	emb := func(i int) []float32 { return m.EmbedResidentForTest((i*2654435761 + 5) % (vocab - 1)) }
	prefill := make([][]float32, P)
	for i := range prefill {
		prefill[i] = append([]float32(nil), emb(i*7+1)...)
	}
	prime := func() {
		rf.faSplit = 16
		if _, e := rf.PrefillLast(context.Background(), prefill, 0); e != nil {
			t.Fatalf("prefill: %v", e)
		}
	}
	anyLaneDiffersFromExact := false
	for _, floor := range []int{0, P + 3, P + 9, 1 << 30} { // 1<<30: no row is a lane row (pure exact path both ways)
		rf.faMinKeys = floor
		for _, M := range []int{1, 2, 5, 9, 16, 20} {
			embs := make([][]float32, M)
			for i := range embs {
				embs[i] = emb(1000 + i)
			}
			// sequential single-row decode (the M=1 lane wherever the floor allows it)
			prime()
			seq := make([][]float32, M)
			for i := range seq {
				l, e := rf.Forward(embs[i], P+i)
				if e != nil {
					t.Fatalf("Forward: %v", e)
				}
				seq[i] = append([]float32(nil), l...)
			}
			// one batched verify pass
			prime()
			rf.faRowLaunches = 0
			bat, e := rf.ForwardN(embs, P)
			if e != nil {
				t.Fatalf("ForwardN: %v", e)
			}
			for i := range seq {
				if !sameLogits(seq[i], bat[i]) {
					t.Fatalf("floor=%d M=%d row %d: batched verify logits differ from sequential decode (multi-row launches %d)", floor, M, i, rf.faRowLaunches)
				}
			}
			if floor == 0 && rf.faRowLaunches == 0 {
				t.Fatalf("floor=0 M=%d: the verify pass never used the multi-row lane", M)
			}
			if floor == 0 && M == 9 { // the exact-path comparator
				prime()
				rf.faSplit = 0 // AFTER prime(), which resets it to 16
				ex := make([][]float32, M)
				for i := range ex {
					l, e := rf.Forward(embs[i], P+i)
					if e != nil {
						t.Fatalf("Forward(exact): %v", e)
					}
					ex[i] = append([]float32(nil), l...)
				}
				for i := range ex {
					if !sameLogits(ex[i], seq[i]) {
						anyLaneDiffersFromExact = true
					}
				}
				rf.faSplit = 16
			}
		}
	}
	if !anyLaneDiffersFromExact {
		t.Fatal("lane logits equal exact-path logits everywhere: the comparison would be the exact path against itself")
	}
}
