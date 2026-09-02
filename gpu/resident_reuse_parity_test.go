//go:build gpu

package gpu

import (
	"context"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestResidentPrefixReuse_tokenIdentical is the gate for resident prefix reuse, and it is a
// TOKEN-IDENTITY gate rather than a similarity one on purpose.
//
// The feature skips re-prefilling leading prompt tokens already committed to the resident
// positional KV. Its failure mode is not a crash or a NaN: a wrong prefix match produces a
// fluent, confident reply conditioned on the wrong context, with no error anywhere. Nothing
// downstream can detect that. So the only acceptable evidence is that the emitted ids are the
// SAME ids a cold prefill emits, across a transcript that actually exercises reuse.
//
// The transcript is shaped like an agent loop, because that is the case reuse exists for:
// each turn is a strict prefix extension of the last (assistant reply + new user text), which
// is exactly the shape a tool loop produces.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags gpu -run TestResidentPrefixReuse -v ./gpu/
func TestResidentPrefixReuse_tokenIdentical(t *testing.T) {
	requireHeavyModel(t)
	path := os.Getenv("GOINFER_REUSE_GGUF")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not found: %s", path)
	}
	newOrSkipHW(t).Close()

	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}

	// run drives a 3-turn transcript and returns each turn's emitted ids plus how many prompt
	// tokens each turn skipped.
	run := func(t *testing.T, reuse bool) ([][]int, []int) {
		if reuse {
			os.Unsetenv("GOINFER_NO_RESIDENT_REUSE")
		} else {
			os.Setenv("GOINFER_NO_RESIDENT_REUSE", "1")
		}
		t.Cleanup(func() { os.Unsetenv("GOINFER_NO_RESIDENT_REUSE") })

		m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer m.Close()
		if !m.ResidentActive() {
			t.Skip("model did not go resident — this gate needs the resident path")
		}

		greedy := decoder.SamplingParams{Temperature: 0}
		prompt, _ := tk.Encode("Write one short line about recursion.", true)
		// Turn 4 DIVERGES: it shares only the opening words with what the cache holds. That
		// turn is what makes this gate real. A first version used three strict prefix
		// extensions only, and a deliberately broken matcher — one that claimed the prefix
		// WITHOUT comparing ids — passed it, because on an ever-extending transcript the
		// wrong answer and the right answer coincide. The bug was invisible in exactly the
		// dimension being tested (CLAUDE.md's minimal-repro trap). Divergence is the only
		// thing that distinguishes "compared the ids" from "assumed they matched".
		divergeAt := 3
		var outs [][]int
		var reused []int
		for turn := range 4 {
			if turn == divergeAt {
				prompt, _ = tk.Encode("Write one short line about pointers instead.", true)
			}
			ch, gen := m.Generate(context.Background(), prompt, 12, greedy)
			var got []int
			for id := range ch {
				got = append(got, id)
			}
			if err := gen.Err(); err != nil {
				t.Fatalf("turn %d: %v", turn+1, err)
			}
			outs = append(outs, got)
			reused = append(reused, gen.PrefillReused)
			// Next turn = this turn's prompt + this turn's reply + a new user line: the
			// strict prefix extension an agent loop produces.
			next := append(append([]int(nil), prompt...), got...)
			more, _ := tk.Encode(" Now say it differently.", false)
			prompt = append(next, more...)
		}
		return outs, reused
	}

	warm, reused := run(t, true)
	cold, coldReused := run(t, false)

	// The test must not pass vacuously: if reuse never fired, identity proves nothing.
	if reused[1] == 0 && reused[2] == 0 {
		t.Fatalf("no prefix was reused on turns 2 or 3 (%v) — this gate would pass without exercising the feature", reused)
	}
	for i, n := range coldReused {
		if n != 0 {
			t.Errorf("cold run turn %d reused %d tokens; GOINFER_NO_RESIDENT_REUSE must disable it entirely", i+1, n)
		}
	}
	t.Logf("prefill tokens reused per turn: warm %v · cold %v", reused, coldReused)

	for i := range warm {
		if len(warm[i]) != len(cold[i]) {
			t.Fatalf("turn %d: warm emitted %d tokens, cold %d — reuse changed the output length",
				i+1, len(warm[i]), len(cold[i]))
		}
		for j := range warm[i] {
			if warm[i][j] != cold[i][j] {
				wt, _ := tk.Decode(warm[i])
				ct, _ := tk.Decode(cold[i])
				t.Fatalf("turn %d diverges at token %d: warm %d vs cold %d\n  warm: %q\n  cold: %q",
					i+1, j, warm[i][j], cold[i][j], wt, ct)
			}
		}
	}
}
