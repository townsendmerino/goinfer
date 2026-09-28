//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"os"
	"runtime/debug"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestSpecNgram_multiTurnMatchesPlain: served with `--spec ngram`, W7's chat turns after the first differed from plain
// decode on Metal, while every first turn matched (docs/measurements/spec-vs-batching-metal-2026-09-27.md §4). This
// replays two W7 turns in-process, W7's shape (4 KV slots, ChatML, 128 greedy tokens), plain Generate against
// GenerateNgramSpeculativeAdaptive (serve's resident spec call), each arm on a fresh load. It reports where the second
// turn diverges, and compares the K/V each arm leaves after the first turn, position by position.
//
//	GOINFER_METAL_MC3=1 go test -tags goinfer_testhooks -count=1 -run '^TestSpecNgram_multiTurnMatchesPlain$' -v ./metal/
func TestSpecNgram_multiTurnMatchesPlain(t *testing.T) {
	path, tokPath := mc3Checkpoint(t)
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1 (loads a real checkpoint twice)")
	}
	tk, err := tokenizer.LoadGGUF(tokPath)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	enc := func(s string) []int {
		ids, err := tk.Encode(s, false)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		return ids
	}
	const maxTok = 128
	turns := []string{
		"What's a good strategy for testing a rate limiter under burst load?",
		"How would that change if the limiter needs to be distributed across multiple servers?",
	}
	type turn struct {
		ids    []int
		reused int
		spec   *decoder.SpecStats
		kv     [][]uint16 // K and V rows per layer, positions 0..committed-1, right after this turn
	}
	snapshot := func(m *decoder.Model, n int) [][]uint16 {
		r := m.ResidentForwardForTest().(*metalResident).r
		var out [][]uint16
		for l := range r.layers {
			kvDim := r.layers[l].geom.kvDim
			o := r.kvHostOff(l, 2)
			out = append(out, append([]uint16(nil), r.kc[l].U16s()[o:o+n*kvDim]...), append([]uint16(nil), r.vc[l].U16s()[o:o+n*kvDim]...))
		}
		return out
	}
	run := func(spec bool) []turn {
		m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 2048, ResidentKVSlots: 4})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer func() { m.Close(); debug.FreeOSMemory() }()
		prompt := enc("<|im_start|>user\n" + turns[0] + "<|im_end|>\n<|im_start|>assistant\n")
		var out []turn
		for tn := range turns {
			var ch <-chan int
			var gen *decoder.Generation
			if spec {
				if ch, gen, err = m.GenerateNgramSpeculativeAdaptive(context.Background(), prompt, maxTok, &decoder.NgramDrafter{},
					&decoder.AdaptiveDepth{MaxDraft: 8}, decoder.SamplingParams{}); err != nil {
					t.Fatalf("spec: %v", err)
				}
			} else {
				ch, gen = m.Generate(context.Background(), prompt, maxTok, decoder.SamplingParams{})
			}
			var ids []int
			for id := range ch {
				ids = append(ids, id)
			}
			if err := gen.Err(); err != nil {
				t.Fatalf("turn %d: %v", tn, err)
			}
			out = append(out, turn{ids: ids, reused: gen.PrefillReused, spec: gen.Spec, kv: snapshot(m, len(prompt)+len(ids)-1)})
			if tn+1 < len(turns) {
				prompt = append(append(slices.Clone(prompt), ids...),
					enc("<|im_end|>\n<|im_start|>user\n"+turns[tn+1]+"<|im_end|>\n<|im_start|>assistant\n")...)
			}
		}
		return out
	}
	plain, spec := run(false), run(true)

	firstDiff := func(a, b []int) int {
		for i := range min(len(a), len(b)) {
			if a[i] != b[i] {
				return i
			}
		}
		if len(a) != len(b) {
			return min(len(a), len(b))
		}
		return -1
	}
	for tn := range turns {
		p, s := plain[tn], spec[tn]
		st := ""
		if s.spec != nil {
			st = fmt.Sprintf("; spec rounds %d drafted %d accepted %d", s.spec.Rounds, s.spec.Drafted, s.spec.Accepted)
		}
		fmt.Fprintf(os.Stderr, "[spec-mt] turn %d: reused plain %d spec %d; %d / %d tokens; first differing token %d%s\n",
			tn, p.reused, s.reused, len(p.ids), len(s.ids), firstDiff(p.ids, s.ids), st)
	}
	// K/V after turn 0, over the positions both arms hold after it
	a, b := plain[0].kv, spec[0].kv
	differ := 0
	for i := range a {
		for j := range min(len(a[i]), len(b[i])) {
			if a[i][j] != b[i][j] {
				differ++
			}
		}
	}
	fmt.Fprintf(os.Stderr, "[spec-mt] K/V after turn 0: %d elements differ over the shared positions\n", differ)
	if differ != 0 {
		t.Errorf("the K/V spec leaves after turn 0 differs from plain decode's in %d elements", differ)
	}
	if d := firstDiff(plain[1].ids, spec[1].ids); d >= 0 {
		t.Errorf("turn 1 differs from plain decode at token %d: n-gram speculation is not token-identical to plain greedy across turns", d)
	}
}

// TestSpecNgram_copyOnStepVerify: on a verbatim-copy prompt, where n-gram drafting does most, speculation verifying on
// the step kernels (PrefillLastNArgmax) emits exactly plain decode's ids, and really does draft deep: rounds well
// under the token count. Reports both arms' wall time (exploratory; the served grading is the measure).
//
//	GOINFER_METAL_MC3=1 go test -tags goinfer_testhooks -count=1 -run '^TestSpecNgram_copyOnStepVerify$' -v ./metal/
func TestSpecNgram_copyOnStepVerify(t *testing.T) {
	path, tokPath := mc3Checkpoint(t)
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1 (loads a real checkpoint)")
	}
	tk, err := tokenizer.LoadGGUF(tokPath)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	src, err := os.ReadFile("backend.go")
	if err != nil {
		t.Fatal(err)
	}
	body := string(src[:3500])
	prompt, err := tk.Encode("<|im_start|>user\nHere is part of a Go file:\n\n```go\n"+body+"\n```\n\nRewrite the code above EXACTLY, character for character, with no changes and no commentary.<|im_end|>\n<|im_start|>assistant\n", false)
	if err != nil {
		t.Fatal(err)
	}
	const maxTok = 192
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 4096, ResidentKVSlots: 2})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if _, ok := m.ResidentForwardForTest().(interface{ VerifyCost() []float64 }); !ok {
		t.Fatal("the Metal resident does not report a verify cost")
	}
	run := func(spec bool) ([]int, *decoder.Generation, time.Duration) {
		t0 := time.Now()
		var ch <-chan int
		var g *decoder.Generation
		if spec {
			if ch, g, err = m.GenerateNgramSpeculativeAdaptive(context.Background(), prompt, maxTok, &decoder.NgramDrafter{},
				&decoder.AdaptiveDepth{MaxDraft: 8}, decoder.SamplingParams{}); err != nil {
				t.Fatal(err)
			}
		} else {
			ch, g = m.Generate(context.Background(), prompt, maxTok, decoder.SamplingParams{})
		}
		var ids []int
		for id := range ch {
			ids = append(ids, id)
		}
		if g.Err() != nil {
			t.Fatal(g.Err())
		}
		return ids, g, time.Since(t0)
	}
	// Plain runs first and cold; spec then reuses its prompt prefix, so spec's wall excludes most of the prefill. The
	// ratio printed is indicative only.
	plain, _, tp := run(false)
	spec, g, ts := run(true)
	if !slices.Equal(plain, spec) {
		i := 0
		for i < min(len(plain), len(spec)) && plain[i] == spec[i] {
			i++
		}
		t.Fatalf("spec differs from plain decode at token %d of %d/%d", i, len(plain), len(spec))
	}
	st := g.Spec
	fmt.Fprintf(os.Stderr, "[spec-copy] %d tokens identical; spec rounds %d drafted %d accepted %d; wall plain %v spec %v (%.2fx)\n",
		len(spec), st.Rounds, st.Drafted, st.Accepted, tp.Round(time.Millisecond), ts.Round(time.Millisecond), tp.Seconds()/ts.Seconds())
	if st.Drafted == 0 || st.Rounds >= len(spec)*3/4 {
		t.Errorf("spec drafted %d tokens over %d rounds for %d tokens: the step verify did not draft on a copy prompt", st.Drafted, st.Rounds, len(spec))
	}
}
