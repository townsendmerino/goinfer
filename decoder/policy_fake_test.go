package decoder

import (
	"context"
	"fmt"
	"hash/fnv"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/constrain"
)

// scriptedForward is a fake Forward (SB5): the logits after a history are a pure function of that history, so a policy
// that rolls back wrongly, or verifies at the wrong position, sees different logits from then on and its stream departs
// from plain decoding. It counts its calls and refuses a truncation past what it holds.
type scriptedForward struct {
	t      *testing.T
	vocab  int
	script func(hist []int) []float32
	hist   []int

	prefills, verifies, steps, truncates int
}

func (f *scriptedForward) Prefill(_ context.Context, ids []int) ([]float32, error) {
	f.prefills++
	f.hist = append(f.hist, ids...)
	return f.script(f.hist), nil
}

func (f *scriptedForward) Verify(_ context.Context, ids []int) ([][]float32, error) {
	f.verifies++
	rows := make([][]float32, len(ids))
	for i, id := range ids {
		f.hist = append(f.hist, id)
		rows[i] = f.script(f.hist)
	}
	return rows, nil
}

func (f *scriptedForward) Step(_ context.Context, id int) ([]float32, error) {
	f.steps++
	f.hist = append(f.hist, id)
	return f.script(f.hist), nil
}

func (f *scriptedForward) TruncateTo(pos int) {
	f.truncates++
	if pos > len(f.hist) {
		f.t.Fatalf("TruncateTo(%d) past the %d tokens the fake holds", pos, len(f.hist))
	}
	f.hist = f.hist[:pos]
}

// hashScript is logits drawn from a hash of the whole history (window 0) or its last window tokens, so the argmax at a
// position changes when anything before it does: every rollback is a branch point.
func hashScript(vocab, window int, seed uint64) func([]int) []float32 {
	return func(hist []int) []float32 {
		h := hist
		if window > 0 && len(h) > window {
			h = h[len(h)-window:]
		}
		out := make([]float32, vocab)
		for v := range vocab {
			x := fnv.New64a()
			fmt.Fprint(x, seed, v, h)
			out[v] = float32(x.Sum64()%1000003) / 1000003
		}
		return out
	}
}

// plainConstrained is constrained greedy decoding over a Forward, as Model.Generate does it with a LogitProcessor: the
// processor masks each row, the argmax is taken, a stop token ends the stream unemitted.
func plainConstrained(fw Forward, prompt []int, n int, isStop func(int) bool, proc func([]int, []float32)) []int {
	ctx := context.Background()
	logits, _ := fw.Prefill(ctx, prompt)
	var gen []int
	for len(gen) < n {
		proc(gen, logits)
		tok := argmax(logits)
		if isStop(tok) {
			break
		}
		gen = append(gen, tok)
		rows, _ := fw.Verify(ctx, []int{tok})
		logits = rows[0]
	}
	return gen
}

// policyScriptDrafter proposes from a fixed continuation of the prompt: at history h it returns the next k tokens of next
// after len(h)-len(prompt), each replaced by wrong(i) at the depths where reject says so.
type policyScriptDrafter struct {
	promptLen int
	next      []int
	wrongAt   func(depth int) bool
	vocab     int
}

func (d *policyScriptDrafter) Draft(ctx []int, k int) []int {
	at := len(ctx) - d.promptLen
	var out []int
	for i := 0; i < k && at+i < len(d.next); i++ {
		tok := d.next[at+i]
		if d.wrongAt != nil && d.wrongAt(i) {
			tok = (tok + 1) % d.vocab
		}
		out = append(out, tok)
	}
	return out
}

// TestGrammarSpecLoop_exactOnTheFake is SB5-P3 for grammar speculation: on scripted models, grammarSpecLoop emits
// exactly constrained greedy decoding, whatever the drafts: the grammar's own forced bytes, every draft right (all
// accepted), every draft wrong, a wrong draft at each depth 0 to 7, maxTokens reached inside an accepted run, and the grammar
// completing (its stop token) inside a draft. It also holds TestGrammarSpecParity's claim, which needed a checkpoint CI
// does not have.
func TestGrammarSpecLoop_exactOnTheFake(t *testing.T) {
	const vocab = 256
	eos := []int{0}
	isStop := func(tok int) bool { return slices.Contains(eos, tok) }
	newMask := func() *constrain.Masker {
		gr, err := constrain.JSONSchema([]byte(censusSchema))
		if err != nil {
			t.Fatal(err)
		}
		return constrain.NewMasker(gr, byteVocab(vocab), eos).StopWhenComplete()
	}
	prompt := bytesIDs("Weather as JSON: ")
	scripts := map[string]func([]int) []float32{
		"whole-history": hashScript(vocab, 0, 1),
		"last-token":    hashScript(vocab, 1, 2),
		"last-3":        hashScript(vocab, 3, 3),
	}
	cases := 0
	for sname, script := range scripts {
		for _, n := range []int{5, 64} {
			ref := plainConstrained(&scriptedForward{t: t, vocab: vocab, script: script}, prompt, n, isStop, newMask().Process)
			if len(ref) == 0 {
				t.Fatalf("%s: constrained decoding emitted nothing: the case tests nothing", sname)
			}
			drafters := map[string]func(*constrain.Masker) Drafter{
				"grammar": func(mk *constrain.Masker) Drafter {
					return &GrammarDrafter{Mask: mk, Encode: func(s string) []int { return bytesIDs(s) }}
				},
				"oracle": func(*constrain.Masker) Drafter { return &policyScriptDrafter{len(prompt), ref, nil, vocab} },
				"all-wrong": func(*constrain.Masker) Drafter {
					return &policyScriptDrafter{len(prompt), ref, func(int) bool { return true }, vocab}
				},
				"past-the-end": func(*constrain.Masker) Drafter {
					return &policyScriptDrafter{len(prompt), append(slices.Clone(ref), 0, '}', 'x', 'y'), nil, vocab}
				},
			}
			for depth := range 8 {
				drafters[fmt.Sprintf("wrong-at-%d", depth)] = func(*constrain.Masker) Drafter {
					return &policyScriptDrafter{len(prompt), ref, func(d int) bool { return d == depth }, vocab}
				}
			}
			for dname, mk := range drafters {
				for _, K := range []int{1, 3, 8} {
					name := fmt.Sprintf("%s/n%d/%s/K%d", sname, n, dname, K)
					fw := &scriptedForward{t: t, vocab: vocab, script: script}
					mask := newMask()
					out := make(chan int, 1024)
					var g Generation
					var stats SpecStats
					grammarSpecLoop(context.Background(), fw, isStop, out, &g, &stats, mask, mk(mask), prompt, 0, n, K, nil, nil)
					close(out)
					got := collectTokens(out)
					if g.err != nil {
						t.Fatalf("%s: %v", name, g.err)
					}
					if !slices.Equal(got, ref) {
						t.Errorf("%s: grammar speculation is not constrained greedy\n got %v\n ref %v", name, got, ref)
					}
					if dname == "oracle" && K > 1 && n == 64 && stats.Accepted == 0 {
						t.Errorf("%s: an oracle drafter had nothing accepted: the accept path is untested", name)
					}
					if dname == "all-wrong" && stats.Accepted != 0 {
						t.Errorf("%s: %d wrong drafts accepted", name, stats.Accepted)
					}
					cases++
				}
			}
		}
	}
	t.Logf("%d cases", cases)
}
