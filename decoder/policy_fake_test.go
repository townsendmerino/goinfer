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

// plainGreedy is greedy decoding over a Forward as Model.Generate does it on the CPU: Prefill the prompt, then one Step
// per emitted token; a stop token ends the stream unemitted.
func plainGreedy(fw Forward, prompt []int, n int, isStop func(int) bool) []int {
	ctx := context.Background()
	logits, _ := fw.Prefill(ctx, prompt)
	var gen []int
	for len(gen) < n {
		tok := argmax(logits)
		if isStop(tok) {
			break
		}
		gen = append(gen, tok)
		logits, _ = fw.Step(ctx, tok)
	}
	return gen
}

// perturbScript is script with its argmax moved at the history lengths where at says so: a draft model that agrees with
// the target except there, so rejections land at chosen depths.
func perturbScript(script func([]int) []float32, at func(n int) bool) func([]int) []float32 {
	return func(h []int) []float32 {
		l := script(h)
		if at(len(h)) {
			l[(argmax(l)+1)%len(l)] += 10
		}
		return l
	}
}

// stopAt is script with the stop token made the argmax once the history reaches n.
func stopAt(script func([]int) []float32, stop, n int) func([]int) []float32 {
	return func(h []int) []float32 {
		l := script(h)
		if len(h) >= n {
			l[stop] += 10
		}
		return l
	}
}

// TestDraftSpecLoop_exactOnTheFake is SB5-P3 for draft-model speculation: on scripted target and draft models,
// draftSpecLoop emits exactly the target's greedy stream, whatever the draft: the target itself (every draft accepted,
// which also holds the draft's own rollback and catch-up step),
// an unrelated model (nearly every draft rejected), the target wrong every 2nd, 3rd or 5th position (rejections at every
// depth), and a target that stops partway; for K 1, 2, 4 and 8 and maxTokens inside and past a round. With a context cap
// it emits a prefix of the same stream and never holds more than the cap.
func TestDraftSpecLoop_exactOnTheFake(t *testing.T) {
	const vocab, stop = 64, 0
	isStop := func(tok int) bool { return tok == stop }
	prompt := []int{5, 9, 11, 3, 7}
	targets := map[string]func([]int) []float32{
		"whole-history": hashScript(vocab, 0, 11),
		"last-token":    hashScript(vocab, 1, 12),
		"stops-at-20":   stopAt(hashScript(vocab, 2, 13), stop, 20),
	}
	cases := 0
	for tname, target := range targets {
		drafts := map[string]func([]int) []float32{
			"itself":     target,
			"unrelated":  hashScript(vocab, 0, 99),
			"wrong-ev-2": perturbScript(target, func(n int) bool { return n%2 == 0 }),
			"wrong-ev-3": perturbScript(target, func(n int) bool { return n%3 == 0 }),
			"wrong-ev-5": perturbScript(target, func(n int) bool { return n%5 == 0 }),
		}
		for _, n := range []int{3, 40} {
			ref := plainGreedy(&scriptedForward{t: t, vocab: vocab, script: target}, prompt, n, isStop)
			if len(ref) == 0 {
				t.Fatalf("%s: greedy emitted nothing: the case tests nothing", tname)
			}
			for dname, draft := range drafts {
				for _, K := range []int{1, 2, 4, 8} {
					name := fmt.Sprintf("%s/n%d/%s/K%d", tname, n, dname, K)
					tf := &scriptedForward{t: t, vocab: vocab, script: target}
					df := &scriptedForward{t: t, vocab: vocab, script: draft}
					out := make(chan int, 1024)
					var g Generation
					var stats SpecStats
					draftSpecLoop(context.Background(), tf, df, isStop, func() int { return 0 }, out, &g, &stats, prompt, n, K)
					close(out)
					got := collectTokens(out)
					if g.err != nil {
						t.Fatalf("%s: %v", name, g.err)
					}
					if !slices.Equal(got, ref) {
						t.Errorf("%s: draft speculation is not greedy\n got %v\n ref %v", name, got, ref)
					}
					// Both are cut back to the same kept prefix each round, so they end holding the same history.
					if !slices.Equal(df.hist, tf.hist) {
						t.Errorf("%s: the draft ends holding %d tokens, the target %d: they fell out of step", name, len(df.hist), len(tf.hist))
					}
					// The target as its own draft is right every time: a draft that falls out of step with the target (a
					// rollback or catch-up step gone wrong) keeps the stream exact but shows here.
					if dname == "itself" && stats.Accepted != stats.Drafted {
						t.Errorf("%s: the target as its own draft had %d of %d drafts accepted, want all", name, stats.Accepted, stats.Drafted)
					}
					cases++
				}
			}
		}
	}

	// A context cap: the stream is a prefix of the uncapped one and the target never holds more than the cap.
	target := hashScript(vocab, 0, 21)
	ref := plainGreedy(&scriptedForward{t: t, vocab: vocab, script: target}, prompt, 40, isStop)
	for _, capN := range []int{len(prompt) + 1, len(prompt) + 4, len(prompt) + 9} {
		tf := &scriptedForward{t: t, vocab: vocab, script: target}
		capped := &cappedForward{scriptedForward: tf, cap: capN, t: t}
		df := &scriptedForward{t: t, vocab: vocab, script: target}
		out := make(chan int, 1024)
		var g Generation
		var stats SpecStats
		draftSpecLoop(context.Background(), capped, df, isStop, func() int { return capN }, out, &g, &stats, prompt, 40, 4)
		close(out)
		got := collectTokens(out)
		if len(got) == 0 || len(got) > len(ref) || !slices.Equal(got, ref[:len(got)]) {
			t.Errorf("cap %d: %v is not a prefix of the uncapped stream %v", capN, got, ref)
		}
		cases++
	}
	t.Logf("%d cases", cases)
}

// cappedForward fails the test if the target is ever asked to hold more than cap tokens.
type cappedForward struct {
	*scriptedForward
	cap int
	t   *testing.T
}

func (f *cappedForward) Verify(ctx context.Context, ids []int) ([][]float32, error) {
	if len(f.hist)+len(ids) > f.cap {
		f.t.Errorf("verify of %d at %d passes the context cap %d", len(ids), len(f.hist), f.cap)
	}
	return f.scriptedForward.Verify(ctx, ids)
}
