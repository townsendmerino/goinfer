package decoder

import (
	"context"
	"fmt"
	"hash/fnv"
	"math"
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

// fakeNgramTarget is ngramTarget over a scripted Forward: verify checks that each position it is given is the one the
// fake holds; with argmax set, idsVerify answers argmax ids as a resident backend computing them on device does.
func fakeNgramTarget(t *testing.T, fw *scriptedForward, isStop func(int) bool, argmaxIDs bool) ngramTarget {
	tg := ngramTarget{
		verify: func(seq []int, base int) ([][]float32, error) {
			if base != len(fw.hist) {
				t.Fatalf("verify at %d, but the fake holds %d tokens", base, len(fw.hist))
			}
			return fw.Verify(context.Background(), seq)
		},
		truncate:       fw.TruncateTo,
		commitResident: func([]int) {},
		isStop:         isStop,
		ctxCap:         func() int { return 0 },
	}
	if argmaxIDs {
		tg.idsVerify = func(seq []int, base int) ([]int, error) {
			rows, err := tg.verify(seq, base)
			ids := make([]int, len(rows))
			for i, r := range rows {
				ids[i] = argmax(r)
			}
			return ids, err
		}
	}
	return tg
}

// TestNgramSpecLoop_exactOnTheFake is SB5-P3 for n-gram speculation (greedy): on scripted models, ngramSpecLoop emits
// exactly greedy decoding with the real NgramDrafter, an oracle drafter, one wrong every time, one wrong at each depth
// 0 to 7, and with adaptive depth; through full-logit verifies and through argmax-only ones; for K 1, 3 and 8 and
// maxTokens inside and past a round, including a target that stops partway.
func TestNgramSpecLoop_exactOnTheFake(t *testing.T) {
	const vocab, stop = 32, 0
	isStop := func(tok int) bool { return tok == stop }
	// A repeating prompt, so the n-gram drafter finds matches; a small vocabulary and a short window, so the target
	// repeats itself and drafts land.
	prompt := []int{4, 7, 9, 4, 7, 9, 4, 7}
	targets := map[string]func([]int) []float32{
		"last-2":      hashScript(vocab, 2, 31),
		"last-3":      hashScript(vocab, 3, 32),
		"stops-at-25": stopAt(hashScript(vocab, 2, 33), stop, 25),
	}
	cases := 0
	for tname, script := range targets {
		for _, n := range []int{4, 48} {
			ref := plainGreedy(&scriptedForward{t: t, vocab: vocab, script: script}, prompt, n, isStop)
			if len(ref) == 0 {
				t.Fatalf("%s: greedy emitted nothing", tname)
			}
			drafters := map[string]func() Drafter{
				"ngram":     func() Drafter { return &NgramDrafter{} },
				"oracle":    func() Drafter { return &policyScriptDrafter{len(prompt), ref, nil, vocab} },
				"all-wrong": func() Drafter { return &policyScriptDrafter{len(prompt), ref, func(int) bool { return true }, vocab} },
			}
			for depth := range 8 {
				drafters[fmt.Sprintf("wrong-at-%d", depth)] = func() Drafter {
					return &policyScriptDrafter{len(prompt), ref, func(d int) bool { return d == depth }, vocab}
				}
			}
			for dname, mk := range drafters {
				for _, K := range []int{1, 3, 8} {
					for _, ids := range []bool{false, true} {
						for _, adaptive := range []bool{false, true} {
							name := fmt.Sprintf("%s/n%d/%s/K%d/ids=%v/ad=%v", tname, n, dname, K, ids, adaptive)
							fw := &scriptedForward{t: t, vocab: vocab, script: script}
							seed, _ := fw.Prefill(context.Background(), prompt)
							var ad *AdaptiveDepth
							if adaptive {
								ad = &AdaptiveDepth{MaxDraft: 8, Theta: 0.5}
							}
							out := make(chan int, 1024)
							var g Generation
							var stats SpecStats
							ngramSpecLoop(context.Background(), fakeNgramTarget(t, fw, isStop, ids), seed, out, &g, &stats, mk(), prompt, n, K, nil, false, nil, ad, nil)
							close(out)
							got := collectTokens(out)
							if g.err != nil {
								t.Fatalf("%s: %v", name, g.err)
							}
							if !slices.Equal(got, ref) {
								t.Errorf("%s: n-gram speculation is not greedy\n got %v\n ref %v", name, got, ref)
							}
							if dname == "oracle" && n == 48 && K > 1 && !adaptive && stats.Accepted == 0 {
								t.Errorf("%s: an oracle drafter had nothing accepted", name)
							}
							cases++
						}
					}
				}
			}
		}
	}
	t.Logf("%d cases", cases)
}

// fakeKV is a scripted Forward as a session's KV: ring, when set, makes a truncation below len-ring inexact (a wrapped
// sliding-window ring cannot restore evicted rows), and recurrent marks state that cannot be rolled back.
type fakeKV struct {
	*scriptedForward
	ring      int
	recurrent bool
}

func (k *fakeKV) truncate(pos int) bool {
	exact := k.ring == 0 || pos >= len(k.hist)-k.ring
	if !exact {
		pos = 0
	}
	k.TruncateTo(pos)
	return exact
}

// fakeSessionTurn is one Session.Generate on the fake, as the CPU decode loop runs it: rewind the KV for the prompt,
// prefill the rest, emit greedy tokens, forward and commit each emitted token but the last, stop on a stop token; cutAt
// > 0 ends the turn after that many commits with the KV one token behind (a forward that failed mid-sweep). It returns
// the stream, the reuse, and the session's new tokens.
func fakeSessionTurn(kv *fakeKV, tokens, prompt []int, clean bool, n, cutAt int, isStop func(int) bool) (gen []int, matched int, after []int) {
	ctx := context.Background()
	matched = sessionRewind(tokens, prompt, clean, false, kv.truncate)
	logits, _ := kv.Prefill(ctx, prompt[matched:])
	seq := slices.Clone(prompt)
	for {
		tok := argmax(logits)
		if isStop(tok) {
			break
		}
		gen = append(gen, tok)
		if len(gen) == n {
			break
		}
		logits, _ = kv.Step(ctx, tok)
		seq = append(seq, tok)
		if cutAt > 0 && len(seq)-len(prompt) == cutAt {
			seq = append(seq, argmax(logits)) // emitted, but its forward failed: the KV is one behind
			gen = append(gen, argmax(logits))
			break
		}
	}
	after, _ = sessionReconcile(len(kv.hist), seq, func() bool { return kv.recurrent }, kv.truncate)
	return gen, matched, after
}

// TestSessionPolicy_exactOnTheFake is SB5-P3 for sessions: a run of turns on one fake KV (a cold start, a resume
// extending the last turn, the same prompt again, a rewind to a shorter prefix, a divergence, a turn cut off mid-write)
// emits, turn by turn, exactly what cold decoding of that turn's prompt emits, with a plain KV, a ring cache that cannot
// truncate exactly below a window, and recurrent state; and the session's tokens are always the KV's contents.
func TestSessionPolicy_exactOnTheFake(t *testing.T) {
	const vocab, stop, n = 48, 0, 12
	isStop := func(tok int) bool { return tok == stop }
	script := hashScript(vocab, 0, 41)
	cold := func(p []int) []int {
		return plainGreedy(&scriptedForward{t: t, vocab: vocab, script: script}, p, n, isStop)
	}
	for _, kvKind := range []struct {
		name      string
		ring      int
		recurrent bool
	}{{"plain", 0, false}, {"ring-6", 6, false}, {"recurrent", 0, true}} {
		kv := &fakeKV{scriptedForward: &scriptedForward{t: t, vocab: vocab, script: script}, ring: kvKind.ring, recurrent: kvKind.recurrent}
		var tokens []int
		p0 := []int{3, 5, 7, 9, 11, 13, 15, 17}
		turns := []struct {
			name  string
			build func(prev []int) []int
			cutAt int
		}{
			{"cold", func([]int) []int { return p0 }, 0},
			{"resume", func(prev []int) []int { return append(slices.Clone(prev), 21, 22) }, 0},
			{"same-again", func(prev []int) []int { return slices.Clone(prev) }, 0},
			{"rewind-short", func([]int) []int { return slices.Clone(p0[:3]) }, 0},
			{"diverge", func([]int) []int { return append(slices.Clone(p0[:4]), 30, 31, 32) }, 0},
			{"cut-mid-write", func(prev []int) []int { return append(slices.Clone(prev), 40) }, 3},
			{"after-the-cut", func(prev []int) []int { return append(slices.Clone(prev), 41) }, 0},
		}
		var lastPrompt []int
		for i, turn := range turns {
			prompt := turn.build(lastPrompt)
			name := fmt.Sprintf("%s/%d-%s", kvKind.name, i, turn.name)
			gen, matched, after := fakeSessionTurn(kv, tokens, prompt, i == 0, n, turn.cutAt, isStop)
			if want := cold(prompt); turn.cutAt == 0 && !slices.Equal(gen, want) {
				t.Errorf("%s: the session's stream differs from cold decoding\n got %v\n ref %v", name, gen, want)
			}
			if matched > len(prompt)-1 || (matched > 0 && !slices.Equal(prompt[:matched], tokens[:matched])) {
				t.Errorf("%s: reused %d tokens, which are not a common prefix of the live tokens and the prompt", name, matched)
			}
			if !slices.Equal(after, kv.hist) && !(after == nil && len(kv.hist) == 0) {
				t.Errorf("%s: the session records %v, the KV holds %v", name, after, kv.hist)
			}
			t.Logf("%s: matched %d of %d, session holds %d", name, matched, len(prompt), len(after))
			// The paths this test exists for must be taken, not just survived.
			switch {
			case turn.name == "resume" && matched == 0:
				t.Errorf("%s: a resume reused nothing", name)
			case kvKind.ring > 0 && turn.name == "rewind-short" && matched != 0:
				t.Errorf("%s: a rewind the ring cannot make exactly reused %d tokens", name, matched)
			case kvKind.recurrent && turn.name == "cut-mid-write" && len(after) != 0:
				t.Errorf("%s: a recurrent KV cut mid-write kept %d tokens", name, len(after))
			case !kvKind.recurrent && turn.name == "cut-mid-write" && len(after) != len(prompt)+turn.cutAt:
				t.Errorf("%s: after a cut the session holds %d tokens, want the %d the KV wrote", name, len(after), len(prompt)+turn.cutAt)
			}
			tokens = after
			lastPrompt = append(slices.Clone(prompt), gen...)
		}
	}
}

// scaled sharpens a script's logits, so its softmax is far from uniform and a wrong acceptance rule moves the
// distribution visibly.
func scaled(script func([]int) []float32, k float32) func([]int) []float32 {
	return func(h []int) []float32 {
		l := script(h)
		for i := range l {
			l[i] *= k
		}
		return l
	}
}

// constDrafter proposes the same token at every depth.
type constDrafter struct{ tok int }

func (d constDrafter) Draft(_ []int, k int) []int {
	out := make([]int, k)
	for i := range out {
		out[i] = d.tok
	}
	return out
}

// TestNgramSpecLoop_sampledIsLosslessOnTheFake is SB5-P3 for sampled n-gram speculation, whose claim is lossless in
// distribution, not in stream: on a fake with a 3-token vocabulary and a 3-token horizon, the distribution over the 27
// sequences the speculative loop emits across 30,000 seeds is within a total variation of 0.03 of plain sampling's,
// computed exactly from the scripted logits (sampling noise at this size is about 0.012), for a drafter that proposes
// the target's likeliest token and one that proposes an unlikely one, K 1 and 3.
func TestNgramSpecLoop_sampledIsLosslessOnTheFake(t *testing.T) {
	if testing.Short() {
		t.Skip("30,000 seeded runs per case")
	}
	const vocab, n, runs = 3, 3, 30000
	script := scaled(hashScript(vocab, 0, 51), 3)
	prompt := []int{1, 2}
	softmax := func(l []float32) []float64 {
		mx := l[0]
		for _, v := range l {
			mx = max(mx, v)
		}
		p := make([]float64, len(l))
		sum := 0.0
		for i, v := range l {
			p[i] = math.Exp(float64(v - mx))
			sum += p[i]
		}
		for i := range p {
			p[i] /= sum
		}
		return p
	}
	// The exact distribution of plain sampling at temperature 1: the product of each step's softmax.
	exact := map[string]float64{}
	var walk func(hist []int, seq []int, pr float64)
	walk = func(hist []int, seq []int, pr float64) {
		if len(seq) == n {
			exact[fmt.Sprint(seq)] += pr
			return
		}
		p := softmax(script(hist))
		for tok := range vocab {
			walk(append(slices.Clone(hist), tok), append(slices.Clone(seq), tok), pr*p[tok])
		}
	}
	walk(prompt, nil, 1)
	likeliest := argmax(script(prompt))
	unlikely := (likeliest + 1) % vocab
	for _, dr := range []struct {
		name string
		d    Drafter
	}{{"likeliest", constDrafter{likeliest}}, {"unlikely", constDrafter{unlikely}}} {
		for _, K := range []int{1, 3} {
			counts := map[string]int{}
			var total SpecStats
			for seed := range runs {
				var stats SpecStats // per run: the loop stops once its own Emitted reaches n
				fw := &scriptedForward{t: t, vocab: vocab, script: script}
				seedLogits, _ := fw.Prefill(context.Background(), prompt)
				sampler := NewSampler(SamplingParams{Temperature: 1, Seed: int64(seed) + 1})
				out := make(chan int, 16)
				var g Generation
				ngramSpecLoop(context.Background(), fakeNgramTarget(t, fw, func(int) bool { return false }, false), seedLogits, out, &g, &stats, dr.d, prompt, n, K, sampler, false, nil, nil, nil)
				sampler.release()
				close(out)
				if g.err != nil {
					t.Fatal(g.err)
				}
				counts[fmt.Sprint(collectTokens(out))]++
				total.Drafted += stats.Drafted
				total.Accepted += stats.Accepted
			}
			tv := 0.0
			for seq, p := range exact {
				tv += math.Abs(float64(counts[seq])/runs - p)
			}
			for seq, c := range counts {
				if _, ok := exact[seq]; !ok {
					tv += float64(c) / runs // a sequence plain sampling cannot emit
				}
			}
			tv /= 2
			name := fmt.Sprintf("%s/K%d", dr.name, K)
			if tv > 0.03 {
				t.Errorf("%s: total variation %.4f from plain sampling (over 0.03): sampled speculation is not lossless", name, tv)
			}
			if total.Accepted == 0 || total.Accepted == total.Drafted {
				t.Errorf("%s: %d of %d drafts accepted: the case does not exercise both acceptance and the residual", name, total.Accepted, total.Drafted)
			}
			t.Logf("%s: TV %.4f, %d of %d drafts accepted", name, tv, total.Accepted, total.Drafted)
		}
	}
}
