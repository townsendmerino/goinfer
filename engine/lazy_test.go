package engine

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
)

// The toy: one token per byte (ids 0..255) and EOS (256).
const toyEOS = 256

func toyTokenBytes() [][]byte {
	tb := make([][]byte, toyEOS+1)
	for i := range 256 {
		tb[i] = []byte{byte(i)}
	}
	return tb
}

func toyIDs(s string) []int {
	ids := make([]int, len(s))
	for i := range len(s) {
		ids[i] = int(s[i])
	}
	return ids
}

func toyText(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		if id < 256 {
			b.WriteByte(byte(id))
		}
	}
	return b.String()
}

// toyLazy is chatml's lazy union over one tool, get(x int), as constrainToolUnion builds it: armed on "<tool_call>".
func toyLazy(t *testing.T) *constrain.LazyMasker {
	t.Helper()
	g, err := constrain.ToolCallsGrammar("<tool_call>", "\n</tool_call>", "arguments", false,
		[]constrain.ToolSpec{{Name: "get", Parameters: []byte(`{"type":"object","properties":{"x":{"type":"integer"}}}`)}})
	if err != nil {
		t.Fatal(err)
	}
	return constrain.NewLazyMasker(constrain.NewMasker(g, toyTokenBytes(), []int{toyEOS}), "<tool_call>")
}

// phaseCalls counts a fake phase's starts: the evidence of which path a turn took.
type phaseCalls struct{ free, constrained int }

// scriptedFree is phase 1 as a speculating server runs it, with no processor: it emits script byte by byte, then EOS. release, when
// non-nil, is read before each token after the first held ones, so a test can see what reached the caller while phase 1 waits.
func scriptedFree(script string, calls *phaseCalls, hold int, release <-chan struct{}) lazyPhase {
	return func(ctx context.Context, prompt []int, max int, sp decoder.SamplingParams) (<-chan int, *decoder.Generation, func(), error) {
		calls.free++
		if sp.LogitProcessor != nil || sp.LogitProcessorGate != nil {
			panic("phase 1 must run with no processor")
		}
		ch := make(chan int)
		go func() {
			defer close(ch)
			for i, id := range toyIDs(script) {
				if release != nil && i >= hold {
					<-release
				}
				select {
				case ch <- id:
				case <-ctx.Done():
					return
				}
			}
		}()
		return ch, nil, nil, nil
	}
}

// rankedConstrained is phase 2: a decoder over the toy vocabulary that, each step, asks the gate whether the next step is masked, applies
// the processor, and takes the best allowed byte by a fixed ranking (closers and digits first, whitespace last); unmasked, it ends
// the turn.
func rankedConstrained(calls *phaseCalls, budgetSeen *int) lazyPhase {
	const ranking = "}\"]0123456789abcdefghijklmnopqrstuvwxyz<>/_{:,[\n " // whitespace last: taken only where the grammar forces it
	return func(ctx context.Context, prompt []int, max int, sp decoder.SamplingParams) (<-chan int, *decoder.Generation, func(), error) {
		calls.constrained++
		*budgetSeen = max
		ch := make(chan int)
		go func() {
			defer close(ch)
			var gen []int
			for step := 0; max <= 0 || step < max; step++ {
				if !sp.LogitProcessorGate(gen) {
					return // the call is complete and the masker free: this toy model then stops
				}
				logits := make([]float32, toyEOS+1)
				for i := range logits {
					logits[i] = -1e9
				}
				for j := range len(ranking) {
					logits[ranking[j]] = float32(100 - j)
				}
				sp.LogitProcessor(gen, logits)
				best := 0
				for i := range logits {
					if logits[i] > logits[best] {
						best = i
					}
				}
				if best == toyEOS || logits[best] < -1e8 {
					return
				}
				gen = append(gen, best)
				select {
				case ch <- best:
				case <-ctx.Done():
					return
				}
			}
		}()
		return ch, nil, nil, nil
	}
}

func drainText(t *testing.T, s *Stream) string {
	t.Helper()
	var ids []int
	timeout := time.After(10 * time.Second)
	for {
		select {
		case id, ok := <-s.Tokens():
			if !ok {
				return toyText(ids)
			}
			ids = append(ids, id)
		case <-timeout:
			t.Fatalf("the turn did not end; got %q so far", toyText(ids))
		}
	}
}

// wellFormedCall reports whether text holds exactly one <tool_call> body that parses as a JSON call to get with object arguments.
func wellFormedCall(text string) bool {
	_, rest, ok := strings.Cut(text, "<tool_call>")
	if !ok {
		return false
	}
	body, _, ok := strings.Cut(rest, "</tool_call>")
	if !ok {
		return false
	}
	var call struct {
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
	}
	return json.Unmarshal([]byte(strings.TrimSpace(body)), &call) == nil && call.Name == "get" && call.Arguments != nil
}

// TestLazyTurn_malformedFreeCallComesBackWellFormed is R2's red-first gate (docs/tasks/task-tool-grammar-union-2026-09.md): a model whose
// free-running call is malformed (a trailing comma, so not JSON), on a speculating server under tool_choice auto. Today that server skips
// the union, so the turn is the free run alone and its call does not parse; through lazyTurn the call is constrained from its opener and
// parses. Nothing phase 1 drafted past the opener reaches the caller.
func TestLazyTurn_malformedFreeCallComesBackWellFormed(t *testing.T) {
	const script = "Let me check. <tool_call>\n{\"name\": \"get\", \"arguments\": {\"x\": 1,}}\n</tool_call>"
	var today phaseCalls
	free := scriptedFree(script, &today, 0, nil)
	s, _, _, _ := free(context.Background(), nil, 0, decoder.SamplingParams{})
	var ids []int
	for id := range s {
		ids = append(ids, id)
	}
	if wellFormedCall(toyText(ids)) {
		t.Fatal("the toy's free-running call parses; the test needs a malformed one")
	}
	var calls phaseCalls
	var budget int
	req := Request{PromptIDs: toyIDs("prompt"), MaxTokens: 200}
	out := drainText(t, lazyTurn(context.Background(), req, toyLazy(t), scriptedFree(script, &calls, 0, nil), rankedConstrained(&calls, &budget)))
	if !wellFormedCall(out) {
		t.Fatalf("through lazyTurn the call is still malformed: %q", out)
	}
	if !strings.HasPrefix(out, "Let me check. <tool_call>") || strings.Contains(out, "1,}") {
		t.Fatalf("output %q: phase 1's text before the opener must be kept and nothing it drafted after it forwarded", out)
	}
	if calls.free != 1 || calls.constrained != 1 {
		t.Fatalf("phase starts free=%d constrained=%d, want 1 and 1", calls.free, calls.constrained)
	}
	if want := 200 - len("Let me check. <tool_call>"); budget != want {
		t.Fatalf("phase 2's budget %d, want %d (the turn's less what phase 1 emitted)", budget, want)
	}
}

// TestLazyTurn_proseIsTheFreeRun holds a prose answer on a speculating server: with the union it is bit-identical to the free run without
// it, and only the speculative path ran (the drafter's phase started, the constrained one never), including a prose answer that starts
// like an opener ("<tool" then not a call), whose every byte comes through.
func TestLazyTurn_proseIsTheFreeRun(t *testing.T) {
	for _, script := range []string{"The answer is 42.", "Use <tools> wisely; <tool_ca is not a call."} {
		var calls phaseCalls
		var budget int
		out := drainText(t, lazyTurn(context.Background(), Request{MaxTokens: 0}, toyLazy(t), scriptedFree(script, &calls, 0, nil), rankedConstrained(&calls, &budget)))
		if out != script {
			t.Fatalf("prose %q came back %q", script, out)
		}
		if calls.free != 1 || calls.constrained != 0 {
			t.Fatalf("%q: phase starts free=%d constrained=%d, want the speculative path alone", script, calls.free, calls.constrained)
		}
	}
}

// TestLazyTurn_noHoldback holds that nothing is held back: the bytes of a possible opener reach the caller while phase 1 is still
// waiting to produce the next token, and a caller that stops early (a stop sequence) ends the turn with nothing left running.
func TestLazyTurn_noHoldback(t *testing.T) {
	const script = "x <tool and more"
	release := make(chan struct{})
	var calls phaseCalls
	var budget int
	ctx, cancel := context.WithCancel(context.Background())
	s := lazyTurn(ctx, Request{}, toyLazy(t), scriptedFree(script, &calls, len("x <tool"), release), rankedConstrained(&calls, &budget))
	var got []int
	for len(got) < len("x <tool") {
		select {
		case id := <-s.Tokens():
			got = append(got, id)
		case <-time.After(5 * time.Second):
			t.Fatalf("only %q reached the caller while phase 1 waited: bytes were held back", toyText(got))
		}
	}
	if toyText(got) != "x <tool" {
		t.Fatalf("got %q", toyText(got))
	}
	cancel() // the caller stops here, as at a stop sequence
	close(release)
	select {
	case <-drainDone(s):
	case <-time.After(5 * time.Second):
		t.Fatal("the turn did not end after the caller stopped")
	}
	if calls.constrained != 0 {
		t.Fatal("a stopped prose turn started the constrained phase")
	}
}

func drainDone(s *Stream) <-chan struct{} {
	done := make(chan struct{})
	go func() {
		for range s.Tokens() {
		}
		close(done)
	}()
	return done
}

// onePool is a SessionPool of fresh sessions on m.
type onePool struct{ m *decoder.Model }

func (p onePool) Acquire([]int) *decoder.Session { return p.m.NewSession(256) }
func (p onePool) Checkin(*decoder.Session)       {}

// TestLazyTurn_realSpeculativeProseKeepsTheDrafter holds gates (1) and (3) on a real model: llama-tiny on a CPU session with n-gram
// speculation, greedy, a prompt that repeats so the drafter has matches. A prose turn through lazyTurn (a lazy union whose token table
// maps no token to '<', so it never arms) emits exactly the plain speculative turn's tokens, its phase 1 is the speculative path with
// tokens drafted (Generation.Spec), and the constrained phase never starts.
func TestLazyTurn_realSpeculativeProseKeepsTheDrafter(t *testing.T) {
	m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{})
	if err != nil {
		t.Skipf("no llama-tiny fixture: %v", err)
	}
	defer m.Close()
	_, _, _, _, _, _, vocab := m.Dims()
	d := Local{Model: m, Pool: onePool{m}, Speculative: true}
	prompt := []int{5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6}
	req := Request{PromptIDs: prompt, MaxTokens: 24}

	var want []int
	plain := d.Start(context.Background(), req)
	for id := range plain.Tokens() {
		want = append(want, id)
	}
	plain.Close()

	tb := make([][]byte, vocab)
	for i := range tb {
		tb[i] = []byte("t ")
	}
	g, err := constrain.ToolCallsGrammar("<tool_call>", "\n</tool_call>", "arguments", false, []constrain.ToolSpec{{Name: "get", Parameters: []byte(`{"type":"object"}`)}})
	if err != nil {
		t.Fatal(err)
	}
	lazy := constrain.NewLazyMasker(constrain.NewMasker(g, tb, nil), "<tool_call>")
	var phase1 *decoder.Generation
	constrainedStarts := 0
	free := func(ctx context.Context, p []int, max int, sp decoder.SamplingParams) (<-chan int, *decoder.Generation, func(), error) {
		s, gen, done, err := d.lazyFree(ctx, p, max, sp)
		phase1 = gen
		return s, gen, done, err
	}
	constrained := func(ctx context.Context, p []int, max int, sp decoder.SamplingParams) (<-chan int, *decoder.Generation, func(), error) {
		constrainedStarts++
		return d.lazyConstrained(ctx, p, max, sp)
	}
	var got []int
	for id := range lazyTurn(context.Background(), req, lazy, free, constrained).Tokens() {
		got = append(got, id)
	}
	if len(got) != len(want) {
		t.Fatalf("the lazy turn emitted %d tokens, the plain speculative turn %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d: lazy turn %d, plain speculative turn %d", i, got[i], want[i])
		}
	}
	if constrainedStarts != 0 {
		t.Fatal("a prose turn started the constrained phase")
	}
	if phase1 == nil || phase1.Spec == nil || phase1.Spec.Drafted == 0 {
		t.Fatal("phase 1 drafted nothing: the turn did not keep the drafter")
	}
	t.Logf("%d tokens, identical to plain speculation; phase 1 drafted %d, accepted %d", len(got), phase1.Spec.Drafted, phase1.Spec.Accepted)
}

// toyModelPhase is one toy model as either phase: at turn position i (counted from origLen, the turn's prompt length) it prefers
// script[i], then the fixed ranking, then EOS once the script is spent; it applies sp's gate and processor when set, as a decoder does.
func toyModelPhase(script string, origLen int, calls *phaseCalls, isFree bool) lazyPhase {
	const ranking = "}\"]0123456789abcdefghijklmnopqrstuvwxyz<>/_{:,[\n "
	return func(ctx context.Context, prompt []int, max int, sp decoder.SamplingParams) (<-chan int, *decoder.Generation, func(), error) {
		if isFree {
			calls.free++
		} else {
			calls.constrained++
		}
		ch := make(chan int)
		go func() {
			defer close(ch)
			var gen []int
			for step := 0; max <= 0 || step < max; step++ {
				pos := len(prompt) - origLen + step
				logits := make([]float32, toyEOS+1)
				for i := range logits {
					logits[i] = -1e9
				}
				for j := range len(ranking) {
					logits[ranking[j]] = float32(100 - j)
				}
				if pos < len(script) {
					logits[script[pos]] = 200
				} else {
					logits[toyEOS] = 300
				}
				if sp.LogitProcessorGate == nil || sp.LogitProcessorGate(gen) {
					if sp.LogitProcessor != nil {
						sp.LogitProcessor(gen, logits)
					}
				}
				best := 0
				for i := range logits {
					if logits[i] > logits[best] {
						best = i
					}
				}
				if best == toyEOS || logits[best] < -1e8 {
					return
				}
				gen = append(gen, best)
				select {
				case ch <- best:
				case <-ctx.Done():
					return
				}
			}
		}()
		return ch, nil, nil, nil
	}
}

// TestLazyTurn_sameOutputWithAndWithoutSpeculation is gate (2): the same toy model, with a malformed call after prose, gives the same
// turn through lazyTurn (phase 1 free, phase 2 constrained) as a non-speculating server's lazy auto turn (the gated processor from token
// 1), before the opener and after it, bit for bit; a prose turn too.
func TestLazyTurn_sameOutputWithAndWithoutSpeculation(t *testing.T) {
	for _, script := range []string{
		"Let me check. <tool_call>\n{\"name\": \"get\", \"arguments\": {\"x\": 1,}}\n</tool_call>",
		"Just prose, no call.",
	} {
		prompt := toyIDs("p")
		req := Request{PromptIDs: prompt, MaxTokens: 120}
		var a, b phaseCalls
		spec := drainText(t, lazyTurn(context.Background(), req, toyLazy(t), toyModelPhase(script, len(prompt), &a, true), toyModelPhase(script, len(prompt), &a, false)))
		lazy := toyLazy(t)
		sp := decoder.SamplingParams{LogitProcessor: lazy.Process, LogitProcessorGate: lazy.Gate}
		s, _, _, _ := toyModelPhase(script, len(prompt), &b, false)(context.Background(), prompt, 120, sp)
		var ids []int
		for id := range s {
			ids = append(ids, id)
		}
		if plain := toyText(ids); spec != plain {
			t.Fatalf("script %q:\n with speculation    %q\n without speculation %q", script, spec, plain)
		}
		if strings.Contains(script, "<tool_call>") && !wellFormedCall(spec) {
			t.Fatalf("script %q: the armed turn's call is not well formed: %q", script, spec)
		}
	}
}
