package decoder

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"testing"
	"time"
)

// fakeDecode is plainDecodeLoop's target over a scriptedForward: the row is the fake's logits after the history, pick
// takes the argmax itself when device is set (a token the backend chose), and advance steps the fake, refusing with
// ErrKVPoolFull at the history length poolFullAt.
type fakeDecode struct {
	fw         *scriptedForward
	cur        []float32
	device     bool
	poolFullAt int
	finishes   [][]int
	commits    []int
}

func (d *fakeDecode) target() decodeTarget {
	return decodeTarget{
		pick: func(*decodeClock) (SampleInfo, int, bool, error) {
			if d.device {
				return SampleInfo{}, argmax(d.cur), true, nil
			}
			return SampleInfo{}, 0, false, nil
		},
		logits: func() []float32 { return d.cur },
		advance: func(next int, _ bool, _ *decodeClock) error {
			if d.poolFullAt > 0 && len(d.fw.hist) == d.poolFullAt {
				return ErrKVPoolFull
			}
			l, err := d.fw.Step(context.Background(), next)
			d.cur = l
			return err
		},
		finish: func(gen []int) { d.finishes = append(d.finishes, slices.Clone(gen)) },
	}
}

// referenceDecode is plain decoding written directly over a Forward: the processor (when the gate opens) masks the row,
// the sampler draws, a stop id ends the stream unemitted, and the token is stepped.
func referenceDecode(fw Forward, prompt []int, n int, sp SamplingParams, isStop func(int) bool) []int {
	ctx := context.Background()
	s := NewSampler(sp)
	defer s.release()
	s.Observe(prompt...)
	logits, _ := fw.Prefill(ctx, prompt)
	var gen []int
	open := sp.LogitProcessorGate == nil || sp.LogitProcessorGate(nil)
	for len(gen) < n {
		if sp.LogitProcessor != nil && open {
			sp.LogitProcessor(gen, logits)
		}
		info, _ := s.SampleWithInfo(logits)
		if isStop(info.ID) {
			break
		}
		gen = append(gen, info.ID)
		if sp.LogitProcessorGate != nil {
			open = sp.LogitProcessorGate(gen)
		}
		logits, _ = fw.Step(ctx, info.ID)
	}
	return gen
}

// fakeExit says how a fake decode is ended early: cancelled before it starts, cancelled from the commit callback once
// atCommit tokens are committed (so the next step's cancel check sees it), or with the reader stopping after readN
// tokens and cancelling while the loop waits to send the next.
type fakeExit struct {
	first    bool
	atCommit int
	readN    int
}

// runFakeDecode runs plainDecodeLoop on d and returns what was emitted, whether the loop completed, and the Generation.
func runFakeDecode(t *testing.T, d *fakeDecode, prompt []int, n int, sp SamplingParams, isStop func(int) bool, ex fakeExit) ([]int, bool, *Generation) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if ex.first {
		cancel()
	}
	s := NewSampler(sp)
	defer s.release()
	s.Observe(prompt...)
	d.cur, _ = d.fw.Prefill(ctx, prompt)
	out := make(chan int)
	g := &Generation{}
	done := make(chan bool, 1)
	commit := func(id int) {
		d.commits = append(d.commits, id)
		if ex.atCommit > 0 && len(d.commits) == ex.atCommit {
			cancel()
		}
	}
	go func() {
		done <- plainDecodeLoop(ctx, d.target(), out, g, s, sp, n, isStop, commit, &decodeClock{})
		close(out)
	}()
	var got []int
	for id := range out {
		got = append(got, id)
		if ex.readN > 0 && len(got) == ex.readN {
			// Stop reading: the loop draws the next token and blocks on its send, long before this cancel lands.
			time.AfterFunc(20*time.Millisecond, cancel)
			break
		}
	}
	return got, <-done, g
}

// TestPlainDecodeLoop_exactOnTheFake is SB5-P3 for plain decoding (docs/tasks/task-subsystems-2026-10.md §8 SB5): on a
// scripted fake, plainDecodeLoop emits what a direct decoding loop over the same fake emits, greedy and sampled, with a
// stop id, a processor, a gated processor and logprobs, at maxTokens 0, 1, 7 and 40; a device-chosen token bypasses the
// sampler; the commit callback sees each forwarded token once, in order, and finish sees what was emitted. A cancel
// before the start, after a commit and at a blocked send each end with one finish holding exactly the forwarded tokens;
// a KV-pool refusal ends with no commit for its token and no finish.
func TestPlainDecodeLoop_exactOnTheFake(t *testing.T) {
	const vocab, stop = 32, 0
	isStop := func(tok int) bool { return tok == stop }
	prompt := []int{5, 9, 2, 7}
	maskOdd := func(_ []int, l []float32) {
		for i := 1; i < len(l); i += 2 {
			l[i] = -1e30
		}
	}
	var gateCalls [][]int
	every3 := func(gen []int) bool {
		gateCalls = append(gateCalls, slices.Clone(gen))
		return len(gen)%3 == 0
	}
	scripts := map[string]func([]int) []float32{}
	for s := range uint64(3) {
		scripts[fmt.Sprintf("hash%d", s)] = hashScript(vocab, 0, s)
		scripts[fmt.Sprintf("stop%d", s)] = stopAt(hashScript(vocab, 3, s), stop, len(prompt)+5+int(s))
	}
	params := map[string]SamplingParams{
		"greedy":    {},
		"sampled":   {Temperature: 0.8, Seed: 7},
		"topk":      {Temperature: 0.9, TopK: 5, Seed: 11},
		"processor": {LogitProcessor: maskOdd},
		"gated":     {LogitProcessor: maskOdd, LogitProcessorGate: every3},
		"logprobs":  {Logprobs: true},
	}
	cases := 0
	for sn, script := range scripts {
		for pn, sp := range params {
			for _, n := range []int{0, 1, 7, 40} {
				name := fmt.Sprintf("%s/%s/n%d", sn, pn, n)
				want := referenceDecode(&scriptedForward{t: t, vocab: vocab, script: script}, prompt, n, sp, isStop)
				gateCalls = nil
				d := &fakeDecode{fw: &scriptedForward{t: t, vocab: vocab, script: script}}
				got, ok, g := runFakeDecode(t, d, prompt, n, sp, isStop, fakeExit{})
				cases++
				if !ok || g.err != nil {
					t.Fatalf("%s: the loop did not complete: %v", name, g.err)
				}
				if !slices.Equal(got, want) {
					t.Errorf("%s: emitted %v, direct decoding %v", name, got, want)
				}
				if !slices.Equal(d.commits, got) {
					t.Errorf("%s: committed %v, emitted %v", name, d.commits, got)
				}
				if len(d.finishes) != 1 || !slices.Equal(d.finishes[0], got) {
					t.Errorf("%s: finish saw %v, emitted %v", name, d.finishes, got)
				}
				if h := d.fw.hist[len(prompt):]; !slices.Equal(h, got) {
					t.Errorf("%s: the fake holds %v after the prompt, emitted %v", name, h, got)
				}
				if sp.Logprobs && len(g.Logprobs) != len(got) {
					t.Errorf("%s: %d logprobs for %d tokens", name, len(g.Logprobs), len(got))
				}
				if sp.LogitProcessorGate != nil {
					// The gate is asked first with nothing generated, then after each emitted token with the list so far.
					wantCalls := [][]int{nil}
					for i := range got {
						wantCalls = append(wantCalls, got[:i+1])
					}
					if len(gateCalls) != len(wantCalls) {
						t.Errorf("%s: the gate was asked %d times, want %d", name, len(gateCalls), len(wantCalls))
					} else {
						for i := range wantCalls {
							if !slices.Equal(gateCalls[i], wantCalls[i]) {
								t.Errorf("%s: gate call %d saw %v, want %v", name, i, gateCalls[i], wantCalls[i])
								break
							}
						}
					}
				}
			}
		}
	}

	// A token the backend chose (pick) is emitted as is, and the sampler is not asked: with a sampled config, the device's
	// argmax stream equals greedy decoding, which the host sampler would not give.
	for sn, script := range scripts {
		want := referenceDecode(&scriptedForward{t: t, vocab: vocab, script: script}, prompt, 40, SamplingParams{}, isStop)
		d := &fakeDecode{fw: &scriptedForward{t: t, vocab: vocab, script: script}, device: true}
		got, ok, _ := runFakeDecode(t, d, prompt, 40, SamplingParams{Temperature: 0.8, Seed: 7}, isStop, fakeExit{})
		cases++
		if !ok || !slices.Equal(got, want) {
			t.Errorf("%s/device: emitted %v, greedy %v", sn, got, want)
		}
	}

	script := hashScript(vocab, 0, 1)
	full := referenceDecode(&scriptedForward{t: t, vocab: vocab, script: script}, prompt, 40, SamplingParams{}, isStop)
	if len(full) < 12 {
		t.Fatalf("the reference stream is too short for the exit cases: %v", full)
	}
	// A cancel before the first step: nothing emitted, finish once with nothing.
	d := &fakeDecode{fw: &scriptedForward{t: t, vocab: vocab, script: script}}
	got, ok, g := runFakeDecode(t, d, prompt, 40, SamplingParams{}, isStop, fakeExit{first: true})
	if ok || !errors.Is(g.err, context.Canceled) || len(got) != 0 || len(d.finishes) != 1 || len(d.finishes[0]) != 0 {
		t.Errorf("cancel first: ok %v err %v emitted %v finishes %v", ok, g.err, got, d.finishes)
	}
	// A cancel after the 6th commit, seen by the next step's check: 6 emitted, committed and forwarded, one finish.
	// A cancel while the loop waits to send: the reader takes 6 tokens and stops, so the 7th is drawn and never sent or
	// forwarded. Both end with one finish holding exactly the 6 forwarded tokens, which the fake holds too.
	for _, ex := range []fakeExit{{atCommit: 6}, {readN: 6}} {
		d = &fakeDecode{fw: &scriptedForward{t: t, vocab: vocab, script: script}}
		got, ok, g = runFakeDecode(t, d, prompt, 40, SamplingParams{}, isStop, ex)
		cases++
		if ok || !errors.Is(g.err, context.Canceled) || !slices.Equal(got, full[:6]) {
			t.Errorf("cancel %+v: ok %v err %v emitted %v", ex, ok, g.err, got)
		}
		if fwd := d.fw.hist[len(prompt):]; len(d.finishes) != 1 || !slices.Equal(d.finishes[0], full[:6]) || !slices.Equal(d.commits, full[:6]) || !slices.Equal(fwd, full[:6]) {
			t.Errorf("cancel %+v: finishes %v, commits %v, the fake holds %v", ex, d.finishes, d.commits, fwd)
		}
	}
	// A KV-pool refusal on the 5th forward: that token was emitted but not committed, and there is no finish.
	d = &fakeDecode{fw: &scriptedForward{t: t, vocab: vocab, script: script}, poolFullAt: len(prompt) + 4}
	got, ok, g = runFakeDecode(t, d, prompt, 40, SamplingParams{}, isStop, fakeExit{})
	cases++
	if ok || g.err != nil || !slices.Equal(got, full[:5]) || !slices.Equal(d.commits, full[:4]) || len(d.finishes) != 0 {
		t.Errorf("pool full: ok %v err %v emitted %v commits %v finishes %v", ok, g.err, got, d.commits, d.finishes)
	}
	t.Logf("%d cases", cases)
}
