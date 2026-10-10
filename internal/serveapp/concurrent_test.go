package serveapp

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestAdmission_capN: with capacity N (MC3c, -max-concurrent), N holders enter without waiting, the next waits, and a
// release hands its turn to the front of the queue in arrival order.
func TestAdmission_capN(t *testing.T) {
	var a admission
	a.setCap(2)
	ctx := context.Background()
	for i := range 2 {
		if !a.enter(ctx, admissionRecord{}) {
			t.Fatalf("holder %d must enter immediately at capacity 2", i+1)
		}
	}
	order := make(chan int, 2)
	var wg sync.WaitGroup
	for i := 1; i <= 2; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if a.enter(ctx, admissionRecord{}) {
				order <- i
			}
		}(i)
		time.Sleep(20 * time.Millisecond) // waiter 1 queues before waiter 2
	}
	select {
	case i := <-order:
		t.Fatalf("waiter %d entered while both turns were held", i)
	case <-time.After(50 * time.Millisecond):
	}
	a.release()
	if got := <-order; got != 1 {
		t.Fatalf("first release admitted waiter %d, want 1 (FIFO)", got)
	}
	a.release()
	if got := <-order; got != 2 {
		t.Fatalf("second release admitted waiter %d, want 2", got)
	}
	wg.Wait()
	a.release()
	a.release()
	if !a.enter(ctx, admissionRecord{}) {
		t.Fatal("after every release a turn must be free again")
	}
}

// TestSessionLRU_checkout: a session a generation holds is never handed to another, a full LRU whose sessions are all
// busy serves a transient session instead of taking one, checkin makes a session reusable, and save skips a busy one.
func TestSessionLRU_checkout(t *testing.T) {
	m, err := decoder.Load(buildSyntheticBase(t), decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	l := newSessionLRU(m, 2, 0, "fp")
	commit := func(s *decoder.Session, prompt []int) {
		ch, gen := s.Generate(context.Background(), prompt, 3, decoder.SamplingParams{})
		for range ch {
		}
		if gen.Err() != nil {
			t.Fatalf("generate: %v", gen.Err())
		}
	}
	a := []int{1, 4, 2, 7, 3}
	s1 := l.acquire(a)
	commit(s1, a)
	// s1 is still checked out: a continuation of the same conversation must NOT get it
	s2 := l.acquire(append(slices.Clone(s1.Tokens()), 5))
	if s2 == s1 {
		t.Fatal("acquire handed out a session another generation holds")
	}
	// both resident sessions busy and the LRU full: a transient session, never one of them
	s3 := l.acquire([]int{9, 9, 9})
	if s3 == s1 || s3 == s2 || slices.Contains(l.order, s3) {
		t.Fatal("with every session busy, acquire must serve a transient session outside the LRU")
	}
	l.checkin(s3)
	if len(l.order) != 2 {
		t.Fatalf("a transient session joined the LRU: %d sessions", len(l.order))
	}
	// save skips a busy session
	dir := t.TempDir()
	commit(s2, append(slices.Clone(s1.Tokens()), 5))
	l.checkin(s2)
	if err := l.save(dir); err != nil {
		t.Fatalf("save: %v", err)
	}
	files, _ := filepath.Glob(filepath.Join(dir, "session-*"))
	if len(files) != 1 {
		t.Errorf("save wrote %d snapshots with one session busy, want 1 (the idle one)", len(files))
	}
	// once checked in, the conversation's own session is reused
	l.checkin(s1)
	cont := append(slices.Clone(s1.Tokens()), 6)
	if got := l.acquire(cont); got != s1 {
		t.Error("after checkin, the continuation did not reuse its own session")
	}
}

// TestServe_maxConcurrentMatchesAlone is MC3c's gate 1 (docs/tasks/task-concurrency-2026-09.md) through the real
// request path: 4 multi-turn conversations driven through drive() at the same time, on one CPU model with
// -max-concurrent 4, each produce byte-identical text to the same conversation driven alone, and reuse their own
// history on every turn. It needs a real checkpoint with a tokenizer: GOINFER_MC3C_MODEL (a .gguf), skipped without.
func TestServe_maxConcurrentMatchesAlone(t *testing.T) {
	path := os.Getenv("GOINFER_MC3C_MODEL")
	if path == "" {
		t.Skip("set GOINFER_MC3C_MODEL to a .gguf checkpoint")
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	const nConv, turns, maxTok = 4, 3, 24
	newLM := func(concurrent int) *loadedModel {
		lm := &loadedModel{tk: tk, model: m, name: "m", fp: "fp", sessions: newSessionLRU(m, 4, 0, "fp")}
		lm.setConcurrency(config{maxConcurrent: concurrent, kvSessions: 4})
		return lm
	}
	type turn struct {
		text   string
		reused int
	}
	conv := func(lm *loadedModel, c int) []turn {
		prompt, _ := tk.Encode("You are a helpful assistant. Conversation "+string(rune('A'+c))+": tell me about rate limiters.", false)
		var out []turn
		for tn := range turns {
			ok, _ := lm.tryEnter(context.Background(), admissionRecord{}, nil)
			if !ok {
				t.Errorf("conversation %d turn %d: not admitted", c, tn)
				return out
			}
			var sb strings.Builder
			_, _, _, _, reused, _, derr := lm.drive(context.Background(),
				genRequest{promptIDs: prompt, maxTokens: maxTok}, nil, nil, func(s string) { sb.WriteString(s) })
			lm.exit()
			if derr != nil {
				t.Errorf("conversation %d turn %d: %v", c, tn, derr)
				return out
			}
			out = append(out, turn{sb.String(), reused})
			next, _ := tk.Encode(sb.String()+" And then?", false)
			prompt = append(slices.Clone(prompt), next...)
		}
		return out
	}
	alone := make([][]turn, nConv)
	for c := range nConv {
		alone[c] = conv(newLM(1), c)
	}
	lm := newLM(4)
	if lm.concurrent != 4 {
		t.Fatalf("setConcurrency gave %d, want 4 (model CPUConcurrentSafe=%v)", lm.concurrent, m.CPUConcurrentSafe())
	}
	together := make([][]turn, nConv)
	var wg sync.WaitGroup
	for c := range nConv {
		wg.Add(1)
		go func(c int) { defer wg.Done(); together[c] = conv(lm, c) }(c)
	}
	wg.Wait()
	for c := range nConv {
		for tn := range alone[c] {
			if tn >= len(together[c]) {
				t.Fatalf("conversation %d: only %d turns concurrently", c, len(together[c]))
			}
			a, b := alone[c][tn], together[c][tn]
			if a.text != b.text {
				t.Errorf("conversation %d turn %d: concurrent text differs from alone:\n alone    %q\n together %q", c, tn, a.text, b.text)
			}
			if tn > 0 && b.reused == 0 {
				t.Errorf("conversation %d turn %d: reused nothing concurrently (alone reused %d)", c, tn, a.reused)
			}
		}
	}
}

// TestAdmission_load: load counts holders plus waiters — what the prefill-memory share reads, so a lone request sees 0
// ahead of it and keeps the whole margin.
func TestAdmission_load(t *testing.T) {
	var a admission
	a.setCap(1)
	if a.load() != 0 {
		t.Fatalf("idle load = %d, want 0", a.load())
	}
	ctx := context.Background()
	a.enter(ctx, admissionRecord{})
	done := make(chan struct{})
	go func() { a.enter(ctx, admissionRecord{}); close(done) }()
	for a.load() != 2 {
		time.Sleep(time.Millisecond)
	}
	a.release()
	<-done
	if a.load() != 1 {
		t.Fatalf("after the hand-off load = %d, want 1", a.load())
	}
	a.release()
}

// TestSetConcurrency_reportsTheDecidedValue: the concurrency line serve prints comes from setConcurrency
// itself, after it has decided, not from the load banner, which runs before and would read the value before
// setConcurrency set it.
func TestSetConcurrency_reportsTheDecidedValue(t *testing.T) {
	m, err := decoder.Load(buildSyntheticBase(t), decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if !m.CPUConcurrentSafe() {
		t.Skip("synthetic base is not CPU-concurrent-safe")
	}
	lm := &loadedModel{model: m, name: "m", fp: "fp", sessions: newSessionLRU(m, 4, 0, "fp")}
	line := lm.setConcurrency(config{maxConcurrent: 4, kvSessions: 4})
	if lm.concurrent != 4 || !strings.Contains(line, "4 generations at once") {
		t.Errorf("setConcurrency decided %d and reported %q", lm.concurrent, line)
	}
	if line := lm.setConcurrency(config{maxConcurrent: 1, kvSessions: 4}); lm.concurrent != 1 || line != "" {
		t.Errorf("-max-concurrent 1: decided %d, reported %q (want 1 and nothing)", lm.concurrent, line)
	}
}
