package serveapp

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestSessionLRU_sharesAPreamble is the CPU's copy through its caller, serve's session LRU
// (docs/measurements/kv2-2026-10-14.md, G3 and G4(c)): a new conversation that opens like an earlier one gets a fresh
// session holding the earlier one's shared whole blocks, and replies as it does cold; with no room, the session the LRU
// would evict, when it holds the hit, is taken in place and cut back to the hit; a busy session is never read; and
// with the model's sharing off nothing is copied.
func TestSessionLRU_sharesAPreamble(t *testing.T) {
	dir := filepath.Join("..", "..", "testdata", "llama-attnfa-tiny")
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("no fixture at %s", dir)
	}
	load := func(off bool) *decoder.Model {
		m, err := decoder.Load(dir, decoder.Options{Backend: "cpu", KVShareOff: off})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		return m
	}
	m := load(false)
	vocab := m.Config().VocabSize
	// A has more history of its own than the shared preamble, so serve's own rule (pickSession) does not give B A's
	// session: B would keep less of it than it overwrites.
	prompt := func(c int) []int {
		p := make([]int, 0, 2400)
		for i := range 1100 {
			p = append(p, (i*37+11)%vocab)
		}
		for i := range 600 + 700*min(1, abs(1-c)) {
			p = append(p, (c*61+i*7+5)%vocab)
		}
		return p
	}
	gen := func(s *decoder.Session, p []int) ([]int, *decoder.Generation) {
		ch, g := s.Generate(context.Background(), p, 6, decoder.SamplingParams{})
		var ids []int
		for id := range ch {
			ids = append(ids, id)
		}
		if g.Err() != nil {
			t.Fatal(g.Err())
		}
		return ids, g
	}
	cold, _ := gen(m.NewSession(0), prompt(1))

	l := newSessionLRU(m, 2, 0, "fp")
	a := l.acquire(prompt(0))
	gen(a, prompt(0))
	l.checkin(a)
	b := l.acquire(prompt(1))
	if b == a || len(b.Tokens()) != 1088 {
		t.Fatalf("room in the LRU: B got A (%v) or a session holding %d tokens, want a fresh one holding 1,088", b == a, len(b.Tokens()))
	}
	got, g := gen(b, prompt(1))
	if g.PrefillShared != 1088 || !slices.Equal(got, cold) {
		t.Errorf("B: shared %d, %v; cold %v", g.PrefillShared, got, cold)
	}
	l.checkin(b)

	// Full, two sessions sharing the preamble (so serve's own rule gives B neither): the least recently used, A, is
	// both the session fresh would evict and a donor, so B takes it in place, cut back to the hit.
	full := newSessionLRU(m, 2, 0, "fp")
	a = full.acquire(prompt(0))
	gen(a, prompt(0))
	full.checkin(a)
	c := full.acquire(prompt(2))
	gen(c, prompt(2))
	full.checkin(c)
	b = full.acquire(prompt(1))
	if b != a || len(b.Tokens()) != 1088 {
		t.Fatalf("a full LRU whose victim holds the hit: B got A %v (C %v), holding %d tokens, want A cut back to 1,088", b == a, b == c, len(b.Tokens()))
	}
	if got, _ := gen(b, prompt(1)); !slices.Equal(got, cold) {
		t.Errorf("B in place: %v, cold %v", got, cold)
	}
	full.checkin(b)

	busy := newSessionLRU(m, 2, 0, "fp")
	a = busy.acquire(prompt(0))
	gen(a, prompt(0)) // still checked out
	if b = busy.acquire(prompt(1)); len(b.Tokens()) != 0 {
		t.Errorf("a busy donor was read: B holds %d tokens", len(b.Tokens()))
	}

	off := load(true)
	lo := newSessionLRU(off, 2, 0, "fp")
	a = lo.acquire(prompt(0))
	gen(a, prompt(0))
	lo.checkin(a)
	if b = lo.acquire(prompt(1)); len(b.Tokens()) != 0 {
		t.Errorf("sharing off: B holds %d copied tokens", len(b.Tokens()))
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
