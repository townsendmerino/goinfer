package stores

import (
	"sync/atomic"
	"testing"
	"time"
)

func TestGenerationRegistry_cancelAndWaitEmpty(t *testing.T) {
	r := NewGenerationRegistry()
	var c1, c2 atomic.Int32
	g1 := r.Register("g1", "m1", func() { c1.Add(1) })
	g2 := r.Register("g2", "m2", func() { c2.Add(1) })
	g1.AddTokens(5)

	if r.Count() != 2 || len(r.List()) != 2 {
		t.Fatalf("count=%d list=%d, want 2", r.Count(), len(r.List()))
	}
	if !r.Cancel("g1", "user stop") || r.Cancel("missing", "x") {
		t.Fatal("Cancel return value mismatch")
	}
	g1.CancelOnce("later reason")
	if g1.Reason() != "user stop" || c1.Load() != 2 {
		t.Fatalf("Reason=%q cancels=%d, want first reason preserved and cancel invoked", g1.Reason(), c1.Load())
	}
	if g2.Reason() != "" {
		t.Fatalf("untouched generation has Reason=%q", g2.Reason())
	}
	if n := r.CancelAll("halt"); n != 2 || g2.Reason() != "halt" || g1.Reason() != "user stop" {
		t.Fatalf("CancelAll=%d g1=%q g2=%q", n, g1.Reason(), g2.Reason())
	}
	r.Remove("g1")
	r.Remove("g2")
	if d := r.WaitEmpty(50 * time.Millisecond); d >= 50*time.Millisecond || r.Count() != 0 {
		t.Fatalf("WaitEmpty=%v count=%d, want immediate return at 0", d, r.Count())
	}
}
