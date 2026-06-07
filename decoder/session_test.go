package decoder

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"slices"
	"testing"
)

// testModelPath picks a checkpoint for the model-gated session tests: the
// GOINFER_TEST_MODEL override (point it at any .gguf or HF dir) else the same
// gemma-3-270m fixture the other tests use. Tests skip if it's absent.
func testModelPath() string {
	if p := os.Getenv("GOINFER_TEST_MODEL"); p != "" {
		return p
	}
	return gemmaModelDir
}

func loadTestModel(t *testing.T) *Model {
	t.Helper()
	path := testModelPath()
	if _, err := os.Stat(path); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no model at %s — set GOINFER_TEST_MODEL to a .gguf/checkpoint", path)
	}
	m, err := Load(path, Options{})
	if err != nil {
		t.Fatalf("Load(%s): %v", path, err)
	}
	return m
}

// gen runs a (stream, *Generation)-returning Generate via the thunk and drains
// the stream, failing on a terminal error. The thunk keeps the two-value call
// out of an argument list (Go won't spread it alongside t).
func gen(t *testing.T, run func() (<-chan int, *Generation)) []int {
	t.Helper()
	stream, g := run()
	var ids []int
	for id := range stream {
		ids = append(ids, id)
	}
	if err := g.Err(); err != nil {
		t.Fatalf("generation error: %v", err)
	}
	return ids
}

// TestKVCache_TruncateTo_kvShared reproduces Gemma 4's awkward cache shape and
// asserts TruncateTo rewinds each layer correctly. Two Gemma 4 facts are baked
// in: (1) a KV-shared tail layer that never Appends (stays empty), and (2)
// layers with DIFFERENT per-position widths (per-layer head_dim) — wider than
// the cache's nominal kvDim. A truncation that used pos*kvDim uniformly would
// slice the wide layer to the wrong length (the bug that made Gemma 4 reuse
// diverge) and slice past the empty shared layer (a panic). Always runs (no
// model needed); the standing guard for the Gemma 4 prefix-reuse path.
func TestKVCache_TruncateTo_kvShared(t *testing.T) {
	const layers, kvHeads, headDim = 3, 1, 4
	kvDim := kvHeads * headDim // nominal = 4
	c := NewKVCache(layers, kvHeads, headDim, 0, 8)
	c.manualPos = true                  // gemma4 advances pos explicitly
	narrow := []float32{1, 2, 3, 4}     // layer 0: width 4 (== nominal kvDim)
	wide := []float32{1, 2, 3, 4, 5, 6} // layer 1: width 6 (per-layer head_dim ≠ nominal)
	for p := 0; p < 5; p++ {
		c.Append(0, narrow, narrow)
		c.Append(1, wide, wide)
		// layer 2 is KV-shared: intentionally never appended (stays empty).
		c.Advance()
	}
	if c.Pos() != 5 {
		t.Fatalf("pos = %d, want 5", c.Pos())
	}
	if got := len(c.Keys(2)); got != 0 {
		t.Fatalf("KV-shared layer 2 len = %d, want 0", got)
	}

	c.TruncateTo(3) // must not panic and must use each layer's own width
	if c.Pos() != 3 {
		t.Fatalf("after TruncateTo(3): pos = %d, want 3", c.Pos())
	}
	if got := len(c.Keys(0)); got != 3*kvDim {
		t.Fatalf("narrow layer 0 len = %d, want %d", got, 3*kvDim)
	}
	if got := len(c.Keys(1)); got != 3*6 {
		t.Fatalf("wide layer 1 len = %d, want %d (per-layer width respected)", got, 3*6)
	}
	if got := len(c.Keys(2)); got != 0 {
		t.Fatalf("shared layer 2 len = %d, want 0", got)
	}

	c.TruncateTo(0)
	if c.Pos() != 0 || len(c.Keys(0)) != 0 || len(c.Keys(1)) != 0 {
		t.Fatalf("after TruncateTo(0): pos=%d len0=%d len1=%d, want 0/0/0", c.Pos(), len(c.Keys(0)), len(c.Keys(1)))
	}
}

func TestCommonPrefixLen(t *testing.T) {
	cases := []struct {
		a, b []int
		want int
	}{
		{nil, nil, 0},
		{[]int{1, 2, 3}, nil, 0},
		{[]int{1, 2, 3}, []int{1, 2, 3}, 3},
		{[]int{1, 2, 3}, []int{1, 2, 3, 4, 5}, 3},
		{[]int{1, 2, 9}, []int{1, 2, 3}, 2},
		{[]int{9}, []int{1}, 0},
	}
	for _, c := range cases {
		if got := commonPrefixLen(c.a, c.b); got != c.want {
			t.Errorf("commonPrefixLen(%v,%v) = %d, want %d", c.a, c.b, got, c.want)
		}
	}
}

// TestSession_reuseParity is the correctness gate for prefix reuse: a Session
// that reuses cached KV must produce bit-identical greedy output to a cold,
// full-prefill Model.Generate. Covers both the pure-extension path (continuing a
// conversation) and the divergent-suffix path (TruncateTo rollback to a shared
// prefix, then re-prefill).
func TestSession_reuseParity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: greedy decode on the naive backend")
	}
	m := loadTestModel(t)
	ctx := context.Background()
	greedy := SamplingParams{} // Temperature 0 ⇒ deterministic argmax

	prompt := []int{1, 100, 200, 300, 400}

	// --- pure extension: generate, then continue from the generated tail ---
	sess := m.NewSession(0)
	first := gen(t, func() (<-chan int, *Generation) { return sess.Generate(ctx, prompt, 6, greedy) })
	if len(first) == 0 {
		t.Fatal("first generation produced no tokens")
	}
	ext := append(append([]int(nil), prompt...), first...) // prompt + generated
	reused := gen(t, func() (<-chan int, *Generation) { return sess.Generate(ctx, ext, 6, greedy) })
	cold := gen(t, func() (<-chan int, *Generation) { return m.Generate(ctx, ext, 6, greedy) })
	if !slices.Equal(reused, cold) {
		t.Fatalf("extension reuse diverged from cold prefill:\n  reused %v\n  cold   %v", reused, cold)
	}

	// --- divergent suffix: reuse the shared prefix, re-prefill a different tail ---
	sess2 := m.NewSession(0)
	gen(t, func() (<-chan int, *Generation) { return sess2.Generate(ctx, prompt, 4, greedy) }) // populate
	diverged := []int{1, 100, 200, 999, 888, 777}                                              // shares prompt[:3], then differs
	got := gen(t, func() (<-chan int, *Generation) { return sess2.Generate(ctx, diverged, 6, greedy) })
	want := gen(t, func() (<-chan int, *Generation) { return m.Generate(ctx, diverged, 6, greedy) })
	if !slices.Equal(got, want) {
		t.Fatalf("divergent-suffix reuse diverged from cold prefill:\n  got  %v\n  want %v", got, want)
	}
}

// TestSession_snapshotRoundTrip serializes a prefilled session, reloads it, and
// asserts (a) the token list survives and (b) the restored KV is genuinely
// correct — continuing from the reloaded session matches a cold prefill of the
// same sequence.
func TestSession_snapshotRoundTrip(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: greedy decode on the naive backend")
	}
	m := loadTestModel(t)
	ctx := context.Background()
	greedy := SamplingParams{}

	prompt := []int{1, 100, 200, 300, 400}
	sess := m.NewSession(0)
	gen(t, func() (<-chan int, *Generation) { return sess.Generate(ctx, prompt, 5, greedy) })
	seq := append([]int(nil), sess.Tokens()...)

	blob := sess.Snapshot("roundtrip")
	restored, err := m.LoadSession(blob, "roundtrip") // matching id passes the identity guard
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if !slices.Equal(restored.Tokens(), seq) {
		t.Fatalf("restored tokens differ:\n  got  %v\n  want %v", restored.Tokens(), seq)
	}

	// Continue from the restored KV (reuses the whole reloaded prefix) and from a
	// cold prefill of the identical sequence — they must agree, proving the KV
	// bytes restored faithfully.
	fromRestored := gen(t, func() (<-chan int, *Generation) { return restored.Generate(ctx, seq, 5, greedy) })
	cold := gen(t, func() (<-chan int, *Generation) { return m.Generate(ctx, seq, 5, greedy) })
	if !slices.Equal(fromRestored, cold) {
		t.Fatalf("restored-KV continuation diverged from cold:\n  restored %v\n  cold     %v", fromRestored, cold)
	}
}

// TestLoadSession_rejectsCorrupt confirms the CRC/geometry guards reject a
// tampered or foreign blob with a typed *SnapshotError rather than panicking.
func TestLoadSession_rejectsCorrupt(t *testing.T) {
	m := loadTestModel(t)
	sess := m.NewSession(0)
	// Hand-build a tiny session state (no generation needed for the guard test).
	sess.tokens = []int{1, 2, 3}
	sess.cache.pos = 0 // empty KV is fine; the guards fire on header/CRC
	blob := sess.Snapshot("model-A")

	if _, err := m.LoadSession(blob, ""); err != nil {
		t.Fatalf("clean blob, no id check should load: %v", err)
	}
	if _, err := m.LoadSession(blob, "model-A"); err != nil {
		t.Fatalf("clean blob, matching id should load: %v", err)
	}
	var se *SnapshotError
	if _, err := m.LoadSession(blob, "model-B"); !errors.As(err, &se) {
		t.Fatalf("identity mismatch: got %v, want *SnapshotError", err)
	}
	bad := append([]byte(nil), blob...)
	bad[len(bad)-1] ^= 0xff // corrupt the CRC word
	if _, err := m.LoadSession(bad, ""); !errors.As(err, &se) {
		t.Fatalf("corrupt blob: got %v, want *SnapshotError", err)
	}
	if _, err := m.LoadSession([]byte("nope"), ""); !errors.As(err, &se) {
		t.Fatalf("garbage blob: got %v, want *SnapshotError", err)
	}
}
