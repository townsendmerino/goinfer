package decoder

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

// G40-1 (docs/tasks/task-cpu-prefill-memory-2026-10.md §4): a text prefill run in chunks equals the single pass bit for
// bit, in its last-row logits and in every K/V element of every layer (the ring contents of a sliding-window layer
// included), cold and warm, under fast attention and under the exact path.

// prefillChunkArm prefills suffix into a fresh cache, after prefix (built as one pass in every arm, so both arms start
// from the same cache), with prefillChunkRows set to chunk for the suffix.
func prefillChunkArm(t *testing.T, m *Model, prefix, suffix []int, chunk int) ([]float32, *KVCache) {
	t.Helper()
	prev := prefillChunkRows
	defer func() { prefillChunkRows = prev }()
	c := m.NewCache(0)
	if len(prefix) > 0 {
		prefillChunkRows = 1 << 30
		if _, err := m.prefillLogits(context.Background(), prefix, c); err != nil {
			t.Fatalf("prefix: %v", err)
		}
	}
	prefillChunkRows = chunk
	lg, err := m.prefillLogits(context.Background(), suffix, c)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}
	return append([]float32(nil), lg...), c
}

// kvDiff counts the K/V elements (and ring slots) that differ between two caches, and names the first.
func kvDiff(a, b *KVCache) (int, string) {
	n, first := 0, ""
	note := func(what string) {
		if n++; first == "" {
			first = what
		}
	}
	f32 := func(x, y []float32, what string) {
		if len(x) != len(y) {
			note(fmt.Sprintf("%s length %d vs %d", what, len(x), len(y)))
			return
		}
		for i := range x {
			if math.Float32bits(x[i]) != math.Float32bits(y[i]) {
				note(fmt.Sprintf("%s[%d]", what, i))
			}
		}
	}
	if a.pos != b.pos {
		note(fmt.Sprintf("pos %d vs %d", a.pos, b.pos))
	}
	for l := range a.numLayers {
		f32(a.keys[l], b.keys[l], fmt.Sprintf("layer %d keys", l))
		f32(a.vals[l], b.vals[l], fmt.Sprintf("layer %d vals", l))
		if ra, rb := a.rings[l], b.rings[l]; ra != nil || rb != nil {
			if ra == nil || rb == nil || ra.count != rb.count {
				note(fmt.Sprintf("layer %d ring", l))
				continue
			}
			f32(ra.k, rb.k, fmt.Sprintf("layer %d ring k", l))
			f32(ra.v, rb.v, fmt.Sprintf("layer %d ring v", l))
		}
	}
	return n, first
}

func logitDiff(a, b []float32) int {
	if len(a) != len(b) {
		return max(len(a), len(b))
	}
	n := 0
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			n++
		}
	}
	return n
}

func chunkPrompt(n, vocab, seed int) []int {
	p := make([]int, n)
	for i := range p {
		p[i] = (i*131 + seed*17 + 7) % vocab
	}
	return p
}

// prefillChunkCell runs every length, cold and warm, fast and exact, at chunk size C.
func prefillChunkCell(t *testing.T, m *Model, C int) {
	_, _, _, _, _, _, vocab := m.Dims()
	if !m.canBatchN(2) {
		t.Fatalf("the model does not take the batched prefill: the cell would compare the sequential path with itself")
	}
	if !m.cpuFastAttention() {
		t.Fatalf("fast attention is off for this model: the fast arm would be the exact one")
	}
	lengths := []int{C - 1, C, C + 1, 2*C + 17, 3*C + 600}
	if 2*C+17 < fastAttnMinPrompt {
		t.Fatalf("C = %d: 2C+17 is under the fast-attention floor, so no case has a short last chunk inside a fast suffix", C)
	}
	t0, done, total := time.Now(), 0, 4*len(lengths)
	for _, exact := range []bool{false, true} {
		m.exactPrefill = exact
		for _, warm := range []bool{false, true} {
			var prefix []int
			if warm {
				prefix = chunkPrompt(100, vocab, 3)
			}
			for _, n := range lengths {
				suffix := chunkPrompt(n, vocab, n)
				lw, cw := prefillChunkArm(t, m, prefix, suffix, 1<<30)
				p0 := atomic.LoadInt64(&prefillPasses)
				lc, cc := prefillChunkArm(t, m, prefix, suffix, C)
				passes := atomic.LoadInt64(&prefillPasses) - p0
				if warm {
					passes-- // the prefix's own single pass
				}
				want := int64((n + C - 1) / C)
				if cc.localAny && !exact && n >= fastAttnMinPrompt {
					want = 1 // a windowed cache under fast attention keeps the single pass
				}
				if passes != want {
					t.Errorf("exact %v, warm %v, suffix %d, C %d: the chunked arm ran %d passes, want %d", exact, warm, n, C, passes, want)
				}
				kd, first := kvDiff(cw, cc)
				if ld := logitDiff(lw, lc); ld != 0 || kd != 0 {
					t.Errorf("exact %v, warm %v, suffix %d, C %d: %d logits and %d K/V elements differ (first: %s)", exact, warm, n, C, ld, kd, first)
				}
				done++
				fmt.Fprintf(os.Stderr, "[g40-1 %5.0fs] %d/%d cases: exact %v, warm %v, suffix %d done\n", time.Since(t0).Seconds(), done, total, exact, warm, n)
			}
		}
	}
	m.exactPrefill = false
}

func TestPrefillChunked_bitIdentical(t *testing.T) {
	for _, name := range []string{"llama-tiny", "llama-attnfa-tiny", "gemma2-tiny", "mixtral-tiny"} {
		t.Run(name, func(t *testing.T) {
			p := filepath.Join("..", "testdata", name)
			if _, err := os.Stat(filepath.Join(p, "config.json")); err != nil {
				t.Fatalf("the committed fixture %s is missing: %v", p, err)
			}
			m, err := Load(p, Options{})
			if err != nil {
				t.Fatalf("load %s: %v", name, err)
			}
			defer m.Close()
			prefillChunkCell(t, m, 256)
		})
	}
}

// TestPrefillChunked_bitIdenticalReal0p5B is G40-1's real-checkpoint cell: Qwen2.5-Coder-0.5B at int4 and the production
// chunk size. Seconds per arm on the CPU; it needs the bench GGUF in ~/models.
func TestPrefillChunked_bitIdenticalReal0p5B(t *testing.T) {
	home, _ := os.UserHomeDir()
	p := filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("%s: %v", p, err)
	}
	m, err := Load(p, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	prefillChunkCell(t, m, prefillChunkRows)
}
