package decoder

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestKV2_G0cpuSplitInvariance is G0 on the CPU (docs/measurements/kv2-2026-10-14.md): a 1,100-token prompt prefilled
// whole against the same prompt in two calls, comparing every layer's K and V. Splits at 512 and 576 keep both passes
// at 512 rows or more (the fast prefill attention's class); a split at 984 and chunks of 64 change the class and are
// reported, not asserted. Each arm runs with the default fast attention and with exact prefill attention. The control (a
// token changed at 300) runs first. It measures; the record grades it.
func TestKV2_G0cpuSplitInvariance(t *testing.T) {
	dir := filepath.Join("..", "testdata", "llama-attnfa-tiny")
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("no fixture at %s", dir)
	}
	m, err := Load(dir, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	const N = 1100
	prompt := make([]int, N)
	for i := range prompt {
		prompt[i] = (i*37 + 11) % m.w.arch.VocabSize
	}
	nL, nKV, hd := m.w.arch.NumLayers, m.w.arch.NumKVHeads, m.w.arch.HeadDim
	fill := func(cuts []int, p []int) *KVCache {
		c := NewKVCache(nL, nKV, hd, 0, N, nil)
		prev := 0
		for _, cut := range append(slices.Clone(cuts), N) {
			if _, err := m.prefillLogits(context.Background(), p[prev:cut], c); err != nil {
				t.Fatal(err)
			}
			prev = cut
		}
		return c
	}
	firstDiff := func(a, b *KVCache) int {
		row := -1
		for l := range a.keys {
			w := len(a.keys[l]) / N
			for _, pair := range [][2][]float32{{a.keys[l], b.keys[l]}, {a.vals[l], b.vals[l]}} {
				for i := range pair[0] {
					if pair[0][i] != pair[1][i] {
						if r := i / w; row < 0 || r < row {
							row = r
						}
						break
					}
				}
			}
		}
		return row
	}
	chunks := func(w int) []int {
		var out []int
		for c := w; c < N; c += w {
			out = append(out, c)
		}
		return out
	}
	for _, exact := range []bool{false, true} {
		m.exactPrefill = exact
		whole := fill(nil, prompt)
		pert := slices.Clone(prompt)
		pert[300] = (pert[300] + 1) % m.w.arch.VocabSize
		if r := firstDiff(fill(nil, pert), whole); r != 300 {
			t.Fatalf("exact %v: control first differs at row %d, want 300", exact, r)
		}
		for _, arm := range []struct {
			name string
			cuts []int
		}{{"split@512 (both passes fast)", []int{512}}, {"split@576 (both passes fast)", []int{576}}, {"split@984 (second pass 116 rows)", []int{984}}, {"chunk=64", chunks(64)}} {
			fmt.Fprintf(os.Stderr, "[kv2-g0 cpu exact-prefill=%v] %s: first KV difference at row %d (-1: none)\n", exact, arm.name, firstDiff(fill(arm.cuts, prompt), whole))
		}
	}
}
