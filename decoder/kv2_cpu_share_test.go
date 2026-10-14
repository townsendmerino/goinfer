package decoder

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestKV2_cpuSessionShare is G2 and G3 for the CPU (docs/measurements/kv2-2026-10-14.md): a session B that copies a
// 1,100-token preamble's whole blocks from session A (SharedHit, CopyPrefixFrom), then generates, emits exactly what B
// emits cold, and its K/V equal the cold prefill's byte for byte. With 600 tokens of its own B takes 1,088 rows; with
// 300 the hit shrinks until B's suffix keeps 512 rows, the fast prefill attention's class. Both with the default fast
// attention and with exact prefill attention.
func TestKV2_cpuSessionShare(t *testing.T) {
	dir := filepath.Join("..", "testdata", "llama-attnfa-tiny")
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("no fixture at %s", dir)
	}
	m, err := Load(dir, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	vocab := m.w.arch.VocabSize
	pre := make([]int, 1100)
	for i := range pre {
		pre[i] = (i*37 + 11) % vocab
	}
	own := func(c, n int) []int {
		p := slices.Clone(pre)
		for i := range n {
			p = append(p, (c*61+i*7+5)%vocab)
		}
		return p
	}
	gen := func(s *Session, prompt []int) ([]int, *Generation) {
		ch, g := s.Generate(context.Background(), prompt, 8, SamplingParams{})
		var ids []int
		for id := range ch {
			ids = append(ids, id)
		}
		if err := g.Err(); err != nil {
			t.Fatal(err)
		}
		return ids, g
	}
	sameKV := func(a, b *KVCache, n int) bool {
		for l := range a.keys {
			w := a.stride[l]
			if !slices.Equal(a.keys[l][:n*w], b.keys[l][:n*w]) || !slices.Equal(a.vals[l][:n*w], b.vals[l][:n*w]) {
				return false
			}
		}
		return true
	}
	for _, exact := range []bool{false, true} {
		m.exactPrefill = exact
		for _, tc := range []struct{ ownLen, wantHit int }{{600, 1088}, {300, 832}} {
			a := m.NewSession(0)
			gen(a, own(0, 600))
			pb := own(1, tc.ownLen)
			hit := a.SharedHit(pb)
			if hit != tc.wantHit {
				t.Errorf("exact %v, %d own tokens: SharedHit = %d, want %d", exact, tc.ownLen, hit, tc.wantHit)
			}
			b := m.NewSession(0)
			if err := b.CopyPrefixFrom(a, hit); err != nil {
				t.Fatal(err)
			}
			got, g := gen(b, pb)
			if g.PrefillReused != hit || g.PrefillShared != hit {
				t.Errorf("exact %v, %d own tokens: reused %d, shared %d, want %d", exact, tc.ownLen, g.PrefillReused, g.PrefillShared, hit)
			}
			cold := m.NewSession(0)
			want, _ := gen(cold, pb)
			if !slices.Equal(got, want) {
				t.Errorf("exact %v, %d own tokens: B with the copy %v, cold %v", exact, tc.ownLen, got, want)
			}
			if !sameKV(b.cache, cold.cache, len(pb)) {
				t.Errorf("exact %v, %d own tokens: B's K/V differ from the cold prefill's", exact, tc.ownLen)
			}
		}
	}
	// A short prompt's cold prefill is not the fast class: nothing is offered.
	a := m.NewSession(0)
	gen(a, own(0, 600))
	if h := a.SharedHit(pre[:400]); h != 0 {
		t.Errorf("a 400-token prompt: SharedHit = %d, want 0", h)
	}
}
