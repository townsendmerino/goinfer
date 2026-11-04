//go:build darwin && goinfer_testhooks

package metal

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"context"

	"github.com/townsendmerino/goinfer/decoder"
)

// ksh1Window copies a fixture directory with its position window raised to n (the weights do not depend on it).
func ksh1Window(t *testing.T, src string, n int) string {
	t.Helper()
	dst := t.TempDir()
	ents, err := os.ReadDir(src)
	if err != nil {
		t.Skipf("no fixture %s: %v", src, err)
	}
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		b, err := os.ReadFile(filepath.Join(src, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if e.Name() == "config.json" {
			var c map[string]any
			if err := json.Unmarshal(b, &c); err != nil {
				t.Fatal(err)
			}
			for _, m := range []map[string]any{c, asMap(c["text_config"])} {
				if m != nil {
					m["max_position_embeddings"] = n
				}
			}
			if b, err = json.Marshal(c); err != nil {
				t.Fatal(err)
			}
		}
		if err := os.WriteFile(filepath.Join(dst, e.Name()), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dst
}

// ksh1State is a resident's state after a prefill: its last logits and 16 greedy steps after them, hashed (which carries any
// recurrent state too), and every attention layer's K/V bytes for positions [0, n).
func ksh1State(t *testing.T, m *decoder.Model, a *metalResident, lg []float32, n int) (uint64, [][]uint16) {
	t.Helper()
	r := a.r
	var kv [][]uint16
	for l := range r.layers {
		if r.layers[l].kvShared || r.layers[l].geom == nil || r.kc[l] == (Buffer{}) {
			continue
		}
		d, o := r.layers[l].geom.kvDim, r.kvHostOff(l, 2)
		kv = append(kv, slices.Clone(r.kc[l].U16s()[o:o+n*d]), slices.Clone(r.vc[l].U16s()[o:o+n*d]))
	}
	h := fnv.New64a()
	put := func(l []float32) {
		for _, x := range l {
			b := math.Float32bits(x)
			h.Write([]byte{byte(b), byte(b >> 8), byte(b >> 16), byte(b >> 24)})
		}
	}
	put(lg)
	next := argmaxF(lg)
	for j := range 16 {
		l, err := a.Forward(m.EmbedResidentForTest(next), n+j)
		if err != nil {
			t.Fatal(err)
		}
		put(l)
		next = argmaxF(l)
	}
	return h.Sum64(), kv
}

// ksh1WholeVsBlocked prefills n rows whole and then in blocks of B (fresh sequence each), and fails unless the two states are
// byte-equal; each block pass's scratch must stay at or under PrefillScratchBytes of its rows.
func ksh1WholeVsBlocked(t *testing.T, m *decoder.Model, a *metalResident, n, B int, label string) {
	t.Helper()
	_, _, _, _, _, _, vocab := m.Dims()
	embs := make([][]float32, n)
	for i := range embs {
		embs[i] = slices.Clone(m.EmbedResidentForTest((i*37 + 11) % vocab))
	}
	a.Reset()
	lg, err := a.PrefillLast(context.Background(), embs, 0)
	if err != nil {
		t.Fatalf("%s: whole: %v", label, err)
	}
	hw, kw := ksh1State(t, m, a, slices.Clone(lg), n)
	a.Reset()
	over := 0
	for lo := 0; lo < n; lo += B {
		hi := min(lo+B, n)
		s0 := prefillScratchSeen.Load()
		if lg, err = a.PrefillLast(context.Background(), embs[lo:hi], lo); err != nil {
			t.Fatalf("%s: block at %d: %v", label, lo, err)
		}
		if seen, want := prefillScratchSeen.Load()-s0, a.PrefillScratchBytes(hi-lo); seen > want {
			over++
			t.Errorf("%s: the block at %d allocated %d B of scratch, over its figure %d", label, lo, seen, want)
		}
	}
	hb, kb := ksh1State(t, m, a, slices.Clone(lg), n)
	kvDiff := 0
	for i := range kw {
		if !slices.Equal(kw[i], kb[i]) {
			kvDiff++
		}
	}
	fmt.Fprintf(os.Stderr, "[KH1] %s: %d rows whole against blocks of %d: state hash %#x / %#x, %d of %d K/V buffers differ, %d passes over their scratch figure\n",
		label, n, B, hw, hb, kvDiff, len(kw), over)
	if hw != hb || kvDiff != 0 {
		t.Errorf("%s: blocked prefill is not byte-equal to whole", label)
	}
}

// TestKSH1_exactAttentionBlocks is KH1-G1 and KH1-G3 (docs/tasks/task-kernel-survey-2026-10.md, "KS-H1 gates"): llama-tiny on
// Metal with the exact attention kernel forced and a 32-key tile, a 300-row prompt whole against 64-row blocks, byte-equal, each
// pass inside its scratch figure; the resident now reports a block, and admission prices a 20,000-token prompt by it.
func TestKSH1_exactAttentionBlocks(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	t.Setenv("GOINFER_METAL_FUSED_ATTENTION", "0")
	t.Setenv("GOINFER_METAL_FAST_PREFILL_FLOOR", "0")
	defer func(v int) { exactAttnTileForTest = v }(exactAttnTileForTest)
	exactAttnTileForTest = 32
	dir := t.TempDir()
	writeDense(t, dir, genTinyWeights(rand.New(rand.NewSource(11))))
	dir = ksh1Window(t, dir, 1024)
	m, err := decoder.Load(dir, decoder.Options{Backend: "metal", Quant: "int8int8", ResidentContext: 1024})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok || !a.r.prefillExactAttn() {
		t.Fatalf("resident %v, exact attention %v: the test would not exercise the exact kernel", ok, ok && a.r.prefillExactAttn())
	}
	if B := a.PrefillBlockRows(); B <= 0 {
		t.Fatalf("PrefillBlockRows = %d on an exact-attention resident at head dim %d, want a block", B, a.r.exactAttnHeadDim())
	}
	ksh1WholeVsBlocked(t, m, a, 300, 64, "llama-tiny, exact kernel")
	price, ok := m.ResidentPrefillScratchForTest(20000)
	fmt.Fprintf(os.Stderr, "[KH1] admission at 20,000 tokens: blocked price %d B (ok %v), block %d rows\n", price, ok, a.PrefillBlockRows())
	rows := min(20000, a.PrefillBlockRows()+8-1) // the decoder's longest pass: a block plus the tail minimum (prefillTailMin, 8) less one
	if want := a.PrefillScratchBytes(rows) + int64(rows)*4*int64(m.ResidentEmbedLen()); !ok || price != want {
		t.Errorf("admission priced %d B (ok %v), want the blocked pass's %d B", price, ok, want)
	}
}

// TestKSH1_qwen35AndEModelBlocks is KH1-G2: qwen35-tiny and gemma4-emodel-tiny (windows raised), Metal, the batched route, a
// 600-row prompt whole against 128-row blocks, byte-equal.
func TestKSH1_qwen35AndEModelBlocks(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	for _, f := range []string{"qwen35-tiny", "gemma4-emodel-tiny"} {
		t.Run(f, func(t *testing.T) {
			dir := ksh1Window(t, filepath.Join("..", "testdata", f), 1024)
			m, err := decoder.Load(dir, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			a, ok := m.ResidentForwardForTest().(*metalResident)
			if !ok {
				t.Fatalf("not resident: %s", m.ResidentDecline())
			}
			if B := a.PrefillBlockRows(); B <= 0 {
				t.Fatalf("PrefillBlockRows = %d, want a block", B)
			}
			ksh1WholeVsBlocked(t, m, a, 600, 128, f)
		})
	}
}
