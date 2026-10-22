//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// kv4LongFixture writes a copy of llama-attnfa-tiny with a context of maxPos positions and returns its directory.
func kv4LongFixture(t *testing.T, maxPos int) string {
	t.Helper()
	src := filepath.Join("..", "testdata", "llama-attnfa-tiny")
	raw, err := os.ReadFile(filepath.Join(src, "config.json"))
	if err != nil {
		t.Skipf("no fixture: %v", err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	cfg["max_position_embeddings"] = maxPos
	dir := t.TempDir()
	out, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), out, 0o644); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"model.safetensors", "generation_config.json"} {
		abs, err := filepath.Abs(filepath.Join(src, f))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(abs, filepath.Join(dir, f)); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// TestKV4_sharedPreambleStoredOnce is G3b at tier 0 (docs/measurements/kv4-cuda-2026-10-21.md §2): four slots whose prompts share a
// 16,384-token preamble, the first prefilling it and the other three taking it by CopyKV, hold the preamble's 256 blocks once. The full
// pool's blocks in use after all four have prefilled their own tails are at most 256 + 4 * (ceil(tail/64) + 1), and each slot's last
// logits equal the same prompt prefilled cold on a fresh resident, bit for bit. A decode step in each slot after it (writing past the
// shared blocks) leaves the others' logits unchanged, and a rollback into a shared block in one slot (copy on write) leaves another
// slot's next step unchanged.
func TestKV4_sharedPreambleStoredOnce(t *testing.T) {
	requireCUDADevice(t)
	const pre, tail, slots, maxPos = 16384, 100, 4, 20480
	dir := kv4LongFixture(t, maxPos)
	load := func(n int) (*decoder.Model, *cudaResident) {
		m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4", KVPrecision: "f32", ResidentContext: maxPos, ResidentKVSlots: n})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		t.Cleanup(func() { _ = m.Close() })
		return m, mustResident(t, m)
	}
	m, r := load(slots)
	if r.KVSlots() != slots {
		t.Fatalf("built %d slots, want %d", r.KVSlots(), slots)
	}
	_, _, _, _, _, _, vocab := m.Dims()
	tok := func(i, salt int) []float32 {
		return append([]float32(nil), m.EmbedResidentForTest((i*31+salt*7+3)%vocab)...)
	}
	prompt := func(s int) [][]float32 {
		e := make([][]float32, pre+tail)
		for i := range e {
			salt := 0
			if i >= pre {
				salt = s + 1 // each conversation's own tail
			}
			e[i] = tok(i, salt)
		}
		return e
	}
	bind := func(s int) {
		if err := r.UseKVSlot(s); err != nil {
			t.Fatal(err)
		}
	}
	got := make([][]float32, slots)
	for s := range slots {
		bind(s)
		e := prompt(s)
		from := 0
		if s > 0 {
			if err := r.CopyKV(0, s, 0, pre); err != nil {
				t.Fatalf("CopyKV into slot %d: %v", s, err)
			}
			from = pre
		}
		lg, err := r.PrefillLast(context.Background(), e[from:], from)
		if err != nil {
			t.Fatalf("slot %d prefill from %d: %v", s, from, err)
		}
		got[s] = append([]float32(nil), lg...)
	}
	inUse := 0
	if err := r.do(func() error { inUse = r.kvPools[kvGroupFull].inUse(); return nil }); err != nil {
		t.Fatal(err)
	}
	limit := pre/kvBlock + slots*(kvBlocksFor(tail)+1)
	t.Logf("full pool: %d blocks in use after %d conversations sharing a %d-token preamble (limit %d; %d if each stored its own)",
		inUse, slots, pre, limit, slots*kvBlocksFor(pre+tail))
	if inUse > limit {
		t.Fatalf("%d blocks in use, past the %d a preamble stored once allows", inUse, limit)
	}
	// Each slot's logits against its own cold prefill on a fresh one-slot resident.
	_, rc := load(1)
	for s := range slots {
		lg, err := rc.PrefillLast(context.Background(), prompt(s), 0)
		if err != nil {
			t.Fatalf("cold prefill %d: %v", s, err)
		}
		if j, ok := sameBits(got[s], lg); !ok {
			t.Fatalf("slot %d: the logits after sharing the preamble differ from its cold prefill at %d", s, j)
		}
	}
	// A rollback in slot 1 to inside the shared blocks rewrites them: slot 1 must take its own copy, so slot 2's next step is unchanged.
	bind(2)
	next2, err := r.Forward(tok(pre+tail, 9), pre+tail)
	if err != nil {
		t.Fatal(err)
	}
	want2 := append([]float32(nil), next2...)
	bind(1)
	for i := range 40 { // positions [pre-32, pre+8): the last shared block and past it
		if _, err := r.Forward(tok(i, 11), pre-kvBlock/2+i); err != nil {
			t.Fatal(err)
		}
	}
	bind(2)
	again2, err := r.Forward(tok(pre+tail, 9), pre+tail)
	if err != nil {
		t.Fatal(err)
	}
	if j, ok := sameBits(again2, want2); !ok {
		t.Fatalf("slot 2's step changed at %d after slot 1 rewrote positions inside the shared preamble: the write reached a shared block", j)
	}
}
