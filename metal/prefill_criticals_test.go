//go:build darwin

package metal

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// The two Criticals of docs/audit-metal-2026-09-30.md. Both were silent: nothing errored, the prompt's KV was simply wrong
// or written past its buffer. Each test asserts the decline that now sends the prompt to the sequential path.

func tinyPrefillResident(t *testing.T, opts decoder.Options, maxPositions int) *metalResident {
	t.Helper()
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	dir := t.TempDir()
	writeDense(t, dir, genTinyWeights(rand.New(rand.NewSource(11))))
	if maxPositions != 256 { // writeDense's window; the cap is clamped to it
		p := filepath.Join(dir, "config.json")
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		c := strings.Replace(string(b), `"max_position_embeddings":256`, fmt.Sprintf(`"max_position_embeddings":%d`, maxPositions), 1)
		if c == string(b) {
			t.Fatal("writeDense's config no longer carries max_position_embeddings 256")
		}
		if err := os.WriteFile(p, []byte(c), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	m, err := decoder.Load(dir, opts)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("build resident: %v", err)
	}
	return &metalResident{r: r, hidden: r.H}
}

func tinyEmbs(n int) [][]float32 {
	embs := make([][]float32, n)
	for i := range embs {
		embs[i] = make([]float32, tmHidden)
		for j := range embs[i] {
			embs[i][j] = float32((i*7+j)%13) * 0.01
		}
	}
	return embs
}

// A-C01: with -kv i8 the prefill kernels wrote half-precision K/V (kv_store_f16, 2 bytes at pos*kvDim) into a cache
// allocated at 1 byte per element, and read it back as half: every position in the wrong layout, and positions at or past
// ctxCap/2 past the buffer.
func TestPrefill_declinesInt8KV(t *testing.T) {
	t.Setenv("GOINFER_METAL_FAST_PREFILL_FLOOR", "0")
	a := tinyPrefillResident(t, decoder.Options{Quant: "int8int8", KVPrecision: "i8"}, 256)
	if !a.r.kvI8 {
		t.Fatal("KVPrecision i8 did not build an int8 KV cache; the test proves nothing")
	}
	if ok, why := a.PrefillPath(); ok || !strings.Contains(why, "int8") {
		t.Errorf("PrefillPath = %v, %q; want sequential, naming the int8 KV cache", ok, why)
	}
	if _, err := a.PrefillLast(context.Background(), tinyEmbs(32), 0); err == nil || !strings.Contains(err.Error(), "-kv i8") {
		t.Errorf("PrefillLast with -kv i8: err %v, want a decline naming -kv i8", err)
	}
	// the control: the same model with an f16 cache still prefills in a batch
	b := tinyPrefillResident(t, decoder.Options{Quant: "int8int8"}, 256)
	if _, err := b.PrefillLast(context.Background(), tinyEmbs(32), 0); err != nil {
		t.Errorf("PrefillLast with the default KV declined: %v", err)
	}
}

// F-C02: the exact attention_prefill kernel keeps `threadgroup float sc[4096]`, indexed by absolute key, and ran past it
// above 4096 keys. A pass past that bound now takes attention_prefill_tiled (P28) instead of declining, and a pass that fits
// still runs attention_prefill. The exact kernel runs when the fused kernel cannot; GOINFER_METAL_FUSED_ATTENTION=0 forces
// it on this hd-16 fixture.
func TestPrefill_exactAttentionTilesPast4096Keys(t *testing.T) {
	t.Setenv("GOINFER_METAL_FAST_PREFILL_FLOOR", "0")
	t.Setenv("GOINFER_METAL_FUSED_ATTENTION", "0")
	a := tinyPrefillResident(t, decoder.Options{Quant: "int8int8", ResidentContext: 8192}, 8192)
	if a.ctxCap() <= prefillExactAttnMaxKeys {
		t.Fatalf("ctxCap %d does not exceed the exact kernel's %d keys; the test proves nothing", a.ctxCap(), prefillExactAttnMaxKeys)
	}
	if fused, _ := a.r.prefillAttnKernels(); fused {
		t.Fatal("the fused kernel is still selected; the test proves nothing")
	}
	if _, err := a.PrefillLast(context.Background(), tinyEmbs(32), prefillExactAttnMaxKeys-16); err != nil {
		t.Errorf("a pass reaching %d keys on the exact kernel declined: %v", prefillExactAttnMaxKeys+16, err)
	}
	if _, err := a.PrefillLast(context.Background(), tinyEmbs(32), prefillExactAttnMaxKeys-32); err != nil {
		t.Errorf("a pass ending exactly at %d keys declined: %v", prefillExactAttnMaxKeys, err)
	}
}

// The decline above is only as good as its constant: prefillExactAttnMaxKeys must be the kernel's own array size.
func TestPrefillExactAttnBound(t *testing.T) {
	i := strings.Index(prefillKernels, "kernel void attention_prefill(")
	if i < 0 {
		t.Fatal("attention_prefill is not in prefillKernels")
	}
	body := prefillKernels[i:]
	if j := strings.Index(body[1:], "kernel void "); j > 0 {
		body = body[:j+1]
	}
	if want := fmt.Sprintf("threadgroup float sc[%d];", prefillExactAttnMaxKeys); !strings.Contains(body, want) {
		t.Errorf("attention_prefill does not declare %q; prefillExactAttnMaxKeys has drifted from the kernel", want)
	}
	// attention_prefill_tiled holds one tile in the same buffer, so its tile may never exceed it.
	i = strings.Index(prefillKernels, "kernel void attention_prefill_tiled(")
	if i < 0 {
		t.Fatal("attention_prefill_tiled is not in prefillKernels")
	}
	tiled := prefillKernels[i:]
	if j := strings.Index(tiled[1:], "kernel void "); j > 0 {
		tiled = tiled[:j+1]
	}
	if want := fmt.Sprintf("threadgroup float sc[%d];", prefillExactAttnMaxKeys); !strings.Contains(tiled, want) {
		t.Errorf("attention_prefill_tiled does not declare %q", want)
	}
	for _, keys := range []int{prefillExactAttnMaxKeys + 1, 32768, 1 << 20} {
		if tile := exactAttnTile(keys); tile < 1 || tile > prefillExactAttnMaxKeys {
			t.Errorf("exactAttnTile(%d) = %d: outside the tiled kernel's 1..%d", keys, tile, prefillExactAttnMaxKeys)
		}
	}
}
