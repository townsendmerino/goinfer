//go:build darwin

package metal

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
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

// A-C01's int8 case, since P3b-2: with -kv i8 the f16 MMA pass stores K/V through kv_store_i8_prefill (int8 codes and scales in
// the cache's own layout) and attends over each layer's cache dequantized into f16 scratch, so a prompt takes the pass and its
// logits agree with the sequential loop's as the f16 pass's agree with f16's (A-C01 was half-precision K/V written into a 1-byte
// cache; the PrefillPath and PrefillLast checks below would see that as garbage logits).
func TestPrefill_int8KVOnThePass(t *testing.T) {
	t.Setenv("GOINFER_METAL_FAST_PREFILL_FLOOR", "0")
	cos := func(kv string) float64 {
		a := tinyPrefillResident(t, decoder.Options{Quant: "int8int8", KVPrecision: kv}, 256)
		if a.r.kvI8 != (kv == "i8") {
			t.Fatalf("KVPrecision %q built kvI8 %v; the test proves nothing", kv, a.r.kvI8)
		}
		if ok, why := a.PrefillPath(); !ok || !strings.Contains(why, "MMA") {
			t.Fatalf("--kv %q: PrefillPath = %v, %q; want the MMA pass", kv, ok, why)
		}
		embs := tinyEmbs(32)
		got, err := a.PrefillLast(context.Background(), embs, 0)
		if err != nil {
			t.Fatalf("--kv %q: PrefillLast: %v", kv, err)
		}
		got = slices.Clone(got)
		a.Reset()
		var ref []float32
		for i, e := range embs {
			if ref, err = a.Forward(e, i); err != nil {
				t.Fatal(err)
			}
		}
		var dot, na, nb float64
		for i := range ref {
			dot += float64(got[i]) * float64(ref[i])
			na += float64(got[i]) * float64(got[i])
			nb += float64(ref[i]) * float64(ref[i])
		}
		return dot / math.Sqrt(na*nb)
	}
	c16, c8 := cos(""), cos("i8")
	fmt.Fprintf(os.Stderr, "[A-C01] 32-token prompt, the pass against the sequential loop: cosine %.6f at f16, %.6f at int8\n", c16, c8)
	if c8 < c16-0.01 {
		t.Errorf("int8 pass vs sequential cosine %.6f, f16's %.6f: the int8 pass is further from its loop than f16's by more than 0.01", c8, c16)
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
