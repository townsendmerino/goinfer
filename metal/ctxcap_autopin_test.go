//go:build darwin && goinfer_testhooks

package metal

import (
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestResolveMetalCtxCap_autoPinIsACeiling (C-C01, docs/audit-metal-2026-09-30.md): a context the load-time fit guard
// auto-pinned (the caller did not choose it) may lower Metal's default, never raise it; an explicit -ctx keeps its own
// rule, honoured up to metalCtxCapMax and refused above. An auto-pin that read as a request would, on a tight machine,
// allocate KV several times the default, and above the ceiling be refused, moving the forward to the CPU.
func TestResolveMetalCtxCap_autoPinIsACeiling(t *testing.T) {
	dir := t.TempDir()
	writeDense(t, dir, genTinyWeights(rand.New(rand.NewSource(11))))
	p := filepath.Join(dir, "config.json")
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	c := strings.Replace(string(b), `"max_position_embeddings":256`, `"max_position_embeddings":131072`, 1)
	if c == string(b) {
		t.Fatal("writeDense's config no longer carries max_position_embeddings 256")
	}
	if err := os.WriteFile(p, []byte(c), 0o644); err != nil {
		t.Fatal(err)
	}
	load := func(ctx int) *decoder.Model {
		m, err := decoder.Load(dir, decoder.Options{Quant: "int8int8", ResidentContext: ctx})
		if err != nil {
			t.Fatalf("load (ResidentContext %d): %v", ctx, err)
		}
		t.Cleanup(func() { m.Close() })
		return m
	}
	for _, tc := range []struct {
		autoPin int
		want    int
	}{
		{20000, metalCtxCapDefault}, // the inflation case: a pin above the default does not raise it
		{40000, metalCtxCapDefault}, // the refusal case: above the ceiling, no longer refused
		{2000, 2000},                // a pin below the default lowers it, which is what the guard pins for
		{metalCtxCapDefault, metalCtxCapDefault},
	} {
		m := load(0)
		m.AutoPinResidentContextForTest(tc.autoPin)
		if got, err := resolveMetalCtxCap(m); err != nil || got != tc.want {
			t.Errorf("auto-pin %d: resolveMetalCtxCap = %d, %v; want %d", tc.autoPin, got, err, tc.want)
		}
	}
	if got, err := resolveMetalCtxCap(load(16384)); err != nil || got != 16384 {
		t.Errorf("-ctx 16384: resolveMetalCtxCap = %d, %v; want 16384, the caller's choice", got, err)
	}
	if _, err := resolveMetalCtxCap(load(metalCtxCapMax + 1)); err == nil {
		t.Errorf("-ctx %d: resolveMetalCtxCap accepted a request above the ceiling", metalCtxCapMax+1)
	}
}
