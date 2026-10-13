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

// wideWindowTiny writes a tiny dense model whose config declares a 131072-position window, so the fit-by-default candidate
// is decoder.MetalFitDefaultCtx itself and not the fixture's own 256 positions.
func wideWindowTiny(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeDense(t, dir, genTinyWeights(rand.New(rand.NewSource(11))))
	widenWindow(t, dir)
	return dir
}

// widenWindow rewrites the 256-position window of a tiny model written by this package's helpers to 131072.
func widenWindow(t *testing.T, dir string) {
	t.Helper()
	p := filepath.Join(dir, "config.json")
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	c := strings.Replace(string(b), `"max_position_embeddings":256`, `"max_position_embeddings":131072`, 1)
	if c == string(b) {
		t.Fatal("the tiny model's config no longer carries max_position_embeddings 256")
	}
	if err := os.WriteFile(p, []byte(c), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestResolveMetalCtxCap_autoPinIsACeiling (C-C01, docs/audit-metal-2026-09-30.md): a context the load-time fit guard
// auto-pinned (the caller did not choose it) may lower what an unpinned load takes, never raise it; an explicit -ctx
// keeps its own rule, honoured up to metalCtxCapMax and refused above. An auto-pin that read as a request would, on a
// tight machine, allocate KV several times what fits, and above the ceiling be refused, moving the forward to the CPU.
// The memory probe is forced roomy and then unknown, so the pin is the only thing lowering the first set and the
// default is all the second set can reach.
func TestResolveMetalCtxCap_autoPinIsACeiling(t *testing.T) {
	dir := wideWindowTiny(t)
	load := func(ctx int) *decoder.Model {
		m, err := decoder.Load(dir, decoder.Options{Quant: "int8int8", ResidentContext: ctx})
		if err != nil {
			t.Fatalf("load (ResidentContext %d): %v", ctx, err)
		}
		t.Cleanup(func() { m.Close() })
		return m
	}
	for _, tc := range []struct {
		autoPin        int
		roomy, unknown int // what the pin resolves to with room for the whole candidate, and with no memory figure
	}{
		{20000, decoder.MetalFitDefaultCtx, metalCtxCapDefault}, // the inflation case: a pin above what an unpinned load takes does not raise it
		{40000, decoder.MetalFitDefaultCtx, metalCtxCapDefault}, // the refusal case: above the ceiling, not refused
		{8192, 8192, metalCtxCapDefault},                        // a pin under the candidate caps it
		{2000, 2000, 2000},                                      // a pin below the default lowers it, which is what the guard pins for
		{metalCtxCapDefault, metalCtxCapDefault, metalCtxCapDefault},
	} {
		for _, probe := range []struct {
			name string
			free int64
			ok   bool
			want int
		}{{"roomy", 64 << 30, true, tc.roomy}, {"unknown", 0, false, tc.unknown}} {
			restore := decoder.SetMemoryProbeForTest("metal", probe.free, probe.ok)
			m := load(0)
			m.AutoPinResidentContextForTest(tc.autoPin)
			if got, err := resolveMetalCtxCap(m); err != nil || got != probe.want {
				t.Errorf("auto-pin %d, %s memory: resolveMetalCtxCap = %d, %v; want %d", tc.autoPin, probe.name, got, err, probe.want)
			}
			restore()
		}
	}
	if got, err := resolveMetalCtxCap(load(16384)); err != nil || got != 16384 {
		t.Errorf("-ctx 16384: resolveMetalCtxCap = %d, %v; want 16384, the caller's choice", got, err)
	}
	if _, err := resolveMetalCtxCap(load(metalCtxCapMax + 1)); err == nil {
		t.Errorf("-ctx %d: resolveMetalCtxCap accepted a request above the ceiling", metalCtxCapMax+1)
	}
}
