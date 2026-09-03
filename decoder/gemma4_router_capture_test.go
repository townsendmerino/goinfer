//go:build realckpt

package decoder

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestGemma4RouterCapture (probe #1) captures the per-token, per-layer top-k expert
// selections for the real gemma-4-26b-a4b-it over a FIXED teacher-forced passage, so the
// int8 run and a 4-bit run can be compared for routing collapse. Set GOINFER_ROUTER_CAPTURE=1
// plus the usual base/scheme/act env; writes JSON to ROUTER_CAPTURE_OUT. Skipped otherwise.
//
// Feeding the true tokens (one linear pass, N×nLayers calls — NOT the O(N²) argmax rig)
// gives both runs identical inputs, so selection differences are purely the quantization's.
func TestGemma4RouterCapture(t *testing.T) {
	requireHeavyModel(t)
	dir := modelPath("gemma-4-26b-a4b-it")
	if _, err := os.Stat(dir); err != nil {
		t.Skip("real gemma-4-26b-a4b-it not present")
	}
	if !routerCapture {
		t.Skip("set GOINFER_ROUTER_CAPTURE=1 to capture router selections")
	}
	base := os.Getenv("ZZBASE")
	if base == "" {
		base = "int4"
	}
	m, err := Load(dir, Options{Quant: base})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	tk, err := tokenizer.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	ids, _ := tk.Encode("The capital of France is Paris, one of the most visited cities in the world, known for the Eiffel Tower and its museums.", true)

	routerCaptureReset() // N-27: clears every buffer AND re-arms the cap
	c := m.NewCache(len(ids))
	for _, id := range ids {
		if _, err := m.runLayers(id, c); err != nil {
			t.Fatal(err)
		}
	}
	sel := routerCaptureBuf

	out := os.Getenv("ROUTER_CAPTURE_OUT")
	if out == "" {
		t.Fatal("set ROUTER_CAPTURE_OUT to a file path")
	}
	b, _ := json.Marshal(map[string]any{
		"base":    base,
		"scheme":  os.Getenv("GOINFER_FAKEQUANT"),
		"act":     os.Getenv("GOINFER_FAKEQUANT_ACT"),
		"experts": os.Getenv("GOINFER_FAKEQUANT_EXPERTS"),
		"nTokens": len(ids),
		"nLayers": m.w.arch.NumLayers,
		"sel":     sel,
	})
	if err := os.WriteFile(out, b, 0644); err != nil {
		t.Fatal(err)
	}
	fmt.Printf("captured %d selections (%d tokens × %d layers) -> %s\n", len(sel), len(ids), m.w.arch.NumLayers, out)
}
