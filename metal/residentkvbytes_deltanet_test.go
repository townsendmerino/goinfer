//go:build darwin

package metal

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestResidentKVBytes_excludesDeltaNetLayers gates N-36 (audit-metal-2026-09-12.md):
// residentKVBytes used to charge every layer the model's default kvDim, including Gated-DeltaNet
// (linear-attention) layers that allocate NO KV cache at all — metal/model.go's own layer-build
// loop leaves r.kc[l]/r.vc[l] zero-value for exactly these layers. testdata/qwen35-tiny is a real
// 3:1 hybrid (3 linear_attention layers, 1 full_attention) — a fixture where the bug and the fix
// give DIFFERENT, easily distinguished answers (4x vs 1x one layer's KV bytes), not just a
// theoretical concern.
func TestResidentKVBytes_excludesDeltaNetLayers(t *testing.T) {
	m, err := decoder.Load("../testdata/qwen35-tiny", decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	_, _, _, _, _, _, dnetOK := m.Qwen35ResidentParams()
	if !dnetOK {
		t.Fatal("test setup: qwen35-tiny did not resolve to a DeltaNet-hybrid model")
	}
	_, nLayers, _, _, _, _, _ := m.Dims()
	var wantAttnLayers, wantLinearLayers int
	for l := 0; l < nLayers; l++ {
		if m.Qwen35LinearLayer(l) {
			wantLinearLayers++
		} else {
			wantAttnLayers++
		}
	}
	if wantLinearLayers == 0 {
		t.Fatal("test setup: qwen35-tiny has no linear-attention layers — fixture changed, this test no longer discriminates")
	}
	if wantAttnLayers == 0 {
		t.Fatal("test setup: qwen35-tiny has no full-attention layers to charge KV for")
	}
	t.Logf("qwen35-tiny: %d layers, %d linear-attention (no KV), %d full-attention (KV)",
		nLayers, wantLinearLayers, wantAttnLayers)

	ctxCap, err := resolveMetalCtxCap(m)
	if err != nil {
		t.Fatalf("resolveMetalCtxCap: %v", err)
	}

	// Hand-compute the expected total from ONLY the attention layers, using the exact same
	// per-layer accessors residentKVBytes itself reads — this is the independent oracle, not a
	// re-implementation of the function under test.
	const bytesPerElem = 2
	var want int64
	for l := 0; l < nLayers; l++ {
		if m.Qwen35LinearLayer(l) {
			continue
		}
		kvDim := int64(m.KVHeadsAtResident(l)) * int64(m.HeadDimAtResident(l))
		want += 2 * int64(ctxCap) * kvDim * bytesPerElem
	}
	if want <= 0 {
		t.Fatal("test setup: expected KV bytes computed as 0 — oracle itself is degenerate")
	}

	got := residentKVBytes(m)
	if got != want {
		// The pre-fix behavior charged every layer, including the linear-attention ones — if the
		// fix regresses, this is what it would look like: got > want by roughly the linear
		// layers' own share.
		t.Errorf("residentKVBytes = %d, want %d (excluding %d DeltaNet layer(s)' KV) — "+
			"got/want ratio %.2f suggests %s",
			got, want, wantLinearLayers, float64(got)/float64(want),
			map[bool]string{true: "DeltaNet layers are still being charged (N-36 regressed)", false: "an unrelated discrepancy"}[got > want])
	}
}
