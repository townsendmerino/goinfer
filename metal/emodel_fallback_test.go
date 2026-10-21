//go:build darwin && goinfer_testhooks

package metal

import (
	"bytes"
	"context"
	"errors"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// loadEModelExact loads gemma4-emodel-tiny Metal-resident with fused attention off, so every attention layer runs the exact
// kernel, and returns the resident and some prompt rows.
func loadEModelExact(t *testing.T, n int) (*decoder.Model, *metalResident, [][]float32) {
	t.Helper()
	if _, err := os.Stat(eModelDir); err != nil {
		t.Skipf("no fixture (%s) — run scripts/pin_gemma4_emodel_tiny.py", eModelDir)
	}
	t.Setenv("GOINFER_METAL_FUSED_ATTENTION", "0")
	m, err := decoder.Load(eModelDir, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 512})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	if !a.emodelBatched() || !a.r.prefillExactAttn() {
		t.Fatalf("batched pass %v, exact attention %v: not the path under test", a.emodelBatched(), a.r.prefillExactAttn())
	}
	rows := make([][]float32, n)
	for i := range rows {
		rows[i] = m.EmbedResidentForTest(3 + (i*37)%200)
	}
	return m, a, rows
}

// TestEModelFallback_visible: when an E-model prompt leaves the batched prefill for the layer-major pass, the fallback is
// announced once per request, with the prompt's length and the decline's reason, and the request's prefill report holds the
// path and the reason (which -log-requests prints). The logits are the layer-major pass's own. The decline is forced
// (prefillDeclineForTest) because the exact kernel's key bound no longer declines a pass: it tiles past it.
func TestEModelFallback_visible(t *testing.T) {
	_, a, rows := loadEModelExact(t, 96)
	why := errors.New("metal: prompt reaches 96 keys (a test decline)")
	prefillDeclineForTest = why
	var out bytes.Buffer
	emodelFallbackOut = &out
	t.Cleanup(func() { prefillDeclineForTest, emodelFallbackOut = nil, os.Stderr })

	a.Reset()
	want := append([]float32(nil), a.r.prefillEModel(rows, 0, true)...)
	if err := a.r.takeExecErr(); err != nil {
		t.Fatal(err)
	}
	for req := 0; req < 2; req++ {
		ctx, rep := decoder.WithPrefillReport(context.Background())
		for call := 0; call < 2; call++ { // a request may call the pass more than once (blocks, chunks)
			got, err := a.PrefillLast(ctx, rows, 0)
			if err != nil {
				t.Fatal(err)
			}
			diff := 0
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					diff++
				}
			}
			if diff != 0 {
				t.Errorf("request %d call %d: %d of %d logits differ from the layer-major pass's", req, call, diff, len(want))
			}
		}
		if path, reason := rep.Fallback(); path != "layer-major" || reason != why.Error() {
			t.Errorf("request %d: the report holds %q, %q; want layer-major and the decline's reason", req, path, reason)
		}
	}
	notes := strings.Count(out.String(), "left the batched prefill for the layer-major pass")
	if notes != 2 || !strings.Contains(out.String(), "(96 tokens)") || !strings.Contains(out.String(), why.Error()) {
		t.Errorf("want one note per request (2), naming the length and the reason; got %d:\n%s", notes, out.String())
	}
}

// TestEModelTiledAttention_matchesTheExactKernel: the batched pass with the exact-attention layers on
// attention_prefill_tiled (forced at 32-key tiles, so a 120-token prompt crosses three tile boundaries) gives the same logits,
// bit for bit, as the untiled exact kernel on the same prompt, and stays within the batched pass's own bar of the
// layer-major pass (TestS9EModelBatched_tiny's: cosine >= 0.98, argmax equal). Past 4,096 keys only the tiled kernel can run;
// this is where the two can be compared.
func TestEModelTiledAttention_matchesTheExactKernel(t *testing.T) {
	_, a, rows := loadEModelExact(t, 120)
	run := func(tile int) []float32 {
		exactAttnTileForTest = tile
		defer func() { exactAttnTileForTest = 0 }()
		lg, err := a.PrefillLast(context.Background(), rows, 0)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), lg...)
	}
	exact, tiled := run(0), run(32)
	diff := 0
	for i := range exact {
		if math.Float32bits(exact[i]) != math.Float32bits(tiled[i]) {
			diff++
		}
	}
	if diff != 0 {
		t.Errorf("%d of %d logits differ between the tiled and the untiled exact kernel", diff, len(exact))
	}
	a.Reset()
	lm := append([]float32(nil), a.r.prefillEModel(rows, 0, true)...)
	if err := a.r.takeExecErr(); err != nil {
		t.Fatal(err)
	}
	if c := cosineV(tiled, lm); c < 0.98 || argmaxF(tiled) != argmaxF(lm) {
		t.Errorf("tiled against the layer-major pass: cosine %.6f, argmax %d vs %d", c, argmaxF(tiled), argmaxF(lm))
	}
	if exactAttnTile(prefillExactAttnMaxKeys) != 0 || exactAttnTile(prefillExactAttnMaxKeys+1) != prefillExactAttnMaxKeys {
		t.Errorf("a pass of 4,096 keys must run the exact kernel unchanged, and one of 4,097 its tiled form")
	}
}
