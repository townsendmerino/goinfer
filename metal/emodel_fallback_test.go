//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestEModelFallback_visible: a Gemma 4 E-model prompt past the exact attention kernel's key bound runs the layer-major pass,
// as before, and the fallback is now visible. The banner says the bound and the path taken past it, the request's prefill
// report holds the path and the reason, and the logits are the layer-major pass's own. On gemma4-emodel-tiny with fused
// attention off, so every layer runs the exact kernel, and the bound lowered to 64 keys (exactAttnKeysForTest), so a
// 96-token prompt crosses it. The kernel's own score buffer is unchanged.
func TestEModelFallback_visible(t *testing.T) {
	if _, err := os.Stat(eModelDir); err != nil {
		t.Skipf("no fixture (%s) — run scripts/pin_gemma4_emodel_tiny.py", eModelDir)
	}
	t.Setenv("GOINFER_METAL_FUSED_ATTENTION", "0")
	exactAttnKeysForTest = 64
	t.Cleanup(func() { exactAttnKeysForTest = 0 })
	m, err := decoder.Load(eModelDir, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 256})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	if !a.emodelBatched() || !a.r.prefillExactAttn() {
		t.Fatalf("batched pass %v, exact attention %v: not the path under test", a.emodelBatched(), a.r.prefillExactAttn())
	}
	if _, banner := a.PrefillPath(); !strings.Contains(banner, "past 4096 keys the exact attention kernel") || !strings.Contains(banner, "the layer-major pass") {
		t.Errorf("the banner does not name the bound and the path past it: %q", banner)
	}
	rows := make([][]float32, 96)
	for i := range rows {
		rows[i] = m.EmbedResidentForTest(3 + (i*37)%200)
	}
	a.Reset()
	want := append([]float32(nil), a.r.prefillEModel(rows, 0, true)...)
	if err := a.r.takeExecErr(); err != nil {
		t.Fatal(err)
	}
	ctx, rep := decoder.WithPrefillReport(context.Background())
	got, err := a.PrefillLast(ctx, rows, 0)
	if err != nil {
		t.Fatal(err)
	}
	path, reason := rep.Fallback()
	if path != "layer-major" || !strings.Contains(reason, "96 keys") || !strings.Contains(reason, "holds 64") {
		t.Errorf("the report holds %q, %q; want layer-major and the bound's reason", path, reason)
	}
	diff := 0
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			diff++
		}
	}
	if diff != 0 {
		t.Errorf("%d of %d logits differ from the layer-major pass's", diff, len(want))
	}
	ctx2, rep2 := decoder.WithPrefillReport(context.Background())
	if _, err := a.PrefillLast(ctx2, rows[:48], 0); err != nil {
		t.Fatal(err)
	}
	if p, r := rep2.Fallback(); p != "" {
		t.Errorf("a prompt under the bound noted a fallback: %q, %q", p, r)
	}
}
