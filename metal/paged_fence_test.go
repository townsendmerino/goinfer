//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/prequant"
)

// TestPagedFence_bitExactAndNotStale is the fence's gate (M-11 / C-B03, paged_fence.go): on the tiny Gemma 4 MoE as a
// metal .giw paged at 3 slots, every logit of 8 positions equals every-expert-resident with the fence on and off; with
// it on, every boundary was fenced (no 50 ms fallback), and with pagedFenceCheckForTest no mirrored route (nor guess,
// with the prefetch on) differed from the router's own buffer.
func TestPagedFence_bitExactAndNotStale(t *testing.T) {
	const ckpt = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
		t.Skipf("no fixture: %v", err)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	giw := filepath.Join(t.TempDir(), "g4.metal.giw")
	if err := prequant.Transcode(context.Background(), ckpt, giw, "int4", false, decoder.GIWTargetMetal); err != nil {
		t.Fatal(err)
	}
	prevF, prevC, prevP := pagedFenceOn, pagedFenceCheckForTest, g4PrefetchOn
	defer func() { pagedFenceOn, pagedFenceCheckForTest, g4PrefetchOn = prevF, prevC, prevP }()
	run := func(slots string, fence, prefetch bool) (lg [][]float32, f *pagedFence) {
		pagedFenceOn, pagedFenceCheckForTest, g4PrefetchOn = fence, fence, prefetch
		if slots == "" {
			os.Unsetenv("GOINFER_METAL_MOE_SLOTS")
		} else {
			t.Setenv("GOINFER_METAL_MOE_SLOTS", slots)
		}
		m, err := decoder.Load(giw, decoder.Options{Backend: "metal", Quant: "int4"})
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		mr := m.ResidentForwardForTest().(*metalResident)
		mr.Reset()
		for i, tok := range twoGeomPrompt {
			l, err := mr.Forward(m.EmbedResidentForTest(tok), i)
			if err != nil {
				t.Fatal(err)
			}
			lg = append(lg, append([]float32(nil), l...))
		}
		return lg, mr.r.g4moe.fence
	}
	base, _ := run("", false, false)
	for _, c := range []struct {
		name            string
		fence, prefetch bool
	}{{"fence off", false, false}, {"fence on", true, false}, {"fence on, prefetch on", true, true}} {
		lg, f := run("3", c.fence, c.prefetch)
		if c.fence {
			if f == nil {
				t.Fatalf("%s: no fence was built (MSL 3.2 did not compile here?)", c.name)
			}
			if f.waits == 0 || f.fallbacks != 0 || f.stale != 0 {
				t.Fatalf("%s: %d fenced boundaries, %d fallbacks, %d stale routes", c.name, f.waits, f.fallbacks, f.stale)
			}
			t.Logf("%s: %d boundaries fenced, 0 fallbacks, 0 stale", c.name, f.waits)
		} else if f != nil {
			t.Fatalf("%s: a fence was built", c.name)
		}
		for i := range base {
			for j := range base[i] {
				if math.Float32bits(base[i][j]) != math.Float32bits(lg[i][j]) {
					t.Fatalf("%s: position %d logit %d differs from every-expert-resident", c.name, i, j)
				}
			}
		}
	}
}
