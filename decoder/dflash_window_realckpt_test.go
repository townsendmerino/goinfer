//go:build realckpt

package decoder

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestDFlashWindow_matchesReference is G1 tier 1: Qwen3.5-9B's drafter on the CPU against z-lab's reference code
// (scripts/pin_dflash_window_trace.py), one block over fused context of 64, 4,000 and 4,500 positions (under, near and past its
// 4,096 window). Every row at cosine >= 0.99999. Plants: with the masks off (every layer full) every case fails; with only the
// window off, only the 4,500 case does.
func TestDFlashWindow_matchesReference(t *testing.T) {
	requireHeavyModel(t)
	dir := os.ExpandEnv("$HOME/models/qwen35-9b-dflash")
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		t.Skipf("no drafter at %s", dir)
	}
	raw, err := readGolden(filepath.Join("..", "testdata", "dflash_window_qwen35_9b_golden.json.gz"))
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Block int `json:"block"`
		Cases []struct {
			Ctx int         `json:"ctx"`
			Out [][]float32 `json:"out"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	d, err := LoadDFlashDrafter(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	causal, window := d.layerCausal, d.layerWindow
	if causal == nil {
		t.Fatal("the drafter loaded with no per-layer masks; this gate needs its sliding layers")
	}
	be, h := &cpuBackend{}, d.hidden
	worst := func(out, want [][]float32) float64 {
		w := 1.0
		for i := range want {
			w = min(w, cosine(out[i], want[i]))
		}
		return w
	}
	for _, c := range g.Cases {
		fv, bv := dflashWindowHashed(1, c.Ctx*h), dflashWindowHashed(2, g.Block*h)
		fused, block := make([][]float32, c.Ctx), make([][]float32, g.Block)
		for i := range fused {
			fused[i] = fv[i*h : (i+1)*h]
		}
		for i := range block {
			block[i] = bv[i*h : (i+1)*h]
		}
		ctx := d.NewContext()
		d.ExtendContext(be, ctx, fused) // the context's K/V do not depend on the masks: projected once for all three runs
		draft := func(c []bool, w []int) [][]float32 {
			d.layerCausal, d.layerWindow = c, w
			out, err := d.DraftBlockCtx(be, ctx, block)
			if err != nil {
				t.Fatal(err)
			}
			return out
		}
		on := worst(draft(causal, window), c.Out)
		off := worst(draft(nil, nil), c.Out)
		noWin := worst(draft(causal, make([]int, len(window))), c.Out)
		t.Logf("ctx %d: worst row cosine %.7f with the masks, %.7f full attention, %.7f causal without the window", c.Ctx, on, off, noWin)
		if on < 0.99999 {
			t.Errorf("ctx %d: worst row cosine %.7f to z-lab's reference, want >= 0.99999", c.Ctx, on)
		}
		if off >= 0.99999 {
			t.Errorf("ctx %d: plant (every layer full) matched the reference at %.7f; the gate cannot see the masks", c.Ctx, off)
		}
		if past := c.Ctx+g.Block > window[0]; (noWin >= 0.99999) == past {
			t.Errorf("ctx %d: plant (no window) at %.7f; it should fail exactly when the context passes the window (%v)", c.Ctx, noWin, past)
		}
	}
	d.layerCausal, d.layerWindow = causal, window
}
