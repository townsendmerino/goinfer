//go:build darwin && goinfer_testhooks

package metal

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/loadflags"
)

// g3MetalNotBuilt are the Metal cells whose fixture builds no Metal resident (a decline outside KV), so the model runs on the CPU
// cache; each is read back against what the model reports delivering (docs/measurements/kv-precision-2026-10/g3.md).
var g3MetalNotBuilt = []string{
	"phi3-tiny/auto", "phi3-tiny/f32", "phi3-tiny/f16", "phi3-tiny/i8",
	"lfm2-tiny/auto", "lfm2-tiny/f32",
	"gemma4-moe-kv-tiny/auto", "gemma4-moe-kv-tiny/f32",
}

// g3MetalPendingD5 are the Metal cells whose cache is not the precision the matrix says: MLA's latent is f32 whatever is asked (D5,
// docs/tasks/task-kv-precision-everywhere-2026-10.md §6). The test requires them to still differ, so a fix shows up here.
// g3MetalMatrixDiffers are the no-resident cells whose CPU cache is not what the matrix's Metal cell says: the matrix reads the
// family's Metal answer, the load runs the CPU's (G3's record).
var g3MetalMatrixDiffers = []string{
	"phi3-tiny/auto", "phi3-tiny/f32", "lfm2-tiny/auto", "lfm2-tiny/f32", "gemma4-moe-kv-tiny/auto", "gemma4-moe-kv-tiny/f32",
}

var g3MetalPendingD5 = []string{"deepseek-tiny/auto", "deepseek-tiny/f16"}

// TestKVDeliveredBytes_Metal is G3's Metal half (docs/tasks/task-kv-precision-everywhere-2026-10.md §4): every Metal cell the
// matrix delivers, loaded with serve's options for --kv and a 100-position resident context, its K/V buffers read back against the
// delivered precision: f16 2 B and f32 4 B per element, int8 1 B plus one f32 scale per position and head for each of K and V; an
// MLA layer's latent at 4 B. A cell whose resident declines is read back on the CPU cache against the model's KVDelivered.
func TestKVDeliveredBytes_Metal(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	fixtures := []string{"llama-tiny", "phi3-tiny", "qwen3moe-tiny", "gemma2-tiny", "cohere-tiny", "qwen25vl-tiny", "deepseek-tiny",
		"qwen35-tiny", "qwen3next-tiny", "nemotron-tiny", "lfm2-tiny", "gemma4-moe-kv-tiny"}
	var notBuilt, mismatched, matrixDiffers []string
	checked := 0
	for _, f := range fixtures {
		for _, v := range []string{"auto", "f32", "f16", "i8"} {
			cell := f + "/" + v
			kv := v
			if v == "auto" {
				kv = ""
			}
			m, err := decoder.Load(filepath.Join("..", "testdata", f), decoder.Options{Backend: "metal", Quant: "int4", KVPrecision: kv, KVQuant: loadflags.CPUKV("", kv), ResidentContext: 100})
			var r *decoder.KVRefusal
			if errors.As(err, &r) {
				continue // a refused cell allocates nothing
			}
			if err != nil {
				t.Fatalf("%s: Load: %v", cell, err)
			}
			want := m.KVDeliveredFor("metal")
			a, ok := m.ResidentForwardForTest().(*metalResident)
			if !ok {
				notBuilt = append(notBuilt, cell)
				got := m.KVDelivered()
				n32, n16, n8, nScale, err := m.KVReadbackForTest(12)
				if err != nil {
					t.Fatalf("%s: CPU readback: %v", cell, err)
				}
				held := map[[4]bool]string{{true, false, false, false}: "f32", {false, true, false, false}: "f16", {false, false, true, true}: "i8"}[[4]bool{n32 > 0, n16 > 0, n8 > 0, nScale > 0}]
				_, _, _, _, hd, _, _ := m.Dims()
				if held != got || (held == "i8" && nScale*int64(hd) != n8) {
					t.Errorf("%s: no resident; the CPU cache reports %s and holds f32 %d, f16 %d, i8 %d, scales %d", cell, got, n32, n16, n8, nScale)
				}
				if held != want {
					matrixDiffers = append(matrixDiffers, cell)
				}
				fmt.Fprintf(os.Stderr, "[G3 metal] %-24s no resident: runs on CPU, holds %-3s, the matrix's Metal cell says %q\n", cell, held, want)
				m.Close()
				continue
			}
			res := a.r
			padded := int64((res.ctxCap + 7) / 8 * 8)
			why := ""
			for l := range res.kc {
				L := res.layers[l]
				if L.kvShared || res.kc[l].Len() == 0 {
					continue
				}
				if res.mla != nil { // the latent rows, f32 (metal/mla.go); V aliases them
					if got, latent := int64(res.kc[l].Len()), padded*int64(res.mla.latDim)*4; got != latent {
						why = fmt.Sprintf("layer %d holds %d B of latent, not %d B at f32", l, got, latent)
					} else if want != "f32" {
						why = fmt.Sprintf("layer %d holds an f32 latent, the matrix says %s", l, want)
					}
					continue
				}
				elems := padded * int64(L.geom.kvDim)
				per := map[string]int64{"f16": 2, "f32": 4, "i8": 1}[want]
				if got := int64(res.kc[l].Len() + res.vc[l].Len()); got != 2*elems*per {
					why = fmt.Sprintf("layer %d holds %d B of K/V for %d elements, not %d B each (%s)", l, got, elems, per, want)
				}
				nScale := int64(0)
				if l < len(res.ks) {
					nScale = int64(res.ks[l].Len() + res.vs[l].Len())
				}
				if wantScale := map[bool]int64{true: 2 * padded * int64(L.geom.nKV), false: 0}[want == "i8"]; nScale != wantScale {
					why = fmt.Sprintf("layer %d holds %d scales, want %d (%s)", l, nScale, wantScale, want)
				}
			}
			fmt.Fprintf(os.Stderr, "[G3 metal] %-24s delivered %-3s %s\n", cell, want, map[bool]string{true: "ok", false: "MISMATCH: " + why}[why == ""])
			if why != "" {
				mismatched = append(mismatched, cell)
				if !slices.Contains(g3MetalPendingD5, cell) {
					t.Errorf("%s: %s", cell, why)
				}
			}
			checked++
			m.Close()
		}
	}
	for _, c := range g3MetalPendingD5 {
		if !slices.Contains(mismatched, c) {
			t.Errorf("%s now holds what the matrix says: D5 is resolved here, so drop it from g3MetalPendingD5", c)
		}
	}
	if !slices.Equal(matrixDiffers, g3MetalMatrixDiffers) {
		t.Errorf("the no-resident cells whose CPU cache differs from the matrix's Metal cell are %v, the list says %v", matrixDiffers, g3MetalMatrixDiffers)
	}
	if !slices.Equal(notBuilt, g3MetalNotBuilt) {
		t.Errorf("the cells that build no Metal resident are %v, the list says %v", notBuilt, g3MetalNotBuilt)
	}
	if checked < 20 {
		t.Fatalf("only %d Metal cells were read back", checked)
	}
	fmt.Fprintf(os.Stderr, "[G3 metal] %d cells read back, %d build no resident, %d pending D5\n", checked, len(notBuilt), len(g3MetalPendingD5))
}
