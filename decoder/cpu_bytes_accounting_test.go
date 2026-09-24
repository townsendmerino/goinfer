//go:build goinfer_testhooks

package decoder

import (
	"os"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestCPUDecode_weightBytesPerToken reports how many bytes of resident weight each decode
// component streams per token, from the LOADED model's real storage (not a size formula), so a
// per-component ms/token split (GOINFER_DECODE_TIMING's DECODE SPLIT lines) can be turned into an
// achieved GB/s and compared against the box's read ceiling. It asserts nothing: it is an
// instrument. docs/measurements/cpu-decode-roofline-2026-09-23.md is what it feeds.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./decoder/ -run TestCPUDecode_weightBytesPerToken -v
func TestCPUDecode_weightBytesPerToken(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads real checkpoints on CPU)")
	}
	models := []struct{ name, env, def string }{
		{"0.5B", "GOINFER_CPU_MODEL_05B", "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"},
		{"1.5B", "GOINFER_CPU_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
		{"7B", "GOINFER_CPU_MODEL_D7", "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf"},
	}
	for _, mc := range models {
		t.Run(mc.name, func(t *testing.T) {
			path := os.Getenv(mc.env)
			if path == "" {
				path = os.ExpandEnv(mc.def)
			}
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no fixture at %s", path)
			}
			m, err := Load(path, Options{Backend: "cpu", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()

			type acc struct{ params, packed, scales int64 }
			var qkv, o, gu, dn acc
			kinds := map[string]int{}
			add := func(a *acc, w *linalg.WeightMat) {
				r, c := int64(w.Rows()), int64(w.Cols())
				if r == 0 {
					return
				}
				a.params += r * c
				kinds[w.Kind()]++
				if q4, s, _, ok := w.Int4(); ok {
					a.packed += int64(len(q4))
					a.scales += int64(len(s)) * 4
				} else if q8, s, _, ok := w.Int8(); ok {
					a.packed += int64(len(q8))
					a.scales += int64(len(s)) * 4
				}
			}
			w := m.Weights()
			for i := range w.Layers {
				ly := &w.Layers[i]
				add(&qkv, &ly.QProj)
				add(&qkv, &ly.KProj)
				add(&qkv, &ly.VProj)
				add(&o, &ly.OProj)
				add(&gu, &ly.GateProj)
				add(&gu, &ly.UpProj)
				add(&dn, &ly.DownProj)
			}
			var head acc
			if w.LMHead.Rows() > 0 {
				add(&head, &w.LMHead)
			} else {
				add(&head, &w.Embed) // tied: the embedding IS the LM head
			}
			total := int64(0)
			row := func(name string, a acc) {
				b := a.packed + a.scales
				total += b
				perParam := 0.0
				if a.params > 0 {
					perParam = float64(b) / float64(a.params)
				}
				t.Logf("BYTES %-5s %-9s params %8.1fM  packed %8.1f MB  scales %6.1f MB  total %8.1f MB  (%.3f B/param)",
					mc.name, name, float64(a.params)/1e6, float64(a.packed)/1e6, float64(a.scales)/1e6, float64(b)/1e6, perParam)
			}
			row("q/k/v", qkv)
			row("o", o)
			row("gate+up", gu)
			row("down", dn)
			row("LM head", head)
			t.Logf("BYTES %-5s TOTAL streamed per token: %.1f MB   kinds=%v", mc.name, float64(total)/1e6, kinds)
		})
	}
}
