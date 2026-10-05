//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMoEExpertRows_bitIdentical is D-B04's routed-expert half's gate: the routed-expert GEMVs through R18's rows form
// (moeExpertRowsOn, gemv_w4a8_moe_rows / gemv_w4a8_moe_wacc_rows) against the one-row kernels, on the tiny Gemma 4 MoE
// and the tiny Qwen3.5 MoE, all experts resident and paged at 2 slots: every logit of 8 teacher-forced positions equal,
// with the rows form engaged (R > 0) on the rows arm and off on the other.
func TestMoEExpertRows_bitIdentical(t *testing.T) {
	for _, c := range []struct {
		name, ckpt string
		env        map[string]string
	}{
		{"gemma4", "../testdata/gemma4-moe-tiny", map[string]string{"GOINFER_GEMMA4_RESIDENT": "1"}},
		{"generic", "../decoder/testdata/qwen3_5_moe-tiny", nil},
	} {
		for _, slots := range []string{"", "2"} {
			t.Run(c.name+"/slots="+slots, func(t *testing.T) {
				if _, err := os.Stat(c.ckpt + "/model.safetensors"); err != nil {
					t.Skipf("no fixture (%s): %v", c.ckpt, err)
				}
				for k, v := range c.env {
					t.Setenv(k, v)
				}
				if slots == "" {
					os.Unsetenv("GOINFER_METAL_MOE_SLOTS")
				} else {
					t.Setenv("GOINFER_METAL_MOE_SLOTS", slots)
				}
				prev := moeExpertRowsOn
				defer func() { moeExpertRowsOn = prev }()
				run := func(on bool) [][]float32 {
					moeExpertRowsOn = on
					m, err := decoder.Load(c.ckpt, decoder.Options{Backend: "metal", Quant: "int4"})
					if err != nil {
						t.Fatalf("load: %v", err)
					}
					defer m.Close()
					mr, ok := m.ResidentForwardForTest().(*metalResident)
					if !ok {
						t.Fatalf("not metal-resident: %s", m.ResidentDecline())
					}
					guR, downR := -1, -1
					if g := mr.r.g4moe; g != nil {
						guR, downR = g.guR, g.downR
					} else if mo := mr.r.moe; mo != nil {
						guR, downR = mo.guR, mo.downR
					}
					if on && (guR <= 0 || downR <= 0) || !on && (guR != 0 || downR != 0) {
						t.Fatalf("rows form %v: gate|up R %d, down R %d", on, guR, downR)
					}
					mr.Reset()
					var out [][]float32
					for i, tok := range twoGeomPrompt {
						lg, err := mr.Forward(m.EmbedResidentForTest(tok), i)
						if err != nil {
							t.Fatal(err)
						}
						out = append(out, append([]float32(nil), lg...))
					}
					return out
				}
				off, on := run(false), run(true)
				for i := range off {
					for j := range off[i] {
						if math.Float32bits(off[i][j]) != math.Float32bits(on[i][j]) {
							t.Fatalf("position %d logit %d: rows %v, one-row %v", i, j, on[i][j], off[i][j])
						}
					}
				}
				t.Logf("%d positions, every logit equal with and without the rows form", len(off))
			})
		}
	}
}

// TestMoEExpertRows_int8BitIdentical: the int8 expert GEMVs' rows form (gemv_w8a8_moe_rows / gemv_w8a8_moe_wacc_rows)
// against the one-row int8 kernels on the native int8 MoE path (int8int8, nativeInt8MoE), on Mixtral and Qwen3-MoE:
// every logit of 8 positions equal, the rows form engaged on the rows arm.
func TestMoEExpertRows_int8BitIdentical(t *testing.T) {
	for _, fx := range []string{"../testdata/mixtral-tiny", "../testdata/qwen3moe-tiny"} {
		t.Run(fx, func(t *testing.T) {
			if _, err := os.Stat(fx + "/model.safetensors"); err != nil {
				t.Skipf("no fixture: %v", err)
			}
			os.Unsetenv("GOINFER_METAL_MOE_SLOTS")
			prevR, prevN, prevM := moeExpertRowsOn, nativeInt8, nativeInt8MoE
			defer func() { moeExpertRowsOn, nativeInt8, nativeInt8MoE = prevR, prevN, prevM }()
			nativeInt8, nativeInt8MoE = true, true
			run := func(on bool) [][]float32 {
				moeExpertRowsOn = on
				m, err := decoder.Load(fx, decoder.Options{Backend: "metal", Quant: "int8int8"})
				if err != nil {
					t.Fatalf("load: %v", err)
				}
				defer m.Close()
				mr, ok := m.ResidentForwardForTest().(*metalResident)
				if !ok || mr.r.moe == nil || !mr.r.moe.w8 {
					t.Fatalf("not the native int8 MoE path (%s)", m.DecodePath())
				}
				if mo := mr.r.moe; on && (mo.guR <= 0 || mo.downR <= 0) || !on && (mo.guR != 0 || mo.downR != 0) {
					t.Fatalf("rows form %v: gate|up R %d, down R %d", on, mo.guR, mo.downR)
				}
				mr.Reset()
				var out [][]float32
				_, _, _, _, _, _, vocab := m.Dims()
				for i := range 8 {
					lg, err := mr.Forward(m.EmbedResidentForTest((i*37+3)%vocab), i)
					if err != nil {
						t.Fatal(err)
					}
					out = append(out, append([]float32(nil), lg...))
				}
				return out
			}
			off, on := run(false), run(true)
			for i := range off {
				for j := range off[i] {
					if math.Float32bits(off[i][j]) != math.Float32bits(on[i][j]) {
						t.Fatalf("position %d logit %d: rows %v, one-row %v", i, j, on[i][j], off[i][j])
					}
				}
			}
			t.Logf("8 positions, every logit equal with and without the int8 rows form")
		})
	}
}
