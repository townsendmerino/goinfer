//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPagedNoLogits_skipsHeadAndKeepsKV: on a paged MoE (the Gemma 4 pager, the one M26 runs, and the generic pager)
// ForwardNoLogits skips the final norm and LM head (task-m26-mac-2026-10.md) and leaves the KV exactly as Forward
// does. Arm A runs Forward at every position; arm B runs ForwardNoLogits for all but the last prompt position, as the
// decoder's sequential prompt loop does, then Forward. The last prompt position's logits and every decode step after
// it must be equal bit for bit, and B's head must not have run: the host logits buffer, set to NaN before each
// ForwardNoLogits, is still NaN after it.
func TestPagedNoLogits_skipsHeadAndKeepsKV(t *testing.T) {
	for _, c := range []struct {
		name, ckpt string
		env        map[string]string
		gemma4     bool
	}{
		{"gemma4", "../testdata/gemma4-moe-tiny", map[string]string{"GOINFER_GEMMA4_RESIDENT": "1"}, true},
		{"generic", "../decoder/testdata/qwen3_5_moe-tiny", nil, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, err := os.Stat(c.ckpt + "/model.safetensors"); err != nil {
				t.Skipf("no fixture (%s): %v", c.ckpt, err)
			}
			for k, v := range c.env {
				t.Setenv(k, v)
			}
			t.Setenv("GOINFER_METAL_MOE_SLOTS", "2")
			m, err := decoder.Load(c.ckpt, decoder.Options{Backend: "metal", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			mr, ok := m.ResidentForwardForTest().(*metalResident)
			if !ok {
				t.Fatalf("not metal-resident: %s", m.ResidentDecline())
			}
			if paged := (c.gemma4 && mr.r.g4moe != nil && mr.r.g4moe.paged) || (!c.gemma4 && mr.r.moe != nil && mr.r.moe.paged); !paged {
				t.Fatal("expected a paged MoE build")
			}
			const promptLen, decodeSteps = 5, 3 // twoGeomPrompt has 8
			toks := twoGeomPrompt[:promptLen+decodeSteps]
			run := func(noLogits bool) [][]float32 {
				mr.Reset()
				var got [][]float32
				for i, tok := range toks {
					emb := m.EmbedResidentForTest(tok)
					if noLogits && i < promptLen-1 {
						for j := range mr.r.logitsHost {
							mr.r.logitsHost[j] = float32(math.NaN())
						}
						if err := mr.ForwardNoLogits(emb, i); err != nil {
							t.Fatalf("ForwardNoLogits[%d]: %v", i, err)
						}
						for j, v := range mr.r.logitsHost {
							if !math.IsNaN(float64(v)) {
								t.Fatalf("ForwardNoLogits[%d] ran the LM head (host logit %d = %v)", i, j, v)
							}
						}
						continue
					}
					lg, err := mr.Forward(emb, i)
					if err != nil {
						t.Fatalf("Forward[%d]: %v", i, err)
					}
					if i >= promptLen-1 {
						got = append(got, append([]float32(nil), lg...))
					}
				}
				return got
			}
			a, b := run(false), run(true)
			for s := range a {
				for j := range a[s] {
					if math.Float32bits(a[s][j]) != math.Float32bits(b[s][j]) {
						t.Fatalf("step %d logit %d: ForwardNoLogits prompt %v, Forward prompt %v", s, j, b[s][j], a[s][j])
					}
				}
			}
			t.Logf("%s: %d prompt positions without the head, then %d positions' logits bit-identical", c.name, promptLen-1, len(a))
		})
	}
}
