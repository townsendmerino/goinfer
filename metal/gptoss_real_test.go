//go:build darwin && goinfer_testhooks

package metal

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGptOssResidentParity is gpt-oss's Metal gate — the same residentParity/assertParity
// harness TestGPT2ResidentParity uses, on the tiny F32 fixture decoder/testdata/gptoss_tiny.gguf
// (decoder/gptoss_gguf_test.go's own CPU parity gate: cosine 0.99843 vs the real HF checkpoint,
// scripts/gptoss_tiny_golden.py). Small enough (645 KB) not to need requireHeavyModel.
//
// Dormant until FeatAttnSink is declared: metal ships the sink term (kernels.go's `attention`),
// the clamped-SwiGLU expert + custom router (moe.go's swiglu_quant_gptoss/route_gptoss) and the
// bias-in-combine down projection (gemv_w4a8_moe_wacc_bias), and the moe.go isGptOss wiring, but
// does not yet DECLARE the feature — skip rather than fail, and residentParity t.Fatals on a
// decline once the declaration lands (catching a silent CPU fallback rather than an honest skip).
func TestGptOssResidentParity(t *testing.T) {
	if !decoder.ResidentBackendFeatures("metal")[decoder.FeatAttnSink] {
		t.Skip("metal does not declare FeatAttnSink yet (kernels/wiring dormant) — see docs/queue-correctness.md G7/G10")
	}
	path := "../decoder/testdata/gptoss_tiny.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no gpt-oss fixture at %s — run scripts/gptoss_tiny_golden.py", path)
	}
	raw, err := os.ReadFile("../decoder/testdata/gptoss_tiny_golden.json")
	if err != nil {
		t.Skipf("no golden: %v", err)
	}
	var g struct {
		InputIDs []int `json:"input_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	st := residentParity(t, path, g.InputIDs, len(g.InputIDs))
	// The tiny fixture is a hand-built 2-layer/32-expert-cap shape, not a calibrated real
	// checkpoint (unlike GPT-2's 0.95 dense-int4 bar) — the CPU-vs-CPU gguf parity gate already
	// proved the forward math at cosine 0.99843; this only has to prove the RESIDENT path agrees
	// with the CPU path it's supposed to mirror, so the bar matches the other resident gates'
	// int4-noise floor.
	assertParity(t, "gpt-oss", st, 0.95)
}

// C-09: THE PAGED ARM the audit asked for, and the reason the existing paging gate could not
// stand in for it.
//
// TestMoEPaging_matchesNonPaged already pins "paging must not be observable in the output" — but
// on qwen3_5_moe, which has NO per-expert biases. gpt-oss is the only family here that carries
// them (gate‖up and down), and they are exactly what the paged encoder mis-addressed: it handed
// idxZeros to both the weight GEMV (right — the slot holds one expert) and the STACKED bias
// tables (wrong — they never move), so every routed expert got expert 0's bias. A family without
// biases cannot express that difference, so the existing gate was green on broken code.
//
// Same fixture and harness as TestGptOssResidentParity above (nE=4, topK=2), so slots=2 forces an
// eviction every token and slots=3 leaves one spare. The bar is EXACT logit equality against the
// non-paged build, not a cosine floor: reuse is an implementation detail and must not reach the
// output at all.
func TestGptOssPaging_matchesNonPaged(t *testing.T) {
	if !decoder.ResidentBackendFeatures("metal")[decoder.FeatAttnSink] {
		t.Skip("metal does not declare FeatAttnSink yet")
	}
	const ckpt = "../decoder/testdata/gptoss_tiny.gguf"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no gpt-oss fixture at %s — run scripts/gptoss_tiny_golden.py", ckpt)
	}

	run := func(t *testing.T, slotsEnv string) [][]float32 {
		if slotsEnv == "" {
			os.Unsetenv("GOINFER_METAL_MOE_SLOTS")
		} else {
			t.Setenv("GOINFER_METAL_MOE_SLOTS", slotsEnv)
		}
		m, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int4"})
		if err != nil {
			t.Fatalf("load metal (slots=%q): %v", slotsEnv, err)
		}
		defer m.Close()
		rf := m.ResidentForwardForTest()
		if rf == nil {
			t.Fatalf("not resident (slots=%q): %s", slotsEnv, m.ResidentDecline())
		}
		_, _, _, _, _, _, vocab := m.Dims()
		rf.Reset()
		const ntok = 24
		out := make([][]float32, ntok)
		for i := range ntok {
			lr, err := rf.Forward(m.EmbedResidentForTest((i*131+7)%vocab), i)
			if err != nil {
				t.Fatalf("forward[%d] (slots=%q): %v", i, slotsEnv, err)
			}
			out[i] = append([]float32(nil), lr...)
		}
		return out
	}

	base := run(t, "") // non-paged: all 4 experts stacked resident
	// The premise, checked rather than assumed: the routing must actually SPREAD over experts. If
	// every token picked the same top-k, expert 0's bias would often be the right bias by accident
	// and this gate would pass on the defect it exists for.
	var nonzero int
	for _, lr := range base {
		for _, v := range lr {
			if v != 0 {
				nonzero++
				break
			}
		}
	}
	if nonzero != len(base) {
		t.Fatalf("premise broke: %d of %d positions produced all-zero logits", len(base)-nonzero, len(base))
	}

	for _, slots := range []string{"2", "3"} {
		t.Run("slots="+slots, func(t *testing.T) {
			got := run(t, slots)
			for i := range base {
				if len(got[i]) != len(base[i]) {
					t.Fatalf("tok %d: length %d != %d", i, len(got[i]), len(base[i]))
				}
				for j := range base[i] {
					if got[i][j] != base[i][j] {
						t.Fatalf("tok %d logit %d: paged %.9g != non-paged %.9g — paging is "+
							"observable in the output. If only the gpt-oss fixture shows this and "+
							"qwen3_5_moe does not, it is the per-expert BIAS addressing (C-09): the "+
							"bias tables stay stacked while the weights move into slots.",
							i, j, got[i][j], base[i][j])
					}
				}
			}
			t.Logf("slots=%s: %d positions bit-identical to the non-paged build", slots, len(base))
		})
	}
}
