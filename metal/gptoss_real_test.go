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
