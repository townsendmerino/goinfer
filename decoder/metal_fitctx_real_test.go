//go:build realckpt

package decoder

import (
	"os"
	"strings"
	"testing"
)

// TestMetalFitCtx_anAgentsFirstRequestFits: on a 16 GB Mac's memory budget (0.70 of RAM, nothing else running), the
// 1.5B and the 7B at int4 resolve, with Metal's two default slots, a context that holds a coding agent's first request
// (opencode's 11,137 tokens). The checkpoints are the GGUFs, loaded onto the heap, which Metal prices with a host copy
// of the weights: the dearer of the two ways a load arrives (a sidecar's mapped weights carry none). Real checkpoints
// from GOINFER_MODELS_DIR (default ~/models); a subtest skips without its file, and when the load-time fit guard
// refuses to put the checkpoint on this machine's heap (the 7B on a 16 GB Mac), which is the guard's call and not a
// reading of the fit.
func TestMetalFitCtx_anAgentsFirstRequestFits(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the real 1.5B and 7B checkpoints)")
	}
	const firstRequest = 11137
	ram := float64(16 << 30)
	budget := int64(ram * WeightsMemFraction)
	for _, c := range []struct{ name, file string }{
		{"1.5B", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
		{"7B", "qwen2.5-7b-instruct-q4_k_m.gguf"},
	} {
		t.Run(c.name, func(t *testing.T) {
			path := modelPath(c.file)
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no checkpoint: %v", err)
			}
			m, err := Load(path, Options{Quant: "int4", ResidentContext: MetalCtxDefault})
			if err != nil && strings.Contains(err.Error(), "was NOT loaded") {
				t.Skipf("the load-time fit guard refused the checkpoint on this machine: %v", err)
			}
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			m.resCtxReq, m.resCtxPinned = 0, false // the pin kept the load-time guard from trimming; the fit is asked unpinned
			f := m.MetalFitCtx(MetalFitInput{FreeBytes: budget, FreeKnown: true, KVSlots: MetalDefaultKVSlots})
			p := m.Plan("metal", budget, PlanRequest{Ctx: f.Ctx, CtxPinned: true})
			t.Logf("%s: ctx %d (%s); weights %.2f GB + host copy %.2f GB of a %.2f GB budget", c.name, f.Ctx, f.Why,
				gb(p.DenseBytes), gb(p.HostCopyBytes), gb(budget))
			if !(f.Ctx > firstRequest) {
				t.Errorf("ctx %d does not hold an agent's first request of %d tokens", f.Ctx, firstRequest)
			}
		})
	}
}
