//go:build darwin

package metal

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPagingBudget reports per-expert W4A8 bytes + the largest sustainable per-layer slot count N
// against recommendedMaxWorkingSetSize (~10.6 GB), from the REAL 26B tensor shapes. N=64 fits with
// ~3.1 GB headroom before KV — the CUDA 38-slot cap (8 GB card) does not bound the Metal budget.
func TestPagingBudget(t *testing.T) {
	if os.Getenv("GOINFER_BUDGET_PROBE") == "" {
		t.Skip("set GOINFER_BUDGET_PROBE=1")
	}
	giw := modelPath("gemma4-26b-int4.giw") // GOINFER_MODELS_DIR (default $HOME/models); see modelsdir_test.go (G-06)
	if _, err := os.Stat(giw); err != nil {
		t.Skip("no .giw")
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	m, err := decoder.Load(giw, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	// first MoE layer bundle
	var b decoder.Gemma4MoEResidentBundle
	nMoE := 0
	firstL := -1
	for l := range 64 {
		if bb, ok := m.Gemma4MoEResidentLayer(l); ok {
			if firstL < 0 {
				b = bb
				firstL = l
			}
			nMoE++
		}
	}
	// per-expert Metal (W4A8) bytes: gate|up + down, from int4DirectWords
	guW, guS, _ := int4DirectWords(b.ExpertsGateUp[0])
	dW, dS, _ := int4DirectWords(b.ExpertsDown[0])
	perGU := len(guW)*4 + len(guS)*2
	perDown := len(dW)*4 + len(dS)*2
	perExpert := perGU + perDown
	mb := func(bytes int) float64 { return float64(bytes) / (1 << 20) }
	gb := func(bytes float64) float64 { return bytes / (1 << 30) }
	t.Logf("gemma4-26b: %d MoE layers, nE=%d, moeInter/hidden via shapes", nMoE, b.NE)
	t.Logf("per-expert W4A8: gate|up=%.2f MB + down=%.2f MB = %.2f MB", mb(perGU), mb(perDown), mb(perExpert))
	t.Logf("full expert set (nE=%d x %d layers): %.2f GB", b.NE, nMoE, gb(float64(perExpert)*float64(b.NE)*float64(nMoE)))

	const budget = 10.6 * (1 << 30) // recommendedMaxWorkingSetSize ~10.6 GB
	const alwaysOn = 1.5 * (1 << 30)
	kvCtx := 4096
	// KV bytes: f16 (kvF32? gemma uses f32 KV) — assume f32 KV for gemma: 2 * kvDim * ctx * nL * 4
	// rough: use hidden-ish; report both f16 and f32
	for _, N := range []int{32, 38, 48, 64, 80} {
		slots := float64(perExpert) * float64(N) * float64(nMoE)
		total := alwaysOn + slots
		t.Logf("  N=%-3d slots=%.2f GB  + alwaysOn 1.5 GB = %.2f GB  headroom vs 10.6 = %.2f GB (KV for ctx=%d extra)",
			N, gb(slots), gb(total), gb(budget-total), kvCtx)
	}
}
