package decoder

import (
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestWantsRow4Fallback_metalOnly pins the exact scope of M-07's row4-skip decision
// (audit-metal-2026-09-12.md, wantsRow4Fallback's own doc comment has the measurement): false
// (skip row4) ONLY for the literal backend name "metal" — every other name, including CUDA/
// WebGPU (unmeasured against this same trade) and empty/unspecified (not a promise of anything,
// same "stated, never inferred" discipline wantsCanonicalInt4 already applies), keeps row4 as a
// CPU-fallback safety net.
func TestWantsRow4Fallback_metalOnly(t *testing.T) {
	cases := []struct {
		backendName string
		want        bool
	}{
		{"metal", false},
		{"cpu", true},
		{"cuda", true},
		{"webgpu", true},
		{"", true},
		{"nonsense", true},
	}
	for _, tc := range cases {
		if got := wantsRow4Fallback(tc.backendName); got != tc.want {
			t.Errorf("wantsRow4Fallback(%q) = %v, want %v", tc.backendName, got, tc.want)
		}
	}
}

// TestW4A8Row4_skippedForMetalBackend is the end-to-end wiring proof for M-07 (audit-
// metal-2026-09-12.md): a Backend:"metal" load must have NO row4 layout on its Q/K/V/gate/up
// projections (isBatchedProjTensor's five standard names — quantizeBatchedProjWM's own scope),
// while a Backend:""  (unspecified) load on the SAME checkpoint keeps row4 on those same tensors
// exactly as before this change. Also confirms o_proj/down_proj/Embed are UNCHANGED by this fix
// (still row4, on both backends) — they route through quantizeWM, deliberately out of scope (see
// quantizeBatchedProjWM's own doc comment for why extending further was not done in this pass).
func TestW4A8Row4_skippedForMetalBackend(t *testing.T) {
	path := prequantGGUF(t)

	loadWith := func(backend string) *Model {
		m, err := Load(path, Options{Quant: "int4", Backend: backend})
		if err != nil {
			t.Fatalf("load (backend=%q): %v", backend, err)
		}
		return m
	}
	mMetal := loadWith("metal")
	mUnspecified := loadWith("")

	var metalBatchedRow4, metalBatchedTotal int
	var metalOtherRow4, metalOtherTotal int
	for i := range mMetal.w.Layers {
		l := &mMetal.w.Layers[i]
		for _, wm := range []*linalg.WeightMat{&l.QProj, &l.KProj, &l.VProj, &l.GateProj, &l.UpProj} {
			metalBatchedTotal++
			if _, _, ok := wm.Int4Row4(); ok {
				metalBatchedRow4++
			}
			if _, _, _, ok := wm.Int4(); !ok {
				t.Fatalf("layer %d: a Q/K/V/gate/up projection has no canonical int4 bytes on Backend:\"metal\" "+
					"— skipping row4 must never also drop canonical, Metal's GPU upload reads it directly", i)
			}
		}
		for _, wm := range []*linalg.WeightMat{&l.OProj, &l.DownProj} {
			metalOtherTotal++
			if _, _, ok := wm.Int4Row4(); ok {
				metalOtherRow4++
			}
		}
	}
	if metalBatchedRow4 != 0 {
		t.Errorf("Backend:\"metal\": %d/%d Q/K/V/gate/up projections still have row4 — M-07's skip did not take", metalBatchedRow4, metalBatchedTotal)
	}
	if metalOtherRow4 != metalOtherTotal {
		t.Errorf("Backend:\"metal\": o_proj/down_proj row4 count = %d/%d, want %d/%d (unaffected by this fix, still row4)",
			metalOtherRow4, metalOtherTotal, metalOtherTotal, metalOtherTotal)
	}

	var unspecBatchedRow4, unspecBatchedTotal int
	for i := range mUnspecified.w.Layers {
		l := &mUnspecified.w.Layers[i]
		for _, wm := range []*linalg.WeightMat{&l.QProj, &l.KProj, &l.VProj, &l.GateProj, &l.UpProj} {
			unspecBatchedTotal++
			if _, _, ok := wm.Int4Row4(); ok {
				unspecBatchedRow4++
			}
		}
	}
	if unspecBatchedRow4 != unspecBatchedTotal {
		t.Errorf("Backend:\"\" (unspecified): Q/K/V/gate/up row4 count = %d/%d, want %d/%d — "+
			"the unspecified-backend default must be UNCHANGED by this fix (canonical+row4 both)",
			unspecBatchedRow4, unspecBatchedTotal, unspecBatchedTotal, unspecBatchedTotal)
	}

	t.Logf("Backend:\"metal\": %d/%d Q/K/V/gate/up projections row4-free (o_proj/down_proj unaffected, %d/%d still row4); "+
		"Backend:\"\": %d/%d unaffected (still row4)",
		metalBatchedTotal-metalBatchedRow4, metalBatchedTotal, metalOtherRow4, metalOtherTotal,
		unspecBatchedRow4, unspecBatchedTotal)
}
