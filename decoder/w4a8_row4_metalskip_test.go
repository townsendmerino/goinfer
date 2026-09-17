package decoder

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
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
// exactly as before this change. o_proj/down_proj are ALSO now row4-free on Metal (M-07's second
// half, 2026-09-13: quantizeWMSkipRow4 threaded through quantizeWM's ~40 family-specific call
// sites) — see TestW4A8Row4_skippedForMetalBackend_MoE for router/expert coverage, which this
// dense-only fixture doesn't exercise.
func TestW4A8Row4_skippedForMetalBackend(t *testing.T) {
	if !linalg.Int4Row4Usable(4, int4GroupSize, int4GroupSize) {
		t.Skip("row4 layout is arm64 dotprod only; non-arm64 builds never build row4")
	}
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
			if _, _, _, ok := wm.Int4(); !ok {
				t.Fatalf("layer %d: o_proj/down_proj has no canonical int4 bytes on Backend:\"metal\"", i)
			}
		}
	}
	if metalBatchedRow4 != 0 {
		t.Errorf("Backend:\"metal\": %d/%d Q/K/V/gate/up projections still have row4 — M-07's skip did not take", metalBatchedRow4, metalBatchedTotal)
	}
	if metalOtherRow4 != 0 {
		t.Errorf("Backend:\"metal\": %d/%d o_proj/down_proj projections still have row4 — M-07's second-half skip did not take",
			metalOtherRow4, metalOtherTotal)
	}

	var unspecBatchedRow4, unspecBatchedTotal int
	var unspecOtherRow4, unspecOtherTotal int
	for i := range mUnspecified.w.Layers {
		l := &mUnspecified.w.Layers[i]
		for _, wm := range []*linalg.WeightMat{&l.QProj, &l.KProj, &l.VProj, &l.GateProj, &l.UpProj} {
			unspecBatchedTotal++
			if _, _, ok := wm.Int4Row4(); ok {
				unspecBatchedRow4++
			}
		}
		for _, wm := range []*linalg.WeightMat{&l.OProj, &l.DownProj} {
			unspecOtherTotal++
			if _, _, ok := wm.Int4Row4(); ok {
				unspecOtherRow4++
			}
		}
	}
	if unspecBatchedRow4 != unspecBatchedTotal {
		t.Errorf("Backend:\"\" (unspecified): Q/K/V/gate/up row4 count = %d/%d, want %d/%d — "+
			"the unspecified-backend default must be UNCHANGED by this fix (canonical+row4 both)",
			unspecBatchedRow4, unspecBatchedTotal, unspecBatchedTotal, unspecBatchedTotal)
	}
	if unspecOtherRow4 != unspecOtherTotal {
		t.Errorf("Backend:\"\" (unspecified): o_proj/down_proj row4 count = %d/%d, want %d/%d — "+
			"unspecified-backend default must be UNCHANGED (canonical+row4 both)",
			unspecOtherRow4, unspecOtherTotal, unspecOtherTotal, unspecOtherTotal)
	}

	t.Logf("Backend:\"metal\": %d/%d Q/K/V/gate/up + %d/%d o_proj/down_proj all row4-free; "+
		"Backend:\"\": %d/%d + %d/%d unaffected (still row4)",
		metalBatchedTotal-metalBatchedRow4, metalBatchedTotal, metalOtherTotal-metalOtherRow4, metalOtherTotal,
		unspecBatchedRow4, unspecBatchedTotal, unspecOtherRow4, unspecOtherTotal)
}

// TestW4A8Row4_skippedForMetalBackend_MoE extends the above to router/MoE-expert/shared-expert
// tensors (M-07's second half, audit-metal-2026-09-12.md) on the two tiny MoE fixtures already
// used elsewhere for MoE correctness testing — prequantGGUF's dense fixture has neither. Covers
// both the generic-MoE safetensors path (loadMatQ, buildWeightsFromSafetensors) and gemma4's own
// (loadGemma4MoE/streamExperts).
func TestW4A8Row4_skippedForMetalBackend_MoE(t *testing.T) {
	if !linalg.Int4Row4Usable(4, int4GroupSize, int4GroupSize) {
		t.Skip("row4 layout is arm64 dotprod only; non-arm64 builds never build row4")
	}
	for _, ckpt := range []string{"testdata/qwen3_5_moe-tiny", "../testdata/gemma4-moe-tiny"} {
		t.Run(ckpt, func(t *testing.T) {
			if _, err := os.Stat(filepath.Join(ckpt, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
				t.Skipf("no checkpoint at %s (model.safetensors gitignored) — run its scripts/pin_*.py", ckpt)
			}
			mMetal, err := Load(ckpt, Options{Quant: "int4", Backend: "metal"})
			if err != nil {
				t.Fatalf("load metal: %v", err)
			}
			mUnspecified, err := Load(ckpt, Options{Quant: "int4"})
			if err != nil {
				t.Fatalf("load unspecified: %v", err)
			}

			row4Free := func(wm *linalg.WeightMat) bool {
				_, _, ok := wm.Int4Row4()
				return !ok
			}
			hasCanonical := func(wm *linalg.WeightMat) bool {
				_, _, _, ok := wm.Int4()
				return ok
			}
			// collect every router/expert/shared-expert WeightMat this fixture actually has,
			// from BOTH the generic MoE (Experts/Router/SharedExpert) and gemma4's own
			// (gemma4moe.expertsGateUp/expertsDown) shapes — whichever this ckpt uses.
			collect := func(m *Model) []*linalg.WeightMat {
				var out []*linalg.WeightMat
				for i := range m.w.Layers {
					l := &m.w.Layers[i]
					if l.Router.Rows() > 0 {
						out = append(out, &l.Router)
					}
					for e := range l.Experts {
						out = append(out, &l.Experts[e].Gate, &l.Experts[e].Up, &l.Experts[e].Down)
					}
					if l.SharedExpert.Gate.Rows() > 0 {
						out = append(out, &l.SharedExpert.Gate, &l.SharedExpert.Up, &l.SharedExpert.Down)
					}
					if l.gemma4moe != nil {
						for e := range l.gemma4moe.expertsGateUp {
							out = append(out, &l.gemma4moe.expertsGateUp[e], &l.gemma4moe.expertsDown[e])
						}
					}
				}
				return out
			}
			metalTensors := collect(mMetal)
			if len(metalTensors) == 0 {
				t.Fatal("no router/expert/shared-expert tensors found — fixture shape changed, this test no longer discriminates")
			}
			for i, wm := range metalTensors {
				// Router (f32, logit-critical) never reaches int4 at all — Int4Row4 correctly
				// reports ok=false for it regardless of this fix, so only check canonical-int4
				// tensors have canonical bytes; row4-freedom is checked for every tensor either way
				// since a non-int4 tensor's Int4Row4() is a harmless always-false no-op.
				if !row4Free(wm) {
					t.Errorf("Backend:\"metal\" tensor %d: still has row4 — M-07's MoE skip did not take", i)
				}
				if wm.IsInt4() && !hasCanonical(wm) {
					t.Errorf("Backend:\"metal\" tensor %d: int4 but no canonical bytes — skipping row4 must never drop canonical", i)
				}
			}

			unspecTensors := collect(mUnspecified)
			int4Total, int4Row4 := 0, 0
			for _, wm := range unspecTensors {
				if !wm.IsInt4() {
					continue
				}
				int4Total++
				if !row4Free(wm) {
					int4Row4++
				}
			}
			if int4Total == 0 {
				t.Fatal("no int4 router/expert tensors on the unspecified-backend load — fixture/quant mismatch")
			}
			if int4Row4 != int4Total {
				t.Errorf("Backend:\"\" (unspecified): %d/%d int4 router/expert tensors have row4, want %d/%d — "+
					"unspecified-backend default must be UNCHANGED", int4Row4, int4Total, int4Total, int4Total)
			}
			t.Logf("%s: %d Metal tensors all row4-free; unspecified backend %d/%d int4 tensors row4 (unaffected)",
				ckpt, len(metalTensors), int4Row4, int4Total)
		})
	}
}
