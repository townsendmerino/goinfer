//go:build darwin

package metal

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestAutoMoESlotsFor is the pure-formula gate for M-13 (audit-metal-2026-09-12.md): no real
// machine holds less RAM than a real MoE model's dense term, so the formula is tested directly
// against a chosen ram value rather than through a live sysctl read.
func TestAutoMoESlotsFor(t *testing.T) {
	const gb = int64(1) << 30
	const topK = 4

	t.Run("ample RAM clamps to the ceiling", func(t *testing.T) {
		// perSlot tiny relative to a normal machine's RAM: the raw quotient is far above
		// autoMoESlotsMax, so the ceiling — not the formula's raw division — should win.
		got := autoMoESlotsFor(topK, 10*1024*1024, 2*gb, 16*uint64(gb))
		if got != autoMoESlotsMax {
			t.Errorf("autoMoESlotsFor = %d, want the ceiling %d", got, autoMoESlotsMax)
		}
	})

	t.Run("tight RAM produces a small N above topK", func(t *testing.T) {
		// 8 GB RAM, 0.7 budget = 5.6 GB; needFixed 4 GB leaves 1.6 GB; perSlot 200 MB ⇒ 8 slots.
		got := autoMoESlotsFor(topK, 200*1024*1024, 4*gb, 8*uint64(gb))
		if got != 8 {
			t.Errorf("autoMoESlotsFor = %d, want 8", got)
		}
	})

	t.Run("clamped up to topK when the raw quotient undershoots it", func(t *testing.T) {
		// Budget leaves only ~1 slot's worth of room, but a token's own routed set needs topK
		// simultaneously resident — the floor must win, not the raw (too-small) quotient.
		got := autoMoESlotsFor(topK, 1*gb, 4*gb, 8*uint64(gb))
		if got != topK {
			t.Errorf("autoMoESlotsFor = %d, want topK=%d (floor)", got, topK)
		}
	})

	t.Run("the fixed part alone doesn't fit: still returns topK, not zero or negative", func(t *testing.T) {
		got := autoMoESlotsFor(topK, 1*gb, 100*gb, 8*uint64(gb))
		if got != topK {
			t.Errorf("autoMoESlotsFor = %d, want topK=%d", got, topK)
		}
	})

	t.Run("degenerate perSlot<=0 falls back to the ceiling, not a divide-by-zero", func(t *testing.T) {
		got := autoMoESlotsFor(topK, 0, 2*gb, 16*uint64(gb))
		if got != autoMoESlotsMax {
			t.Errorf("autoMoESlotsFor = %d, want the ceiling %d", got, autoMoESlotsMax)
		}
	})
}

// TestMoeTopK_realFixtures confirms moeTopK reads the right Config field for both MoE shapes this
// backend admits — generic (mixtral-tiny) and Gemma-4 (gemma4-moe-tiny, if present).
func TestMoeTopK_realFixtures(t *testing.T) {
	m, err := decoder.Load("../testdata/mixtral-tiny", decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load mixtral-tiny: %v", err)
	}
	defer m.Close()
	_, wantK, _, _, _, _, _, _, _, _, ok := m.MoEResidentParams()
	if !ok {
		t.Fatal("mixtral-tiny did not report MoE resident params")
	}
	if got := moeTopK(m); got != wantK {
		t.Errorf("moeTopK(mixtral-tiny) = %d, want %d (from MoEResidentParams)", got, wantK)
	}

	const g4 = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(g4); err != nil {
		t.Skipf("no gemma4-moe-tiny fixture: %v", err)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	mg, err := decoder.Load(g4, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load gemma4-moe-tiny: %v", err)
	}
	defer mg.Close()
	if !mg.HasGemma4MoEResident() {
		t.Fatal("gemma4-moe-tiny did not resolve to a Gemma-4 MoE resident")
	}
	if want := mg.Config().TopKExperts; want > 0 {
		if got := moeTopK(mg); got != want {
			t.Errorf("moeTopK(gemma4-moe-tiny) = %d, want %d (Config.TopKExperts)", got, want)
		}
	}
}

// TestMetalMoESlotsRequest_autoOnlyWithCacheExperts is the wiring half of M-13: --moe-cache-experts
// alone (no --moe-cache-slots) must now reach autoMoESlots and produce a real, clamped slot
// request — not the old "" (unpaged, every expert resident) metalMoESlotsRequest returned before
// this fix, regardless of what number the machine's own RAM happens to produce.
func TestMetalMoESlotsRequest_autoOnlyWithCacheExperts(t *testing.T) {
	m, err := decoder.Load("../testdata/mixtral-tiny", decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if m.MoECacheExperts() {
		t.Fatal("test setup: MoECacheExperts should be false without the option")
	}
	if got := metalMoESlotsRequest(m); got != "" {
		t.Errorf("metalMoESlotsRequest with MoECacheExperts unset = %q, want \"\" (unchanged default)", got)
	}

	mAuto, err := decoder.Load("../testdata/mixtral-tiny", decoder.Options{Quant: "int4", MoECacheExperts: true})
	if err != nil {
		t.Fatalf("load with MoECacheExperts: %v", err)
	}
	defer mAuto.Close()
	if !mAuto.MoECacheExperts() {
		t.Fatal("test setup: MoECacheExperts should be true")
	}
	if mAuto.MoECacheSlotsRequest() != 0 {
		t.Fatal("test setup: no explicit --moe-cache-slots should have been set")
	}
	got := metalMoESlotsRequest(mAuto)
	if got == "" {
		t.Fatal("metalMoESlotsRequest with MoECacheExperts set returned \"\" — auto-sizing did not engage (M-13 regressed)")
	}
	t.Logf("auto-derived slot request on mixtral-tiny: %q", got)
}

// The sizer must choose a slot count the guard then accepts, under the guard's OWN budget. With a
// live budget tighter than the static fraction (memory pressure), sizing against the static figure
// picked more slots than the live budget holds — the guard refused, and the load fell back to CPU.
func TestAutoMoESlots_fitsTheGuardsBudget(t *testing.T) {
	const gb = int64(1 << 30)
	topK, perSlot, needFixed := 4, int64(200*1024*1024), 4*gb
	ram := uint64(16 * gb)
	static := metalStaticCeiling(ram)
	for _, budget := range []int64{static, static / 2, needFixed + 10*perSlot} {
		n := autoMoESlotsForBudget(topK, perSlot, needFixed, budget)
		if n > topK && needFixed+int64(n)*perSlot > budget {
			t.Errorf("budget %.2f GB: chose %d slots needing %.2f GB — the guard would refuse it",
				float64(budget)/float64(gb), n, float64(needFixed+int64(n)*perSlot)/float64(gb))
		}
	}
	if a, b := autoMoESlotsForBudget(topK, perSlot, needFixed, static/2), autoMoESlotsForBudget(topK, perSlot, needFixed, static); a >= b {
		t.Errorf("a tighter budget did not choose fewer slots: %d vs %d", a, b)
	}
}
