package decoder

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/runtime/giw"
)

// TestMoEHitRatePrior pins the calibration points against benchmarks.md §B4.1 directly (not a
// rederivation — the exact measured pairs) plus the required monotonic, clamped shape between and
// beyond them.
func TestMoEHitRatePrior(t *testing.T) {
	const eps = 1e-9
	for _, c := range []struct {
		name       string
		residency  float64
		want       float64
		approxOnly bool
	}{
		{"zero residency -> zero hit rate", 0, 0, false},
		{"negative residency clamps to zero", -1, 0, false},
		{"12.5% residency (16/128 slots) matches §B4.1", 0.125, 0.573, true},
		{"23.4% residency (30/128 slots) matches §B4.1", 0.234, 0.761, true},
		{"31.25% residency (40/128 slots) matches §B4.1", 0.3125, 0.822, true},
		{"full residency clamps below 1.0", 1.0, 0.999, false},
		{"over-full residency still clamps below 1.0", 5.0, 0.999, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := moeHitRatePrior(c.residency)
			if c.approxOnly {
				if math.Abs(got-c.want) > 0.01 {
					t.Errorf("moeHitRatePrior(%.4f) = %.4f, want ~%.3f (calibration point)", c.residency, got, c.want)
				}
				return
			}
			if math.Abs(got-c.want) > eps {
				t.Errorf("moeHitRatePrior(%.4f) = %.6f, want %.6f", c.residency, got, c.want)
			}
		})
	}
	t.Run("monotonic non-decreasing across the whole domain", func(t *testing.T) {
		prev := moeHitRatePrior(0)
		for r := 0.0; r <= 1.5; r += 0.01 {
			got := moeHitRatePrior(r)
			if got < prev-1e-12 {
				t.Fatalf("moeHitRatePrior(%.2f) = %.6f is LESS than moeHitRatePrior(prev) = %.6f — not monotonic", r, got, prev)
			}
			prev = got
		}
	})
}

// TestPredictedMoETokPerSec pins the arithmetic (1 / (missBytes/preadRate)) against a hand-computed
// case, plus every "don't know -> 0" edge this repo's own "every unknown proceeds" discipline
// requires.
func TestPredictedMoETokPerSec(t *testing.T) {
	t.Run("hand-computed case", func(t *testing.T) {
		// 1 GB active bytes/token, 50% hit rate -> 512 MB missed -> missBytes/preadRate seconds/token.
		const activeBytes = int64(1) << 30
		const hitRate = 0.5
		missBytes := float64(activeBytes) * (1 - hitRate)
		wantSec := missBytes / moePreadRateBytesPerSec
		want := 1 / wantSec
		got := predictedMoETokPerSec(activeBytes, hitRate)
		if math.Abs(got-want) > want*1e-9 {
			t.Errorf("predictedMoETokPerSec(1 GiB, 0.5) = %.6f, want %.6f", got, want)
		}
	})
	t.Run("zero/negative activeBytesPerToken is unknown", func(t *testing.T) {
		if got := predictedMoETokPerSec(0, 0.5); got != 0 {
			t.Errorf("predictedMoETokPerSec(0, 0.5) = %v, want 0", got)
		}
		if got := predictedMoETokPerSec(-1, 0.5); got != 0 {
			t.Errorf("predictedMoETokPerSec(-1, 0.5) = %v, want 0", got)
		}
	})
	t.Run("hitRate clamped to 0.999, never reaches +Inf", func(t *testing.T) {
		got := predictedMoETokPerSec(1<<30, 1.0)
		if got <= 0 || math.IsInf(got, 1) {
			t.Errorf("predictedMoETokPerSec(1 GiB, 1.0) = %v, want a finite positive number (clamped hit rate)", got)
		}
	})
	t.Run("negative hitRate clamps to 0, not treated as free", func(t *testing.T) {
		got := predictedMoETokPerSec(1<<30, -1)
		want := predictedMoETokPerSec(1<<30, 0)
		if got != want {
			t.Errorf("predictedMoETokPerSec(1 GiB, -1) = %v, want == hitRate=0 case (%v)", got, want)
		}
	})
	t.Run("more active bytes predicts a slower rate, all else equal", func(t *testing.T) {
		small := predictedMoETokPerSec(1<<20, 0.5)
		big := predictedMoETokPerSec(1<<30, 0.5)
		if big >= small {
			t.Errorf("predictedMoETokPerSec did not decrease with more active bytes: small=%.4f big=%.4f", small, big)
		}
	})
}

// TestMoEWorkingSetRefusal is the pure threshold-gate test moeworkingset.go's own doc comment
// promises: synthetic predicted rates, no fixture needed, because a fixture genuinely large enough
// to predict a rate below moeSlowTokPerSecThreshold is exactly what this session cannot safely
// load (see S5's own scoping note in the task doc).
func TestMoEWorkingSetRefusal(t *testing.T) {
	orig := moeSlowTokPerSecThreshold
	moeSlowTokPerSecThreshold = 2.0
	t.Cleanup(func() { moeSlowTokPerSecThreshold = orig })

	t.Run("below floor without acceptSlow refuses", func(t *testing.T) {
		err := moeWorkingSetRefusal("my-model", 1.0, false)
		if err == nil {
			t.Fatal("moeWorkingSetRefusal(1.0 tok/s, acceptSlow=false) = nil, want a refusal")
		}
		if !strings.Contains(err.Error(), "my-model") || !strings.Contains(err.Error(), "1.00 tok/s") {
			t.Errorf("refusal message missing the model name or predicted rate: %v", err)
		}
	})
	t.Run("below floor WITH acceptSlow proceeds", func(t *testing.T) {
		if err := moeWorkingSetRefusal("my-model", 1.0, true); err != nil {
			t.Errorf("moeWorkingSetRefusal(1.0 tok/s, acceptSlow=true) = %v, want nil", err)
		}
	})
	t.Run("at or above floor proceeds regardless of acceptSlow", func(t *testing.T) {
		if err := moeWorkingSetRefusal("my-model", 2.0, false); err != nil {
			t.Errorf("moeWorkingSetRefusal(2.0 tok/s == floor, acceptSlow=false) = %v, want nil", err)
		}
		if err := moeWorkingSetRefusal("my-model", 50.0, false); err != nil {
			t.Errorf("moeWorkingSetRefusal(50.0 tok/s, acceptSlow=false) = %v, want nil", err)
		}
	})
	t.Run("predicted <= 0 (don't know) never refuses", func(t *testing.T) {
		if err := moeWorkingSetRefusal("my-model", 0, false); err != nil {
			t.Errorf("moeWorkingSetRefusal(0, acceptSlow=false) = %v, want nil (unknown proceeds)", err)
		}
		if err := moeWorkingSetRefusal("my-model", -1, false); err != nil {
			t.Errorf("moeWorkingSetRefusal(-1, acceptSlow=false) = %v, want nil (unknown proceeds)", err)
		}
	})
}

// TestLoad_moeWorkingSetGateWiring is item 5's own real-Load wiring gate. gemma4-moe-tiny
// (testdata/, gitignored but present on this box) is the only local checkpoint small enough to
// build a real paged-MoE .giw without the risk this session's own scoping note excludes — but at
// its real (tiny) byte counts it always predicts a fast rate, so moeSlowTokPerSecThreshold is
// pushed absurdly high for this one test to force the refusal branch, proving Load actually reads
// moeWorkingSetPrediction's real numbers and calls moeWorkingSetRefusal with them — not just that
// the two functions exist in isolation (TestMoEWorkingSetRefusal above already covers that).
// int8GIWFixture serializes the safetensors checkpoint ckpt at int8int8 into a .giw under t.TempDir and returns its path.
func int8GIWFixture(t *testing.T, ckpt string) string {
	t.Helper()
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no %s fixture: %v", ckpt, err)
	}
	m0, err := Load(ckpt, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load safetensors: %v", err)
	}
	blob, err := SerializeWeights(m0.w, "moe-workingset-fixture")
	m0.Close()
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	giwPath := filepath.Join(t.TempDir(), filepath.Base(ckpt)+".giw")
	if err := os.WriteFile(giwPath, giw.Write(blob, nil), 0o644); err != nil {
		t.Fatalf("write .giw: %v", err)
	}
	return giwPath
}

// TestLoad_mappedMoEWorkingSetRefusal: a MoE .giw loaded the default way (mapped, no -stream-weights) and decoded on the CPU
// is held to the same working-set refusal as a paged one: below the floor it refuses and names --accept-slow, AcceptSlowMoE
// loads it, a prediction above the floor loads it, and a dense .giw is never asked.
func TestLoad_mappedMoEWorkingSetRefusal(t *testing.T) {
	moe := int8GIWFixture(t, "../testdata/mixtral-tiny")
	dense := int8GIWFixture(t, "../testdata/llama-tiny")
	t.Run("a prediction above the floor loads", func(t *testing.T) {
		m, err := Load(moe, Options{Backend: "cpu"})
		if err != nil {
			t.Fatalf("Load refused at the real floor: %v", err)
		}
		m.Close()
	})
	orig := moeSlowTokPerSecThreshold
	moeSlowTokPerSecThreshold = 1e18 // above any prediction: every MoE load is "slow"
	t.Cleanup(func() { moeSlowTokPerSecThreshold = orig })
	t.Run("refuses without AcceptSlowMoE", func(t *testing.T) {
		m, err := Load(moe, Options{Backend: "cpu"})
		if err == nil {
			m.Close()
			t.Fatal("the default mapped MoE load proceeded below the floor")
		}
		if !strings.Contains(err.Error(), "tok/s") || !strings.Contains(err.Error(), "--accept-slow") {
			t.Errorf("refusal does not say why or name --accept-slow: %v", err)
		}
	})
	t.Run("AcceptSlowMoE loads it, mapped and unpaged", func(t *testing.T) {
		m, err := Load(moe, Options{Backend: "cpu", AcceptSlowMoE: true})
		if err != nil {
			t.Fatalf("Load with AcceptSlowMoE refused: %v", err)
		}
		defer m.Close()
		if m.pager != nil || m.mmap == nil {
			t.Errorf("pager %v, mapping %d bytes: want the mapped load, unpaged", m.pager != nil, len(m.mmap))
		}
	})
	t.Run("a dense .giw is never refused", func(t *testing.T) {
		m, err := Load(dense, Options{Backend: "cpu"})
		if err != nil {
			t.Fatalf("a dense .giw was refused: %v", err)
		}
		m.Close()
	})
}

func TestLoad_moeWorkingSetGateWiring(t *testing.T) {
	giwPath := int8GIWFixture(t, "../testdata/mixtral-tiny")

	orig := moeSlowTokPerSecThreshold
	moeSlowTokPerSecThreshold = 1e18 // absurdly high: even this fixture's fast prediction is "slow"
	t.Cleanup(func() { moeSlowTokPerSecThreshold = orig })

	t.Run("refuses without AcceptSlowMoE", func(t *testing.T) {
		_, err := Load(giwPath, Options{StreamWeights: true, WeightCacheBytes: 1 << 20})
		if err == nil {
			t.Fatal("Load succeeded even though moeSlowTokPerSecThreshold was pushed above any possible prediction — the gate is not wired into Load")
		}
		if !strings.Contains(err.Error(), "tok/s") {
			t.Errorf("refusal error does not look like the working-set refusal: %v", err)
		}
	})
	t.Run("AcceptSlowMoE bypasses the same refusal", func(t *testing.T) {
		m, err := Load(giwPath, Options{StreamWeights: true, WeightCacheBytes: 1 << 20, AcceptSlowMoE: true})
		if err != nil {
			t.Fatalf("Load with AcceptSlowMoE=true refused anyway: %v", err)
		}
		defer m.Close()
		if m.pager == nil {
			t.Fatal("Load succeeded but built no pager")
		}
	})
}

// TestMappedMoEWorkingSet_isThePagersArithmetic: at the same requested budget, the mapped load's prediction is the paged
// load's, input for input (the experts' bytes and count, the budget a pager takes), at a budget under the expert total and
// at the automatic one.
func TestMappedMoEWorkingSet_isThePagersArithmetic(t *testing.T) {
	path := int8GIWFixture(t, "../testdata/mixtral-tiny")
	mapped, err := Load(path, Options{Backend: "cpu"})
	if err != nil {
		t.Fatal(err)
	}
	defer mapped.Close()
	total, _, n := mappedExpertBytes(mapped.w, mapped.mmap)
	if total <= 0 || n == 0 {
		t.Fatalf("VOID: the fixture maps no expert bytes (total %d, %d experts)", total, n)
	}
	for _, budget := range []int64{total / 3, 0} {
		paged, err := Load(path, Options{Backend: "cpu", StreamWeights: true, WeightCacheBytes: budget, AcceptSlowMoE: true})
		if err != nil {
			t.Fatal(err)
		}
		if paged.pager == nil || paged.pager.total != total || paged.pager.nExperts != n {
			t.Fatalf("budget %d: the pager manages %+v; the mapped accounting reads %d bytes over %d experts", budget, paged.pager, total, n)
		}
		want, wok := moeWorkingSetPrediction(paged)
		got, gok := mappedMoEWorkingSetPrediction(mapped, resolveWeightCacheBudget(budget))
		paged.Close()
		if !wok || !gok || got != want {
			t.Errorf("budget %d: mapped predicts %v (ok %v), the pager %v (ok %v)", budget, got, gok, want, wok)
		}
	}
}
