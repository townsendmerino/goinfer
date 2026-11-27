package decoder

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/runtime/giw"
	"github.com/townsendmerino/goinfer/internal/runtime/paging"
	"github.com/townsendmerino/goinfer/internal/testgate"
)

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

// skipUnlessExpertsPage skips when this platform's pages leave the .giw at path no pageable expert bytes: a pager keeps
// only each tensor's page-aligned interior (mmap.PageAlignedInterior), and a tiny fixture's expert projections, smaller
// than an Apple silicon page, have none there. A real checkpoint's experts span many pages; this is a fixture-size fact.
func skipUnlessExpertsPage(t *testing.T, path string) {
	t.Helper()
	m, err := Load(path, Options{Backend: "cpu", AcceptSlowMoE: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if total, _, _ := paging.MappedExpertBytes(m.w, m.mmap); total == 0 {
		t.Skipf("%d-byte pages leave %s's expert tensors no page-aligned interior to page", os.Getpagesize(), filepath.Base(path))
	}
}

// TestLoad_mappedMoEWorkingSetRefusal: a MoE .giw loaded the default way (mapped, no -stream-weights) and decoded on the CPU
// is held to the same working-set refusal as a paged one: below the floor it refuses and names --accept-slow, AcceptSlowMoE
// loads it, a prediction above the floor loads it, and a dense .giw is never asked.
func TestLoad_mappedMoEWorkingSetRefusal(t *testing.T) {
	moe := int8GIWFixture(t, "../testdata/mixtral-tiny")
	skipUnlessExpertsPage(t, moe)
	dense := int8GIWFixture(t, "../testdata/llama-tiny")
	t.Run("a prediction above the floor loads", func(t *testing.T) {
		m, err := Load(moe, Options{Backend: "cpu"})
		if err != nil {
			t.Fatalf("Load refused at the real floor: %v", err)
		}
		m.Close()
	})
	testgate.SetForTest(t, &moeSlowTokPerSecThreshold, 1e18)
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
	skipUnlessExpertsPage(t, giwPath)

	testgate.SetForTest(t, &moeSlowTokPerSecThreshold, 1e18)

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
	skipUnlessExpertsPage(t, path)
	mapped, err := Load(path, Options{Backend: "cpu"})
	if err != nil {
		t.Fatal(err)
	}
	defer mapped.Close()
	total, _, n := paging.MappedExpertBytes(mapped.w, mapped.mmap)
	if total <= 0 || n == 0 {
		t.Fatalf("VOID: the fixture maps no expert bytes (total %d, %d experts)", total, n)
	}
	for _, budget := range []int64{total / 3, 0} {
		paged, err := Load(path, Options{Backend: "cpu", StreamWeights: true, WeightCacheBytes: budget, AcceptSlowMoE: true})
		if err != nil {
			t.Fatal(err)
		}
		if paged.pager == nil || paged.pager.TotalMappedBytes() != total || paged.pager.ExpertCount() != n {
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
