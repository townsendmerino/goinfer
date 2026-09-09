//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMoESlotsViaOptions_engagesPaging is Phase 2's own gate (docs/task-gpu-paths-2026-09.md —
// "Metal slots become an Option and a flag"): decoder.Options.MoECacheSlots (the SAME
// --moe-cache-slots CUDA's own auto-cap already reads) must reach the REAL dispatch-building code
// in metal/moe.go and metal/gemma4_moe.go, not just residentNeedBytes' guard estimate
// (TestMetalMoESlotsFromEnv, metal/resident_memguard_test.go, already covers the guard half in
// isolation). Confirms on the real gemma4-moe-tiny fixture that paging actually ENGAGES from
// Options alone, with GOINFER_METAL_MOE_SLOTS explicitly unset so the env var cannot be silently
// doing the work instead.
func TestMoESlotsViaOptions_engagesPaging(t *testing.T) {
	const ckpt = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no fixture (%s)", ckpt)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	os.Unsetenv("GOINFER_METAL_MOE_SLOTS")

	mProbe, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load probe: %v", err)
	}
	nE, topK := 0, 0
	for l := range 64 {
		if _, _, e, k, _, ok := mProbe.Gemma4MoERouterForTest(l); ok {
			nE, topK = e, k
			break
		}
	}
	mProbe.Close()
	if nE <= topK {
		t.Skipf("fixture nE=%d <= topK=%d — can't force paging; width fixture (nE=128) needed", nE, topK)
	}
	N := topK + topK/2
	if N >= nE {
		N = nE - 1
	}
	if N < topK {
		N = topK
	}

	m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4", MoECacheSlots: N})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if got := m.MoECacheSlotsRequest(); got != N {
		t.Fatalf("MoECacheSlotsRequest() = %d, want %d — Options.MoECacheSlots not wired through decoder.Load", got, N)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	defer r.Close()
	if r.g4moe == nil {
		t.Fatal("resident has no gemma4moe state")
	}
	if !r.g4moe.paged {
		t.Fatalf("expected paging to engage from Options.MoECacheSlots=%d alone (env unset) — nE=%d topK=%d", N, nE, topK)
	}
	if r.g4moe.slots != N {
		t.Errorf("r.g4moe.slots = %d, want %d (the Options request, not the env var default)", r.g4moe.slots, N)
	}
}

// TestMoESlotsViaOptions_engagesPaging_genericMoE is the generic (non-gemma4) MoE twin of the
// test above — metal/moe.go's own real dispatch-engagement check (as opposed to
// gemma4_moe.go's), which shares metalMoESlotsRequest's resolution but is otherwise a completely
// separate code path (a separate moeResident struct, a separate paged/slots pair). Runs
// unconditionally in CI: testdata/mixtral-tiny is TRACKED in git (nE=8, topK=2 — a real,
// meaningful paging case), unlike gemma4-moe-tiny above.
func TestMoESlotsViaOptions_engagesPaging_genericMoE(t *testing.T) {
	const ckpt = "../testdata/mixtral-tiny"
	os.Unsetenv("GOINFER_METAL_MOE_SLOTS")

	mProbe, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load probe: %v", err)
	}
	nE, topK, _, _, _, _, _, _, _, _, ok := mProbe.MoEResidentParams()
	mProbe.Close()
	if !ok || nE <= topK {
		t.Fatalf("fixture nE=%d topK=%d (ok=%v) — expected a real paging-eligible MoE shape", nE, topK, ok)
	}
	N := topK + 1
	if N >= nE {
		t.Fatalf("fixture too narrow to force paging: nE=%d, wanted N=%d < nE", nE, N)
	}

	m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4", MoECacheSlots: N})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if got := m.MoECacheSlotsRequest(); got != N {
		t.Fatalf("MoECacheSlotsRequest() = %d, want %d — Options.MoECacheSlots not wired through decoder.Load", got, N)
	}
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	defer r.Close()
	if r.moe == nil {
		t.Fatal("resident has no generic MoE state")
	}
	if !r.moe.paged {
		t.Fatalf("expected paging to engage from Options.MoECacheSlots=%d alone (env unset) — nE=%d topK=%d", N, nE, topK)
	}
	if r.moe.slots != N {
		t.Errorf("r.moe.slots = %d, want %d (the Options request, not the env var default)", r.moe.slots, N)
	}
}

// TestMoESlotsViaOptions_belowTopKRefusesWithNumbers is G6 (docs/task-gpu-paths-2026-09.md §6):
// an explicit --moe-cache-slots below top-k cannot be honoured (one token's own routed set must
// be simultaneously resident) — it must be REFUSED, with the numbers, never silently rounded up
// or ignored. Uses testdata/mixtral-tiny (tracked, runs in CI unconditionally); the request comes
// through decoder.Options, not the deprecated env var, so this is specifically testing the NEW
// Phase 2 surface's error path, not the pre-existing env-var one metal/moe.go already had.
func TestMoESlotsViaOptions_belowTopKRefusesWithNumbers(t *testing.T) {
	const ckpt = "../testdata/mixtral-tiny"
	os.Unsetenv("GOINFER_METAL_MOE_SLOTS")

	mProbe, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load probe: %v", err)
	}
	_, topK, _, _, _, _, _, _, _, _, ok := mProbe.MoEResidentParams()
	mProbe.Close()
	if !ok || topK < 1 {
		t.Fatalf("fixture has no usable topK (topK=%d, ok=%v)", topK, ok)
	}
	below := topK - 1
	if below < 0 {
		t.Skipf("topK=%d — no integer below it to request", topK)
	}

	m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4", MoECacheSlots: below})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	_, err = buildResident(m)
	if err == nil {
		t.Fatalf("BuildResident succeeded with MoECacheSlots=%d < topK=%d — should have refused", below, topK)
	}
	wantSub := []string{fmt.Sprintf("%d", below), fmt.Sprintf("topK=%d", topK)}
	for _, sub := range wantSub {
		if !strings.Contains(err.Error(), sub) {
			t.Errorf("error %q does not mention %q — a refusal must name the numbers, not just fail silently", err, sub)
		}
	}
}
