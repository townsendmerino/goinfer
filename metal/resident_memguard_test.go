//go:build darwin

package metal

import (
	"errors"
	"io/fs"
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestResidentMemGuard pins BOTH directions of the fits-in-memory guard, because each failure
// mode is real and they are opposite: a guard that never fires leaves the swap-exhaustion hang
// it exists to prevent, and one that fires too eagerly silently moves every model to the CPU
// path — a large, invisible slowdown that no test would otherwise catch.
func TestResidentMemGuard(t *testing.T) {
	const gb = int64(1) << 30
	cases := []struct {
		name string
		need int64
		ram  uint64
		want bool
	}{
		// THE MEASURED CASE. gpt-oss-20b's 11.28 GB of weights on a 16 GB MacBook drove swap to
		// 35.98 GB of 36 GB and never completed or declined. It must be refused.
		{"gptoss20b_on_16gb", 11280 * gb / 1000, 16 * uint64(gb), false},
		// Ordinary models on the same machine must still go resident. qwen2.5-coder 0.5B/1.5B
		// int4 are ~0.4/1.2 GB; a guard that refused these would be worse than none.
		{"qwen_0_5b_on_16gb", 400 * gb / 1000, 16 * uint64(gb), true},
		{"qwen_1_5b_on_16gb", 1200 * gb / 1000, 16 * uint64(gb), true},
		// The SAME model fits a bigger machine — the guard is about the ratio, not the model.
		{"gptoss20b_on_64gb", 11280 * gb / 1000, 64 * uint64(gb), true},
		// Exactly at the bar passes; a hair over does not.
		{"exactly_at_bar", 70 * gb / 10, 10 * uint64(gb), true},
		{"just_over_bar", 71 * gb / 10, 10 * uint64(gb), false},
		// Unknown inputs must never refuse: an unreadable hw.memsize or a model reporting zero
		// bytes would otherwise disable residency for everyone, silently.
		{"unknown_ram", 8 * gb, 0, true},
		{"unknown_need", 0, 16 * uint64(gb), true},
		{"negative_need", -1, 16 * uint64(gb), true},
	}
	for _, c := range cases {
		if got := fitsResidentBudget(c.need, c.ram); got != c.want {
			t.Errorf("%s: fitsResidentBudget(%.2f GB, %.0f GB) = %v, want %v",
				c.name, float64(c.need)/float64(gb), float64(c.ram)/float64(gb), got, c.want)
		}
	}
}

// tinyDenseModelWithMoESlots loads a minimal dense fixture (genTinyWeights/writeDense,
// metal/moe_model_test.go) with the given Options.MoECacheSlots — real MoE structure is
// irrelevant to what's under test here (the slot REQUEST's resolution, not expert dispatch), so
// the simplest already-available fixture stands in for "any *decoder.Model".
func tinyDenseModelWithMoESlots(t *testing.T, slots int) *decoder.Model {
	t.Helper()
	dir := t.TempDir()
	writeDense(t, dir, genTinyWeights(rand.New(rand.NewSource(1))))
	m, err := decoder.Load(dir, decoder.Options{Quant: "int8int8", MoECacheSlots: slots})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// TestMetalMoESlotsFromEnv is M-02's gate for the guard's half of the ordering fix:
// residentFitsMemory must ask ResidentWeightBytesPaged for the SAME N that metal/moe.go and
// metal/gemma4_moe.go are about to honor, not silently fall back to the unpaged number on
// anything it cannot parse cleanly. Mirrors those two files' resolution exactly (metalMoESlotsRequest),
// except an invalid/unset value means "assume unpaged" here (safe: buildResident still validates
// and declines on a bad value) rather than a hard error.
//
// Phase 2 (docs/task-gpu-paths-2026-09.md — "Metal slots become an Option and a flag"): the
// env-var cases below now go through a model whose Options.MoECacheSlots is 0 (unset), so
// metalMoESlotsRequest's fallback to GOINFER_METAL_MOE_SLOTS is what's actually exercised — an
// additional case pins the NEW priority order directly (Options wins over the env var when both
// are set).
func TestMetalMoESlotsFromEnv(t *testing.T) {
	for _, tc := range []struct {
		name, val string
		optSlots  int
		want      int
	}{
		{"unset", "", 0, 0},
		{"valid", "64", 0, 64},
		{"zero", "0", 0, 0},
		{"negative", "-1", 0, 0},
		{"not a number", "sixty-four", 0, 0},
		{"Options wins over a conflicting env var", "16", 64, 64},
		{"Options alone, no env var", "", 64, 64},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.val == "" {
				orig, wasSet := os.LookupEnv("GOINFER_METAL_MOE_SLOTS")
				os.Unsetenv("GOINFER_METAL_MOE_SLOTS")
				t.Cleanup(func() {
					if wasSet {
						os.Setenv("GOINFER_METAL_MOE_SLOTS", orig)
					}
				})
			} else {
				t.Setenv("GOINFER_METAL_MOE_SLOTS", tc.val)
			}
			m := tinyDenseModelWithMoESlots(t, tc.optSlots)
			if got := metalMoESlotsFromEnv(m); got != tc.want {
				t.Errorf("metalMoESlotsFromEnv() with env=%q Options.MoECacheSlots=%d = %d, want %d", tc.val, tc.optSlots, got, tc.want)
			}
		})
	}
}

// TestResidentNeedBytes_honorsPagingSlots is M-02's gate for the actual wiring gap: reverting
// residentNeedBytes to always call ResidentWeightBytes() (the pre-fix behavior) compiles clean
// and TestMetalMoESlotsFromEnv above still passes, because that test only exercises the parsing
// function in isolation — it never proves the guard USES what it parses. This does, by loading a
// real (tiny) MoE checkpoint and comparing residentNeedBytes' output against
// ResidentWeightBytes/ResidentWeightBytesPaged/ResidentHostCopyBytes/residentKVBytes directly,
// with no real RAM or a checkpoint large enough to swing residentFitsMemory's verdict required.
//
// 2026-09-09 (M-02 continued): residentNeedBytes gained two more additive terms (the host-copy
// addend and KV bytes) beside the paged-weight term this test originally gated alone — so "==
// unpaged weight bytes" is no longer residentNeedBytes' own contract; the assertions below add
// the SAME two terms back in, computed independently via the public accessors, so this still
// catches a regression in the paging wiring specifically without needing to be rewritten every
// time another additive term is found.
func TestResidentNeedBytes_honorsPagingSlots(t *testing.T) {
	// testdata/gemma4-moe-tiny is gitignored (a real, if small, checkpoint) — never present in CI,
	// so skip rather than fail when it's absent, matching decoder's own convention for this fixture.
	const ckpt = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no tiny checkpoint (%s) — run scripts/pin_gemma4_moe_forward.py", ckpt)
	}
	m, err := decoder.Load(ckpt, decoder.Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	unpaged := m.ResidentWeightBytes()
	kv := residentKVBytes(m)

	t.Run("unset env == unpaged", func(t *testing.T) {
		orig, wasSet := os.LookupEnv("GOINFER_METAL_MOE_SLOTS")
		os.Unsetenv("GOINFER_METAL_MOE_SLOTS")
		t.Cleanup(func() {
			if wasSet {
				os.Setenv("GOINFER_METAL_MOE_SLOTS", orig)
			}
		})
		want := unpaged + m.ResidentHostCopyBytes(0) + kv
		if got := residentNeedBytes(m); got != want {
			t.Errorf("residentNeedBytes() with no slots env = %d, want unpaged+hostcopy+kv %d", got, want)
		}
	})

	t.Run("slots=1 matches ResidentWeightBytesPaged and is strictly smaller", func(t *testing.T) {
		t.Setenv("GOINFER_METAL_MOE_SLOTS", "1")
		wantWeights := m.ResidentWeightBytesPaged(1)
		if wantWeights >= unpaged {
			t.Fatalf("test fixture has too few experts to make this case meaningful (paged(1)=%d, unpaged=%d)", wantWeights, unpaged)
		}
		want := wantWeights + m.ResidentHostCopyBytes(1) + kv
		if got := residentNeedBytes(m); got != want {
			t.Errorf("residentNeedBytes() with GOINFER_METAL_MOE_SLOTS=1 = %d, want %d (paged weights+hostcopy+kv) — "+
				"the guard is not asking for the paged estimate", got, want)
		}
		// The host-copy addend must ITSELF shrink under paging (that's the whole point of M-02's
		// distinction): paged experts stream, so ResidentHostCopyBytes(1) must be strictly smaller
		// than ResidentHostCopyBytes(0) whenever paging actually caps anything on this fixture.
		if hc0, hc1 := m.ResidentHostCopyBytes(0), m.ResidentHostCopyBytes(1); hc1 >= hc0 {
			t.Errorf("ResidentHostCopyBytes(1)=%d not smaller than ResidentHostCopyBytes(0)=%d — "+
				"paging should exempt streamed experts from the host-copy addend", hc1, hc0)
		}
	})
}
