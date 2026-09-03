//go:build darwin

package metal

import (
	"errors"
	"io/fs"
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

// TestMetalMoESlotsFromEnv is M-02's gate for the guard's half of the ordering fix:
// residentFitsMemory must ask ResidentWeightBytesPaged for the SAME N that metal/moe.go and
// metal/gemma4_moe.go are about to honor, not silently fall back to the unpaged number on
// anything it cannot parse cleanly. Mirrors those two files' os.Getenv/strconv.Atoi read exactly,
// except an invalid/unset value means "assume unpaged" here (safe: buildResident still validates
// and declines on a bad value) rather than a hard error.
func TestMetalMoESlotsFromEnv(t *testing.T) {
	for _, tc := range []struct {
		name, val string
		want      int
	}{
		{"unset", "", 0},
		{"valid", "64", 64},
		{"zero", "0", 0},
		{"negative", "-1", 0},
		{"not a number", "sixty-four", 0},
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
			if got := metalMoESlotsFromEnv(); got != tc.want {
				t.Errorf("metalMoESlotsFromEnv() with %q = %d, want %d", tc.val, got, tc.want)
			}
		})
	}
}

// TestResidentNeedBytes_honorsPagingSlots is M-02's gate for the actual wiring gap: reverting
// residentNeedBytes to always call ResidentWeightBytes() (the pre-fix behavior) compiles clean
// and TestMetalMoESlotsFromEnv above still passes, because that test only exercises the parsing
// function in isolation — it never proves the guard USES what it parses. This does, by loading a
// real (tiny) MoE checkpoint and comparing residentNeedBytes' output against
// ResidentWeightBytes/ResidentWeightBytesPaged directly, with no real RAM or a checkpoint large
// enough to swing residentFitsMemory's verdict required.
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

	t.Run("unset env == unpaged", func(t *testing.T) {
		orig, wasSet := os.LookupEnv("GOINFER_METAL_MOE_SLOTS")
		os.Unsetenv("GOINFER_METAL_MOE_SLOTS")
		t.Cleanup(func() {
			if wasSet {
				os.Setenv("GOINFER_METAL_MOE_SLOTS", orig)
			}
		})
		if got := residentNeedBytes(m); got != unpaged {
			t.Errorf("residentNeedBytes() with no slots env = %d, want unpaged %d", got, unpaged)
		}
	})

	t.Run("slots=1 matches ResidentWeightBytesPaged and is strictly smaller", func(t *testing.T) {
		t.Setenv("GOINFER_METAL_MOE_SLOTS", "1")
		want := m.ResidentWeightBytesPaged(1)
		if want >= unpaged {
			t.Fatalf("test fixture has too few experts to make this case meaningful (paged(1)=%d, unpaged=%d)", want, unpaged)
		}
		if got := residentNeedBytes(m); got != want {
			t.Errorf("residentNeedBytes() with GOINFER_METAL_MOE_SLOTS=1 = %d, want %d (ResidentWeightBytesPaged(1)) — "+
				"the guard is not asking for the paged estimate", got, want)
		}
	})
}
