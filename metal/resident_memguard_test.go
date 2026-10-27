//go:build darwin

package metal

import (
	"errors"
	"io/fs"
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"strings"

	"golang.org/x/sys/unix"
	"path/filepath"
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
		// The measured failure behind residentMemFraction: gpt-oss-20b's weights on a 16 GB MacBook. It must be refused.
		{"gptoss20b_on_16gb", 11280 * gb / 1000, 16 * uint64(gb), false},
		// Ordinary models on the same machine (qwen2.5-coder 0.5B and 1.5B int4) must still go resident; a guard that refused these would
		// be worse than none.
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

// TestMetalMemoryCeiling_takesTheStricterBound pins that the combined ceiling equals the static ram*residentMemFraction figure
// whenever the live probe is unknown or looser, and the live figure whenever that is tighter, never wider: a live figure allowed to
// widen the ceiling would reopen the darwin-UBC "guard that inverts under pressure" failure metalMemoryCeiling's doc comment
// explains.
func TestMetalMemoryCeiling_takesTheStricterBound(t *testing.T) {
	const gb = int64(1) << 30
	const ram = 16 * uint64(gb) // static ceiling = 11.2 GB
	staticCeiling := metalStaticCeiling(ram)
	for _, c := range []struct {
		name string
		live int64
		want int64
	}{
		{"live unknown (0) leaves the static ceiling untouched", 0, staticCeiling},
		{"live negative (unknown) leaves the static ceiling untouched", -1, staticCeiling},
		{"live LOOSER than static does not widen the ceiling", 15 * gb, staticCeiling},
		{"live equal to static does not widen the ceiling", staticCeiling, staticCeiling},
		{"live TIGHTER than static wins", 3 * gb, 3 * gb},
		{"live tighter by a small margin still wins", staticCeiling - 1, staticCeiling - 1},
	} {
		t.Run(c.name, func(t *testing.T) {
			orig := metalLiveAvailable
			metalLiveAvailable = func() int64 { return c.live }
			t.Cleanup(func() { metalLiveAvailable = orig })
			if got := metalMemoryCeiling(ram); got != c.want {
				t.Errorf("metalMemoryCeiling(ram=%d GB) with live=%d = %d, want %d",
					ram/uint64(gb), c.live, got, c.want)
			}
		})
	}
	t.Run("ram=0 is unknown regardless of the live probe", func(t *testing.T) {
		orig := metalLiveAvailable
		metalLiveAvailable = func() int64 { return 3 * gb }
		t.Cleanup(func() { metalLiveAvailable = orig })
		if got := metalMemoryCeiling(0); got != 0 {
			t.Errorf("metalMemoryCeiling(0) = %d, want 0", got)
		}
	})
}

// TestResidentFitsMemory_honorsLiveCeiling pins the wiring: TestMetalMemoryCeiling_takesTheStricterBound proves the combined
// ceiling in isolation; this proves residentFitsMemory, the real load-time guard, reads through it. A residentFitsMemory reverted to a
// static-only budget would pass every other test in this file (they never touch metalLiveAvailable) while ignoring a live probe
// reporting almost no memory.
func TestResidentFitsMemory_honorsLiveCeiling(t *testing.T) {
	if os.Getenv("GOINFER_NO_RESIDENT_MEM_GUARD") != "" {
		t.Skip("GOINFER_NO_RESIDENT_MEM_GUARD is set in the environment — the guard is disabled, nothing to exercise")
	}
	m := tinyDenseModelWithMoESlots(t, 0)
	need := residentNeedBytes(m)
	if need <= 0 {
		t.Skip("tiny fixture reports zero resident bytes — nothing to price")
	}
	orig := metalLiveAvailable
	// Almost nothing available — far below what this real dev machine's static 70% ceiling would
	// allow for a model this tiny, so a decline here can ONLY come from the live probe actually
	// being consulted.
	metalLiveAvailable = func() int64 { return 1 }
	t.Cleanup(func() { metalLiveAvailable = orig })
	if residentFitsMemory(m) {
		t.Fatalf("residentFitsMemory() = true with metalLiveAvailable() = 1 byte (need=%d bytes) — "+
			"the guard is not reading the live-tightened ceiling", need)
	}
}

// TestResidentFitsMemory_refusesExactlyBelowTheReading: the guard's decision moves with the available-memory reading at a fixed
// need, whatever formula produced the reading (B41, docs/measurements/hostmem-available-2026-10/README.md, gate 3): a reading one
// byte short of the need refuses, the need itself and anything above it admits.
func TestResidentFitsMemory_refusesExactlyBelowTheReading(t *testing.T) {
	if os.Getenv("GOINFER_NO_RESIDENT_MEM_GUARD") != "" {
		t.Skip("GOINFER_NO_RESIDENT_MEM_GUARD is set in the environment: the guard is disabled, nothing to exercise")
	}
	m := tinyDenseModelWithMoESlots(t, 0)
	need := residentNeedBytes(m)
	if need <= 0 {
		t.Skip("tiny fixture reports zero resident bytes: nothing to price")
	}
	orig := metalLiveAvailable
	t.Cleanup(func() { metalLiveAvailable = orig })
	for _, c := range []struct {
		avail int64
		fits  bool
	}{{need - 1, false}, {need, true}, {2 * need, true}} {
		metalLiveAvailable = func() int64 { return c.avail }
		if got := residentFitsMemory(m); got != c.fits {
			t.Errorf("reading %d for a need of %d: fits %v, want %v", c.avail, need, got, c.fits)
		}
	}
}

// TestResidentMemoryDecline_workingSetCap: the guard refuses, by name, a load whose Metal allocation (the need less the host copy) is
// over the device's recommendedMaxWorkingSetSize, and admits one at it (B41 gate 4). The live ceiling is set far above the need so the
// cap is the only bound.
func TestResidentMemoryDecline_workingSetCap(t *testing.T) {
	if os.Getenv("GOINFER_NO_RESIDENT_MEM_GUARD") != "" {
		t.Skip("GOINFER_NO_RESIDENT_MEM_GUARD is set in the environment: the guard is disabled, nothing to exercise")
	}
	m := tinyDenseModelWithMoESlots(t, 0)
	need := residentNeedBytes(m)
	if need <= 0 {
		t.Skip("tiny fixture reports zero resident bytes: nothing to price")
	}
	alloc := need - m.ResidentHostCopyBytes(metalMoESlotsFromEnv(m))
	origCap, origLive := metalWorkingSetCap, metalLiveAvailable
	t.Cleanup(func() { metalWorkingSetCap, metalLiveAvailable = origCap, origLive })
	metalLiveAvailable = func() int64 { return 1 << 40 }
	metalWorkingSetCap = func() int64 { return alloc - 1 }
	if why := residentMemoryDecline(m); !strings.Contains(why, "recommendedMaxWorkingSetSize") {
		t.Errorf("an allocation of %d over a cap of %d: decline %q, want one naming recommendedMaxWorkingSetSize", alloc, alloc-1, why)
	}
	metalWorkingSetCap = func() int64 { return alloc }
	if why := residentMemoryDecline(m); why != "" {
		t.Errorf("an allocation of %d at a cap of %d: declined %q", alloc, alloc, why)
	}
}

// TestMetalWorkingSetCap_readsTheDevice: the device's recommendedMaxWorkingSetSize is read, and is a plausible share of RAM.
func TestMetalWorkingSetCap_readsTheDevice(t *testing.T) {
	ws := deviceWorkingSet()
	ram, _ := unix.SysctlUint64("hw.memsize")
	if ws <= 0 || ram == 0 || ws > int64(ram) {
		t.Fatalf("recommendedMaxWorkingSetSize %d against %d of RAM", ws, ram)
	}
	t.Logf("recommendedMaxWorkingSetSize %d bytes (%.2f GiB, %.0f%% of RAM)", ws, float64(ws)/(1<<30), 100*float64(ws)/float64(ram))
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

// TestMetalMoESlotsFromEnv pins the guard's reader of the slot request: residentFitsMemory must ask ResidentWeightBytesPaged for
// the same N that moe.go and gemma4_moe.go are about to honor, not silently fall back to the unpaged number on a value it cannot
// parse. It mirrors their resolution (metalMoESlotsRequest) except that an invalid or unset value means "assume unpaged" here
// (buildResident still validates and declines a bad one). The env cases run with Options.MoECacheSlots unset, so the fallback
// GOINFER_METAL_MOE_SLOTS is what is exercised; one case pins the priority order: Options wins over the env var.
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

// TestResidentNeedBytes_honorsPagingSlots pins that the guard USES what TestMetalMoESlotsFromEnv parses: a residentNeedBytes
// reverted to always call ResidentWeightBytes() compiles clean and passes that parsing test. It loads a real (tiny) MoE checkpoint
// and compares residentNeedBytes against ResidentWeightBytes, ResidentWeightBytesPaged, ResidentHostCopyBytes and residentKVBytes
// directly, with no real RAM needed. The assertions add the host-copy and KV terms back in, computed independently through the public
// accessors, so the test catches a regression in the paging wiring specifically without a rewrite each time another additive term
// appears.
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

	// The slot request is a per-model knob, read at Load (docs/tasks/task-env-config-2026-09.md, phase 4), so
	// each arm prices its own model loaded with that request, not one model with the environment changed under it.
	needWith := func(t *testing.T, slots string) int64 {
		t.Helper()
		mk, err := decoder.Load(ckpt, decoder.Options{Quant: "f32", Knobs: &decoder.Knobs{"GOINFER_METAL_MOE_SLOTS": slots}})
		if err != nil {
			t.Fatalf("Load (slots %q): %v", slots, err)
		}
		defer mk.Close()
		return residentNeedBytes(mk)
	}

	t.Run("unset env == unpaged", func(t *testing.T) {
		want := unpaged + m.ResidentHostCopyBytes(0) + kv
		if got := needWith(t, ""); got != want {
			t.Errorf("residentNeedBytes() with no slots env = %d, want unpaged+hostcopy+kv %d", got, want)
		}
	})

	t.Run("slots=1 matches ResidentWeightBytesPaged and is strictly smaller", func(t *testing.T) {
		wantWeights := m.ResidentWeightBytesPaged(1)
		if wantWeights >= unpaged {
			t.Fatalf("test fixture has too few experts to make this case meaningful (paged(1)=%d, unpaged=%d)", wantWeights, unpaged)
		}
		want := wantWeights + m.ResidentHostCopyBytes(1) + kv
		if got := needWith(t, "1"); got != want {
			t.Errorf("residentNeedBytes() with GOINFER_METAL_MOE_SLOTS=1 = %d, want %d (paged weights+hostcopy+kv) — "+
				"the guard is not asking for the paged estimate", got, want)
		}
		// The host-copy addend must itself shrink under paging: paged experts stream, so ResidentHostCopyBytes(1) must be strictly smaller
		// than ResidentHostCopyBytes(0) whenever paging caps anything on this fixture.
		if hc0, hc1 := m.ResidentHostCopyBytes(0), m.ResidentHostCopyBytes(1); hc1 >= hc0 {
			t.Errorf("ResidentHostCopyBytes(1)=%d not smaller than ResidentHostCopyBytes(0)=%d — "+
				"paging should exempt streamed experts from the host-copy addend", hc1, hc0)
		}
	})
}

// demandConfigs are the configs of the tracked tiny checkpoints that load on the CPU.
func demandConfigs(t *testing.T) []*decoder.Config {
	t.Helper()
	dirs, _ := filepath.Glob(filepath.Join("..", "testdata", "*", "config.json"))
	var cfgs []*decoder.Config
	for _, c := range dirs {
		dir := filepath.Dir(c)
		if ws, _ := filepath.Glob(filepath.Join(dir, "*.safetensors")); len(ws) == 0 {
			continue
		}
		m, err := decoder.Load(dir, decoder.Options{Backend: "cpu"})
		if err != nil {
			continue // a tower or an encoder, not a decoder family
		}
		cfgs = append(cfgs, m.Config())
		_ = m.Close()
	}
	return cfgs
}

// TestBuildDemand_matchesDecoderDefault (MM2a gate G-MM2a-2): this package's registered report equals decoder's default copy
// on every tracked family's config, across the option cells and a ladder of priced bytes. It first shows the comparison sees a
// change to each field.
func TestBuildDemand_matchesDecoderDefault(t *testing.T) {
	cfgs := demandConfigs(t)
	if len(cfgs) < 10 {
		t.Fatalf("only %d configs loaded: a comparison over few configs proves little", len(cfgs))
	}
	var queries []decoder.BuildQuery
	for _, cfg := range cfgs {
		for _, be := range []string{"metal", "cpu", "cuda", ""} {
			for _, stream := range []bool{false, true} {
				for _, moe := range []bool{false, true} {
					for _, gguf := range []bool{false, true} {
						for _, w := range []int64{0, 1 << 30, 7 << 30} {
							for _, e := range []int64{0, 2 << 30} {
								o := decoder.Options{Backend: be, StreamWeights: stream, MoECacheExperts: moe}
								queries = append(queries, decoder.BuildQueryFor(cfg, o, w, e, gguf))
							}
						}
					}
				}
			}
		}
	}
	differs := func(f func(decoder.BuildQuery) decoder.BuildDemand) int {
		n := 0
		for _, q := range queries {
			if f(q) != decoder.DefaultBuildDemand("metal", q) {
				n++
			}
		}
		return n
	}
	// One planted change per field Metal's report sets (a flag flipped): each must be seen. Metal reports no byte counts.
	for name, plant := range map[string]func(*decoder.BuildDemand){
		"Resident":   func(d *decoder.BuildDemand) { d.Resident = !d.Resident },
		"KVOnDevice": func(d *decoder.BuildDemand) { d.KVOnDevice = !d.KVOnDevice },
	} {
		planted := func(q decoder.BuildQuery) decoder.BuildDemand {
			d := buildDemand(q)
			plant(&d)
			return d
		}
		if differs(planted) == 0 {
			t.Errorf("a planted change to %s was not seen: no query exercises it", name)
		}
	}
	if n := differs(buildDemand); n > 0 {
		t.Errorf("the registered metal report differs from decoder's default on %d of %d queries", n, len(queries))
	}
	t.Logf("%d configs, %d queries", len(cfgs), len(queries))
}
