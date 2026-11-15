//go:build darwin

package metal

import (
	"errors"
	"fmt"
	gpu "github.com/townsendmerino/goinfer/backend/metal/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"math/rand"
	"os"
	"slices"
	"sync"
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
// accessors, and at a paged request the experts' slot indices as arithmetic on the top k, so the test catches a regression in the
// paging wiring specifically without a rewrite each time another additive term appears.
func TestResidentNeedBytes_honorsPagingSlots(t *testing.T) {
	// testdata/gemma4-moe-tiny is gitignored (a real, if small, checkpoint) — never present in CI,
	// so skip rather than fail when it's absent, matching decoder's own convention for this fixture.
	const ckpt = "../../testdata/gemma4-moe-tiny"
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
	// The buffers Metal's build derives beside the tensors: the excess of its dense figure over the tensors' sum, which on this f32
	// load, with no q4k lane or Mamba layer to take anything off, is that term alone.
	derived := m.ResidentDenseWeightBytesFor("metal") - m.ResidentDenseWeightBytes()

	// The slot request is a per-model knob, read at Load (docs/completed/task-env-config-2026-09.md, phase 4), so
	// each arm prices its own model loaded with that request, not one model with the environment changed under it.
	// needWith is the guard's figure at a slot request less decode's scratch with every expert resident, which the request does not
	// move: the weights, the derived buffers, host copy and KV at that request, and the paged experts' slot indices where it pages.
	needWith := func(t *testing.T, slots string) int64 {
		t.Helper()
		mk, err := decoder.Load(ckpt, decoder.Options{Quant: "f32", Knobs: &decoder.Knobs{"GOINFER_METAL_MOE_SLOTS": slots}})
		if err != nil {
			t.Fatalf("Load (slots %q): %v", slots, err)
		}
		defer mk.Close()
		return residentNeedBytes(mk) - mk.MetalDecodeScratchBytes(0)
	}
	index := int64(4 * moeTopK(m)) // the paged experts' slot indices, 4 bytes for each of the top k

	t.Run("unset env == unpaged", func(t *testing.T) {
		want := unpaged + derived + m.ResidentHostCopyBytes(0) + kv
		if got := needWith(t, ""); got != want {
			t.Errorf("residentNeedBytes() with no slots env = %d, want unpaged+derived+hostcopy+kv %d", got, want)
		}
	})

	t.Run("slots=1 matches ResidentWeightBytesPaged and is strictly smaller", func(t *testing.T) {
		wantWeights := m.ResidentWeightBytesPaged(1)
		if wantWeights >= unpaged {
			t.Fatalf("test fixture has too few experts to make this case meaningful (paged(1)=%d, unpaged=%d)", wantWeights, unpaged)
		}
		want := wantWeights + derived + m.ResidentHostCopyBytes(1) + kv + index
		if got := needWith(t, "1"); got != want {
			t.Errorf("residentNeedBytes() with GOINFER_METAL_MOE_SLOTS=1 = %d, want %d (paged weights+derived+hostcopy+kv+slot indices) — "+
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
	dirs, _ := filepath.Glob(filepath.Join("../../testdata/*/config.json"))
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
	// One planted change per field Metal's report sets (a flag flipped, the state doubled): each must be seen.
	for name, plant := range map[string]func(*decoder.BuildDemand){
		"Resident":   func(d *decoder.BuildDemand) { d.Resident = !d.Resident },
		"KVOnDevice": func(d *decoder.BuildDemand) { d.KVOnDevice = !d.KVOnDevice },
		"StateBytes": func(d *decoder.BuildDemand) { d.StateBytes *= 2 },
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

// TestResident_holdsTheModelsProgram (OP4m-a's A4, docs/completed/task-forward-architecture-2026-10.md §4.9): the resident a Metal
// build returns holds the model's own program, the one copy decoder walks, for a synthetic dense model and the tracked qwen3.5
// fixture.
func TestResident_holdsTheModelsProgram(t *testing.T) {
	probe, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	probe.ReleaseObjects()
	dense := t.TempDir()
	writeDense(t, dense, genTinyWeights(rand.New(rand.NewSource(41))))
	for _, c := range []struct{ name, dir, quant string }{
		{"synthetic dense", dense, "int8int8"},
		{"qwen35-tiny", "../../testdata/qwen35-tiny", "int4"}, // a Gated-DeltaNet model builds resident at int4 only
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, err := os.Stat(filepath.Join(c.dir, "config.json")); err != nil {
				t.Fatalf("no fixture (both are tracked or written above): %v", err)
			}
			m, err := decoder.Load(c.dir, decoder.Options{Quant: c.quant})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			r, err := buildResident(m)
			if err != nil {
				t.Fatalf("build resident: %v", err)
			}
			defer r.Close()
			want := m.Program()
			if r.prog != want {
				t.Fatalf("the resident holds %p, not the model's program %p", r.prog, want)
			}
			ops := 0
			for _, l := range r.prog.Layers {
				ops += len(l)
			}
			if len(r.prog.Layers) != len(m.Weights().Layers) || ops == 0 {
				t.Fatalf("the resident's program has %d layers and %d operations, for a %d-layer model", len(r.prog.Layers), ops, len(m.Weights().Layers))
			}
			t.Logf("%d layers, %d operations", len(r.prog.Layers), ops)
		})
	}
}

// shapeFixtures are the tracked tiny checkpoints B1-1 loads on the CPU: every directory under the two testdata roots with a
// config.json and safetensors weights.
func shapeFixtures(t *testing.T) []string {
	t.Helper()
	var dirs []string
	for _, root := range []string{"../../testdata", "../../decoder/testdata"} {
		ents, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, e := range ents {
			d := filepath.Join(root, e.Name())
			ws, _ := filepath.Glob(filepath.Join(d, "*.safetensors"))
			if _, err := os.Stat(filepath.Join(d, "config.json")); e.IsDir() && err == nil && len(ws) > 0 {
				dirs = append(dirs, d)
			}
		}
	}
	return dirs
}

// shapeDiffs compares program.ShapeOf(m's program) with decoder's accessors, field by field, and names each that differs.
func shapeDiffs(m *decoder.Model) ([]string, error) {
	s, err := program.ShapeOf(m.Program())
	if err != nil {
		return nil, err
	}
	var d []string
	for _, c := range []struct {
		name      string
		got, want bool
	}{
		{"sandwich", s.Sandwich, m.BackendView().SandwichNorm()},
		{"postOnly", s.PostOnly, m.BackendView().PostOnlyNorm()},
		{"parallel", s.Parallel, m.BackendView().ParallelBlock()},
		{"nonGated", s.NonGated, m.BackendView().NonGatedMLP()},
		{"layerNorm", s.LayerNorm, m.BackendView().LayerNorm()},
		{"qkNorm", s.QKNorm, m.BackendView().HasQKNorm()},
		{"qkNormWhole", s.QKNormWhole, m.BackendView().QKNormWhole()},
		{"outBias", s.OutBias, m.BackendView().OutBias()},
		{"learnedPos", s.LearnedPos, m.BackendView().LearnedPos()},
	} {
		if c.got != c.want {
			d = append(d, fmt.Sprintf("%s: program %v, accessor %v", c.name, c.got, c.want))
		}
	}
	return d, nil
}

// shapeNotDecoders are the fixture directories B1-1 expects decoder.Load to refuse: encoders and vision or audio towers, which no
// decode runs. Any other load failure fails the test.
var shapeNotDecoders = map[string]bool{
	"embeddinggemma2-tiny": true, "gemma4-audio-tiny": true, "gemma4-vision-tiny": true, "glm-ocr-vision-tiny": true,
	"qwen3vl-vision-tiny": true, "siglip-tiny": true, "whisper-tiny-rand": true, "whisper-tiny-ts": true,
}

// shapeMaxWeightBytes bounds a B1-1 fixture's weights: a slice of a real model (mellum's is 4.2 GB) is not a tiny checkpoint, and
// loading one beside the test's other work risks swap on a 16 GB Mac.
const shapeMaxWeightBytes = 1 << 30

// eachShapeFixture loads every tracked tiny checkpoint the decoder loads on the CPU and calls f with it, closing each model as soon
// as f returns. It fails the test on an unexpected load failure, and returns how many fixtures it compared.
func eachShapeFixture(t *testing.T, f func(name string, m *decoder.Model)) int {
	t.Helper()
	compared, skipped := 0, 0
	for _, dir := range shapeFixtures(t) {
		name := filepath.Base(dir)
		ws, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
		var bytes int64
		for _, w := range ws {
			if fi, err := os.Stat(w); err == nil {
				bytes += fi.Size()
			}
		}
		if bytes > shapeMaxWeightBytes {
			t.Logf("NOT RUN: %s: %d MB of weights, not a tiny checkpoint", name, bytes>>20)
			skipped++
			continue
		}
		m, err := decoder.Load(dir, decoder.Options{})
		if err != nil {
			if shapeNotDecoders[name] {
				t.Logf("NOT RUN: %s: not a decoder (%v)", name, err)
				skipped++
				continue
			}
			t.Errorf("%s: unexpected load failure: %v", name, err)
			continue
		}
		f(name, m)
		m.Close()
		compared++
	}
	if compared < 10 {
		t.Fatalf("only %d fixtures compared: a comparison over few fixtures proves little", compared)
	}
	t.Logf("%d fixtures compared, %d NOT RUN", compared, skipped)
	return compared
}

// TestDenseShape_matchesTheAccessors is OP4m-b1's B1-1 and B1-2 (docs/completed/task-forward-architecture-2026-10.md §4.9): on every
// tracked tiny checkpoint the decoder loads on the CPU, the layer shape Metal derives from the program equals the nine
// Architecture accessors it replaces, and the derivation returns no error. The derivation is internal/program's ShapeOf; its
// plants are decoder's TestProgramShape_matchesTheAccessors, which runs on every box.
func TestDenseShape_matchesTheAccessors(t *testing.T) {
	eachShapeFixture(t, func(name string, m *decoder.Model) {
		if d, err := shapeDiffs(m); err != nil {
			t.Errorf("%s: %v", name, err)
		} else if len(d) > 0 {
			t.Errorf("%s: %v", name, d)
		}
	})
}

// walkerAccepts reports whether Metal's step derivation accepts every layer of p: layerStepsOf and checkStepClaims, without the
// table.
func walkerAccepts(p *program.Program) error {
	for l, ops := range p.Layers {
		spans, err := layerStepsOf(ops)
		if err == nil {
			err = checkStepClaims(ops, spans)
		}
		if err != nil {
			return fmt.Errorf("layer %d: %w", l, err)
		}
	}
	return nil
}

// tableHasSteps reports whether s has a step for every operation of p (program.Support.HasStep: the kinds and FFN forms, not
// the parameters), naming the first it lacks.
func tableHasSteps(s *program.Support, p *program.Program) error {
	for l, ops := range p.Layers {
		for i := range ops {
			if o := &ops[i]; !s.HasStep(o) && o.Kind == program.OpFFN {
				return fmt.Errorf("layer %d: operation %d (ffn, form %v)", l, i, o.FFN.Form)
			} else if !s.HasStep(o) {
				return fmt.Errorf("layer %d: operation %d (%v)", l, i, o.Kind)
			}
		}
	}
	return nil
}

// opLayer is a hand-written layer holding o: a mixer's sublayer, an FFN-only sublayer, a step of one operation, or, for a part
// of the attention sublayer, that sublayer with o inside it.
func opLayer(o program.Op) []program.Op {
	norm := program.Op{Kind: program.OpNorm, Norm: program.NormParams{Slot: program.SlotPreAttn, Form: program.NormRMS}}
	res := program.Op{Kind: program.OpResidual}
	switch o.Kind {
	case program.OpGemma4MoE, program.OpPLE, program.OpLayerScale:
		return []program.Op{o}
	case program.OpProjQKV, program.OpProjO, program.OpAttention:
		o = program.Op{Kind: program.OpAttention}
		fallthrough
	case program.OpQKNorm, program.OpRope, program.OpAttnTemp, program.OpOutGate:
		ops := []program.Op{norm, {Kind: program.OpProjQKV}}
		if o.Kind != program.OpAttention {
			ops = append(ops, o)
		}
		return append(ops, program.Op{Kind: program.OpAttention}, program.Op{Kind: program.OpProjO}, res)
	}
	return []program.Op{norm, o, res}
}

// TestLayerSteps_tableEqualsWalker is OP4m-c1's C1-1 over synthetic layers: for a layer of each operation kind, and an FFN-only
// layer of each FFN form (and one past the last), Metal's table has a step exactly when the walker accepts the layer.
func TestLayerSteps_tableEqualsWalker(t *testing.T) {
	var cases []program.Op
	for k := program.OpNorm; k <= program.OpLayerScale+1; k++ {
		if k != program.OpFFN && k != program.OpNorm && k != program.OpResidual {
			cases = append(cases, program.Op{Kind: k})
		}
	}
	for f := program.FfnGated; f <= program.FfnGemma4Dense+1; f++ {
		cases = append(cases, program.Op{Kind: program.OpFFN, FFN: program.FFNParams{Form: f}})
	}
	accepted := 0
	for _, o := range cases {
		p := &program.Program{Layers: [][]program.Op{opLayer(o)}}
		walk, table := walkerAccepts(p), tableHasSteps(program.Metal, p)
		if (walk == nil) != (table == nil) {
			t.Errorf("%v (FFN form %d): the walker gives %v and the table %v", o.Kind, o.FFN.Form, walk, table)
		}
		if walk == nil {
			accepted++
		}
	}
	if accepted == 0 || accepted == len(cases) {
		t.Fatalf("%d of %d synthetic layers accepted: the sweep must hold both outcomes", accepted, len(cases))
	}
}

// TestBuildLayerSteps_matchesTheLoadedBlocks (Gemini on #544, points 3 and 4): buildLayerSteps declines when the program's steps
// differ from the blocks the layer's build loaded, including a layer scale the build did not read (a buffer left empty because
// the scalar is 1 is not a missing read), and when the q4k lane would run a step beyond attention and a dense FFN.
func TestBuildLayerSteps_matchesTheLoadedBlocks(t *testing.T) {
	norm := func(slot program.NormSlot) program.Op {
		return program.Op{Kind: program.OpNorm, Norm: program.NormParams{Slot: slot, Form: program.NormRMS}}
	}
	res := program.Op{Kind: program.OpResidual}
	dense := []program.Op{norm(program.SlotPreAttn), {Kind: program.OpProjQKV}, {Kind: program.OpAttention}, {Kind: program.OpProjO}, res,
		norm(program.SlotPreMLP), {Kind: program.OpFFN, FFN: program.FFNParams{Form: program.FfnGated, Act: program.ActSiLU}}, res}
	scaled := append(slices.Clone(dense), program.Op{Kind: program.OpLayerScale})
	parallel := append(slices.Clone(dense[:4]), dense[6], program.Op{Kind: program.OpResidual, Res: program.ResidualParams{Two: true}})
	build := func(ops []program.Op, L residLayer, q4k bool) error {
		parallel := slices.ContainsFunc(ops, func(o program.Op) bool { return o.Kind == program.OpResidual && o.Res.Two })
		r := &resident{layers: []residLayer{L}, q4kLane: q4k, parallelBlock: parallel, decode: decodeShape{act: uint32(program.ActSiLU)}}
		return r.buildLayerSteps(&program.Program{Layers: [][]program.Op{ops}})
	}
	nonEmpty := (Buffer{}).At(1)
	loaded := residLayer{geom: &attnGeom{}, ffn: true, preNorm: nonEmpty, postNorm: nonEmpty}
	read := loaded
	read.layerScaleRead = true
	for _, c := range []struct {
		name string
		ops  []program.Op
		L    residLayer
		q4k  bool
		ok   bool
	}{
		{"a dense layer as loaded", dense, loaded, false, true},
		{"a dense layer on the q4k lane", dense, loaded, true, true},
		{"a layer scale whose scalar was read", scaled, read, false, true},
		{"the build loaded no FFN", dense, residLayer{geom: &attnGeom{}, preNorm: nonEmpty}, false, false},
		{"the build loaded no attention", dense, residLayer{ffn: true, postNorm: nonEmpty}, false, false},
		{"a layer scale the build did not read", scaled, loaded, false, false},
		{"a read layer scale the program lacks", dense, read, false, false},
		{"the q4k lane with a layer scale", scaled, read, true, false},
		// Gemini on #547: a parallel block's steps flatten to attention and a dense FFN, which the q4k lane's sequential FFN
		// does not run.
		{"a parallel block", parallel, loaded, false, true},
		{"a parallel block on the q4k lane", parallel, loaded, true, false},
	} {
		if err := build(c.ops, c.L, c.q4k); (err == nil) != c.ok {
			t.Errorf("%s: %v, want accepted %v", c.name, err, c.ok)
		}
	}
}

var (
	releasedMu  sync.Mutex
	releasedBad []string
	releasedN   int
)

// TestMain (G-MM3c-2): a test's own buffers lease on the test line, and every device the run releases must have had nothing
// unplanned and must close to zero.
func TestMain(m *testing.M) {
	switchesBefore := readSwitches(testSwitches())
	gpu.UnlinedLine = mem.LineTest
	gpu.ReleasedForTest = func(d *Device) {
		b := metalLedgerBreak(metalLedgerOf(d))
		releasedMu.Lock()
		defer releasedMu.Unlock()
		releasedN++
		if b != "" {
			releasedBad = append(releasedBad, b)
		}
	}
	code := m.Run()
	releasedMu.Lock()
	fmt.Fprintf(os.Stderr, "ledger: %d device(s) released in this run\n", releasedN)
	if len(releasedBad) > 0 {
		fmt.Fprintf(os.Stderr, "FAIL: %d released device(s) broke the ledger's invariants (I1 nothing unplanned, I2 close to zero):\n", len(releasedBad))
		for _, b := range releasedBad {
			fmt.Fprintln(os.Stderr, "  "+b)
		}
		code = 1
	}
	releasedMu.Unlock()
	if leaks := switchLeaks(testSwitches(), switchesBefore); len(leaks) > 0 {
		fmt.Fprintf(os.Stderr, "FAIL: %d package-level switch(es) were left changed by a test; each test restores the value it found:\n", len(leaks))
		for _, l := range leaks {
			fmt.Fprintln(os.Stderr, "  "+l)
		}
		code = 1
	}
	os.Exit(code)
}

// leakSwitch reads one package-level variable a test may assign.
type leakSwitch struct {
	name string
	read func() any
}

func sw[T comparable](name string, p *T) leakSwitch {
	return leakSwitch{name, func() any { return *p }}
}

// testSwitches lists every package-level scalar a test in this package assigns; TestMain fails the run when a test leaves one
// changed. A new switch is one line here, and TestSwitchGuard_coversEveryAssignedSwitch fails until it has one. A test that
// changes a switch restores what it found (`defer func(v bool) { x = v }(x)`), never a constant.
func testSwitches() []leakSwitch {
	return []leakSwitch{
		sw("attnFABlk64On", &attnFABlk64On),
		sw("attnFABlkAnyG", &attnFABlkAnyG),
		sw("attnSoftcapDropForTest", &attnSoftcapDropForTest),
		sw("chainFallbackAtForTest", &chainFallbackAtForTest),
		sw("chainNaNAtForTest", &chainNaNAtForTest),
		sw("costPlantForTest", &costPlantForTest),
		sw("dg01Mutation", &dg01Mutation),
		sw("dnSeqOldForTest", &dnSeqOldForTest),
		sw("dnetInt8DeclineOffForTest", &dnetInt8DeclineOffForTest),
		sw("dnetPrefillOn", &dnetPrefillOn),
		sw("eg2Block", &eg2Block),
		sw("emodelBatchDefect", &emodelBatchDefect),
		sw("emodelBatchedOn", &emodelBatchedOn),
		sw("exactAttnTileForTest", &exactAttnTileForTest),
		sw("expertPoolDefect", &expertPoolDefect),
		sw("ffnWalkPlantForTest", &ffnWalkPlantForTest),
		sw("g4DropLayerScalarForTest", &g4DropLayerScalarForTest),
		sw("g4DropVNormForTest", &g4DropVNormForTest),
		sw("g4ExpertBatchOn", &g4ExpertBatchOn),
		sw("g4KVSrcOffForTest", &g4KVSrcOffForTest),
		sw("g4KeepSharedKVStoreForTest", &g4KeepSharedKVStoreForTest),
		sw("g4LayerMajorOn", &g4LayerMajorOn),
		sw("g4OneFFNWidthForTest", &g4OneFFNWidthForTest),
		sw("g4PrefetchOn", &g4PrefetchOn),
		sw("g4SkipPLEForTest", &g4SkipPLEForTest),
		sw("gemmTilePolicy", &gemmTilePolicy),
		sw("gemma4FFNSplitOffForTest", &gemma4FFNSplitOffForTest),
		sw("gemvExtOn", &gemvExtOn),
		sw("greedyChainOff", &greedyChainOff),
		sw("gridIdleRelease", &gridIdleRelease),
		sw("gridReleaseCeiling", &gridReleaseCeiling),
		sw("gridReleaseOffForTest", &gridReleaseOffForTest),
		sw("gridScratchAllForTest", &gridScratchAllForTest),
		sw("gvKeepF32ForAB", &gvKeepF32ForAB),
		sw("ksc2OldDownForTest", &ksc2OldDownForTest),
		sw("ksd7OldDequantForTest", &ksd7OldDequantForTest),
		sw("kvCopyFailAfterForTest", &kvCopyFailAfterForTest),
		sw("kvCopyFailForTest", &kvCopyFailForTest),
		sw("kvEvictOrderPlantForTest", &kvEvictOrderPlantForTest),
		sw("kvKeepShortForTest", &kvKeepShortForTest),
		sw("kvOwnOffForTest", &kvOwnOffForTest),
		sw("kvScrambleForTest", &kvScrambleForTest),
		sw("kvTakeFailAfterForTest", &kvTakeFailAfterForTest),
		sw("kvTakeFailForTest", &kvTakeFailForTest),
		sw("kvWindowReadPlantForTest", &kvWindowReadPlantForTest),
		sw("kvRetainOffForTest", &kvRetainOffForTest),
		sw("kvRetainEarlyPlantForTest", &kvRetainEarlyPlantForTest),
		sw("kvReapEarlyPlantForTest", &kvReapEarlyPlantForTest),
		sw("kvNoRepointPlantForTest", &kvNoRepointPlantForTest),
		sw("kvReusableBasePlantForTest", &kvReusableBasePlantForTest),
		sw("ledgerChargeDeltaForTest", &ledgerChargeDeltaForTest),
		sw("ledgerChargeLineForTest", &ledgerChargeLineForTest),
		sw("mambaPrefillOn", &mambaPrefillOn),
		sw("mambaResetSkipForTest", &mambaResetSkipForTest),
		sw("mc3AdjRowsOn", &mc3AdjRowsOn),
		sw("mc3DeviceArgmaxOn", &mc3DeviceArgmaxOn),
		sw("mc3FARowsOn", &mc3FARowsOn),
		sw("mc3W8RowsOn", &mc3W8RowsOn),
		sw("metalDeepstackPrefillOn", &metalDeepstackPrefillOn),
		sw("metalFitFreeUnknownForTest", &metalFitFreeUnknownForTest),
		sw("metalMRoPEPrefillOn", &metalMRoPEPrefillOn),
		sw("metalMemoryCeilingForTest", &metalMemoryCeilingForTest),
		sw("mixerWalkPlantForTest", &mixerWalkPlantForTest),
		sw("mlaKVCopyDeclineOffForTest", &mlaKVCopyDeclineOffForTest),
		sw("moeExpertBatchOn", &moeExpertBatchOn),
		sw("moeExpertRowsOn", &moeExpertRowsOn),
		sw("moeKSlotsOn", &moeKSlotsOn),
		sw("moeLayerMajorOn", &moeLayerMajorOn),
		sw("nativeInt4Mix", &nativeInt4Mix),
		sw("nativeInt8", &nativeInt8),
		sw("nativeInt8MoE", &nativeInt8MoE),
		sw("pagedAsyncPhase2On", &pagedAsyncPhase2On),
		sw("pagedFenceCheckForTest", &pagedFenceCheckForTest),
		sw("pagedFenceOn", &pagedFenceOn),
		sw("preciseMathCompile", &preciseMathCompile),
		sw("prefillAttnSkipForTest", &prefillAttnSkipForTest),
		sw("prefillBlockRowsForTest", &prefillBlockRowsForTest),
		sw("prefillParallelDefectForTest", &prefillParallelDefectForTest),
		sw("prefillR20Copy", &prefillR20Copy),
		sw("prefillResidScaleForTest", &prefillResidScaleForTest),
		sw("prefillScratchCopy", &prefillScratchCopy),
		sw("prefillScratchPoison", &prefillScratchPoison),
		sw("prefillSteelAttnOff", &prefillSteelAttnOff),
		sw("promptStepAboveFloor", &promptStepAboveFloor),
		sw("q4kNativeOffForTest", &q4kNativeOffForTest),
		sw("siglipForceInt8", &siglipForceInt8),
		sw("skipPagedInt4CheckForTest", &skipPagedInt4CheckForTest),
		sw("stepNonGatedRow0PlantForTest", &stepNonGatedRow0PlantForTest),
		sw("stepPrefillAllHeads", &stepPrefillAllHeads),
		sw("stepSlotOffsetPlantForTest", &stepSlotOffsetPlantForTest),
		sw("swigluFusedOffForTest", &swigluFusedOffForTest),
		sw("w8FastMath", &w8FastMath),
	}
}

func readSwitches(ss []leakSwitch) map[string]any {
	v := make(map[string]any, len(ss))
	for _, s := range ss {
		v[s.name] = s.read()
	}
	return v
}

// switchLeaks names each switch whose value differs from before, sorted.
func switchLeaks(ss []leakSwitch, before map[string]any) []string {
	var out []string
	for _, s := range ss {
		if now := s.read(); now != before[s.name] {
			out = append(out, fmt.Sprintf("%s: %v at start, %v now", s.name, before[s.name], now))
		}
	}
	slices.Sort(out)
	return out
}

// TestSwitchGuard_namesALeak: the guard reports a switch a test left changed, and nothing when it is restored.
func TestSwitchGuard_namesALeak(t *testing.T) {
	ss := testSwitches()
	before := readSwitches(ss)
	if l := switchLeaks(ss, before); len(l) != 0 {
		t.Fatalf("no switch changed, yet the guard reports %v", l)
	}
	prev := gemvExtOn
	defer func() { gemvExtOn = prev }()
	gemvExtOn = !prev
	l := switchLeaks(ss, before)
	if len(l) != 1 || !strings.HasPrefix(l[0], "gemvExtOn: ") {
		t.Fatalf("gemvExtOn flipped: the guard reports %v, want one line naming gemvExtOn", l)
	}
}

// TestSwitchGuard_coversEveryAssignedSwitch: every package-level scalar (bool, int, string or a literal-initialised one) that a
// _test.go file in this directory assigns is in testSwitches, so a new switch cannot be left out of the guard.
func TestSwitchGuard_coversEveryAssignedSwitch(t *testing.T) {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, ".", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	scalar := map[string]bool{"bool": true, "int": true, "int64": true, "float32": true, "string": true}
	vars := map[string]bool{}
	for _, p := range pkgs {
		for fn, f := range p.Files {
			if strings.HasSuffix(fn, "_test.go") {
				continue
			}
			for _, d := range f.Decls {
				g, ok := d.(*ast.GenDecl)
				if !ok || g.Tok != token.VAR {
					continue
				}
				for _, sp := range g.Specs {
					vs := sp.(*ast.ValueSpec)
					for i, n := range vs.Names {
						if id, ok := vs.Type.(*ast.Ident); ok && scalar[id.Name] {
							vars[n.Name] = true
						} else if vs.Type == nil && i < len(vs.Values) {
							if lit, ok := vs.Values[i].(*ast.BasicLit); ok && lit.Kind != token.IMAG {
								vars[n.Name] = true
							} else if id, ok := vs.Values[i].(*ast.Ident); ok && (id.Name == "true" || id.Name == "false") {
								vars[n.Name] = true
							}
						}
					}
				}
			}
		}
	}
	listed := map[string]bool{}
	for _, s := range testSwitches() {
		listed[s.name] = true
	}
	var missing []string
	for _, p := range pkgs {
		for fn, f := range p.Files {
			if !strings.HasSuffix(fn, "_test.go") {
				continue
			}
			ast.Inspect(f, func(n ast.Node) bool {
				if a, ok := n.(*ast.AssignStmt); ok {
					for _, l := range a.Lhs {
						if id, ok := l.(*ast.Ident); ok && vars[id.Name] && !listed[id.Name] && !slices.Contains(missing, id.Name) {
							missing = append(missing, id.Name)
						}
					}
				}
				return true
			})
		}
	}
	slices.Sort(missing)
	if len(missing) > 0 {
		t.Errorf("a test assigns these package-level switches, which testSwitches does not list (add one sw line each): %v", missing)
	}
	if len(listed) < 40 {
		t.Errorf("only %d switches listed: the scan or the table is broken", len(listed))
	}
}

// metalLedgerOf is the ledger the metal package attached to d, or nil.
func metalLedgerOf(d *Device) *mem.Ledger {
	if l, ok := d.Leaser().(*deviceLedger); ok {
		return &l.Ledger
	}
	return nil
}

// metalLedgerBreak says how a released device's ledger breaks I1 (an unplanned byte ever) or I2 (a lease open after release),
// or "". A device with no ledger breaks I1.
func metalLedgerBreak(l *mem.Ledger) string {
	if l == nil {
		return "a device with no ledger: its allocations were not leased"
	}
	if u := l.Usage(); u.PeakUnplannedCharged > 0 || u.Leases > 0 {
		return fmt.Sprintf("unplanned peak %d bytes; %d leases (%d bytes) open after release, by line %v",
			u.PeakUnplannedCharged, u.Leases, u.Charged, u.ByLine)
	}
	return ""
}

// unlinedMetal are the constructors that lease on gpu.UnlinedLine: gpudev's and this package's helpers. Each has a lined twin
// (…On); the unlined name is a wrapper for tests.
var unlinedMetal = map[string]bool{"NewBufferLen": true, "NewBufferBytes": true, "NewBufferNoCopy": true, "MustBuf": true, "NewBufferOf": true,
	"NewBufferLenOf": true, "byteBuf": true, "NewBufferU32": true, "NewBufferFloats": true, "NewBufferInt8": true, "NewBufferU16s": true,
	"NewBufferUint32s": true, "int4BufA": true, "int4Concat": true, "q4kLaneBuf": true, "scales16": true, "concatNibbles": true, "window": true,
	"int4Buf": true, "int8Buf": true, "int8Concat": true, "q4kLaneProj": true, "int4ConcatA": true, "int8BufA": true, "int8Codes": true,
	"nibbles": true, "f32Mat": true}

// calleeName is the name a call's function is reached by (gpu.X, X, gpu.X[T], .X), or "".
func calleeName(c *ast.CallExpr) string {
	fun := c.Fun
	if ix, ok := fun.(*ast.IndexExpr); ok {
		fun = ix.X
	}
	switch e := fun.(type) {
	case *ast.SelectorExpr:
		return e.Sel.Name
	case *ast.Ident:
		return e.Name
	}
	return ""
}

// unlinedArg reports whether e is the unlined line: UnlinedLine (any qualifier) or the empty string.
func unlinedArg(e ast.Expr) bool {
	switch x := e.(type) {
	case *ast.Ident:
		return x.Name == "UnlinedLine"
	case *ast.SelectorExpr:
		return x.Sel.Name == "UnlinedLine"
	case *ast.BasicLit:
		return x.Value == `""`
	}
	return false
}

// isWrapper reports whether fd is an unlined wrapper: a listed name whose body hands UnlinedLine to its own …On twin.
func isWrapper(fd *ast.FuncDecl) bool {
	if !unlinedMetal[fd.Name.Name] || fd.Body == nil {
		return false
	}
	found := false
	ast.Inspect(fd.Body, func(x ast.Node) bool {
		if c, ok := x.(*ast.CallExpr); ok && calleeName(c) == fd.Name.Name+"On" {
			for _, a := range c.Args {
				if unlinedArg(a) {
					found = true
				}
			}
		}
		return true
	})
	return found
}

// unlinedMetalCalls counts the calls in f that allocate on no plan line: a call to an unlined constructor, or a lined (…On) call
// handed UnlinedLine or "", outside the unlined wrappers' own declarations.
func unlinedMetalCalls(f *ast.File) int {
	n := 0
	ast.Inspect(f, func(x ast.Node) bool {
		if fd, ok := x.(*ast.FuncDecl); ok && isWrapper(fd) {
			return false
		}
		c, ok := x.(*ast.CallExpr)
		if !ok {
			return true
		}
		name := calleeName(c)
		if unlinedMetal[name] {
			n++
		} else if strings.HasSuffix(name, "On") && unlinedMetal[strings.TrimSuffix(name, "On")] {
			for _, a := range c.Args {
				if unlinedArg(a) {
					n++
				}
			}
		}
		return true
	})
	return n
}

// TestLedger_everyMetalAllocationNamesALine (G-MM3c-1, I1 static): no non-test file of the metal module calls an unlined
// constructor outside the unlined wrappers, so every production MTLBuffer names its plan line. It reads every file and counts
// a planted call of each form. Device-free: CI's metal job runs it.
func TestLedger_everyMetalAllocationNamesALine(t *testing.T) {
	// Counted: the method, the generic, the helper, a lined call on UnlinedLine, a lined call on "", and a call inside a function
	// that only shares a wrapper's name; not counted: a lined call on a real line, and the true wrapper's own call.
	plant := "package x\nfunc f() { _ = d.NewBufferLen(4); _ = gpu.NewBufferLenOf[float32](d, 4); _ = NewBufferU32(d, 1); _ = d.NewBufferLenOn(\"kv\", 4)\n" +
		"\t_ = d.MustBufOn(UnlinedLine, id, 4, \"x\"); _ = gpu.NewBufferLenOfOn[float32](d, \"\", 4) }\n" +
		"func NewBufferU32(d *Device, v uint32) Buffer { return NewBufferU32On(gpu.UnlinedLine, d, v) }\n" +
		"func (o *other) window(d *Device) Buffer { return d.NewBufferLen(1) }\n"
	pf, err := parser.ParseFile(token.NewFileSet(), "plant.go", plant, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := unlinedMetalCalls(pf); got != 6 {
		t.Fatalf("the planted file counts %d unlined calls, want 6", got)
	}
	read := 0
	err = filepath.WalkDir(".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && d.Name() == "testdata" {
			return filepath.SkipDir
		}
		if d.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		f, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
		if err != nil {
			return err
		}
		read++
		if n := unlinedMetalCalls(f); n > 0 {
			t.Errorf("%s calls an unlined constructor %d time(s): name the plan line (the …On twin)", path, n)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if read < 60 { // 71 non-test files, gpudev's included, when written: far fewer means the walk is mis-scoped
		t.Fatalf("read %d source files", read)
	}
	t.Logf("read %d non-test files", read)
}
