package decoder

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/townsendmerino/goinfer/internal/giw"
)

// TestPlanColdLoad is the rule, row by row: a bundle is read first only when it is small-matrix, cold and fits with
// room; its read-around is turned off only when it certainly cannot stay; everything else is left alone.
func TestPlanColdLoad(t *testing.T) {
	const gb = int64(1) << 30
	base := coldLoadFacts{smallMatrices: true, residentFrac: 0, fileBytes: 20 * gb, availBytes: 60 * gb, ramBytes: 62 * gb}
	with := func(edit func(*coldLoadFacts)) coldLoadFacts { f := base; edit(&f); return f }
	for _, c := range []struct {
		name string
		f    coldLoadFacts
		want coldLoadAction
	}{
		{"cold, fits with room", base, coldLoadReadFirst},
		{"a dense bundle", with(func(f *coldLoadFacts) { f.smallMatrices = false }), coldLoadNothing},
		{"already in the page cache", with(func(f *coldLoadFacts) { f.residentFrac = 1 }), coldLoadNothing},
		{"half in the page cache is warm", with(func(f *coldLoadFacts) { f.residentFrac = coldResidentWarm }), coldLoadNothing},
		{"just under half is cold", with(func(f *coldLoadFacts) { f.residentFrac = 0.49 }), coldLoadReadFirst},
		{"exactly three quarters of the available memory fits", with(func(f *coldLoadFacts) { f.fileBytes, f.availBytes = 24*gb, 32*gb }), coldLoadReadFirst},
		{"one byte over three quarters does not", with(func(f *coldLoadFacts) { f.fileBytes, f.availBytes = 24*gb+1, 32*gb }), coldLoadNothing},
		{"fits the box but not with room: left alone", with(func(f *coldLoadFacts) { f.availBytes = 22 * gb; f.ramBytes = 32 * gb }), coldLoadNothing},
		{"larger than the installed memory", with(func(f *coldLoadFacts) { f.availBytes = 12 * gb; f.ramBytes = 16 * gb }), coldLoadNoReadAround},
		{"paged against a budget, though it would fit", with(func(f *coldLoadFacts) { f.paged = true }), coldLoadNoReadAround},
		{"paged and warm: left alone", with(func(f *coldLoadFacts) { f.paged = true; f.residentFrac = 1 }), coldLoadNothing},
		{"available memory unknown", with(func(f *coldLoadFacts) { f.availBytes = 0 }), coldLoadNothing},
		{"both sizes unknown", with(func(f *coldLoadFacts) { f.availBytes, f.ramBytes = 0, 0 }), coldLoadNothing},
		{"an empty mapping", with(func(f *coldLoadFacts) { f.fileBytes = 0 }), coldLoadNothing},
	} {
		if got := planColdLoad(c.f); got != c.want {
			t.Errorf("%s: plan %d, want %d (facts %+v)", c.name, got, c.want, c.f)
		}
	}
}

// TestExpertMatricesAreSmall: only an MoE whose separate expert matrices average at most coldSmallMatrixBytes counts.
func TestExpertMatricesAreSmall(t *testing.T) {
	const gb = int64(1) << 30
	moe := func(layers, experts int) *Architecture {
		return &Architecture{NumLayers: layers, MoE: &MoEConfig{NumExperts: experts}}
	}
	for _, c := range []struct {
		name string
		arch *Architecture
		file int64
		want bool
	}{
		{"40 layers of 256 experts in 20 GB (0.65 MB a matrix)", moe(40, 256), 20 * gb, true},
		{"28 layers of 64 experts in 7 GB", moe(28, 64), 7 * gb, true},
		{"32 layers of 8 experts in 26 GB (34 MB a matrix)", moe(32, 8), 26 * gb, false},
		{"a dense model", &Architecture{NumLayers: 28}, 5 * gb, false},
		{"no architecture", nil, 5 * gb, false},
		{"an MoE descriptor with no experts", moe(28, 0), 5 * gb, false},
	} {
		if got := expertMatricesAreSmall(c.arch, c.file); got != c.want {
			t.Errorf("%s: %v, want %v", c.name, got, c.want)
		}
	}
}

// coldLoadCalls drives Load on a baked tiny bundle with the plan switched on and its probes and actions replaced, and
// returns what the load did to the page cache, in order.
func coldLoadCalls(t *testing.T, path string, opts Options, resident float64, avail, ram int64) []string {
	t.Helper()
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	var calls []string
	wantLen := int(fi.Size())
	check := func(mapping []byte) {
		if len(mapping) != wantLen {
			t.Errorf("the action was given %d bytes, want the whole mapping of %d", len(mapping), wantLen)
		}
	}
	saved := []any{coldLoadSupported, coldResidentFraction, coldAvailBytes, coldRAMBytes, coldReadThrough, coldAdviseReadAround}
	t.Cleanup(func() {
		coldLoadSupported = saved[0].(bool)
		coldResidentFraction = saved[1].(func([]byte) float64)
		coldAvailBytes = saved[2].(func() int64)
		coldRAMBytes = saved[3].(func() int64)
		coldReadThrough = saved[4].(func([]byte))
		coldAdviseReadAround = saved[5].(func([]byte, bool))
	})
	coldLoadSupported = true
	coldResidentFraction = func(m []byte) float64 { check(m); return resident }
	coldAvailBytes = func() int64 { return avail }
	coldRAMBytes = func() int64 { return ram }
	coldReadThrough = func(m []byte) { check(m); calls = append(calls, "read through") }
	coldAdviseReadAround = func(m []byte, on bool) {
		check(m)
		calls = append(calls, map[bool]string{false: "read-around off", true: "read-around on"}[on])
	}
	m, err := Load(path, opts)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	m.Close()
	return calls
}

// bakeTiny writes the fixture at dir as an int4 .giw and returns its path.
func bakeTiny(t *testing.T, dir string) string {
	t.Helper()
	m, err := Load(dir, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load %s: %v", dir, err)
	}
	defer m.Close()
	blob, err := SerializeWeights(m.w, "cold")
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "cold.giw")
	if err := os.WriteFile(p, giw.Write(blob, nil), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// TestLoad_coldLoadPlan: Load acts on the plan, on a real MoE bundle. A cold bundle that fits is read through once
// and nothing else; one in the page cache is left alone; one larger than the box, or paged, has its read-around
// turned off and then back on. A dense bundle is never touched, whatever the probes say.
func TestLoad_coldLoadPlan(t *testing.T) {
	const huge, tiny = int64(1) << 50, int64(1)
	moe := bakeTiny(t, "../testdata/qwen3moe-tiny")
	for _, c := range []struct {
		name       string
		opts       Options
		resident   float64
		avail, ram int64
		want       []string
	}{
		{"cold and fits", Options{}, 0, huge, huge, []string{"read through"}},
		{"in the page cache", Options{}, 1, huge, huge, nil},
		{"larger than the installed memory", Options{}, 0, tiny, tiny, []string{"read-around off", "read-around on"}},
		{"paged", Options{StreamWeights: true}, 0, huge, huge, []string{"read-around off", "read-around on"}},
		{"fits the box, not with room", Options{}, 0, tiny, huge, nil},
	} {
		if got := coldLoadCalls(t, moe, c.opts, c.resident, c.avail, c.ram); !reflect.DeepEqual(got, c.want) {
			t.Errorf("MoE bundle, %s: the load did %v, want %v", c.name, got, c.want)
		}
	}
	dense := bakeTiny(t, "../testdata/llama-tiny")
	if got := coldLoadCalls(t, dense, Options{}, 0, huge, huge); got != nil {
		t.Errorf("dense bundle, cold and fits: the load did %v, want nothing", got)
	}
	if got := coldLoadCalls(t, dense, Options{StreamWeights: true}, 0, tiny, tiny); got != nil {
		t.Errorf("dense bundle, paged and larger than the box: the load did %v, want nothing", got)
	}
}

// TestLoad_coldLoadOffWhereUnsupported: where the plan is not supported, Load asks nothing of the page cache.
func TestLoad_coldLoadOffWhereUnsupported(t *testing.T) {
	moe := bakeTiny(t, "../testdata/qwen3moe-tiny")
	saved, savedRead := coldLoadSupported, coldReadThrough
	t.Cleanup(func() { coldLoadSupported, coldReadThrough = saved, savedRead })
	coldLoadSupported = false
	called := false
	coldReadThrough = func([]byte) { called = true }
	m, err := Load(moe, Options{})
	if err != nil {
		t.Fatal(err)
	}
	m.Close()
	if called {
		t.Error("the bundle was read through with the plan unsupported")
	}
}
