package decoder

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/townsendmerino/goinfer/internal/giw"
)

// TestPlanColdLoad is the rule, row by row: a bundle is read first only when the policy covers it, it is cold and it
// fits with room; one that certainly cannot stay is never read first, and has its read-around turned off only where
// the policy says that matters; everything else is left alone.
func TestPlanColdLoad(t *testing.T) {
	const gb = int64(1) << 30
	base := coldLoadFacts{qualifies: true, dropReadAround: true, residentFrac: 0, fileBytes: 20 * gb, availBytes: 60 * gb, ramBytes: 62 * gb}
	with := func(edit func(*coldLoadFacts)) coldLoadFacts { f := base; edit(&f); return f }
	for _, c := range []struct {
		name string
		f    coldLoadFacts
		want coldLoadAction
	}{
		{"cold, fits with room", base, coldLoadReadFirst},
		{"a bundle the policy does not cover", with(func(f *coldLoadFacts) { f.qualifies = false }), coldLoadNothing},
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
		// A policy with a small read-around never turns it off, and never reads first what cannot stay.
		{"small read-around: cold, fits with room", with(func(f *coldLoadFacts) { f.dropReadAround = false }), coldLoadReadFirst},
		{"small read-around: larger than the installed memory", with(func(f *coldLoadFacts) { f.dropReadAround = false; f.availBytes = 12 * gb; f.ramBytes = 16 * gb }), coldLoadNothing},
		{"small read-around: paged", with(func(f *coldLoadFacts) { f.dropReadAround = false; f.paged = true }), coldLoadNothing},
	} {
		if got := planColdLoad(c.f); got != c.want {
			t.Errorf("%s: plan %d, want %d (facts %+v)", c.name, got, c.want, c.f)
		}
	}
}

// TestColdLoadPolicy_qualifies: Linux's policy covers only a small-matrix MoE, on any backend; darwin's every bundle,
// on the CPU only; and a system with no policy none.
func TestColdLoadPolicy_qualifies(t *testing.T) {
	const gb = int64(1) << 30
	moe := &Architecture{NumLayers: 40, MoE: &MoEConfig{NumExperts: 256}}
	dense := &Architecture{NumLayers: 28}
	for _, c := range []struct {
		name   string
		policy coldLoadPolicy
		arch   *Architecture
		onCPU  bool
		want   bool
	}{
		{"linux, a small-matrix MoE", coldPolicyLinux, moe, true, true},
		{"linux, a small-matrix MoE, not on the CPU", coldPolicyLinux, moe, false, true},
		{"linux, a dense model", coldPolicyLinux, dense, true, false},
		{"darwin, a small-matrix MoE", coldPolicyDarwin, moe, true, true},
		{"darwin, a dense model", coldPolicyDarwin, dense, true, true},
		{"darwin, a small-matrix MoE, not on the CPU", coldPolicyDarwin, moe, false, false},
		{"darwin, a dense model, not on the CPU", coldPolicyDarwin, dense, false, false},
		{"no policy, an MoE", coldLoadPolicy{}, moe, true, false},
		{"no policy, a dense model", coldLoadPolicy{}, dense, true, false},
	} {
		if got := c.policy.qualifies(c.arch, 20*gb, c.onCPU); got != c.want {
			t.Errorf("%s: qualifies = %v, want %v", c.name, got, c.want)
		}
	}
	if !coldPolicyLinux.dropReadAround || coldPolicyDarwin.dropReadAround {
		t.Errorf("the read-around is turned off under Linux's policy (%v) and not under darwin's (%v)", coldPolicyLinux.dropReadAround, coldPolicyDarwin.dropReadAround)
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

// coldLoadCalls drives Load on a baked tiny bundle under policy, with the probes and actions replaced, and returns
// what the load did to the page cache, in order, and how many times it asked how much memory is available.
func coldLoadCalls(t *testing.T, policy coldLoadPolicy, path string, opts Options, resident float64, avail, ram int64) (calls []string, memProbes int) {
	t.Helper()
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	wantLen := int(fi.Size())
	check := func(mapping []byte) {
		if len(mapping) != wantLen {
			t.Errorf("the action was given %d bytes, want the whole mapping of %d", len(mapping), wantLen)
		}
	}
	savedPolicy, savedRes, savedAvail, savedRAM, savedRead, savedAdvise := coldPolicy, coldResidentFraction, coldAvailBytes, coldRAMBytes, coldReadThrough, coldAdviseReadAround
	t.Cleanup(func() {
		coldPolicy, coldResidentFraction, coldAvailBytes, coldRAMBytes, coldReadThrough, coldAdviseReadAround = savedPolicy, savedRes, savedAvail, savedRAM, savedRead, savedAdvise
	})
	coldPolicy = policy
	coldResidentFraction = func(m []byte) float64 { check(m); return resident }
	coldAvailBytes = func() int64 { memProbes++; return avail }
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
	return calls, memProbes
}

// notCPUBackend is the CPU backend under another name.
type notCPUBackend struct{ Backend }

func (notCPUBackend) Name() string { return "not-cpu" }

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

// TestLoad_coldLoadPlan: Load acts on the plan, on real bundles, under each system's policy.
//
// Linux's: a cold MoE bundle that fits is read through once and nothing else; one in the page cache is left alone; one
// larger than the box, or paged, has its read-around turned off and then back on; a dense bundle is never touched.
// darwin's: a cold bundle that fits is read through, dense or MoE, when the load is on the CPU and not when it is on
// another backend; one that cannot stay, or is paged, is left alone, and its read-around is never touched. Under
// either, a warm bundle is not asked how much memory there is.
func TestLoad_coldLoadPlan(t *testing.T) {
	const huge, tiny = int64(1) << 50, int64(1)
	// A backend that computes as the CPU does and is not the CPU by name, which is what the policy reads.
	other := "cold-load-not-cpu-" + t.Name()
	RegisterBackend(other, func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		return notCPUBackend{cpu}, nil
	})
	moe := bakeTiny(t, "../testdata/qwen3moe-tiny")
	dense := bakeTiny(t, "../testdata/llama-tiny")
	readThrough, offOn := []string{"read through"}, []string{"read-around off", "read-around on"}
	for _, c := range []struct {
		name       string
		policy     coldLoadPolicy
		bundle     string
		opts       Options
		resident   float64
		avail, ram int64
		want       []string
	}{
		{"linux, MoE, cold and fits", coldPolicyLinux, moe, Options{}, 0, huge, huge, readThrough},
		{"linux, MoE, in the page cache", coldPolicyLinux, moe, Options{}, 1, huge, huge, nil},
		{"linux, MoE, larger than the installed memory", coldPolicyLinux, moe, Options{}, 0, tiny, tiny, offOn},
		{"linux, MoE, paged", coldPolicyLinux, moe, Options{StreamWeights: true}, 0, huge, huge, offOn},
		{"linux, MoE, fits the box but not with room", coldPolicyLinux, moe, Options{}, 0, tiny, huge, nil},
		{"linux, dense, cold and fits", coldPolicyLinux, dense, Options{}, 0, huge, huge, nil},
		{"linux, dense, paged and larger than the box", coldPolicyLinux, dense, Options{StreamWeights: true}, 0, tiny, tiny, nil},
		{"darwin, MoE, cold and fits", coldPolicyDarwin, moe, Options{}, 0, huge, huge, readThrough},
		{"darwin, dense, cold and fits", coldPolicyDarwin, dense, Options{}, 0, huge, huge, readThrough},
		{"darwin, dense, in the page cache", coldPolicyDarwin, dense, Options{}, 1, huge, huge, nil},
		{"darwin, MoE, larger than the installed memory", coldPolicyDarwin, moe, Options{}, 0, tiny, tiny, nil},
		{"darwin, MoE, paged", coldPolicyDarwin, moe, Options{StreamWeights: true}, 0, huge, huge, nil},
		{"darwin, dense, fits the box but not with room", coldPolicyDarwin, dense, Options{}, 0, tiny, huge, nil},
		{"darwin, dense, cold and fits, not on the CPU", coldPolicyDarwin, dense, Options{Backend: other}, 0, huge, huge, nil},
		{"darwin, MoE, cold and fits, not on the CPU", coldPolicyDarwin, moe, Options{Backend: other}, 0, huge, huge, nil},
		{"linux, MoE, cold and fits, not on the CPU", coldPolicyLinux, moe, Options{Backend: other}, 0, huge, huge, readThrough},
		{"no policy, MoE, cold and fits", coldLoadPolicy{}, moe, Options{}, 0, huge, huge, nil},
	} {
		got, probes := coldLoadCalls(t, c.policy, c.bundle, c.opts, c.resident, c.avail, c.ram)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s: the load did %v, want %v", c.name, got, c.want)
		}
		if c.resident >= coldResidentWarm && probes != 0 {
			t.Errorf("%s: a warm bundle asked for the available memory %d time(s), want none", c.name, probes)
		}
	}
}
