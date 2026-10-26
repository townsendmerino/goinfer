package decoder

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/internal/mem/probe"
)

// useProbe installs r behind every machine reading the decoder takes (the fit guard's, the prefill budget's, the cold
// load's, the sidecar check's and the expert pager's fault counts) for one test.
func useProbe(t *testing.T, r probe.Reader) {
	t.Helper()
	savedRAM, savedAvail, savedColdAvail, savedColdRAM, savedColdRes, savedSidecar, savedFaults :=
		hostRAM, hostRAMAvailable, coldAvailBytes, coldRAMBytes, coldResidentFraction, sidecarHostRAM, processFaultCounts
	hostRAM, hostRAMAvailable = r.Physical, r.Available
	coldAvailBytes, coldRAMBytes, coldResidentFraction = r.Available, r.Physical, r.ResidentFraction
	sidecarHostRAM, processFaultCounts = r.Physical, r.FaultCounts
	resetAvailProbeCache()
	t.Cleanup(func() {
		hostRAM, hostRAMAvailable, coldAvailBytes, coldRAMBytes, coldResidentFraction, sidecarHostRAM, processFaultCounts =
			savedRAM, savedAvail, savedColdAvail, savedColdRAM, savedColdRes, savedSidecar, savedFaults
		resetAvailProbeCache()
	})
}

// bracketed reports whether got equals one of two reference readings taken around it, or lies between them.
func bracketed(lo, got, hi int64) bool {
	if lo > hi {
		lo, hi = hi, lo
	}
	return got >= lo && got <= hi
}

// TestMemProbe_seamsReadTheProbe (MM0 gate G-MM0-1c): every machine reading the decoder takes, through its exported
// readers and through each package seam, is the probe's. A live reading can move between two calls, so those are
// compared against two probe readings that bracket them, five tries at most.
func TestMemProbe_seamsReadTheProbe(t *testing.T) {
	for name, got := range map[string]int64{"HostRAMBytes": HostRAMBytes(), "hostRAM": hostRAM(), "coldRAMBytes": coldRAMBytes(),
		"sidecarHostRAM": sidecarHostRAM()} {
		if want := probe.Physical(); got != want {
			t.Errorf("%s = %d, probe.Physical = %d", name, got, want)
		}
	}
	live := map[string]func() int64{
		"HostRAMAvailableBytes": HostRAMAvailableBytes, "hostRAMAvailable": hostRAMAvailable, "coldAvailBytes": coldAvailBytes,
		"SwapUsedBytes": func() int64 { v, _ := SwapUsedBytes(); return v },
	}
	ref := map[string]func() int64{
		"HostRAMAvailableBytes": probe.Available, "hostRAMAvailable": probe.Available, "coldAvailBytes": probe.Available,
		"SwapUsedBytes": func() int64 { v, _ := probe.SwapUsed(); return v },
	}
	for name, read := range live {
		ok := false
		var lo, got, hi int64
		for range 5 {
			lo, got, hi = ref[name](), read(), ref[name]()
			if ok = bracketed(lo, got, hi); ok {
				break
			}
		}
		if !ok {
			t.Errorf("%s = %d, outside the probe's %d..%d on five tries", name, got, lo, hi)
		}
	}
	if _, ok := SwapUsedBytes(); func() bool { _, pok := probe.SwapUsed(); return pok }() != ok {
		t.Error("SwapUsedBytes and probe.SwapUsed disagree on whether swap is known")
	}
	r, why := SelfSwapReader()
	pr, pwhy := probe.SelfSwapReader()
	if (r == nil) != (pr == nil) || why != pwhy {
		t.Errorf("SelfSwapReader = (nil %v, %q), probe's (nil %v, %q)", r == nil, why, pr == nil, pwhy)
	}
	if _, _, ok := processFaultCounts(); func() bool { _, _, pok := probe.FaultCounts(); return pok }() != ok {
		t.Error("processFaultCounts and probe.FaultCounts disagree on ok")
	}
	page := make([]byte, 4*os.Getpagesize())
	for i := range page {
		page[i] = 1
	}
	if got, want := coldResidentFraction(page), probe.ResidentFraction(page); got != want {
		t.Errorf("coldResidentFraction = %v, probe.ResidentFraction = %v", got, want)
	}
}

// longContextFixture copies the tracked internlm2-tiny checkpoint into a temp dir with its context raised to ctx, so the
// fit guard's auto-pin, which only shrinks to ctxFloor and above, is reachable.
func longContextFixture(t *testing.T, ctx int) string {
	t.Helper()
	src, dir := "testdata/internlm2-tiny", t.TempDir()
	b, err := os.ReadFile(filepath.Join(src, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(b, &cfg); err != nil {
		t.Fatal(err)
	}
	cfg["max_position_embeddings"] = ctx
	if b, err = json.Marshal(cfg); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	w, err := os.ReadFile(filepath.Join(src, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), w, 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// lowestAvail is the smallest available-memory reading in [1, 1<<46] at which ok holds, given that ok is monotone in it.
func lowestAvail(t *testing.T, ok func(avail int64) bool) int64 {
	t.Helper()
	lo, hi := int64(1), int64(1)<<46
	if !ok(hi) {
		t.Fatalf("no reading up to %d satisfies the predicate", hi)
	}
	for lo < hi {
		mid := lo + (hi-lo)/2
		if ok(mid) {
			hi = mid
		} else {
			lo = mid + 1
		}
	}
	return lo
}

// TestFitGuard_fakeProbeReachesEveryOutcome (MM0 gate G-MM0-2): a probe.Fake, installed through useProbe and changed in
// nothing but its readings (and the pin, where the outcome is about a pin), drives each load-time memory decision to each
// of its outcomes. The thresholds are found by searching the fake's available reading, so the test names no byte figure of
// its own.
func TestFitGuard_fakeProbeReachesEveryOutcome(t *testing.T) {
	fake := &probe.Fake{PhysicalBytes: 64 << 30}
	useProbe(t, fake)
	setAvail := func(v int64) {
		fake.Set(func(f *probe.Fake) { f.AvailableBytes = v })
		resetAvailProbeCache()
	}

	t.Run("guardFit", func(t *testing.T) {
		const maxCtx = 1 << 17
		dir := longContextFixture(t, maxCtx)
		run := func(avail int64, opts Options) (int, error) {
			setAvail(avail)
			return guardFit(fitCheckFor(dir, "int4", quantInt4, opts))
		}
		fitsAt := lowestAvail(t, func(a int64) bool { ctx, err := run(a, Options{}); return err == nil && ctx == 0 })
		loadsAt := lowestAvail(t, func(a int64) bool { _, err := run(a, Options{}); return err == nil })

		setAvail(0)
		if f := fitCheckFor(dir, "int4", quantInt4, Options{}); f.known() {
			t.Error("unknown: a 0 reading priced the load as known")
		}
		if ctx, err := run(0, Options{}); err != nil || ctx != 0 {
			t.Errorf("unknown: (%d, %v), want proceed unchanged", ctx, err)
		}
		if ctx, err := run(fitsAt, Options{}); err != nil || ctx != 0 {
			t.Errorf("fits: (%d, %v) at %d, want proceed unchanged", ctx, err, fitsAt)
		}
		if loadsAt >= fitsAt {
			t.Fatalf("auto-pin: no reading pins (loads from %d, fits from %d)", loadsAt, fitsAt)
		}
		if ctx, err := run(loadsAt, Options{}); err != nil || ctx < ctxFloor || ctx >= maxCtx {
			t.Errorf("auto-pin: (%d, %v) at %d, want a context in [%d, %d)", ctx, err, loadsAt, ctxFloor, maxCtx)
		}
		var decline *FitDeclineError
		if _, err := run(loadsAt-1, Options{}); !errors.As(err, &decline) {
			t.Errorf("refuse at the floor: %v at %d, want a *FitDeclineError", err, loadsAt-1)
		}
		if _, err := run(loadsAt, Options{ResidentContext: maxCtx}); !errors.As(err, &decline) {
			t.Errorf("refuse a pin: %v at %d with -ctx %d, want a *FitDeclineError", err, loadsAt, maxCtx)
		}
	})

	t.Run("decideGIWFit", func(t *testing.T) {
		cfg := &Config{NumLayers: 32, NumKVHeads: 8, HeadDim: 128, MaxPositions: 1 << 17}
		run := func(avail int64, opts Options) giwFit {
			setAvail(avail)
			return decideGIWFit(cfg, opts)
		}
		fitsAt := lowestAvail(t, func(a int64) bool { d := run(a, Options{}); return d.err == nil && d.pinCtx == 0 })
		loadsAt := lowestAvail(t, func(a int64) bool { return run(a, Options{}).err == nil })

		if d := run(0, Options{}); d.known || d.err != nil || d.pinCtx != 0 {
			t.Errorf("unknown: %+v, want proceed unpriced", d)
		}
		if d := run(fitsAt, Options{}); !d.known || d.err != nil || d.pinCtx != 0 {
			t.Errorf("fits: %+v at %d, want priced and unchanged", d, fitsAt)
		}
		if loadsAt >= fitsAt {
			t.Fatalf("auto-pin: no reading pins (loads from %d, fits from %d)", loadsAt, fitsAt)
		}
		if d := run(loadsAt, Options{}); d.err != nil || d.pinCtx < ctxFloor || d.pinCtx >= cfg.MaxPositions {
			t.Errorf("auto-pin: %+v at %d, want a context in [%d, %d)", d, loadsAt, ctxFloor, cfg.MaxPositions)
		}
		if d := run(loadsAt-1, Options{}); d.err == nil || d.ctx != ctxFloor {
			t.Errorf("refuse at the floor: %+v at %d, want a refusal priced at %d", d, loadsAt-1, ctxFloor)
		}
		if d := run(loadsAt, Options{ResidentContext: cfg.MaxPositions}); d.err == nil || !d.pinned {
			t.Errorf("refuse a pin: %+v at %d, want a refusal of the pinned context", d, loadsAt)
		}
	})

	t.Run("prepareColdLoad", func(t *testing.T) {
		savedPolicy, savedRead, savedAdvise := coldPolicy, coldReadThrough, coldAdviseReadAround
		t.Cleanup(func() { coldPolicy, coldReadThrough, coldAdviseReadAround = savedPolicy, savedRead, savedAdvise })
		coldPolicy = coldPolicyLinux
		var calls []string
		coldReadThrough = func([]byte) { calls = append(calls, "read through") }
		coldAdviseReadAround = func(_ []byte, on bool) { calls = append(calls, map[bool]string{false: "off", true: "on"}[on]) }
		be, err := NewBackend("cpu")
		if err != nil {
			t.Fatal(err)
		}
		// An MoE of small expert matrices, which the Linux policy covers.
		arch := &Architecture{NumLayers: 2, MoE: &MoEConfig{NumExperts: 8}}
		mapping := make([]byte, 1<<20)
		for _, c := range []struct {
			name       string
			resident   float64
			avail, ram int64
			want       []string
		}{
			{"read first", 0, 1 << 40, 1 << 40, []string{"read through"}},
			{"no read-around", 0, 1, 1, []string{"off", "on"}},
			{"nothing (warm)", 1, 1 << 40, 1 << 40, nil},
			{"nothing (fits the box, not with room)", 0, 1, 1 << 40, nil},
		} {
			calls = nil
			fake.Set(func(f *probe.Fake) {
				f.ResidentFractionRead, f.AvailableBytes, f.PhysicalBytes = c.resident, c.avail, c.ram
			})
			prepareColdLoad("x.giw", mapping, arch, Options{}, be)()
			if !slices.Equal(calls, c.want) {
				t.Errorf("%s: the load did %v, want %v", c.name, calls, c.want)
			}
		}
	})
}
