package decoder

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// P13: the safetensors loader used to hold the SOURCE mapping for the model's whole life. On a
// 55.6 GB bf16 checkpoint that is 46.8 GB RSS against GGUF's 24.5 GB for an IDENTICAL 17.9 GB Go
// heap, and 1.69x slower decode — dead bf16 pages competing with the hot quantized weights for
// page cache. loadWeights now closes the source at end of load when nothing can alias it.
//
// THE GATE IS mmapAliasRisk, AND THIS TESTS THE GATE RATHER THAN THE OUTCOME. Getting it wrong in
// the safe direction costs the old behaviour; getting it wrong in the unsafe direction is a
// use-after-free on weights the decode is reading, which would surface as garbage output or a
// SIGBUS far from here.
func TestP13_mmapAliasRisk_classifiesDTypes(t *testing.T) {
	// BF16 and F16 are widened into fresh slices on read, so they cannot alias. Everything else
	// may be served as a zero-copy view. Table-driven so a new dtype has to be classified
	// deliberately rather than inheriting "safe" by falling through.
	for _, tc := range []struct {
		dtype string
		risky bool
	}{
		{"BF16", false},
		{"F16", false},
		{"F32", true}, // reinterpretLE zero-copy view when aligned — the aliasing case
		{"F64", true},
		{"I64", true},
		{"I32", true},
		{"U8", true},
	} {
		got := riskyDType(tc.dtype)
		if got != tc.risky {
			t.Errorf("dtype %s: risky=%v, want %v", tc.dtype, got, tc.risky)
		}
	}
}

// TestP13_realCheckpointReleases is the end-to-end half: on a real all-BF16 checkpoint the loader
// must actually drop the mapping. Asserted on w.st rather than on RSS — RSS is the symptom P13
// measured, but it is confounded by page-cache behaviour (and on darwin the OS reclaims lazily),
// so the structural fact is the testable one.
func TestP13_realCheckpointReleases(t *testing.T) {
	requireHeavyModel(t)
	dir := os.Getenv("GOINFER_MELLUM_CKPT")
	if dir == "" {
		dir = os.Getenv("HOME") + "/models/mellum2-unq"
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		t.Skipf("no safetensors checkpoint at %s", dir)
	}
	// openCheckpointMmap, not a hardcoded shard index: a checkpoint may be a single
	// model.safetensors or a sharded set, and the loader under test handles both.
	st, err := openCheckpointMmap(dir)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	risk := mmapAliasRisk(st)
	_ = st.Close()
	if risk != "" {
		t.Skipf("checkpoint has a %q-class tensor (%s) — release correctly declined, nothing to assert here", "non-BF16", risk)
	}

	m, err := Load(dir, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	if m.w.st != nil {
		t.Errorf("source mapping still retained after load — P13's release did not fire on an " +
			"all-BF16 checkpoint, so the bf16 source is still competing with the quantized weights")
	}
	// The weights must still be usable: a released mapping that took live data with it is the
	// failure this whole gate exists to prevent, and it would not show up as a load error.
	if _, err := m.forward(785, m.NewCache(8)); err != nil {
		t.Fatalf("forward after releasing the source mapping: %v", err)
	}
}

// TestP13_rssAfterLoad reports resident set size after a load, so the release can be measured
// rather than asserted. DIAGNOSTIC — P13's own entry says "do not assume the 1.69x transfers",
// because that figure came from a box holding 46.8 GB of 62 GB and is a page-pressure artifact as
// much as a mapping one. This prints; it does not gate.
func TestP13_rssAfterLoad(t *testing.T) {
	if os.Getenv("GOINFER_DIAG") == "" {
		t.Skip("DIAGNOSTIC (set GOINFER_DIAG=1)")
	}
	requireHeavyModel(t)
	dir := os.Getenv("GOINFER_MELLUM_CKPT")
	if dir == "" {
		dir = os.Getenv("HOME") + "/models/mellum2-unq"
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		t.Skipf("no checkpoint at %s", dir)
	}
	rss := func() int64 {
		out, err := exec.Command("ps", "-o", "rss=", "-p", fmt.Sprint(os.Getpid())).Output()
		if err != nil {
			return -1
		}
		var kb int64
		fmt.Sscan(strings.TrimSpace(string(out)), &kb)
		return kb * 1024
	}
	before := rss()
	m, err := Load(dir, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	after := rss()
	var src int64
	for _, g := range []string{"model*.safetensors"} {
		fs, _ := filepath.Glob(filepath.Join(dir, g))
		for _, f := range fs {
			if fi, err := os.Stat(f); err == nil {
				src += fi.Size()
			}
		}
	}
	fmt.Fprintf(os.Stderr,
		"P13 RSS: before %.2f GB -> after %.2f GB (delta %.2f GB); source on disk %.2f GB; "+
			"mapping retained: %v\n",
		float64(before)/1e9, float64(after)/1e9, float64(after-before)/1e9, float64(src)/1e9,
		m.w.st != nil)
}
