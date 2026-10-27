package kernelprint

import (
	"encoding/json"
	"errors"
	"flag"
	"os"
	"sort"
	"strings"
	"testing"
	"time"
)

// tablePath is the committed kernel fingerprint table, one entry per dispatch configuration CI runs on. The parity key
// mixes in its digest (paritykey.KernelTableDigest) in place of aikit's version (VK5's switch).
const tablePath = "../../testdata/kernel_fingerprints.json"

var updateTable = flag.Bool("update", false, "write this configuration's entry of testdata/kernel_fingerprints.json")

// TestCompute_stable is K1: two Computes in one process agree on every kernel.
func TestCompute_stable(t *testing.T) {
	t0 := time.Now()
	a := Compute()
	d := time.Since(t0)
	b := Compute()
	if len(a.Kernels) != len(suite) {
		t.Fatalf("%d fingerprints for %d kernels", len(a.Kernels), len(suite))
	}
	for _, name := range Kernels() {
		if a.Kernels[name] != b.Kernels[name] {
			t.Errorf("%s: %s then %s", name, a.Kernels[name], b.Kernels[name])
		}
	}
	t.Logf("%d kernels under %s; one Compute took %s", len(suite), a.Config, d.Round(time.Millisecond))
}

// TestCompute_child runs Compute for K2 and K3, which run this package's test binary as a child with a different aikit
// or with coverage. It writes the report to GOINFER_KERNELPRINT_OUT, and does nothing without it.
func TestCompute_child(t *testing.T) {
	out := os.Getenv("GOINFER_KERNELPRINT_OUT")
	if out == "" {
		t.Skip("a child of K2 and K3; set GOINFER_KERNELPRINT_OUT")
	}
	b, err := json.MarshalIndent(Compute(), "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(out, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestCompute_table holds this runner's fingerprints to its configuration's entry in the committed table. A configuration
// the table has no entry for is reported as not checked on a developer's machine, and fails on CI (GITHUB_ACTIONS), where
// a skip is invisible without -v and would leave the parity key on a table nobody checks. Each runner checks only its own
// entry, so a kernel change on one configuration is caught on a PR only when a runner of that configuration runs it
// (docs/measurements/vk5-2026-10.md §7).
func TestCompute_table(t *testing.T) {
	r := Compute()
	cfg := r.Config.String()
	tab := map[string]map[string]string{}
	b, err := os.ReadFile(tablePath)
	switch {
	case err == nil:
		if err := json.Unmarshal(b, &tab); err != nil {
			t.Fatalf("%s: %v", tablePath, err)
		}
	case !*updateTable || !errors.Is(err, os.ErrNotExist):
		t.Fatal(err)
	}
	if *updateTable {
		tab[cfg] = r.Kernels
		b, err := json.MarshalIndent(tab, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(tablePath, append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote the %s entry of %s", cfg, tablePath)
		return
	}
	want, ok := tab[cfg]
	if !ok {
		if os.Getenv("GITHUB_ACTIONS") == "true" {
			t.Fatalf("%s has no entry for %s, a configuration CI runs: add it (-update on such a machine), or the parity key rests on an unchecked table", tablePath, cfg)
		}
		t.Skipf("%s has no entry for %s: not checked here (write one with -update on such a machine)", tablePath, cfg)
	}
	var diffs []string
	for k, v := range r.Kernels {
		if want[k] != v {
			diffs = append(diffs, k)
		}
	}
	for k := range want {
		if _, ok := r.Kernels[k]; !ok {
			diffs = append(diffs, k+" (no longer in the suite)")
		}
	}
	sort.Strings(diffs)
	if len(diffs) > 0 {
		t.Errorf("%s: %d kernel fingerprint(s) differ from %s; a kernel changed (or the suite did): every parity family's key moves with the table, so update it (-update) in the change that moves it:\n  %s",
			cfg, len(diffs), tablePath, strings.Join(diffs, "\n  "))
		return
	}
	t.Logf("%d kernels under %s match the table", len(r.Kernels), cfg)
}
