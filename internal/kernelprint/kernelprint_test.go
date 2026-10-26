package kernelprint

import (
	"encoding/json"
	"os"
	"testing"
	"time"
)

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
