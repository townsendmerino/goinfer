// Package testgate holds the skip gates tests share across goinfer's modules: the heavy opt-in, the device probe's
// verdict, the model directory and the measure opt-in; and SetForTest, the one way a test sets a package-level switch. Each skip message begins with a fixed token (SkipHeavy, SkipDevice, SkipAsset, SkipMeasure), which
// cmd/gate's skip buckets and scripts/cuda_device_suite.py classify before their wording patterns.
//
// It is imported only by _test.go files (TestNoProductionImporter), so its environment reads are not production reads.
package testgate

import (
	"os"
	"path/filepath"
	"testing"
)

// The leading token of each gate's skip message. Classifiers match the token anywhere in the reason line, since
// `go test` prefixes it with file:line.
const (
	SkipHeavy   = "SKIP-HEAVY:"
	SkipDevice  = "SKIP-DEVICE:"
	SkipAsset   = "SKIP-ASSET:"
	SkipMeasure = "SKIP-MEASURE:"
)

// RequireHeavy skips t unless GOINFER_HEAVY_TESTS is set: a test that loads a multi-GB checkpoint runs only on request,
// not because the file happens to be on disk. The caller's own os.Stat skip on the asset stays as a second guard.
func RequireHeavy(t testing.TB) {
	t.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip(SkipHeavy + " heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a multi-GB model from ~/models)")
	}
}

// RequireMeasure skips t unless GOINFER_MEASURE_TESTS is set. It marks a test that measures rather than guards (a sweep, a
// timing, a dump, a survey whose decision is recorded): it runs on demand, never in a gate, since cmd/gate drops the
// variable from every cell (cellEnv). GOINFER_HEAVY_TESTS does not imply it. The list of what moved here and why is
// docs/measurements/h1-heavy-tier-triage-2026-11-23.md, § "9. Proposed as approved".
func RequireMeasure(t testing.TB) {
	t.Helper()
	if os.Getenv("GOINFER_MEASURE_TESTS") == "" {
		t.Skip(SkipMeasure + " a measurement, not a guard: set GOINFER_MEASURE_TESTS=1 to run it (on demand; no gate runs it)")
	}
}

// RequireDevice skips t when err, the caller's device probe (cuInit, the Metal device open, the WebGPU adapter request),
// is non-nil. backend names the device in the message ("cuda", "metal", "webgpu"). The probe stays with the caller: this
// package cannot import a backend.
func RequireDevice(t testing.TB, backend string, err error) {
	t.Helper()
	if err != nil {
		t.Skipf("%s no %s device: %v", SkipDevice, backend, err)
	}
}

// ModelPath resolves name under GOINFER_MODELS_DIR, default $HOME/models; with no home directory, under models in the
// working directory.
func ModelPath(name string) string {
	root := os.Getenv("GOINFER_MODELS_DIR")
	if root == "" {
		root = "models"
		if h, err := os.UserHomeDir(); err == nil {
			root = filepath.Join(h, "models")
		}
	}
	return filepath.Join(root, name)
}

// RequireModel returns ModelPath(name), skipping t when nothing is there.
func RequireModel(t testing.TB, name string) string {
	t.Helper()
	p := ModelPath(name)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("%s %s not found under the models directory: %v", SkipAsset, name, err)
	}
	return p
}

// SetForTest sets *p to v for the rest of t and restores the value it had when t ends, so a test that sets a package-level
// switch or seam cannot leave it set for the next. A test running in parallel with another that reads *p must not use it.
func SetForTest[T any](t testing.TB, p *T, v T) {
	t.Helper()
	old := *p
	*p = v
	t.Cleanup(func() { *p = old })
}
