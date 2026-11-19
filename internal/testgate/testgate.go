// Package testgate holds the skip gates tests share across goinfer's modules: the heavy opt-in, the device probe's
// verdict and the model directory. Each skip message begins with a fixed token (SkipHeavy, SkipDevice, SkipAsset), which
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
	SkipHeavy  = "SKIP-HEAVY:"
	SkipDevice = "SKIP-DEVICE:"
	SkipAsset  = "SKIP-ASSET:"
)

// RequireHeavy skips t unless GOINFER_HEAVY_TESTS is set: a test that loads a multi-GB checkpoint runs only on request,
// not because the file happens to be on disk. The caller's own os.Stat skip on the asset stays as a second guard.
func RequireHeavy(t testing.TB) {
	t.Helper()
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip(SkipHeavy + " heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a multi-GB model from ~/models)")
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
