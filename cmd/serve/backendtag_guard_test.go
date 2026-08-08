package main

import (
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestBackendTagGuardFailsBuild is the gate for audit D-B / field-report F1: since the M-19
// submodule split the root cmd/serve builds no backend, so `go build -tags cuda ./cmd/serve`
// (the command in every pre-v0.10.0 doc) used to exit 0 and silently produce a CPU binary. The
// backendtag_guard_*.go files turn that into a compile error whose text names the submodule
// entrypoint to run instead. This test proves BOTH halves for all three backends: the build
// fails (non-zero exit) AND stderr carries the exact replacement command.
//
// It runs in the DEFAULT `go test ./...` — no -short, no build tag. A finding about a silent
// build earns a gate that always runs. (The guard files carry //go:build tags, so they are
// absent from this default-build test binary; the test shells out to `go build -tags …`, a
// separate process, to exercise them.)
func TestBackendTagGuardFailsBuild(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go toolchain not on PATH")
	}
	// The package to (fail to) build is this directory's package.
	const pkg = "github.com/townsendmerino/goinfer/cmd/serve"

	cases := []struct {
		tag     string
		wantCmd string // the exact replacement command that must appear in stderr
	}{
		{"cuda", "go build -tags cuda github.com/townsendmerino/goinfer/cuda/cmd/serve"},
		{"gpu", "go build -tags gpu github.com/townsendmerino/goinfer/gpu/cmd/serve"},
		{"metal", "go build github.com/townsendmerino/goinfer/metal/cmd/serve"},
	}
	for _, c := range cases {
		t.Run(c.tag, func(t *testing.T) {
			out := filepath.Join(t.TempDir(), "should-never-exist")
			cmd := exec.Command("go", "build", "-tags", c.tag, "-o", out, pkg)
			b, err := cmd.CombinedOutput()
			stderr := string(b)
			if err == nil {
				t.Fatalf("`go build -tags %s %s` exited 0 — the guard did not fire; it would produce a silent CPU binary.\noutput:\n%s", c.tag, pkg, stderr)
			}
			if !strings.Contains(stderr, c.wantCmd) {
				t.Errorf("build failed (good) but stderr does not name the replacement command %q.\nstderr:\n%s", c.wantCmd, stderr)
			}
		})
	}
}
