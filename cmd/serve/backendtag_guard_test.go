package main

import (
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestBackendTagGuardFailsBuild pins that `go build -tags cuda|gpu|metal ./cmd/serve` fails to compile, and that stderr
// carries the exact submodule entrypoint to run instead: the root cmd/serve builds no backend, so without the
// backendtag_guard_*.go files that command would exit 0 and silently produce a CPU binary. Those files carry //go:build
// tags and are absent from this test binary, so the test shells out to `go build`. It runs in the default
// `go test ./...` on purpose (no -short, no tag). Origin: docs/code-notes/cmd-serve.md#TestBackendTagGuardFailsBuild.
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
