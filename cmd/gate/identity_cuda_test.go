package main

import (
	"io"
	"os"
	"slices"
	"strings"
	"testing"
)

// identityStderr runs runIdentity and returns its exit code and what it wrote to os.Stderr (the flag errors go there).
func identityStderr(t *testing.T, argv ...string) (int, string) {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	old := os.Stderr
	os.Stderr = w
	code := runIdentity(argv, io.Discard)
	os.Stderr = old
	w.Close()
	b, _ := io.ReadAll(r)
	return code, string(b)
}

// TestIdentityCUDA_backendIsKnown: -backend cuda is accepted by identity's flag handling, builds its dumper in cuda/ under
// -tags cuda, runs the two quantized paths on the tiny fixtures (a GPU backend declines f32), and counts only a path that starts
// with the backend's name as run on it, so a CPU fallback reads NOT RUN and not IDENTICAL.
func TestIdentityCUDA_backendIsKnown(t *testing.T) {
	// -assets is checked after -backend, so a bad one proves the backend was accepted.
	if code, msg := identityStderr(t, "-backend", "cuda", "-assets", "bogus", "a", "b"); code != 2 || strings.Contains(msg, "-backend") || !strings.Contains(msg, "-assets") {
		t.Fatalf("-backend cuda was refused as a backend (exit %d): %s", code, msg)
	}
	if code, msg := identityStderr(t, "-backend", "rocm", "a", "b"); code != 2 || !strings.Contains(msg, "want cpu, metal, webgpu or cuda") {
		t.Fatalf("-backend rocm: exit %d, stderr %q, want exit 2 naming the four backends", code, msg)
	}
	if got := defaultIdentityQuants("cuda", "tiny"); !slices.Equal(got, []string{"int4", "int8int8"}) {
		t.Errorf("default tiny quants on cuda = %v, want int4 and int8int8", got)
	}
	if got := defaultIdentityQuants("cuda", "real"); !slices.Equal(got, []string{"int4"}) {
		t.Errorf("default real quants on cuda = %v, want int4", got)
	}
	if d := identityBuildDesc("cuda"); !strings.Contains(d, "cuda/") || !strings.Contains(d, "-tags cuda") {
		t.Errorf("build description %q does not name the cuda module and its tag", d)
	}
	if !pathOnBackend("cuda-resident (int4)", "cuda") || pathOnBackend("cpu (int4)", "cuda") {
		t.Error("pathOnBackend: a cuda-resident path must count as on cuda, and a cpu path must not")
	}
}
