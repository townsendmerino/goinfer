package gatecore

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func movedRepo(t *testing.T) (root, pre, post string) {
	t.Helper()
	root = t.TempDir()
	git := func(args ...string) string {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git("init", "-q")
	if err := os.MkdirAll(filepath.Join(root, "metal"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "metal", "k.txt"), []byte("kernel\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "metal")
	git("commit", "-q", "-m", "before the move")
	pre = git("rev-parse", "HEAD")
	if err := os.MkdirAll(filepath.Join(root, "backend"), 0o755); err != nil {
		t.Fatal(err)
	}
	git("mv", "metal", "backend/metal")
	git("commit", "-q", "-m", "after the move")
	post = git("rev-parse", "HEAD")
	return root, pre, post
}

func TestCanonicalPath_reportsTheCurrentName(t *testing.T) {
	for in, want := range map[string]string{
		"cuda":         "backend/cuda",
		"metal/x.go":   "backend/metal/x.go",
		"gpu/kernels":  "backend/webgpu/kernels",
		"backend/cuda": "backend/cuda",
		"decoder":      "decoder",
		"cudaish/x":    "cudaish/x",
	} {
		if got := CanonicalPath(in); got != want {
			t.Errorf("CanonicalPath(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestPreMovePath(t *testing.T) {
	for _, c := range []struct {
		in, out string
		moved   bool
	}{
		{"backend/cuda", "cuda", true},
		{"backend/metal/internal/x", "metal/internal/x", true},
		{"backend/webgpu", "gpu", true},
		{"backend/webgpu/kernels", "gpu/kernels", true},
		{"backend/cudaish", "backend/cudaish", false},
		{"decoder", "decoder", false},
	} {
		got, moved := PreMovePath(c.in)
		if got != c.out || moved != c.moved {
			t.Errorf("PreMovePath(%q) = %q, %v; want %q, %v", c.in, got, moved, c.out, c.moved)
		}
	}
}

func TestTreeAt_preMoveCommitReadsThroughTheFallback(t *testing.T) {
	root, pre, post := movedRepo(t)
	want, err := GitTree(root, pre, "metal")
	if err != nil || want == "" {
		t.Fatalf("pre-move tree of metal: %q, %v", want, err)
	}
	calls := map[string]int{}
	lookup := func(rev, dir string) (string, error) {
		calls[rev+" "+dir]++
		return GitTree(root, rev, dir)
	}
	got, err := TreeAt(lookup, pre, "backend/metal")
	if err != nil || got != want {
		t.Fatalf("TreeAt at the pre-move commit = %q, %v; want %q", got, err, want)
	}
	if calls[pre+" backend/metal"] != 1 || calls[pre+" metal"] != 1 {
		t.Fatalf("lookups at the pre-move commit: %v; want the new path once, then the old one once", calls)
	}

	calls = map[string]int{}
	got, err = TreeAt(lookup, post, "backend/metal")
	if err != nil || got == "" {
		t.Fatalf("TreeAt at the post-move commit = %q, %v", got, err)
	}
	if len(calls) != 1 || calls[post+" backend/metal"] != 1 {
		t.Fatalf("lookups at the post-move commit: %v; want the new path only", calls)
	}
}

func TestObjectAt_emptyLsTreeAtPreMoveCommitReadsThroughTheFallback(t *testing.T) {
	root, pre, _ := movedRepo(t)
	want, err := GitObject(root, pre, "metal/k.txt")
	if err != nil || want == "" {
		t.Fatalf("pre-move object of metal/k.txt: %q, %v", want, err)
	}
	got, err := ObjectAt(func(rev, p string) (string, error) { return GitObject(root, rev, p) }, pre, "backend/metal/k.txt")
	if err != nil || got != want {
		t.Fatalf("ObjectAt at the pre-move commit = %q, %v; want %q", got, err, want)
	}
}
