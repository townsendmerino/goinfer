package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// movedRepo is a git repository whose first commit has the pre-move layout (metal/k.txt) and whose second has the
// post-move one (backend/metal/k.txt).
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
		if got := canonicalPath(in); got != want {
			t.Errorf("canonicalPath(%q) = %q, want %q", in, got, want)
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
		got, moved := preMovePath(c.in)
		if got != c.out || moved != c.moved {
			t.Errorf("preMovePath(%q) = %q, %v; want %q, %v", c.in, got, moved, c.out, c.moved)
		}
	}
}

// A commit from before the move reads through the fallback: the new path is absent there, the old one is read once, and
// the tree id is the one the old path has at that commit.
func TestTreeAt_preMoveCommitReadsThroughTheFallback(t *testing.T) {
	root, pre, post := movedRepo(t)
	want, err := gitTree(root, pre, "metal")
	if err != nil || want == "" {
		t.Fatalf("pre-move tree of metal: %q, %v", want, err)
	}
	calls := map[string]int{}
	lookup := func(rev, dir string) (string, error) {
		calls[rev+" "+dir]++
		return gitTree(root, rev, dir)
	}
	got, err := treeAt(lookup, pre, "backend/metal")
	if err != nil || got != want {
		t.Fatalf("treeAt at the pre-move commit = %q, %v; want %q", got, err, want)
	}
	if calls[pre+" backend/metal"] != 1 || calls[pre+" metal"] != 1 {
		t.Fatalf("lookups at the pre-move commit: %v; want the new path once, then the old one once", calls)
	}

	// At a post-move commit the new path resolves, and the fallback is not taken.
	calls = map[string]int{}
	got, err = treeAt(lookup, post, "backend/metal")
	if err != nil || got == "" {
		t.Fatalf("treeAt at the post-move commit = %q, %v", got, err)
	}
	if len(calls) != 1 || calls[post+" backend/metal"] != 1 {
		t.Fatalf("lookups at the post-move commit: %v; want the new path only", calls)
	}
}

// `git ls-tree` prints nothing for a missing path instead of failing; objectAt reads the pre-move name for it.
func TestObjectAt_emptyLsTreeAtPreMoveCommitReadsThroughTheFallback(t *testing.T) {
	root, pre, _ := movedRepo(t)
	want, err := gitObject(root, pre, "metal/k.txt")
	if err != nil || want == "" {
		t.Fatalf("pre-move object of metal/k.txt: %q, %v", want, err)
	}
	got, err := objectAt(func(rev, p string) (string, error) { return gitObject(root, rev, p) }, pre, "backend/metal/k.txt")
	if err != nil || got != want {
		t.Fatalf("objectAt at the pre-move commit = %q, %v; want %q", got, err, want)
	}
}

// A pre-move revision reads through the fallback for the layout Tier 2 regroup as well: gitSource (the changelog's reader
// at a base) reads and lists internal/app/serveapp by its pre-move name, which is what git has at that commit.
func TestGitSource_preMoveRevisionReadsTheInternalRegroup(t *testing.T) {
	root := t.TempDir()
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
	if err := os.MkdirAll(filepath.Join(root, "internal", "serveapp"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "internal", "serveapp", "routes.go"), []byte("package serveapp\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "internal")
	git("commit", "-q", "-m", "before the regroup")
	pre := git("rev-parse", "HEAD")
	if err := os.MkdirAll(filepath.Join(root, "internal", "app"), 0o755); err != nil {
		t.Fatal(err)
	}
	git("mv", "internal/serveapp", "internal/app/serveapp")
	git("commit", "-q", "-m", "after the regroup")

	src := gitSource{root: root, rev: pre}
	b, err := src.read("internal/app/serveapp/routes.go")
	if err != nil || string(b) != "package serveapp\n" {
		t.Fatalf("read at the pre-move revision = %q, %v; want the pre-move file", b, err)
	}
	names, err := src.list("internal/app/serveapp")
	if err != nil || len(names) != 1 || names[0] != "routes.go" {
		t.Fatalf("list at the pre-move revision = %v, %v; want [routes.go]", names, err)
	}
}

// regroupedRepo is a repository with internal/serveapp/routes.go at the first commit and internal/app/serveapp at the
// second. It returns the root, the pre-move commit and the post-move commit.
func regroupedRepo(t *testing.T, body string) (root, pre string) {
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
	if err := os.MkdirAll(filepath.Join(root, "internal", "serveapp"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "internal", "serveapp", "routes.go"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "internal")
	git("commit", "-q", "-m", "before the regroup")
	pre = git("rev-parse", "HEAD")
	if err := os.MkdirAll(filepath.Join(root, "internal", "app"), 0o755); err != nil {
		t.Fatal(err)
	}
	git("mv", "internal/serveapp", "internal/app/serveapp")
	git("commit", "-q", "-m", "after the regroup")
	return root, pre
}

// serveRoutes at a pre-move revision finds the routes under the pre-move directory: the route scan of `gate changelog`
// at a base tag (mac-rel's v0.23.0 run reported "0 routes" before the fallback).
func TestServeRoutes_preMoveRevisionFindsTheRoutes(t *testing.T) {
	root, pre := regroupedRepo(t, `package serveapp

import "net/http"

func register(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/jobs/{id}/events", nil)
}
`)
	routes, err := serveRoutes(gitSource{root: root, rev: pre})
	if err != nil {
		t.Fatalf("serveRoutes at the pre-move revision: %v", err)
	}
	if !routes["POST /v1/jobs/{}/events"] || len(routes) != 1 {
		t.Fatalf("routes at the pre-move revision = %v; want the one route under internal/serveapp", routes)
	}
}

// When the new path and the pre-move path both come back with nothing, the error reported is the new path's: the fallback
// must not replace it with the pre-move path's (or with an empty result that looks like success).
func TestGitSource_newPathErrorIsReportedWhenTheFallbackFindsNothing(t *testing.T) {
	root, _ := regroupedRepo(t, "package serveapp\n")
	src := gitSource{root: root, rev: "no-such-revision"}
	if _, err := src.read("internal/app/serveapp/routes.go"); err == nil || !strings.Contains(err.Error(), "internal/app/serveapp/routes.go") {
		t.Fatalf("read with nothing at either name: error = %v; want the new path's error", err)
	}
	if _, err := src.list("internal/app/serveapp"); err == nil || !strings.Contains(err.Error(), "internal/app/serveapp") {
		t.Fatalf("list with nothing at either name: error = %v; want the new path's error", err)
	}
}
