package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

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
