package gatecore

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

// GatherProvenance collects git commit/dirty status, UTC timestamp, and host uname in the current working directory.
func GatherProvenance(fields [][2]string) Provenance {
	p := Provenance{Commit: "?", Date: time.Now().UTC().Format(time.RFC3339), Fields: fields}
	if out, err := exec.Command("git", "rev-parse", "--short", "HEAD").Output(); err == nil {
		p.Commit = strings.TrimSpace(string(out))
	}
	if out, err := exec.Command("git", "status", "--porcelain").Output(); err == nil && len(strings.TrimSpace(string(out))) > 0 {
		p.Dirty = true
	}
	if out, err := exec.Command("uname", "-sm").Output(); err == nil {
		p.Host = strings.TrimSpace(string(out))
	}
	return p
}

// GatherProvenanceIn collects provenance for the git repository rooted at root.
func GatherProvenanceIn(root string, fields [][2]string) Provenance {
	p := Provenance{Commit: "?", Date: time.Now().UTC().Format(time.RFC3339), Fields: fields}
	git := func(args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		out, _ := cmd.Output()
		return strings.TrimSpace(string(out))
	}
	if c := git("rev-parse", "--short", "HEAD"); c != "" {
		p.Commit = c
	}
	p.Dirty = git("status", "--porcelain") != ""
	if out, err := exec.Command("uname", "-sm").Output(); err == nil {
		p.Host = strings.TrimSpace(string(out))
	}
	return p
}

// RepoRoot returns `git rev-parse --show-toplevel` for the current working directory.
func RepoRoot() (string, error) {
	out, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		return "", fmt.Errorf("not inside a git repository: %w", err)
	}
	return strings.TrimSpace(string(out)), nil
}

// GitOut runs `git args...` inside dir and returns trimmed combined output.
func GitOut(dir string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return strings.TrimSpace(string(out)), fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	return strings.TrimSpace(string(out)), nil
}

// GitTree returns the git tree SHA of dir at rev.
func GitTree(root, rev, dir string) (string, error) {
	return GitOut(root, "rev-parse", rev+":"+dir)
}

// GitObject returns the git object SHA of path at rev, or "" when path is absent at rev.
func GitObject(root, rev, path string) (string, error) {
	out, err := GitOut(root, "ls-tree", rev, "--", path)
	if err != nil {
		return "", err
	}
	f := strings.Fields(out)
	if len(f) < 3 {
		return "", nil
	}
	return f[2], nil
}

// TailLines returns up to the last n lines of path, or a single diagnostic line when unreadable.
func TailLines(path string, n int) []string {
	b, err := os.ReadFile(path)
	if err != nil {
		return []string{fmt.Sprintf("(no log: %v)", err)}
	}
	lines := strings.Split(strings.TrimRight(string(bytes.TrimSpace(b)), "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return lines
}
