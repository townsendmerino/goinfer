// Package paritykey computes a parity family's validation key from a tree: today's key, which is
// decoder/parity_manifest_test.go:freshDepsHash as a library, and the shadow key VK5 logs beside it, which replaces the
// aikit pin with a digest of the kernel fingerprints (internal/kernelprint).
//
// Contract. Key hashes, in sorted order, each dependency path, a NUL, its content (a Go file's token stream, comments and
// layout dropped, directives kept; any other file's bytes) and a NUL, then the tail. TodayTail and ShadowTail are the two
// tails. A tree is read through a Source, so the same code hashes a working tree and any commit. TestParityKey_matchesFreshDepsHash
// (decoder) holds today's key equal to freshDepsHash for every family.
//
// Imports: the standard library only.
//
// Plan: docs/tasks/task-subsystems-2026-10.md §5.8.
package paritykey

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"go/scanner"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// ManifestPath is the parity manifest, relative to the repository root.
const ManifestPath = "testdata/parity_manifest.json"

// Source reads a tree's files by repository-relative path. A missing file is an error wrapping os.ErrNotExist.
type Source interface {
	ReadFile(path string) ([]byte, error)
}

// Dir is a Source over a working tree rooted at the named directory.
type Dir string

func (d Dir) ReadFile(path string) ([]byte, error) {
	return os.ReadFile(filepath.Join(string(d), path))
}

// Manifest is the part of testdata/parity_manifest.json a key reads.
type Manifest struct {
	SharedSets map[string][]string `json:"shared_sets"`
	Families   map[string]Family   `json:"families"`
}

// Family is one family's row: the sets it uses, its own files, and the hash and status recorded for it.
type Family struct {
	Uses     []string `json:"uses"`
	Own      []string `json:"own"`
	DepsHash string   `json:"deps_hash"`
	Status   string   `json:"status"`
}

// ReadManifest reads the parity manifest from src.
func ReadManifest(src Source) (*Manifest, error) {
	b, err := src.ReadFile(ManifestPath)
	if err != nil {
		return nil, err
	}
	var m Manifest
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("%s: %w", ManifestPath, err)
	}
	return &m, nil
}

// DepFiles is the sorted, deduplicated union of a family's shared sets and own files.
func DepFiles(m *Manifest, fam Family) []string {
	seen := map[string]bool{}
	for _, set := range fam.Uses {
		for _, f := range m.SharedSets[set] {
			seen[f] = true
		}
	}
	for _, f := range fam.Own {
		seen[f] = true
	}
	out := make([]string, 0, len(seen))
	for f := range seen {
		out = append(out, f)
	}
	sort.Strings(out)
	return out
}

var aikitPinRE = regexp.MustCompile(`(?m)^\s*github\.com/townsendmerino/aikit (v[0-9][^\s]*)`)

// AikitPin reads the root go.mod's aikit requirement.
func AikitPin(src Source) (string, error) {
	b, err := src.ReadFile("go.mod")
	if err != nil {
		return "", err
	}
	m := aikitPinRE.FindSubmatch(b)
	if m == nil {
		return "", fmt.Errorf("root go.mod does not require github.com/townsendmerino/aikit")
	}
	return string(m[1]), nil
}

// TodayTail is the tail of today's key: the aikit pin.
func TodayTail(pin string) string { return "aikit_version=" + pin }

// ShadowTail is the tail of the shadow key: the digest of the kernel fingerprints (kernelprint.Report.Digest).
func ShadowTail(digest string) string { return "kernelprint=" + digest }

// Content is what the key hashes for one path: a Go file's token stream, any other file's bytes.
func Content(path string, b []byte) ([]byte, error) {
	if !strings.HasSuffix(path, ".go") {
		return b, nil
	}
	return goCodeTokens(path, b)
}

// Key is a family's key over src with the given tail.
func Key(src Source, m *Manifest, fam Family, tail string) (string, error) {
	h := sha256.New()
	for _, p := range DepFiles(m, fam) {
		b, err := src.ReadFile(p)
		if err != nil {
			return "", err
		}
		if b, err = Content(p, b); err != nil {
			return "", err
		}
		h.Write([]byte(p))
		h.Write([]byte{0})
		h.Write(b)
		h.Write([]byte{0})
	}
	h.Write([]byte(tail))
	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}

// ContentDigest is a short digest of Content(path, b), for naming which input of a key moved.
func ContentDigest(src Source, path string) (string, error) {
	b, err := src.ReadFile(path)
	if err != nil {
		return "", err
	}
	if b, err = Content(path, b); err != nil {
		return "", err
	}
	d := sha256.Sum256(b)
	return hex.EncodeToString(d[:8]), nil
}

// goCodeTokens is decoder/parity_manifest_test.go:goCodeTokens.
func goCodeTokens(name string, src []byte) ([]byte, error) {
	fset := token.NewFileSet()
	file := fset.AddFile(name, -1, len(src))
	var s scanner.Scanner
	var scanErr error
	s.Init(file, src, func(pos token.Position, msg string) {
		if scanErr == nil {
			scanErr = fmt.Errorf("%s: %s", pos, msg)
		}
	}, scanner.ScanComments)
	var out bytes.Buffer
	for {
		_, tok, lit := s.Scan()
		if tok == token.EOF {
			break
		}
		if tok == token.COMMENT && !strings.HasPrefix(lit, "//go:") && !strings.HasPrefix(lit, "// +build") {
			continue
		}
		out.WriteString(tok.String())
		out.WriteByte(' ')
		out.WriteString(lit)
		out.WriteByte('\n')
	}
	if scanErr != nil {
		return nil, scanErr
	}
	return out.Bytes(), nil
}
