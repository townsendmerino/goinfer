// Package paritykey computes a parity family's validation key from a tree: today's key, which is
// decoder/parity_manifest_test.go:freshDepsHash as a library, and the shadow key VK5's week 2 logs beside it, in which the
// program builders (BuilderFile) leave the key and the family's program fingerprints take their place.
//
// Contract. A key hashes, in sorted order, each dependency path, a NUL, its content (a Go file's token stream, comments
// and layout dropped, directives kept; any other file's bytes) and a NUL, then a tail. Today's tail is the committed
// kernel fingerprint table's digest, or the aikit pin in a tree from before the table. The shadow key of a family with
// programs hashes its files less the builders, with "programs=<digest>" and today's tail as its tail; a family with no
// programs keeps today's key. A tree is read through a Source, so the same code hashes a working tree and any commit.
// decoder's TestParityKey_matchesFreshDepsHash holds today's key equal to freshDepsHash, and
// decoder's TestParityKey_shadowMatchesDecoder holds the shadow key to decoder's own computation. LoaderDigest reads SB4's
// loader identity table, which no key reads yet.
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
	"errors"
	"fmt"
	"go/scanner"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// RecordsDir holds the parity records: SharedSetsFile and one <family>.json per family. LegacyManifestPath is the single
// file they replaced (VK3); commits before the change carry it, and ManifestJSON still reads it there.
const (
	RecordsDir         = "testdata/parity"
	SharedSetsFile     = RecordsDir + "/shared_sets.json"
	LegacyManifestPath = "testdata/parity_manifest.json"
)

// Source reads a tree's files by repository-relative path, and lists a directory's file names. A missing file or
// directory is an error wrapping os.ErrNotExist.
type Source interface {
	ReadFile(path string) ([]byte, error)
	ListDir(path string) ([]string, error)
}

// Dir is a Source over a working tree rooted at the named directory.
type Dir string

func (d Dir) ReadFile(path string) ([]byte, error) {
	return os.ReadFile(filepath.Join(string(d), path))
}

func (d Dir) ListDir(path string) ([]string, error) {
	ents, err := os.ReadDir(filepath.Join(string(d), path))
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range ents {
		if !e.IsDir() {
			out = append(out, e.Name())
		}
	}
	return out, nil
}

// ManifestJSON assembles the parity manifest as one JSON object, {"shared_sets": {...}, "families": {"<name>": {...}}},
// from src's records, each family's record carried verbatim. A tree without records (a commit before VK3) yields its
// single manifest file unchanged. A tree with neither is an error, never an empty manifest.
func ManifestJSON(src Source) ([]byte, error) {
	sets, err := src.ReadFile(SharedSetsFile)
	if errors.Is(err, os.ErrNotExist) {
		if legacy, lerr := src.ReadFile(LegacyManifestPath); lerr == nil {
			return legacy, nil
		}
		return nil, fmt.Errorf("no parity records (%s) and no %s: %w", SharedSetsFile, LegacyManifestPath, err)
	} else if err != nil {
		return nil, err
	}
	names, err := src.ListDir(RecordsDir)
	if err != nil {
		return nil, err
	}
	fams := map[string]json.RawMessage{}
	for _, n := range names {
		if !strings.HasSuffix(n, ".json") || RecordsDir+"/"+n == SharedSetsFile {
			continue
		}
		b, err := src.ReadFile(RecordsDir + "/" + n)
		if err != nil {
			return nil, err
		}
		if !json.Valid(b) {
			return nil, fmt.Errorf("%s/%s: not valid JSON", RecordsDir, n)
		}
		fams[strings.TrimSuffix(n, ".json")] = b
	}
	if len(fams) == 0 {
		return nil, fmt.Errorf("%s holds no family record", RecordsDir)
	}
	if !json.Valid(sets) {
		return nil, fmt.Errorf("%s: not valid JSON", SharedSetsFile)
	}
	return json.Marshal(struct {
		SharedSets json.RawMessage            `json:"shared_sets"`
		Families   map[string]json.RawMessage `json:"families"`
	}{sets, fams})
}

// Manifest is the part of the parity manifest a key reads.
type Manifest struct {
	SharedSets map[string][]string `json:"shared_sets"`
	Families   map[string]Family   `json:"families"`
}

// Family is one family's row: the sets it uses, its own files, the fixture directories whose program fingerprints stand
// for its builders, and the hash and status recorded for it.
type Family struct {
	Uses     []string `json:"uses"`
	Own      []string `json:"own"`
	Programs []string `json:"programs"`
	DepsHash string   `json:"deps_hash"`
	Status   string   `json:"status"`
}

// ReadManifest reads the parity manifest from src (ManifestJSON).
func ReadManifest(src Source) (*Manifest, error) {
	b, err := ManifestJSON(src)
	if err != nil {
		return nil, err
	}
	var m Manifest
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("parity manifest: %w", err)
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

// KernelTablePath is the committed kernel fingerprint table: configuration -> kernel -> fingerprint. internal/kernelprint's
// TestCompute_table holds each runner's own entry to the suite.
const KernelTablePath = "testdata/kernel_fingerprints.json"

// ReadKernelTable reads src's kernel fingerprint table. A tree without one gives an error wrapping os.ErrNotExist.
func ReadKernelTable(src Source) (map[string]map[string]string, error) {
	b, err := src.ReadFile(KernelTablePath)
	if err != nil {
		return nil, err
	}
	var tab map[string]map[string]string
	if err := json.Unmarshal(b, &tab); err != nil {
		return nil, fmt.Errorf("%s: %w", KernelTablePath, err)
	}
	if len(tab) == 0 {
		return nil, fmt.Errorf("%s has no configuration", KernelTablePath)
	}
	return tab, nil
}

// KernelTableDigest is the table's digest, independent of its layout: each configuration in sorted order, a newline,
// then its kernels as "name=fingerprint" lines in sorted order.
func KernelTableDigest(tab map[string]map[string]string) string {
	h := sha256.New()
	for _, cfg := range sortedKeys(tab) {
		h.Write([]byte(cfg + "\n"))
		h.Write([]byte(kernelLines(tab[cfg])))
	}
	return hex.EncodeToString(h.Sum(nil))
}

func kernelLines(kernels map[string]string) string {
	var b strings.Builder
	for _, k := range sortedKeys(kernels) {
		b.WriteString(k + "=" + kernels[k] + "\n")
	}
	return b.String()
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// TodayTail is the tail of today's key in src: "kernel_fingerprints=" and the kernel table's digest, or, in a tree from
// before the table, "aikit_version=" and the aikit pin.
func TodayTail(src Source) (string, error) {
	tab, err := ReadKernelTable(src)
	if errors.Is(err, os.ErrNotExist) {
		pin, err := AikitPin(src)
		if err != nil {
			return "", err
		}
		return AikitTailPrefix + pin, nil
	}
	if err != nil {
		return "", err
	}
	return KernelTailPrefix + KernelTableDigest(tab), nil
}

// The two definitions of today's tail, before and after the kernel table replaced the aikit pin.
const (
	AikitTailPrefix  = "aikit_version="
	KernelTailPrefix = "kernel_fingerprints="
)

// ProgramTablePath is VK4's committed program fingerprint table: GOARCH -> fixture directory -> fingerprint. decoder's
// TestProgramFingerprint_table holds each GOARCH's entry to the programs the fixtures' configs build.
const ProgramTablePath = "testdata/program_fingerprints.json"

// BuilderFile reports whether path is a program builder: a file whose effect on numerics is the program it builds,
// which the program fingerprints cover. The executors, which compute each op, are not builders and stay in every key.
func BuilderFile(path string) bool {
	switch path {
	case "decoder/program.go", "decoder/blockspec.go", "decoder/arch.go", "decoder/config.go", "decoder/registry.go",
		"decoder/gemma12.go": // gemma's and gemma2's adapters, under another name
		return true
	}
	return strings.HasPrefix(path, "decoder/arch_") && strings.HasSuffix(path, ".go") && !strings.HasSuffix(path, "_test.go")
}

// ReadProgramTable reads src's program fingerprint table, or nil in a tree from before it.
func ReadProgramTable(src Source) (map[string]map[string]string, error) {
	b, err := src.ReadFile(ProgramTablePath)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var tab map[string]map[string]string
	if err := json.Unmarshal(b, &tab); err != nil {
		return nil, fmt.Errorf("%s: %w", ProgramTablePath, err)
	}
	return tab, nil
}

// ProgramsDigest is the digest of the fixtures' entries in the program table, for every GOARCH in it: each GOARCH in
// sorted order, then "fixture=fingerprint" lines in sorted fixture order. A fixture missing from a GOARCH, or a table
// with no GOARCH, is an error.
func ProgramsDigest(fixtures []string, tab map[string]map[string]string) (string, error) {
	if len(tab) == 0 {
		return "", fmt.Errorf("%s is missing or empty, so it cannot stand for the builders", ProgramTablePath)
	}
	fx := append([]string(nil), fixtures...)
	sort.Strings(fx)
	h := sha256.New()
	for _, arch := range sortedKeys(tab) {
		h.Write([]byte(arch + "\n"))
		for _, f := range fx {
			fp, ok := tab[arch][f]
			if !ok {
				return "", fmt.Errorf("%s has no %s entry for %s", ProgramTablePath, arch, f)
			}
			h.Write([]byte(f + "=" + fp + "\n"))
		}
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// LoaderTablePath is SB4's committed loader identity table: GOARCH -> fixture -> quant mode -> the digest of the fixture's
// per-tensor identities as loaded in that mode. decoder's TestLoaderIdentity_table holds each GOARCH's entry to the
// loaders. No key reads it yet: it is the input that would let the loader files leave a family's key the way the program
// fingerprints let the builders leave it (docs/tasks/task-subsystems-2026-10.md §8 SB4).
const LoaderTablePath = "testdata/loader_identity.json"

// ReadLoaderTable reads src's loader identity table, or nil in a tree from before it.
func ReadLoaderTable(src Source) (map[string]map[string]map[string]string, error) {
	b, err := src.ReadFile(LoaderTablePath)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var tab map[string]map[string]map[string]string
	if err := json.Unmarshal(b, &tab); err != nil {
		return nil, fmt.Errorf("%s: %w", LoaderTablePath, err)
	}
	return tab, nil
}

// LoaderDigest is the digest of the fixtures' entries in the loader table, for every GOARCH in it: each GOARCH in sorted
// order, then for each fixture in sorted order its "fixture mode=digest" lines in sorted mode order. An empty fixture list,
// a fixture with no mode in a GOARCH, or a table with no GOARCH, is an error. Which GOARCHes the table must hold is the
// key's rule to set, not this digest's.
func LoaderDigest(fixtures []string, tab map[string]map[string]map[string]string) (string, error) {
	if len(fixtures) == 0 {
		return "", fmt.Errorf("no fixtures: an empty list cannot stand for the loaders")
	}
	if len(tab) == 0 {
		return "", fmt.Errorf("%s is missing or empty, so it cannot stand for the loaders", LoaderTablePath)
	}
	fx := append([]string(nil), fixtures...)
	sort.Strings(fx)
	h := sha256.New()
	for _, arch := range sortedKeys(tab) {
		h.Write([]byte(arch + "\n"))
		for _, f := range fx {
			modes := tab[arch][f]
			if len(modes) == 0 {
				return "", fmt.Errorf("%s has no %s entry for %s", LoaderTablePath, arch, f)
			}
			for _, m := range sortedKeys(modes) {
				h.Write([]byte(f + " " + m + "=" + modes[m] + "\n"))
			}
		}
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// ShadowFiles is the files the shadow key hashes for fam: its dependency files less the builders when it has programs,
// all of them when it has none.
func ShadowFiles(m *Manifest, fam Family) []string {
	files := DepFiles(m, fam)
	if len(fam.Programs) == 0 {
		return files
	}
	var out []string
	for _, f := range files {
		if !BuilderFile(f) {
			out = append(out, f)
		}
	}
	return out
}

// ShadowKey is fam's week-2 shadow key over src: today's key for a family with no programs; otherwise its files less
// the builders, then "programs=<ProgramsDigest>", a newline and todayTail.
func ShadowKey(src Source, m *Manifest, fam Family, todayTail string, programs map[string]map[string]string) (string, error) {
	if len(fam.Programs) == 0 {
		return Key(src, m, fam, todayTail)
	}
	d, err := ProgramsDigest(fam.Programs, programs)
	if err != nil {
		return "", err
	}
	return hashFiles(src, ShadowFiles(m, fam), "programs="+d+"\n"+todayTail)
}

// Content is what the key hashes for one path: a Go file's token stream, any other file's bytes.
func Content(path string, b []byte) ([]byte, error) {
	if !strings.HasSuffix(path, ".go") {
		return b, nil
	}
	return goCodeTokens(path, b)
}

// Key is a family's key over src with the given tail.
func Key(src Source, m *Manifest, fam Family, tail string) (string, error) {
	return hashFiles(src, DepFiles(m, fam), tail)
}

func hashFiles(src Source, files []string, tail string) (string, error) {
	h := sha256.New()
	for _, p := range files {
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
