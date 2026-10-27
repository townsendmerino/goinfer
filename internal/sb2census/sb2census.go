// Package sb2census is the shared plumbing of the backend layers of SB2's admission and routing census
// (docs/tasks/task-subsystems-2026-10.md, "SB2 step 2"): layer D's fixtures and option cells, read from its golden so every
// layer censuses the same checkpoints, and the golden each layer is held to. Only tests import it.
package sb2census

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
)

// Cell is one option set of layer D (decoder's TestAdmissionCensus), whose table this must match: Fixtures checks it.
type Cell struct {
	Name, Quant     string
	ActGroup        int
	SSM             string // the GOINFER_SSM_RESIDENT knob, "" for unset
	KVPrec, KVQuant string // Options.KVPrecision and Options.KVQuant
}

var Cells = []Cell{
	{"int4", "int4", 0, "", "", ""}, {"int4/g32", "int4", 32, "", "", ""},
	{"int8int8", "int8int8", 0, "", "", ""}, {"int8int8/g32", "int8int8", 32, "", "", ""},
	{"f32", "f32", 0, "", "", ""}, {"f32/g32", "f32", 32, "", "", ""},
	{"int4/ssm", "int4", 0, "1", "", ""},
	{"int4/kv-f16", "int4", 0, "", "f16", ""}, {"int4/kv-i8", "int4", 0, "", "", "i8"},
}

// LayerD is layer D's census for this architecture, read from root (the repository root, relative to the caller).
type LayerD struct {
	Fixtures []string        // fixture names, in the golden's order: paths from the root module or the decoder package
	Resident map[string]bool // "<fixture> <cell>" for every cell layer D built resident on backend
}

// ReadLayerD reads layer D's golden for runtime.GOARCH and returns its fixtures and the cells it built resident on backend.
// It fails when layer D's option cells are not Cells.
func ReadLayerD(root, backend string) (LayerD, error) {
	b, err := readGz(filepath.Join(root, "testdata", "sb2_admission_census."+runtime.GOARCH+".txt.gz"))
	if err != nil {
		return LayerD{}, fmt.Errorf("layer D's golden: %w", err)
	}
	d := LayerD{Resident: map[string]bool{}}
	seen, cells := map[string]bool{}, map[string]bool{}
	for _, l := range strings.Split(string(b), "\n") {
		w := strings.Fields(l)
		if len(w) < 5 || w[0] != "L" || w[3] != backend {
			continue
		}
		if !seen[w[1]] {
			seen[w[1]] = true
			d.Fixtures = append(d.Fixtures, w[1])
		}
		cells[w[2]] = true
		if w[4] == "resident=true" {
			d.Resident[w[1]+" "+w[2]] = true
		}
	}
	want := map[string]bool{}
	for _, c := range Cells {
		want[c.Name] = true
	}
	if !maps.Equal(cells, want) {
		return LayerD{}, fmt.Errorf("layer D's option cells %v are not sb2census.Cells %v", slices.Sorted(maps.Keys(cells)), slices.Sorted(maps.Keys(want)))
	}
	return d, nil
}

// Path finds a layer D fixture name under root: layer D names a fixture by its path from the root module or from the decoder
// package, and no name is in both.
func Path(root, name string) (string, error) {
	var found []string
	for _, base := range []string{root, filepath.Join(root, "decoder")} {
		p := filepath.Join(base, filepath.FromSlash(name))
		if _, err := os.Stat(p); err == nil {
			found = append(found, p)
		}
	}
	if len(found) != 1 {
		return "", fmt.Errorf("fixture %s is at %d places (%v), not one", name, len(found), found)
	}
	return found[0], nil
}

// Clean makes a recorded reason independent of where the repository is checked out and of the registered family list.
func Clean(s string) string {
	if wd, err := os.Getwd(); err == nil {
		s = strings.ReplaceAll(s, filepath.Dir(wd), "<repo>")
	}
	s, _, _ = strings.Cut(s, " (have:")
	return strings.ReplaceAll(s, "\n", " / ")
}

// Golden is a layer's golden for runtime.GOARCH under dir: dir/<name>.<goarch>.txt.gz.
func Golden(dir, name string) string {
	return filepath.Join(dir, name+"."+runtime.GOARCH+".txt.gz")
}

// Write rewrites golden with lines.
func Write(golden string, lines []string) error {
	var buf bytes.Buffer
	zw, err := gzip.NewWriterLevel(&buf, gzip.BestCompression)
	if err != nil {
		return err
	}
	zw.Write([]byte(strings.Join(lines, "\n") + "\n"))
	if err := zw.Close(); err != nil {
		return err
	}
	return os.WriteFile(golden, buf.Bytes(), 0o644)
}

// Diff compares lines with golden, keyed by each line's first keyFields fields, and returns one entry per added, removed or
// changed cell, sorted. A missing golden is an error.
func Diff(golden string, lines []string, keyFields int) ([]string, error) {
	b, err := readGz(golden)
	if err != nil {
		return nil, err
	}
	text := strings.Join(lines, "\n") + "\n"
	if string(b) == text {
		return nil, nil
	}
	key := func(l string) string {
		f := strings.Fields(l)
		return strings.Join(f[:min(len(f), keyFields)], " ")
	}
	want := map[string]string{}
	for _, l := range strings.Split(strings.TrimSuffix(string(b), "\n"), "\n") {
		want[key(l)] = l
	}
	var diffs []string
	seen := map[string]bool{}
	for _, l := range lines {
		k := key(l)
		seen[k] = true
		switch w, ok := want[k]; {
		case !ok:
			diffs = append(diffs, "added: "+l)
		case w != l:
			diffs = append(diffs, "changed: "+k+"\n  was: "+w+"\n  now: "+l)
		}
	}
	for k, w := range want {
		if !seen[k] {
			diffs = append(diffs, "removed: "+w)
		}
	}
	slices.Sort(diffs)
	if len(diffs) == 0 {
		diffs = []string{"the same cells in a different order"}
	}
	return diffs, nil
}

func readGz(p string) ([]byte, error) {
	f, err := os.Open(p)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	return io.ReadAll(zr)
}
