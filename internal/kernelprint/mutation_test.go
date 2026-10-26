package kernelprint

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

// plant is a one-ULP change to one kernel's output, as a text edit of an aikit source file.
type plant struct {
	name, kernel, file, anchor, insert string
}

func ulpPlants() []plant {
	var out []plant
	for _, dir := range []struct{ name, inf string }{{"+1 ULP", "1"}, {"-1 ULP", "-1"}} {
		out = append(out,
			plant{"MatmulAVAcc64 " + dir.name, "linalg.MatmulAVAcc64", "linalg/matmul_av_acc64.go",
				"\tcheckMatmulAVAcc64(len(scores), len(vals), len(dst), len(acc), M, nKeys, hd, headOff, rowStride)\n",
				"\tdefer func() { dst[0] = math.Nextafter32(dst[0], float32(math.Inf(" + dir.inf + "))) }()\n"},
			plant{"Q5_0 dequant " + dir.name, "embed.GGUFFile(Q5_0).Tensor", "embed/gguf_dequant.go",
				"\tqs := raw[base+6 : base+22]\n\tout = out[:32:32]\n\tqs = qs[:16:16]\n",
				"\tdefer func() { out[0] = math.Nextafter32(out[0], float32(math.Inf(" + dir.inf + "))) }()\n"},
		)
	}
	return out
}

// TestCompute_mutation is K2. On a copy of the aikit this package builds against, each plant must change exactly its
// kernel's fingerprint, and the copy with no plant must reproduce this process's fingerprints for every kernel. It
// builds aikit five times, so it runs only with GOINFER_KERNELPRINT_MUTATION=1.
func TestCompute_mutation(t *testing.T) {
	if os.Getenv("GOINFER_KERNELPRINT_MUTATION") != "1" {
		t.Skip("K2 builds aikit five times; set GOINFER_KERNELPRINT_MUTATION=1")
	}
	tmp := t.TempDir()
	copyDir := filepath.Join(tmp, "aikit")
	copyTree(t, aikitDir(t), copyDir)
	modfile := filepath.Join(tmp, "go.mod")
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	gomod, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	gosum, err := os.ReadFile(filepath.Join(root, "go.sum"))
	if err != nil {
		t.Fatal(err)
	}
	writeFile(t, modfile, append(gomod, []byte("\nreplace "+aikitModule+" => "+copyDir+"\n")...))
	writeFile(t, filepath.Join(tmp, "go.sum"), gosum)

	child := func(label string) Report {
		t0 := time.Now()
		out := filepath.Join(tmp, "report.json")
		cmd := exec.Command("go", "test", "-modfile="+modfile, "-count=1", "-run", "^TestCompute_child$", ".")
		cmd.Env = append(os.Environ(), "GOWORK=off", "GOINFER_KERNELPRINT_OUT="+out)
		if b, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("%s: %v\n%s", label, err, b)
		}
		var r Report
		b, err := os.ReadFile(out)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(b, &r); err != nil {
			t.Fatal(err)
		}
		fmt.Fprintf(os.Stderr, "K2 %s: built and computed in %s\n", label, time.Since(t0).Round(time.Second))
		return r
	}

	base := Compute()
	if d := changed(base, child("unplanted copy")); len(d) != 0 {
		t.Fatalf("the unplanted copy changed %v: the copy is not the aikit this package builds against", d)
	}
	for _, p := range ulpPlants() {
		path := filepath.Join(copyDir, p.file)
		orig, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		src := string(orig)
		if strings.Count(src, p.anchor) != 1 {
			t.Fatalf("VOID %s: the anchor occurs %d times in %s; fix the plant", p.name, strings.Count(src, p.anchor), p.file)
		}
		writeFile(t, path, []byte(withMath(strings.Replace(src, p.anchor, p.anchor+p.insert, 1))))
		d := changed(base, child(p.name))
		writeFile(t, path, orig)
		if len(d) != 1 || d[0] != p.kernel {
			t.Errorf("%s: changed %v, want exactly [%s]", p.name, d, p.kernel)
		} else {
			t.Logf("%s: changed exactly %s", p.name, p.kernel)
		}
	}
	if d := changed(base, child("plants removed")); len(d) != 0 {
		t.Errorf("removing the plants left %v changed", d)
	}
}

// changed lists the kernels whose fingerprints differ between a and b, or that only one has.
func changed(a, b Report) []string {
	var out []string
	for k, v := range a.Kernels {
		if b.Kernels[k] != v {
			out = append(out, k)
		}
	}
	for k := range b.Kernels {
		if _, ok := a.Kernels[k]; !ok {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

// withMath makes sure a Go source file imports "math".
func withMath(src string) string {
	if strings.Contains(src, "\"math\"") {
		return src
	}
	if i := strings.Index(src, "import (\n"); i >= 0 {
		return src[:i+len("import (\n")] + "\t\"math\"\n" + src[i+len("import (\n"):]
	}
	i := strings.Index(src, "\npackage ")
	if strings.HasPrefix(src, "package ") {
		i = -1
	}
	j := strings.Index(src[i+1:], "\n") + i + 1
	return src[:j+1] + "\nimport \"math\"\n" + src[j+1:]
}

// copyTree copies a module directory, skipping testdata, with writable modes (the module cache is read-only).
func copyTree(t *testing.T, from, to string) {
	err := filepath.WalkDir(from, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(from, path)
		if d.IsDir() {
			if d.Name() == "testdata" || d.Name() == ".git" {
				return filepath.SkipDir
			}
			return os.MkdirAll(filepath.Join(to, rel), 0o755)
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(filepath.Join(to, rel), b, 0o644)
	})
	if err != nil {
		t.Fatal(err)
	}
}

func writeFile(t *testing.T, path string, b []byte) {
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}
