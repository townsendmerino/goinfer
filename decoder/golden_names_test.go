package decoder

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// goldenLiteral matches a quoted golden file name in test source: "…golden….json" or "…golden….json.gz".
var goldenLiteral = regexp.MustCompile(`"([^"\s]*golden[^"\s]*\.json(?:\.gz)?)"`)

// Every real-checkpoint gate SKIPS when its golden is missing ("no golden — run the pin script"), so a
// test that names x_golden.json while the tree holds x_golden.json.gz (or the reverse) does not fail — it
// quietly stops gating. That is the one mistake the 2026-09-25 compression of the large goldens could
// make, and a fixture-less run cannot see it, so this reads the source instead: for every golden name
// quoted in a *_test.go anywhere in the repo, wherever the OTHER spelling exists on disk, this one must
// too. A golden that is absent in both spellings (gitignored, not pinned on this box) is not its concern.
func TestGoldenNames_matchTheFileOnDisk(t *testing.T) {
	root, err := filepath.Abs("..")
	if err != nil {
		t.Fatal(err)
	}
	var literals, gz int
	err = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", "testdata", "node_modules":
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, "_test.go") {
			return nil
		}
		src, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		dir := filepath.Dir(path)
		for _, m := range goldenLiteral.FindAllStringSubmatch(string(src), -1) {
			name := m[1]
			literals++
			other := name + ".gz"
			if strings.HasSuffix(name, ".gz") {
				gz++
				other = strings.TrimSuffix(name, ".gz")
			}
			// Callers pass a path relative to the package ("../testdata/x", "testdata/x") or a bare
			// name joined onto a testdata dir; check each place the name could resolve.
			for _, base := range []string{dir, filepath.Join(dir, "testdata"), filepath.Join(root, "testdata")} {
				if exists(filepath.Join(base, other)) && !exists(filepath.Join(base, name)) {
					rel, _ := filepath.Rel(root, path)
					t.Errorf("%s names %q, but only %q exists in %s — the gate would skip, not run",
						rel, name, other, base)
					break
				}
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	// The scan must have found the compressed goldens' readers, or it checked nothing.
	if gz < 20 {
		t.Fatalf("found %d golden names (%d .gz) in test source — expected the 24 compressed goldens' readers; the pattern no longer matches how tests name them", literals, gz)
	}
	t.Logf("checked %d golden names (%d .gz)", literals, gz)
}

func exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
