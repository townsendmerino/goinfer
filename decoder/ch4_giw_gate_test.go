//go:build goinfer_testhooks

package decoder

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestCH4_giwByteIdentity is code-health CH4's gate (docs/tasks/task-code-health.md): the .giw a real GGUF transcodes to must be
// byte-identical before and after a gguf.go refactor. GOINFER_CH4_GGUF names the fixtures (comma-separated paths to real K-quant
// .gguf files); each one's .giw weights body, what gguf.go produces (StreamTranscodeGGUF, no target), is hashed as it streams at q4k
// (the source's Q4_K kept), int4 and int8int8, twice each (the output must be deterministic), and its SHA-256 is written to
// GOINFER_CH4_RECORD's file (one "basename quant sha" line each) or checked against GOINFER_CH4_WANT's. Nothing is written to disk.
// Its red, on every run: each fixture is transcoded once more at int4 with one dequantized value of its first layer row set past its
// group's scale (ggufRowFilter, the dequantizer's own seam), and that hash must differ, or the gate could not see a decode change. The seam does not
// reach the kept-Q4_K copy (q4kMat), so q4k is held by its recorded hash alone. Without GOINFER_CH4_GGUF it skips.
func TestCH4_giwByteIdentity(t *testing.T) {
	paths := strings.FieldsFunc(os.Getenv("GOINFER_CH4_GGUF"), func(r rune) bool { return r == ',' })
	if len(paths) == 0 {
		t.Skip("GOINFER_CH4_GGUF names no fixture")
	}
	record, wantFile := os.Getenv("GOINFER_CH4_RECORD"), os.Getenv("GOINFER_CH4_WANT")
	if (record == "") == (wantFile == "") {
		t.Fatal("set exactly one of GOINFER_CH4_RECORD (write the hashes) and GOINFER_CH4_WANT (check them)")
	}
	want := map[string]string{}
	if wantFile != "" {
		f, err := os.Open(wantFile)
		if err != nil {
			t.Fatal(err)
		}
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			if fs := strings.Fields(sc.Text()); len(fs) == 3 {
				want[fs[0]+" "+fs[1]] = fs[2]
			}
		}
		f.Close()
	}
	sum := func(path, quant string) string {
		t.Helper()
		h := sha256.New()
		if _, err := StreamTranscodeGGUF(context.Background(), path, h, quant, false, GIWTargetNone, "ch4"); err != nil {
			t.Fatalf("%s at %s: transcode: %v", filepath.Base(path), quant, err)
		}
		return hex.EncodeToString(h.Sum(nil))
	}
	var lines []string
	for _, p := range paths {
		for _, q := range []string{"q4k", "int4", "int8int8"} {
			key := filepath.Base(p) + " " + q
			a, b := sum(p, q), sum(p, q)
			if a != b {
				t.Fatalf("%s: two transcodes differ (%s, %s); the gate needs a deterministic .giw", key, a, b)
			}
			t.Logf("%s %s", key, a)
			lines = append(lines, fmt.Sprintf("%s %s", key, a))
			if wantFile == "" {
				continue
			}
			switch w, ok := want[key]; {
			case !ok:
				t.Errorf("%s: no recorded hash in %s", key, wantFile)
			case w != a:
				t.Errorf("%s: .giw sha256 %s, recorded %s: the transcode's bytes changed", key, a, w)
			}
		}
	}
	for _, p := range paths {
		clean := sum(p, "int4")
		once := false
		SetGGUFRowFilterForTest(func(name string, row []float32) {
			if !once && len(row) > 0 {
				// twice the row's largest magnitude, plus one: past any int4 group's scale, so its quantized bytes must change
				var peak float32
				for _, v := range row {
					peak = max(peak, v, -v)
				}
				row[0] = 2*peak + 1
				once = true
			}
		})
		planted := sum(p, "int4")
		SetGGUFRowFilterForTest(nil)
		if !once {
			t.Fatalf("%s: the row filter saw no layer row; the plant did not run", filepath.Base(p))
		}
		if planted == clean {
			t.Fatalf("%s: a planted dequantized value left the int4 .giw hash unchanged; the gate cannot see a decode change", filepath.Base(p))
		}
		t.Logf("%s: plant (one dequantized value past its group scale) changes the int4 hash (red, as it must be)", filepath.Base(p))
	}
	if record != "" {
		if err := os.WriteFile(record, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("recorded %d hashes to %s", len(lines), record)
	}
}
