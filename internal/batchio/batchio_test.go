package batchio

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseInput(t *testing.T) {
	good := `{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"messages":[]}}

{"custom_id":"b","body":{}}
`
	lines, err := ParseInput([]byte(good))
	if err != nil || len(lines) != 2 || lines[0].CustomID != "a" || lines[1].CustomID != "b" {
		t.Fatalf("got %+v, %v", lines, err)
	}
	if lines[0].URL != "/v1/chat/completions" || string(lines[0].Body) != `{"messages":[]}` {
		t.Errorf("fields not carried: %+v", lines[0])
	}

	// The messages are the ones serve answers a bad upload with, and they name the physical line (blank lines count).
	for name, tc := range map[string]struct{ in, want string }{
		"bad json":    {"{\"custom_id\":\"a\"}\n{nope\n", "input file line 2: invalid JSON"},
		"no id":       {"\n{\"body\":{}}\n", "input file line 2: custom_id is required"},
		"empty":       {"\n  \n", "input file has no request lines"},
		"line counts": {"{\"custom_id\":\"a\"}\n\n\n{\"custom_id\":\"\"}\n", "input file line 4: custom_id is required"},
	} {
		if _, err := ParseInput([]byte(tc.in)); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: got %v, want it to contain %q", name, err, tc.want)
		}
	}
}

func TestCheckUnique(t *testing.T) {
	ls := func(ids ...string) []Line {
		var out []Line
		for _, id := range ids {
			out = append(out, Line{CustomID: id})
		}
		return out
	}
	if err := CheckUnique(ls("a", "b", "c")); err != nil {
		t.Errorf("distinct ids refused: %v", err)
	}
	err := CheckUnique(ls("a", "b", "a"))
	if err == nil || !strings.Contains(err.Error(), `"a" (entries 1 and 3)`) {
		t.Errorf("duplicate not named with both positions: %v", err)
	}
	var many []string
	for range 9 {
		many = append(many, "x", "x") // nine duplicates
	}
	if err := CheckUnique(ls(many...)); err == nil || !strings.Contains(err.Error(), "more") {
		t.Errorf("a long duplicate list must be summarised: %v", err)
	}
}

// The byte layout is the wire contract with serve's output file: keys sorted (json.Marshal of a map), one line, newline-ended.
func TestLineLayout(t *testing.T) {
	ok := OKLine("batch_req_1", "a", 200, map[string]any{"x": 1})
	want := `{"custom_id":"a","error":null,"id":"batch_req_1","response":{"body":{"x":1},"status_code":200}}` + "\n"
	if string(ok) != want {
		t.Errorf("OKLine:\n got %s want %s", ok, want)
	}
	bad := ErrLine("batch_req_2", "b", "cancelled", "boom")
	want = `{"custom_id":"b","error":{"code":"cancelled","message":"boom"},"id":"batch_req_2","response":null}` + "\n"
	if string(bad) != want {
		t.Errorf("ErrLine:\n got %s want %s", bad, want)
	}
}

func write(t *testing.T, name string, parts ...[]byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, bytes.Join(parts, nil), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestScanOutput(t *testing.T) {
	a, b := OKLine("i1", "a", 200, map[string]any{}), OKLine("i2", "b", 200, map[string]any{})
	e := ErrLine("i3", "c", "api_error", "x")

	// Missing file: an empty run, not an error.
	p, err := ScanOutput(filepath.Join(t.TempDir(), "nope.jsonl"))
	if err != nil || len(p.Done) != 0 || p.GoodBytes != 0 || p.Torn {
		t.Fatalf("missing file: %+v, %v", p, err)
	}

	// Complete file. An error line in it is NOT done (only a response counts), so a retry picks it up.
	path := write(t, "out.jsonl", a, b, e)
	p, err = ScanOutput(path)
	if err != nil || !p.Done["a"] || !p.Done["b"] || p.Done["c"] || p.Lines != 3 || p.Torn || p.GoodBytes != int64(len(a)+len(b)+len(e)) {
		t.Fatalf("complete file: %+v, %v", p, err)
	}

	// Torn tail: half a line after the last newline. The complete lines count, the partial one does not, and GoodBytes is
	// where to truncate.
	path = write(t, "torn.jsonl", a, b, b[:len(b)/2])
	p, err = ScanOutput(path)
	if err != nil || !p.Torn || len(p.Done) != 2 || p.GoodBytes != int64(len(a)+len(b)) {
		t.Fatalf("torn tail: %+v, %v", p, err)
	}
	// A VALID line that just lacks its newline is torn too: appending would glue the next line onto it.
	path = write(t, "nonl.jsonl", a, bytes.TrimSuffix(b, []byte("\n")))
	if p, err = ScanOutput(path); err != nil || !p.Torn || p.Done["b"] || p.GoodBytes != int64(len(a)) {
		t.Fatalf("unterminated valid line: %+v, %v", p, err)
	}

	// A complete-but-corrupt line in the middle is refused, naming the line.
	path = write(t, "bad.jsonl", a, []byte("not json\n"), b)
	if _, err = ScanOutput(path); err == nil || !strings.Contains(err.Error(), "line 2") {
		t.Fatalf("mid-file corruption must be refused, naming the line: %v", err)
	}
	path = write(t, "noid.jsonl", []byte("{\"response\":{}}\n"))
	if _, err = ScanOutput(path); err == nil || !strings.Contains(err.Error(), "no custom_id") {
		t.Fatalf("a line without custom_id must be refused: %v", err)
	}
}

// Resume end to end at the file level: write, tear, rescan, truncate, append — the file ends byte-identical to an
// uninterrupted one.
func TestWriterResume(t *testing.T) {
	lines := [][]byte{
		OKLine("i1", "a", 200, map[string]any{"n": 1}),
		OKLine("i2", "b", 200, map[string]any{"n": 2}),
		OKLine("i3", "c", 200, map[string]any{"n": 3}),
	}
	path := filepath.Join(t.TempDir(), "out.jsonl")

	w, err := OpenAppend(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range lines[:2] {
		if err := w.Append(l); err != nil {
			t.Fatal(err)
		}
	}
	w.Close()
	// The process dies partway through a line that was LONGER than the one that will replace it (a rerun need not reproduce the
	// same text), so a torn tail that is not truncated would outlast the overwrite and corrupt the file.
	long := OKLine("i3", "c", 200, map[string]any{"n": 3, "text": strings.Repeat("long reply ", 40)})
	f, _ := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o644)
	f.Write(long[:len(long)/2])
	f.Close()

	p, err := ScanOutput(path)
	if err != nil || !p.Torn || len(p.Done) != 2 {
		t.Fatalf("scan after the crash: %+v, %v", p, err)
	}
	w, err = OpenAppend(path, p.GoodBytes)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.Append(lines[2]); err != nil {
		t.Fatal(err)
	}
	w.Close()

	got, _ := os.ReadFile(path)
	if want := bytes.Join(lines, nil); !bytes.Equal(got, want) {
		t.Errorf("resumed file differs from an uninterrupted one:\n got %q\nwant %q", got, want)
	}
	if !json.Valid(bytes.Split(got, []byte("\n"))[0]) {
		t.Error("first line is not JSON")
	}
}

func TestErrorPathAndOrphans(t *testing.T) {
	for in, want := range map[string]string{
		"out.jsonl": "out.errors.jsonl", "d/run.jsonl": "d/run.errors.jsonl", "out": "out.errors", "out.json": "out.json.errors",
	} {
		if got := ErrorPath(in); got != want {
			t.Errorf("ErrorPath(%q) = %q, want %q", in, got, want)
		}
	}
	got := Orphans(map[string]bool{"a": true, "z": true, "m": true}, []Line{{CustomID: "a"}})
	if strings.Join(got, ",") != "m,z" {
		t.Errorf("Orphans = %v", got)
	}
}
