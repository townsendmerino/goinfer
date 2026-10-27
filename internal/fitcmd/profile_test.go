package fitcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// writeProfile writes a machine profile with avail bytes available to a temp file.
func writeProfile(t *testing.T, name string, avail int64) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name+".json")
	b, _ := json.Marshal(map[string]any{"name": name, "available_bytes": avail})
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// longCopy is a tracked tiny checkpoint with its window raised to 131,072, so an auto-pin is reachable.
func longCopy(t *testing.T, src string) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), filepath.Base(src))
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	ents, err := os.ReadDir(src)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range ents {
		b, err := os.ReadFile(filepath.Join(src, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if e.Name() == "config.json" {
			var cfg map[string]any
			if err := json.Unmarshal(b, &cfg); err != nil {
				t.Fatal(err)
			}
			cfg["max_position_embeddings"] = 131072
			b, _ = json.Marshal(cfg)
		}
		if err := os.WriteFile(filepath.Join(dir, e.Name()), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// TestProfile_answersAsFitQueryAvailable (MM6 gate G-MM6-1): fit -profile's exit code, verdict and answer are
// decoder.FitQueryAvailable's at the profile's reading, for each outcome: fits, capped, refused (with Load's own text),
// and unpriced.
func TestProfile_answersAsFitQueryAvailable(t *testing.T) {
	dir := longCopy(t, filepath.Join("..", "..", "testdata", "llama-tiny"))
	opts := decoder.Options{Quant: "int4"}
	at := func(avail int64) decoder.FitAnswer {
		a, err := decoder.FitQueryAvailable(dir, opts, avail)
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	// The lowest reading at which the load is not refused: there it is auto-pinned.
	lo, hi := int64(1), int64(1)<<40
	for lo < hi {
		if mid := lo + (hi-lo)/2; at(mid).Fits {
			hi = mid
		} else {
			lo = mid + 1
		}
	}
	for _, c := range []struct {
		name    string
		avail   int64
		verdict string
		code    int
	}{
		{"ample", 1 << 40, "fits", 0},
		{"edge", lo, "capped", 0},
		{"short", lo - 1, "refused", ExitRefused},
		{"unknown", 0, "unpriced", 0},
	} {
		var out bytes.Buffer
		code := runProfile(&out, dir, writeProfile(t, c.name, c.avail), opts, true)
		var v profileVerdict
		if err := json.Unmarshal(out.Bytes(), &v); err != nil {
			t.Fatalf("%s: output is not one JSON line: %v\n%s", c.name, err, out.String())
		}
		want := at(c.avail)
		if code != c.code || v.Verdict != c.verdict || v.Profile != c.name || v.AvailableBytes != c.avail {
			t.Errorf("%s: exit %d verdict %q profile %q available %d; want exit %d verdict %q", c.name, code, v.Verdict, v.Profile, v.AvailableBytes, c.code, c.verdict)
		}
		if g, w := mustJSON(t, v.Answer), mustJSON(t, want); g != w {
			t.Errorf("%s: answer %s, FitQueryAvailable %s", c.name, g, w)
		}
		if c.verdict == "refused" {
			var text bytes.Buffer
			runProfile(&text, dir, writeProfile(t, c.name, c.avail), opts, false)
			if want.Decline == "" || !strings.Contains(text.String(), want.Decline) {
				t.Errorf("%s: the text output does not carry Load's refusal:\n%s", c.name, text.String())
			}
		}
	}
	if code := runProfile(&bytes.Buffer{}, "hf:owner/repo", writeProfile(t, "x", 1<<40), opts, true); code != 1 {
		t.Errorf("a reference was priced (exit %d): -profile must not fetch", code)
	}
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}
