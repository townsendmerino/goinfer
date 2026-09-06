package decoder

import (
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// R2 (docs/measurements/cold-user-2026-09-06.md, finding #3): on a Mac the runtime printed
// "decoder: metal backend not built in ... using cpu" and then, on the very next line,
// "loaded 28-layer model ... [backend=metal quant=int4]". The warning scrolls past; the status
// line is what gets pasted into an issue, and it named a backend that was not executing. The
// cost on that box was 37.9 vs 82.3 tok/s.
//
// These two tests are the gate. The first pins the semantics of the report; the second pins that
// every banner in the tree actually uses it, which is the half that would have gone red on
// v0.16.0 (all three banners formatted the REQUEST).
func TestBackendReport_namesTheEffectiveBackend(t *testing.T) {
	decline := errors.New("decoder: metal backend not built in; rebuild with -tags metal")

	for _, tc := range []struct {
		name      string
		requested string
		beErr     error
		wantEff   string
		wantReq   string
		// wantIn are substrings the banner MUST contain; wantNotPrefix rejects the v0.16.0
		// shape, where the whole field was just the requested name.
		wantIn        []string
		wantNotEqual  string
		wantNotSubstr string
	}{
		{
			name: "honoured request reads as the bare name",
			// No decline: the banner is just the backend, no transition noise.
			requested: "cuda", beErr: nil, wantReq: "cuda", wantEff: "cuda",
			wantIn: []string{"cuda"}, wantNotSubstr: "requested",
		},
		{
			name:      "declined request never names the backend alone",
			requested: "metal", beErr: decline, wantReq: "metal", wantEff: "cpu",
			wantIn:       []string{"requested metal", "running on cpu", "not built in"},
			wantNotEqual: "metal",
		},
		{
			name: "empty request normalises to cpu",
			// A caller that passed nothing must not report an empty backend.
			requested: "", beErr: nil, wantReq: "cpu", wantEff: "cpu",
			wantIn: []string{"cpu"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := (&Model{}).withBackendNames(tc.requested, tc.beErr)
			if got := m.RequestedBackend(); got != tc.wantReq {
				t.Errorf("RequestedBackend() = %q, want %q", got, tc.wantReq)
			}
			if got := m.EffectiveBackend(); got != tc.wantEff {
				t.Errorf("EffectiveBackend() = %q, want %q", got, tc.wantEff)
			}
			got := m.BackendReport()
			for _, want := range tc.wantIn {
				if !strings.Contains(got, want) {
					t.Errorf("BackendReport() = %q, missing %q", got, want)
				}
			}
			if tc.wantNotEqual != "" && got == tc.wantNotEqual {
				t.Errorf("BackendReport() = %q — that is the REQUEST, not what is executing", got)
			}
			if tc.wantNotSubstr != "" && strings.Contains(got, tc.wantNotSubstr) {
				t.Errorf("BackendReport() = %q, should not contain %q when nothing was declined", got, tc.wantNotSubstr)
			}
		})
	}
}

// TestBackendBanner_usesTheReport is the half that goes red on v0.16.0. A correct BackendReport
// that no banner calls is exactly the state the cold run found, so the semantics test above
// cannot stand alone: this walks every .go file in the tree, finds each site that formats
// "[backend=%s", and requires its argument list to carry BackendReport(). A new command that
// prints its own load banner from the request is caught the day it is written.
func TestBackendBanner_usesTheReport(t *testing.T) {
	// The banner is a bracketed field so this cannot match cmd/gate's prose "(backend=%s)".
	site := regexp.MustCompile(`\[backend=%s`)

	checked := 0
	err := filepath.WalkDir("..", func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", "testdata", "node_modules":
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return nil
		}
		b, rerr := os.ReadFile(p)
		if rerr != nil {
			return nil
		}
		lines := strings.Split(string(b), "\n")
		for i, line := range lines {
			if !site.MatchString(line) {
				continue
			}
			checked++
			// The arguments usually wrap onto the following lines; read forward to the line
			// that closes the call, capped so a malformed file cannot swallow the whole source.
			args := line
			for j := i + 1; j < i+6 && j < len(lines); j++ {
				args += "\n" + lines[j]
				if strings.HasSuffix(strings.TrimSpace(lines[j]), ")") {
					break
				}
			}
			if !strings.Contains(args, "BackendReport()") {
				t.Errorf("%s:%d formats a load banner without BackendReport():\n%s\n"+
					"a banner must name what is EXECUTING, not what was requested "+
					"(docs/measurements/cold-user-2026-09-06.md finding #3)", p, i+1, args)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk: %v", err)
	}
	// A zero-site pass would be a green that vouches for nothing — the banners were renamed and
	// this test silently stopped looking.
	if checked == 0 {
		t.Fatal("no load banners found: the regex no longer matches anything, so this gate is inert")
	}
	t.Logf("checked %d load banner site(s)", checked)
}
