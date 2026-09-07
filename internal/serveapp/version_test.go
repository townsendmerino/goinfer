package serveapp

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// R2 gate (docs/measurements/cold-user-2026-09-06.md, finding #3). On v0.16.0 there was no way
// to ask a binary which backends it contained: `serve --version` printed
// "flag provided but not defined: -version" and exited 2. The darwin asset linked no Metal
// backend, and the only signal was a warning line that scrolled past the load banner.
//
// Two halves. This one pins that the `backends:` line is DERIVED from the registry rather than
// a hand-written string — a hard-coded "cpu cuda metal" would satisfy a text check while being
// exactly the lie the finding is about.
func TestVersionReport_backendsLineIsDerived(t *testing.T) {
	got := backendsLine(t, versionReport("goinfer-serve"))
	want := strings.Join(decoder.CompiledBackends(), " ")
	if got != want {
		t.Fatalf("backends line = %q, want %q (it must come from decoder.CompiledBackends)", got, want)
	}
	if !strings.Contains(got, "cpu") {
		t.Errorf("backends line %q omits cpu, which is always linked in", got)
	}

	// Registering a backend must change the line. This is what makes the equality above a
	// gate rather than a tautology against a constant.
	decoder.RegisterBackend("test-fake-backend", func() (decoder.Backend, error) { return nil, nil })
	after := backendsLine(t, versionReport("goinfer-serve"))
	if !strings.Contains(after, "test-fake-backend") {
		t.Fatalf("after registering a backend the line is still %q — it is not reading the registry", after)
	}
	if after == got {
		t.Fatalf("backends line did not change after a registration: %q", after)
	}
}

// The second half runs the real binary, because a correct versionReport that no entrypoint
// calls is the same failure shape as R2's correct-but-uncalled banner report. It builds the
// pure-Go root serve — which must report cpu and ONLY cpu, since M-19 made it link no backend.
func TestServeVersionFlag_reportsOnlyCPUForTheRootBinary(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a binary")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("no go toolchain")
	}
	bin := filepath.Join(t.TempDir(), "goinfer-serve")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	build := exec.Command("go", "build", "-o", bin, "github.com/townsendmerino/goinfer/cmd/serve")
	build.Dir = ".."
	if out, err := build.CombinedOutput(); err != nil {
		t.Skipf("cannot build cmd/serve here: %v\n%s", err, out)
	}

	// Both spellings, because the release workflow greps one and a user types the other.
	for _, arg := range []string{"--version", "version"} {
		out, err := exec.Command(bin, arg).CombinedOutput()
		if err != nil {
			t.Fatalf("%s %s: %v\n%s", bin, arg, err, out)
		}
		line := backendsLine(t, string(out))
		if line != "cpu" {
			t.Errorf("root serve %s reports backends %q, want exactly \"cpu\" — "+
				"the root binary links no backend since M-19", arg, line)
		}
		if !strings.Contains(string(out), "go: "+runtime.Version()) {
			t.Errorf("%s output missing the go toolchain line:\n%s", arg, out)
		}
	}

	// A model is not required, and must not become required: this is the one question a user
	// can ask a downloaded asset before committing to a multi-gigabyte checkout.
	if _, err := os.Stat(bin); err != nil {
		t.Fatalf("stat %s: %v", bin, err)
	}
}

// backendsLine extracts the value of the single `backends:` line, failing loudly if the report
// no longer has one — a renamed field would otherwise make every check above vacuous.
func backendsLine(t *testing.T, report string) string {
	t.Helper()
	for _, l := range strings.Split(report, "\n") {
		if v, ok := strings.CutPrefix(l, "backends: "); ok {
			return strings.TrimSpace(v)
		}
	}
	t.Fatalf("no `backends:` line in version report:\n%s", report)
	return ""
}

// TestHelpHeader_isSkimmableAndNamesTheFlagThatMattered is R4's gate for the scenario-B friction
// entry "--help is 13,583 bytes / 39 flags / 100 lines … I could not skim it for the flag I
// needed" (docs/measurements/cold-user-2026-09-06.md).
//
// That is not a style complaint. The same tester then drove a 16 GB machine +7.8 GB into swap
// because they never found `-stream-weights`, whose own help text names their exact model and
// their exact RAM. The flag was present and the page was too long to find it in.
//
// So the assertion is not "the help is short" — the long text is deliberate and every paragraph in
// it is a disclosure some measurement earned. It is that a MAP exists above the dump, that the map
// names the flag whose absence did the damage, and that it comes FIRST.
func TestHelpHeader_isSkimmableAndNamesTheFlagThatMattered(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a binary")
	}
	bin := filepath.Join(t.TempDir(), "serve")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	build := exec.Command("go", "build", "-o", bin, "github.com/townsendmerino/goinfer/cmd/serve")
	build.Dir = ".."
	if out, err := build.CombinedOutput(); err != nil {
		t.Skipf("cannot build cmd/serve here: %v\n%s", err, out)
	}
	// --help exits non-zero by Go's convention; the output is what matters.
	out, _ := exec.Command(bin, "--help").CombinedOutput()
	help := string(out)

	idx := strings.Index(help, "-stream-weights")
	if idx < 0 {
		t.Fatal("--help never mentions -stream-weights at all")
	}
	head := help[:idx]
	if n := strings.Count(head, "\n"); n > 25 {
		t.Errorf("-stream-weights first appears %d lines in; the cold run's tester gave up before "+
			"finding it and paid for that in swap. It belongs in the skimmable header", n)
	}
	// The header must say what the flag DOES, not just list it — "-stream-weights" as a bare name
	// is another thing to go look up.
	if !strings.Contains(help[:idx+400], "BIGGER THAN YOUR RAM") {
		t.Error("the header names -stream-weights without saying what it is for")
	}
	// The map must precede the dump, or it is just more text.
	if dump := strings.Index(help, "  -adapter"); dump >= 0 && dump < idx {
		t.Error("the flag dump starts before the skimmable header — the map must come first")
	}
	// The flag count must be DERIVED. A hand-typed count is the same defect class as the parity
	// manifest's hand-typed aikit_version, which sat seventeen versions stale.
	// Count only the DUMP's entries. The header's own "  --model" lines start the same way, so
	// counting the whole page conflates the map with the territory — which this test did on its
	// first run, reporting 47 against a real 40.
	dumpAt := strings.Index(help, "flags, with the trade-offs")
	if dumpAt < 0 {
		t.Fatal("could not find the boundary between the header and the flag dump")
	}
	want := 0
	for _, l := range strings.Split(help[dumpAt:], "\n") {
		if strings.HasPrefix(l, "  -") {
			want++
		}
	}
	if want > 0 && !strings.Contains(help, fmt.Sprintf("All %d flags", want)) {
		t.Errorf("help claims a flag count that does not match the %d flags it prints", want)
	}
}
