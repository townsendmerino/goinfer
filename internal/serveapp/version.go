package serveapp

import (
	"flag"
	"fmt"
	"runtime"
	"runtime/debug"
	"strings"

	"github.com/townsendmerino/goinfer/decoder"
)

// versionReport is what `serve --version` prints. Its load-bearing line is `backends:` — the
// list of backends COMPILED INTO this binary, which is not the same thing as the list
// --backend accepts.
//
// R2 (docs/measurements/cold-user-2026-09-06.md, finding #3): the v0.16.0 darwin release asset
// was built from the root cmd/serve, which links no backend at all. `--backend metal` on it
// therefore ran on CPU — 37.9 tok/s against the 82.3 the same box does with Metal — and the
// only signal was one warning line that scrolled past before the banner. Nothing on the binary
// could be asked. Now it can be, without loading a model, which is also what the release
// workflow greps to prove each asset carries the backend for its platform.
// injectedVersion is set via `-ldflags -X` on a release-built binary. R6 (docs/measurements/
// cold-user-2026-09-06-nobara-pc.md): the v0.17.0 linux-amd64 release asset's `--version`
// printed "v0.0.0-20260907045005-f36b095ac9a1+dirty", not "v0.17.0" — R2-follow-on's
// `go mod edit -replace` on the ephemeral submodule checkout is an uncommitted go.mod edit, and
// that alone is enough for the VCS stamp to read "modified". A binary built the ordinary way
// (`go install .../cmd/serve@v0.17.0`, no replace, no ephemeral checkout) is unaffected and
// keeps reporting its real tag from buildIdent with no injection at all — this only overrides
// the release workflow's own path.
var injectedVersion string

func versionReport(prog string) string {
	var b strings.Builder
	version, revision := buildIdent()
	fmt.Fprintf(&b, "%s %s", prog, version)
	// A pseudo-version already carries the commit ("v0.16.1-0.2026…-e57fef116b97"), so the
	// parenthetical would just repeat it back.
	if revision != "" && !strings.Contains(version, strings.TrimSuffix(revision, "-dirty")) {
		fmt.Fprintf(&b, " (%s)", revision)
	}
	b.WriteString("\n")
	// Space-separated, one line, lowercase: greppable from a shell without jq.
	fmt.Fprintf(&b, "backends: %s\n", strings.Join(decoder.CompiledBackends(), " "))
	fmt.Fprintf(&b, "go: %s %s/%s\n", runtime.Version(), runtime.GOOS, runtime.GOARCH)
	return b.String()
}

// buildIdent reads the module version and VCS revision the toolchain stamped in. A binary from
// `go install …@v0.16.0` reports that tag; one built from a working tree reports "(devel)" plus
// the commit, which is the honest answer rather than a version constant someone forgot to bump.
func buildIdent() (version, revision string) {
	version = "(unknown)"
	info, ok := debug.ReadBuildInfo()
	// Collected independently of Settings ORDER: vcs.modified is emitted after vcs.revision
	// today, but appending "-dirty" as we go would silently lose it if that ever flipped.
	dirty := false
	if ok {
		for _, s := range info.Settings {
			switch s.Key {
			case "vcs.revision":
				if len(s.Value) > 12 {
					revision = s.Value[:12]
				} else {
					revision = s.Value
				}
			case "vcs.modified":
				dirty = s.Value == "true"
			}
		}
	}
	if injectedVersion != "" {
		version = injectedVersion
	} else if ok && info.Main.Version != "" {
		version = info.Main.Version
	}
	if dirty && revision != "" {
		revision += "-dirty"
	}
	return version, revision
}

// isVersionArg matches the forms a user actually types. `version` (no dashes) is included
// because `serve check` and `serve pull` are subcommands, so a bare word is the shape this
// binary has taught people to expect.
func isVersionArg(a string) bool {
	switch a {
	case "--version", "-version", "version":
		return true
	}
	return false
}

// countFlags reports how many flags are registered, so the help header's "all N flags" line cannot
// drift from reality the way a hand-typed count would. That drift is the same defect class as the
// parity manifest's hand-typed aikit_version, which sat seventeen versions stale.
func countFlags() int {
	n := 0
	flag.VisitAll(func(*flag.Flag) { n++ })
	return n
}
