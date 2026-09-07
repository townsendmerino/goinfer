package chatapp

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"strings"

	"github.com/townsendmerino/goinfer/decoder"
)

// injectedVersion is set via `-ldflags -X` on a release-built binary, where a plain
// runtime/debug.ReadBuildInfo() reads a "+dirty" pseudo-version (R6, docs/measurements/
// cold-user-2026-09-06-nobara-pc.md): the release workflow's GPU assets are built from an
// ephemeral submodule checkout with a local `go mod edit -replace` applied (R2-follow-on), and
// that uncommitted go.mod edit is enough for the VCS stamp to read "modified" even though the
// tree is exactly the tagged release. A binary built with `go install .../cmd/chat@v0.17.0`
// (no replace, no ephemeral checkout) is unaffected and continues to report its real tag via
// buildIdent alone — this only overrides what that path would otherwise get wrong.
var injectedVersion string

// embeddedTier and embeddedQuant are set via `-ldflags -X` by build-embed.sh (empty in a plain
// `-tags embed`/`-tags prequant` build run by hand without them, and always empty on the
// no-embedded-model build — see noembed.go). They exist because `-quant`'s help text states one
// default ("int4") that does not apply to the embed release binaries at all: a `-tags prequant`
// build (build-embed.sh's default mode, and what the release workflow uses for
// goinfer-chat-0.5b/1.5b) bakes its weights at a FIXED quant chosen at build time
// (cmd/prequant's own default, int8int8) and never reads the --quant flag — confirmed by
// internal/chatapp/prequant.go's loadEmbedded, which passes the deserialized bundle straight to
// decoder.NewModel with no reference to opts.Quant. Rather than rewrite the shared --quant help
// string per build tag, --version says what actually shipped.
var embeddedTier, embeddedQuant string

// versionReport is what `goinfer-chat --version` prints. R6 (docs/measurements/
// cold-user-2026-09-06-nobara-pc.md): on v0.17.0 this binary had NO way to report its version at
// all — `--version`/`-v` both hit "flag provided but not defined", and a bare `version`
// positional was silently swallowed and started an interactive chat session with the embedded
// model instead of erroring or printing anything.
func versionReport(prog string) string {
	var b strings.Builder
	version, revision := buildIdent()
	fmt.Fprintf(&b, "%s %s", prog, version)
	if revision != "" && !strings.Contains(version, strings.TrimSuffix(revision, "-dirty")) {
		fmt.Fprintf(&b, " (%s)", revision)
	}
	b.WriteString("\n")
	fmt.Fprintf(&b, "backends: %s\n", strings.Join(decoder.CompiledBackends(), " "))
	fmt.Fprintf(&b, "go: %s %s/%s\n", runtime.Version(), runtime.GOOS, runtime.GOARCH)
	if hasEmbeddedModel {
		tier, quant := embeddedTier, embeddedQuant
		if tier == "" {
			tier = "(unset — not built by build-embed.sh, or built without its version ldflags)"
		}
		if quantIsFixedAtBuildTime {
			if quant == "" {
				quant = "(unset — not built by build-embed.sh, or built without its version ldflags)"
			}
			fmt.Fprintf(&b, "embedded: tier=%s quant=%s (baked at build time; --quant has no effect on this binary)\n", tier, quant)
		} else {
			// The --gguf embed mode (not what release assets use) quantizes at LAUNCH per
			// --quant like an ordinary --model load, so it has no fixed quant to report here.
			fmt.Fprintf(&b, "embedded: tier=%s quant=runtime-selectable (see --quant; not fixed at build time)\n", tier)
		}
	}
	return b.String()
}

// buildIdent prefers the version build-embed.sh / the release workflow injected; falling back to
// the toolchain's own VCS stamp is what makes `go install …@v0.17.0` (never touched by this
// binary's own release automation) report correctly with no injection at all.
func buildIdent() (version, revision string) {
	version = "(unknown)"
	info, ok := debug.ReadBuildInfo()
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

// isVersionArg matches the forms a user actually types — see internal/serveapp/version.go's
// twin; kept in sync deliberately rather than shared, so a change to one binary's dispatch does
// not silently reach the other.
func isVersionArg(a string) bool {
	switch a {
	case "--version", "-version", "version":
		return true
	}
	return false
}
