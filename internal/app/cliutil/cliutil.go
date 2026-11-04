// Package cliutil holds the command-line pieces goinfer-serve (internal/serveapp) and goinfer-chat
// (internal/chatapp) had each carried an identical copy of: a lenient on/off flag and the build
// identity that --version prints. Only what is binary-agnostic lives here — each binary's version
// dispatch (isVersionArg) and its -ldflags -X targets stay in its own package.
package cliutil

import (
	"errors"
	"runtime/debug"
	"strconv"
	"strings"
)

// OnOff is a lenient bool flag.Value: plain flag.Bool only accepts strconv.ParseBool's spellings
// (1/t/T/TRUE/true/True/0/f/F/FALSE/false/False), so --fit=off — the spelling --fit's own help and
// tasks/task-fit-to-hardware.md print — was rejected with exit 2 (M-14, audit-2026-09-10). Accepts
// on/off in addition to the usual set, case-insensitively.
type OnOff bool

func (f *OnOff) String() string {
	if f == nil {
		return "true"
	}
	return strconv.FormatBool(bool(*f))
}

// Set errors on anything unrecognised rather than picking a default. The flag package prefixes
// the message with the value and flag name ("invalid value "x" for flag -fit: ...").
func (f *OnOff) Set(v string) error {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "on", "1", "t", "true":
		*f = true
	case "off", "0", "f", "false":
		*f = false
	default:
		return errors.New("want on|off|true|false")
	}
	return nil
}

// IsBoolFlag lets a bare `--fit` (no `=value`) mean `--fit=true`, matching flag.Bool's own UX.
func (f *OnOff) IsBoolFlag() bool { return true }

// BuildIdent reads the module version and VCS revision the toolchain stamped in. A binary from
// `go install …@v0.16.0` reports that tag; one built from a working tree reports "(devel)" plus
// the commit, the honest answer rather than a version constant someone forgot to bump.
//
// injected, when non-empty, wins over the stamped version. It is the caller's own package-level
// var that a release build sets with `-ldflags -X` (R6, docs/measurements/
// cold-user-2026-09-06-nobara-pc.md: the release workflow builds from an ephemeral checkout with
// a `go mod edit -replace` applied, and that uncommitted edit makes the stamp read "+dirty" even
// on the exact tagged tree). The var deliberately stays in each binary's package: -X names a
// symbol by its full package path, and an -X naming a symbol that no longer exists is silently
// ignored, so moving it would quietly bring the "+dirty" release version back.
// TestLdflagsTargetsExist holds those paths to the source.
func BuildIdent(injected string) (version, revision string) {
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
	if injected != "" {
		version = injected
	} else if ok && info.Main.Version != "" {
		version = info.Main.Version
	}
	if dirty && revision != "" {
		revision += "-dirty"
	}
	return version, revision
}
