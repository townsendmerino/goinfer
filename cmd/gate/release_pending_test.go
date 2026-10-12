package main

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// A pending gate may ride through one release unconfirmed, never two (audit-2026-09-10 G-03).
//
// awaitingFirstConfirmation exists so a newly required gate that has not run yet does not block
// everything on the day it is added. But a failure in a pending gate reads as an ITEM, not a
// blocker, so an entry that sits there indefinitely is a required gate that can never stop a
// release. The bound is the release, not a calendar age (a day count fails CI at the maintainer's
// pace, which says nothing about what shipped): at tag time, an entry dated before the PREVIOUS
// release has already ridden through one release unconfirmed, and it blocks this one. Between
// releases nothing is enforced.

// TestParity_noPendingGateOutlivesARelease applies the rule to the real map. It enforces only when
// GOINFER_RELEASE_TAG names the version being released: RELEASING.md's pre-flight sets it before
// tagging, and release-assets.yml sets it as the backstop. Otherwise it skips, and the skip names
// what would block the next release. It is a skip, not a pass, so a gate cell cannot count it as
// coverage (G-13(j)).
func TestParity_noPendingGateOutlivesARelease(t *testing.T) {
	root, err := repoRoot()
	if err != nil {
		t.Fatalf("cannot locate the repo root: %v", err)
	}
	cl, err := os.ReadFile(filepath.Join(root, "CHANGELOG.md"))
	if err != nil {
		t.Fatalf("cannot read CHANGELOG.md, so the previous release cannot be dated: %v", err)
	}
	releasing := os.Getenv("GOINFER_RELEASE_TAG")
	probe := releasing
	if probe == "" {
		probe = "(the next release)"
	}
	prevVer, prevDate, blockers, err := pendingOutlivingRelease(awaitingFirstConfirmation, string(cl), probe)
	if err != nil {
		t.Fatal(err)
	}
	remedy := "Run each and promote it (go run ./cmd/gate ledger promote --gate <gate> --value PASS --by <you>), " +
		"or move it to neverConfirmed with the reason it will never be confirmed."
	if releasing == "" {
		if len(blockers) == 0 {
			t.Skipf("not a release (GOINFER_RELEASE_TAG unset); nothing has been pending since before %s (%s)", prevVer, prevDate)
		}
		t.Skipf("not a release (GOINFER_RELEASE_TAG unset). The NEXT release is blocked by %d gate(s) pending "+
			"since before %s (%s). %s\n  %s", len(blockers), prevVer, prevDate, remedy, strings.Join(blockers, "\n  "))
	}
	if len(blockers) > 0 {
		t.Fatalf("releasing %s: %d required gate(s) have been pending since before the previous release %s (%s). "+
			"Each rode through that release unconfirmed, where a failure in it could not block the tag. %s\n  %s",
			releasing, len(blockers), prevVer, prevDate, remedy, strings.Join(blockers, "\n  "))
	}
	t.Logf("releasing %s: nothing has been pending since before %s (%s)", releasing, prevVer, prevDate)
}

// TestParity_pendingOutlivingRelease holds the rule itself; it runs everywhere.
func TestParity_pendingOutlivingRelease(t *testing.T) {
	cl := "# Changelog\n\n## [Unreleased]\n\n## [v0.18.0] — 2026-09-20\n\n### Fixed\n\n## [v0.17.2] — 2026-09-08\n\n## [v0.17.1] — 2026-09-07\n"
	pending := map[string]string{
		"TestOld":     "2026-09-02 — added before v0.17.2",
		"TestSameDay": "2026-09-08 — added on v0.17.2's day",
		"TestNew":     "2026-09-15 — added between v0.17.2 and v0.18.0",
	}
	for _, tc := range []struct {
		name, releasing, wantPrev string
		want                      []string
	}{
		// Tagging v0.18.0 with its header written: v0.17.2 is the previous release. Only the entry that
		// predates it has ridden through a release; same-day and later entries ride this one.
		{"header written", "v0.18.0", "v0.17.2", []string{"TestOld (pending since 2026-09-02)"}},
		// The header for the version being tagged not written yet: the newest dated release counts.
		{"header not yet written", "v0.18.1", "v0.18.0", []string{
			"TestNew (pending since 2026-09-15)", "TestOld (pending since 2026-09-02)", "TestSameDay (pending since 2026-09-08)"}},
	} {
		prev, _, got, err := pendingOutlivingRelease(pending, cl, tc.releasing)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if prev != tc.wantPrev || !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: previous %s, blockers %q; want %s, %q", tc.name, prev, got, tc.wantPrev, tc.want)
		}
	}
	if _, _, _, err := pendingOutlivingRelease(pending, "## [Unreleased]\n", "v0.18.0"); err == nil {
		t.Error("a CHANGELOG with no dated release must be an error, not an empty blocker list")
	}
	if _, _, _, err := pendingOutlivingRelease(map[string]string{"TestUndated": "added some time"}, cl, "v0.18.0"); err == nil {
		t.Error("an undated pending entry must be an error: it cannot be placed against a release")
	}
}
