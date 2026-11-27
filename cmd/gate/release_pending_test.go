package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/guard/gaterelease"
)

var isoDatePrefix = gaterelease.ISODatePrefix

// TestParity_noPendingGateOutlivesARelease applies gaterelease.PendingOutlivingRelease to the real map. It enforces only when
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
	prevVer, prevDate, blockers, err := gaterelease.PendingOutlivingRelease(awaitingFirstConfirmation, string(cl), probe)
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
