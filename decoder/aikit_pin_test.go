package decoder

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"testing"
)

// TestAikitPinsAgree pins every module's aikit require to the ROOT's.
//
// WHY. goinfer is five Go modules, each with its own go.mod, and each pinning
// github.com/townsendmerino/aikit independently. Nothing checked that they agree. Measured
// 2026-09-06: the root sat at v1.37.0 while cuda, gpu, metal and demo/agent were all still on
// v1.35.0 — a drift discovered only by reading four files by hand, prompted by a months-old git
// stash that had once tried to fix the same thing at v1.33.0.
//
// It matters more here than "tidiness" suggests, for two reasons:
//
//   - The RELEASE ASSETS ARE BUILT FROM THESE MODULES. The Mac and Linux `goinfer-serve` binaries
//     come from metal/cmd/serve and cuda/cmd/serve, so a submodule pinning an older aikit ships an
//     older aikit to users — the same failure shape as the release-asset finding recorded in
//     docs/tasks/task-first-hour.md, one dependency over.
//   - The parity manifest's aikit_version tracks the ROOT's pin only. A submodule on a different
//     aikit is numerics the staleness gate cannot see, which is exactly the hole that let
//     aikit_version sit at v1.19.0 for seventeen versions.
//
// aikit's own `gpupins` gate (`gpu backend version pins` in its CI, with `gpupins --fix`) checks both
// the root aikit pin and the aikit/gpu pin across aikit's eight gpu backends. This test covers the
// same two pins on the goinfer side, as two subtests: "aikit" and "aikit/gpu". (Until 2026-09-24 this comment claimed that parity while the test matched only
// `aikit v…` — the regex's space after "aikit" never matches `aikit/gpu v…` — so metal sat on
// gpu v0.33.1 while cuda was on v0.33.3, unseen.) Neither test can check that a pin is the LATEST
// tag — that needs the network — only that the modules agree. Mid-cycle drift is EXPECTED between
// releases — RELEASING.md's two-step tag bumps the submodules after the root tag — but expected is
// not the same as unchecked, and the two-step is precisely the ritual a person can forget.
func TestAikitPinsAgree(t *testing.T) {
	t.Run("aikit", testAikitRootPinsAgree)
	t.Run("aikit/gpu", testAikitGPUPinsAgree)
}

// testAikitRootPinsAgree pins every module's aikit require to the ROOT's.
func testAikitRootPinsAgree(t *testing.T) {
	req := regexp.MustCompile(`(?m)^\s*github\.com/townsendmerino/aikit (v[0-9][^\s]*)`)

	read := func(mod string) string {
		b, err := os.ReadFile(filepath.Join("..", mod, "go.mod"))
		if err != nil {
			t.Fatalf("read %s/go.mod: %v", mod, err)
		}
		m := req.FindStringSubmatch(string(b))
		if m == nil {
			return "" // module does not depend on aikit directly; nothing to agree with
		}
		return m[1]
	}

	root := read(".")
	if root == "" {
		t.Fatal("the ROOT go.mod does not require aikit — this gate has nothing to compare against " +
			"and would pass vacuously")
	}
	t.Logf("root pins aikit %s", root)

	checked := 0
	for _, mod := range []string{"cuda", "gpu", "metal", "demo/agent"} {
		got := read(mod)
		if got == "" {
			continue
		}
		checked++
		if got != root {
			t.Errorf("%s/go.mod pins aikit %s, root pins %s — a submodule on a different aikit is "+
				"numerics the parity manifest's aikit_version cannot see, and the release assets "+
				"are built FROM these modules", mod, got, root)
		}
	}
	// A zero-module pass would be a green that vouches for nothing — the same shape as the
	// staleness gate enforcing zero families.
	if checked == 0 {
		t.Fatal("no submodule required aikit: this gate checked nothing")
	}
	t.Logf("%d submodule(s) agree with the root", checked)
}

// testAikitGPUPinsAgree pins every module that requires github.com/townsendmerino/aikit/gpu to the SAME
// version. There is no root pin to compare against (the root and gpu/go.mod require only aikit), so the
// rule is agreement: today cuda and metal, which build the Linux and Mac `goinfer-serve` release
// binaries. A module on an older gpu ships older kernels to users, and the parity manifest tracks the
// ROOT's aikit_version only, so a gpu-only difference is numerics its staleness gate cannot see.
// RELEASING.md's B-07 states this rule; before this test nothing enforced it (metal sat on v0.33.1
// while cuda was on v0.33.3, found 2026-09-24).
func testAikitGPUPinsAgree(t *testing.T) {
	req := regexp.MustCompile(`(?m)^\s*github\.com/townsendmerino/aikit/gpu (v[0-9][^\s]*)`)
	pins := map[string]string{}
	for _, mod := range []string{".", "cuda", "gpu", "metal", "demo/agent"} {
		b, err := os.ReadFile(filepath.Join("..", mod, "go.mod"))
		if err != nil {
			t.Fatalf("read %s/go.mod: %v", mod, err)
		}
		if m := req.FindStringSubmatch(string(b)); m != nil {
			pins[mod] = m[1]
		}
	}
	// Fewer than two pinning modules is an agreement check that compares nothing: the same
	// never-a-vacuous-green rule as the aikit subtest above.
	if len(pins) < 2 {
		t.Fatalf("only %d module(s) require aikit/gpu (%v): this gate needs at least two to compare — "+
			"if a backend stopped requiring it, update this test deliberately", len(pins), pins)
	}
	mods := make([]string, 0, len(pins))
	for m := range pins {
		mods = append(mods, m)
	}
	sort.Strings(mods)
	ref := mods[0]
	for _, m := range mods[1:] {
		if pins[m] != pins[ref] {
			t.Errorf("%s/go.mod pins aikit/gpu %s, %s/go.mod pins %s — the modules that build the release "+
				"binaries must agree on aikit/gpu (RELEASING.md B-07); bump them together",
				m, pins[m], ref, pins[ref])
		}
	}
	t.Logf("aikit/gpu pins: %v", pins)
}
