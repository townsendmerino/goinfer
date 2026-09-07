package decoder

import (
	"os"
	"path/filepath"
	"regexp"
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
//     docs/task-first-hour.md, one dependency over.
//   - The parity manifest's aikit_version tracks the ROOT's pin only. A submodule on a different
//     aikit is numerics the staleness gate cannot see, which is exactly the hole that let
//     aikit_version sit at v1.19.0 for seventeen versions.
//
// aikit has the same gate for its own eight gpu backends (`gpu backend version pins` in its CI,
// with `gpupins --fix`); this is the goinfer-side counterpart. Mid-cycle drift is EXPECTED between
// releases — RELEASING.md's two-step tag bumps the submodules after the root tag — but expected is
// not the same as unchecked, and the two-step is precisely the ritual a person can forget.
func TestAikitPinsAgree(t *testing.T) {
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
