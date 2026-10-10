package decoder

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"testing"
)

// TestAikitPinsAgree pins every module's aikit require to the ROOT's. Each Go module has its own go.mod and
// pins github.com/townsendmerino/aikit independently, and nothing else checks that they agree. It matters
// for two reasons: the release binaries (metal/cmd/serve, cuda/cmd/serve) are built from the submodules, so
// one pinning an older aikit ships older numerics to users; and the parity manifest's aikit_version tracks
// only the ROOT's pin, so a submodule on a different aikit is numerics the staleness gate cannot see.
//
// Two subtests, "aikit" and "aikit/gpu": the goinfer-side twin of aikit's own gpupins gate (its CI step
// `gpu backend version pins`). Neither can check that a pin is the LATEST tag (that needs the network),
// only that the modules agree. Mid-cycle drift is expected between releases (RELEASING.md's two-step tag
// bumps the submodules after the root tag), but expected is not unchecked, and the two-step is the ritual
// a person can forget.
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
// version. There is no root pin to compare against (the root and gpu/go.mod require only aikit), so the rule
// is agreement. A module on an older gpu ships older kernels to users, and the parity manifest tracks only the
// ROOT's aikit_version, so a gpu-only difference is numerics its staleness gate cannot see. RELEASING.md's
// B-07 states the rule.
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
