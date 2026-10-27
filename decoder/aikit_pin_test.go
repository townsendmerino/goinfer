package decoder

import (
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

// TestAikitPinsAgree pins every module's aikit require to the ROOT's. Each Go module has its own go.mod and
// pins github.com/townsendmerino/aikit independently, and nothing else checks that they agree. It matters
// for two reasons: the release binaries (metal/cmd/serve, cuda/cmd/serve) are built from the submodules, so
// one pinning an older aikit ships older numerics to users; and the parity manifest's aikit_version tracks
// only the ROOT's pin, so a submodule on a different aikit is numerics the staleness gate cannot see.
//
// It cannot check that a pin is the LATEST tag (that needs the network), only that the modules agree. Mid-cycle drift
// is expected between releases (RELEASING.md's two-step tag bumps the submodules after the root tag), but expected is
// not unchecked, and the two-step is the ritual a person can forget. No module requires aikit/gpu any more:
// internal/movedpaths refuses one.
func TestAikitPinsAgree(t *testing.T) {
	t.Run("aikit", testAikitRootPinsAgree)
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
