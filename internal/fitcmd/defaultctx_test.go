package fitcmd

import (
	"os"
	"regexp"
	"strconv"
	"testing"
)

// The dry run and the real load must agree on the default context: cuda/resident.go's fitDefaultCtx says goinfer-chat fit's -ctx default is "the same
// figure, so the dry run and the real load agree" — and nothing checked it, so R19's change (8192 to 16384, docs/tasks/task-first-hour.md) had two
// places to forget. The cuda module cannot be imported from here (it is its own module, behind a build tag), so the constant is read from its source.
// It must also hold a coding agent's first request: opencode's was 11,137 tokens.
func TestDefaultCtx_agreesWithTheCudaPlannerAndHoldsAnAgentTurn(t *testing.T) {
	src, err := os.ReadFile("../../cuda/resident.go")
	if err != nil {
		t.Fatal(err)
	}
	m := regexp.MustCompile(`(?m)^const fitDefaultCtx = (\d+)\s*$`).FindSubmatch(src)
	if m == nil {
		t.Fatal("const fitDefaultCtx not found in cuda/resident.go — this guard is watching nothing")
	}
	want, _ := strconv.Atoi(string(m[1]))

	if defaultCtx != want {
		t.Errorf("fit's -ctx defaults to %d but the CUDA planner's candidate is %d — the dry run prices a different context than the load picks", defaultCtx, want)
	}
	fitSrc, err := os.ReadFile("fit.go")
	if err != nil {
		t.Fatal(err)
	}
	if !regexp.MustCompile(`fs\.Int\("ctx", defaultCtx,`).Match(fitSrc) {
		t.Error("the -ctx flag no longer defaults to defaultCtx, so the constant above proves nothing")
	}
	if want < 11137+1024 {
		t.Errorf("the default context %d does not hold a coding agent's first request (11,137 tokens, opencode) with headroom (R19)", want)
	}
}
