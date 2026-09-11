//go:build cuda

package cuda

import (
	"os"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// TestPTXModules_coverEveryEmbed holds TestKernelLocalMemoryCensus's module list to what kernels.go
// actually embeds (audit-2026-09-10 G-13(b)). The list was written by hand and covered 15 of 22
// modules, so the census's "moe_route declares the maximum local memory" precondition was never
// checked against the gpt-oss expert-cache path it exists for (gptoss_act.ptx was missing).
// TestKernelFMALint_coversEmbeddedPTX closes the same gap for the FMA lint.
func TestPTXModules_coverEveryEmbed(t *testing.T) {
	kb, err := os.ReadFile("kernels.go")
	if err != nil {
		t.Fatalf("read kernels.go: %v", err)
	}
	embedded := map[string]bool{}
	for _, m := range regexp.MustCompile(`//go:embed testdata/([A-Za-z0-9_]+\.ptx)`).FindAllStringSubmatch(string(kb), -1) {
		embedded[m[1]] = true
	}
	if len(embedded) < 10 {
		t.Fatalf("found %d embedded PTX in kernels.go — the scan is broken", len(embedded))
	}
	listed := map[string]bool{}
	for _, m := range ptxModules() {
		listed[m.name] = true
	}
	var missing, stale []string
	for f := range embedded {
		if !listed[f] {
			missing = append(missing, f)
		}
	}
	for f := range listed {
		if !embedded[f] {
			stale = append(stale, f)
		}
	}
	sort.Strings(missing)
	sort.Strings(stale)
	if len(missing) > 0 {
		t.Errorf("ptxModules() omits %d embedded module(s): %s — the local-memory census cannot see them",
			len(missing), strings.Join(missing, " "))
	}
	if len(stale) > 0 {
		t.Errorf("ptxModules() lists %d module(s) kernels.go no longer embeds: %s", len(stale), strings.Join(stale, " "))
	}
	t.Logf("%d embedded modules, %d in ptxModules()", len(embedded), len(listed))
}
