//go:build cuda

package cuda

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// TestPipelineLint_boundKernelsAreLaunched closes the FIFTH state a kernel can be in.
//
// The launch-site census sorts kernels into: launched by production and covered by an asserting
// gate; launched by production and covered by nothing; launched only from tests; and embedded but
// never launched. `gemv_w4a8_batched` was in none of them. It was BOUND into a production pipeline
// field at every model load — paying NVRTC JIT time — and launched by nothing, anywhere, while
// carrying a parity test AND a bandwidth benchmark that made it look like the shipping batched
// int4 kernel. It is not: bGemvB dispatches int4 to `gemv_w4a8_rn` unconditionally
// (prefill.go, bGemvB), so the kernel named for the feature was not the one the feature used.
//
// A runtime launch trace CANNOT find this class — it sees no launch for a bound-and-dead field
// and no launch for a field that was never bound, and those are the same observation with
// opposite causes. Only source can tell them apart, so this is a static lint, in the same family
// as TestKernelFMALint_coversEmbeddedPTX: fix the class, not the instance.
//
// Plain `-tags cuda` (no testhooks, no device) so gpu_gate.sh group 2a runs it.
func TestPipelineLint_boundKernelsAreLaunched(t *testing.T) {
	src := packageSources(t)

	// Pipeline-typed fields of cudaResident: "a, b, c Pipeline" possibly followed by a comment.
	fieldDecl := regexp.MustCompile(`(?m)^\s+([A-Za-z0-9_, \t]+?)\s+Pipeline\b`)
	fields := map[string]bool{}
	for _, m := range fieldDecl.FindAllStringSubmatch(src["resident.go"], -1) {
		for _, name := range strings.Split(m[1], ",") {
			if name = strings.TrimSpace(name); name != "" {
				fields[name] = true
			}
		}
	}
	if len(fields) < 20 {
		// Guard the guard: if the field regex stops matching (a struct reformat, a type rename),
		// every field silently drops out and the lint passes having checked nothing — the same
		// "zero means either" shape as the skip census reading 0 for want of -v.
		t.Fatalf("pipeline-field scan found only %d fields; the declaration shape changed and this "+
			"lint would pass having checked nothing", len(fields))
	}

	names := make([]string, 0, len(fields))
	for f := range fields {
		names = append(names, f)
	}
	sort.Strings(names)

	all := strings.Join(valuesOf(src), "\n")
	var dead, unbound []string
	for _, f := range names {
		// Bound: &r.F (load helpers, table entries) or `r.F, ... = ...NewComputePipeline/qgemv`.
		bound := regexp.MustCompile(`&r\.`+f+`\b`).MatchString(all) ||
			regexp.MustCompile(`\br\.`+f+`\b[^=\n]*=\s*(r\.dev\.NewComputePipeline|qgemv\.)`).MatchString(all)
		// Launched: r.launch(r.F or stream.Launch(r.F.
		launched := regexp.MustCompile(`[Ll]aunch\(\s*r\.` + f + `\b`).MatchString(all)
		switch {
		case bound && !launched:
			dead = append(dead, f)
		case !bound && launched:
			unbound = append(unbound, f)
		}
	}

	if len(unbound) > 0 {
		t.Errorf("pipeline field(s) LAUNCHED but never bound — a nil pipeline launch: %v", unbound)
	}
	if len(dead) > 0 {
		t.Errorf("pipeline field(s) BOUND but never launched: %v\n"+
			"Each is NVRTC-compiled into every model load and run by nothing. Either launch it or "+
			"delete the binding (and the kernel, if nothing else uses it). A bound-and-dead kernel "+
			"reads as shipping code to every reader and to every benchmark quoting its throughput.",
			dead)
	}
	t.Logf("pipeline lint: %d Pipeline fields, %d bound-and-dead, %d launched-but-unbound", len(names), len(dead), len(unbound))
}

// packageSources reads the package's non-test .go sources, keyed by base name.
//
// EXCLUDING _test.go IS THE POINT, not an oversight — do not "fix" it. The lint asks whether a
// pipeline bound in PRODUCTION is launched in PRODUCTION. gemv_w4a8_batched, the defect this was
// written for, was launched by a parity test AND a bandwidth benchmark; counting those would have
// hidden it, and made the benchmark's throughput read as the shipping kernel's. A field launched
// only from tests is precisely what this must flag.
//
// It DOES include build-tagged non-test files (testhooks_gen.go among them), since it reads every
// .go in the directory rather than a tag-filtered set — so a field launched only from
// goinfer_testhooks production code is seen. That is the opposite exposure and it is not present.
func packageSources(t *testing.T) map[string]string {
	t.Helper()
	ents, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}
	out := map[string]string{}
	for _, e := range ents {
		n := e.Name()
		if !strings.HasSuffix(n, ".go") || strings.HasSuffix(n, "_test.go") {
			continue
		}
		b, err := os.ReadFile(filepath.Clean(n))
		if err != nil {
			t.Fatalf("read %s: %v", n, err)
		}
		out[n] = string(b)
	}
	if out["resident.go"] == "" {
		t.Fatal("resident.go not readable — the lint cannot see the pipeline fields")
	}
	return out
}

func valuesOf(m map[string]string) []string {
	out := make([]string, 0, len(m))
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		out = append(out, m[k])
	}
	return out
}
