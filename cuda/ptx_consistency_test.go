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

// TestPTX_matchesSourcesAndBindings is PTX freshness without a device or an NVRTC (audit-2026-09-10
// G-09). A .cu edit committed without regenerating its .ptx compiles, vets and passes -short CI.
// Then at runtime cuModuleGetFunction fails, BuildResident declines, and every family runs on the
// CPU; 23c46b1 and 5b44383 each describe that shape. This test is its static guard:
//
//  1. every __global__ a .cu declares is a .visible .entry in its .ptx, with the same parameter
//     count and kinds, and the .ptx carries no entry the .cu no longer declares;
//  2. every kernel name the production Go code binds is an entry in some embedded PTX.
//
// It reads files only, so it runs in CI's -short cuda job.
func TestPTX_matchesSourcesAndBindings(t *testing.T) {
	kb, err := os.ReadFile("kernels.go")
	if err != nil {
		t.Fatalf("read kernels.go: %v", err)
	}
	var mods []string
	for _, m := range regexp.MustCompile(`//go:embed testdata/([A-Za-z0-9_]+)\.ptx`).FindAllStringSubmatch(string(kb), -1) {
		mods = append(mods, m[1])
	}
	if len(mods) < 10 {
		t.Fatalf("found %d embedded PTX modules in kernels.go — the scan is broken", len(mods))
	}
	entryMod := map[string]string{}
	checked, kernels := 0, 0
	for _, mod := range mods {
		ptx, err := os.ReadFile(filepath.Join("testdata", mod+".ptx"))
		if err != nil {
			t.Errorf("%s.ptx is embedded but unreadable: %v", mod, err)
			continue
		}
		entries := ptxEntries(string(ptx))
		if len(entries) == 0 {
			t.Errorf("%s.ptx has no .visible .entry — empty or unparseable", mod)
		}
		for name := range entries {
			entryMod[name] = mod
		}
		cu, err := os.ReadFile(mod + ".cu")
		if err != nil {
			t.Errorf("%s.ptx is embedded but %s.cu is missing — a shipped PTX must regenerate from its source", mod, mod)
			continue
		}
		globals := cuGlobals(string(cu))
		checked++
		for name, want := range globals {
			kernels++
			got, ok := entries[name]
			if !ok {
				t.Errorf("%s.cu declares __global__ %s but %s.ptx has no .visible .entry %s — the PTX lags "+
					"its source; rebuild it with build_ptx.sh (REGEN.md)", mod, name, mod, name)
				continue
			}
			if len(got) != len(want) {
				t.Errorf("%s: %s.cu has %d parameters, %s.ptx has %d — the PTX lags its source", name, mod, len(want), mod, len(got))
				continue
			}
			for i := range want {
				if want[i] != "?" && got[i] != "?" && want[i] != got[i] {
					t.Errorf("%s parameter %d: %s.cu kind %s, %s.ptx kind %s — the PTX lags its source", name, i, mod, want[i], mod, got[i])
				}
			}
		}
		for name := range entries {
			if _, ok := globals[name]; !ok {
				t.Errorf("%s.ptx has .visible .entry %s, which %s.cu no longer declares — stale PTX", mod, name, mod)
			}
		}
	}
	bound := goBoundKernelNames(t)
	if len(bound) < 30 {
		t.Fatalf("found only %d kernel names bound by the production Go code — the binding scan is broken", len(bound))
	}
	for _, name := range bound {
		if _, ok := entryMod[name]; !ok {
			t.Errorf("production code binds kernel %q, which no embedded PTX defines — BuildResident would "+
				"fail to find it at runtime and decline to CPU", name)
		}
	}
	t.Logf("%d .cu/.ptx pairs, %d kernels checked by name and signature; %d bound kernel names resolved", checked, kernels, len(bound))
}

var (
	ptxEntryHead = regexp.MustCompile(`\.visible\s+\.entry\s+([A-Za-z_][A-Za-z0-9_$]*)\s*\(`)
	ptxParamKind = regexp.MustCompile(`\.param\s+(?:\.align\s+\d+\s+)?\.([a-z]+[0-9]*)`)
	cuComment    = regexp.MustCompile(`(?s)/\*.*?\*/|//[^\n]*`)
	cuGlobalHead = regexp.MustCompile(`__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?void\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(`)
)

// ptxEntries maps each .visible .entry to its parameters' kinds, normalised by width.
func ptxEntries(ptx string) map[string][]string {
	out := map[string][]string{}
	for _, loc := range ptxEntryHead.FindAllStringSubmatchIndex(ptx, -1) {
		name := ptx[loc[2]:loc[3]]
		end := strings.Index(ptx[loc[1]:], ")")
		if end < 0 {
			continue
		}
		var kinds []string
		for _, pm := range ptxParamKind.FindAllStringSubmatch(ptx[loc[1]:loc[1]+end], -1) {
			kinds = append(kinds, ptxKind(pm[1]))
		}
		out[name] = kinds
	}
	return out
}

func ptxKind(t string) string {
	switch t {
	case "u64", "s64", "b64":
		return "64"
	case "u32", "s32", "b32":
		return "32i"
	case "f32":
		return "f32"
	case "f64":
		return "f64"
	case "u16", "s16", "b16", "f16":
		return "16"
	case "u8", "s8", "b8":
		return "8"
	}
	return "?"
}

// cuGlobals maps each __global__ kernel in a .cu (comments stripped) to its parameters' kinds.
func cuGlobals(cu string) map[string][]string {
	src := cuComment.ReplaceAllString(cu, "")
	out := map[string][]string{}
	for _, loc := range cuGlobalHead.FindAllStringSubmatchIndex(src, -1) {
		name := src[loc[2]:loc[3]]
		depth, i := 1, loc[1]
		for ; i < len(src) && depth > 0; i++ {
			switch src[i] {
			case '(':
				depth++
			case ')':
				depth--
			}
		}
		params := strings.TrimSpace(src[loc[1] : i-1])
		var kinds []string
		if params != "" && params != "void" {
			for _, p := range strings.Split(params, ",") {
				kinds = append(kinds, cKind(p))
			}
		}
		out[name] = kinds
	}
	return out
}

// cKind maps a C parameter declaration to the width PTX gives it.
func cKind(p string) string {
	if strings.Contains(p, "*") {
		return "64"
	}
	f := strings.Fields(strings.NewReplacer("const", " ", "__restrict__", " ").Replace(p))
	if len(f) < 2 {
		return "?"
	}
	switch strings.Join(f[:len(f)-1], " ") {
	case "float":
		return "f32"
	case "double":
		return "f64"
	case "int", "unsigned", "unsigned int", "uint32_t", "int32_t", "uint":
		return "32i"
	case "long", "long long", "int64_t", "uint64_t", "size_t", "unsigned long long":
		return "64"
	case "__half", "half", "short", "unsigned short":
		return "16"
	case "bool", "char", "unsigned char", "int8_t", "uint8_t", "signed char":
		return "8"
	}
	return "?"
}

// goBoundKernelNames collects the kernel-name literals the production Go code binds, in each form
// the cuda module uses: NewComputePipeline(mod, "x"), load*(&r.f, [mod,] "x"), the {&r.f, "x"}
// tables, and vision_encoder's {xPTX, "x", &r.f} table.
func goBoundKernelNames(t *testing.T) []string {
	t.Helper()
	files, _ := filepath.Glob("*.go")
	res := []*regexp.Regexp{
		regexp.MustCompile(`NewComputePipeline\(\s*\w+\s*,\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*\)`),
		regexp.MustCompile(`\bload\w*\(\s*&[\w.\[\]]+\s*,\s*(?:\w+\s*,\s*)?"([A-Za-z_][A-Za-z0-9_]*)"\s*\)`),
		regexp.MustCompile(`\{\s*&[\w.\[\]]+\s*,\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*\}`),
		regexp.MustCompile(`\{\s*\w+PTX\s*,\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*,\s*&`),
	}
	seen := map[string]bool{}
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("read %s: %v", f, err)
		}
		for _, re := range res {
			for _, m := range re.FindAllStringSubmatch(string(b), -1) {
				seen[m[1]] = true
			}
		}
	}
	var out []string
	for n := range seen {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}
