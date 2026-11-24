package main

import (
	"flag"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
)

// Every listed test lands in exactly one group, and each group's -run pattern selects exactly its own tests.
func TestHeavyPartition_everyTestInExactlyOneGroup(t *testing.T) {
	refs := map[string][]string{
		"TestA26":      {"gemma4-26b-int4.giw"},
		"TestB7":       {"$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
		"TestCTiny":    {"../testdata/gemma4-moe-tiny"},
		"TestDNone":    nil,
		"TestE15":      {"qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
		"TestFGemma3":  {"gemma-3-4b-it", "glm-ocr"},
		"TestG35":      {"qwen3.6-35b-a3b-int4.giw", "qwen36-35b-dflash"},
		"TestHTower":   {"qwen3.6-35b-a3b-vision", "qwen3.5-0.8b"},
		"TestI_Regexp": {"aya-expanse-8b"},
		"TestJKV10":    {"$HOME/models/qwen3.5-0.8b", "KV10_CKPT"},
	}
	var listed []string
	for k := range refs {
		listed = append(listed, k)
	}
	slices.Sort(listed)
	groups, problems := heavyPartition(listed, refs, nil)
	if len(problems) > 0 {
		t.Fatalf("problems: %v", problems)
	}
	owner := map[string]string{}
	for _, g := range groups {
		re := regexp.MustCompile(g.Run)
		for _, n := range g.Tests {
			if prev, ok := owner[n]; ok {
				t.Errorf("%s is in both %s and %s", n, prev, g.Label)
			}
			owner[n] = g.Label
		}
		for _, n := range listed {
			if re.MatchString(n) != slices.Contains(g.Tests, n) {
				t.Errorf("%s's pattern matches %s: %v, but owning it: %v", g.Label, n, re.MatchString(n), slices.Contains(g.Tests, n))
			}
		}
	}
	want := map[string]string{
		"TestA26": "gemma4-26b", "TestB7": "7b-8b", "TestCTiny": "small", "TestDNone": "small", "TestE15": "small",
		"TestFGemma3": "gemma3-4b", "TestG35": "qwen3.6-35b", "TestHTower": "small", "TestI_Regexp": "7b-8b",
		"TestJKV10": "kv10",
	}
	for n, g := range want {
		if owner[n] != g {
			t.Errorf("%s: in group %q, want %q", n, owner[n], g)
		}
	}
	if last := groups[len(groups)-1].Label; last != heavySmallGroup {
		t.Errorf("the shared group runs last; got %q last", last)
	}
}

// KV10's harness runs at the gate's depths in its own process: its group is not the shared one, and every heavy process
// pins GOINFER_KV10_DEPTHS, so a deeper value in the calling environment cannot reach it.
func TestHeavyGroup_kv10RunsAloneAtTheGatesDepths(t *testing.T) {
	refs := map[string][]string{"TestKV10_longContextFidelity": {"$HOME/models/qwen3.5-0.8b", "KV10_CKPT"}, "TestSmall": {"qwen3.5-0.8b"}}
	groups, problems := heavyPartition([]string{"TestKV10_longContextFidelity", "TestSmall"}, refs, nil)
	if len(problems) > 0 {
		t.Fatalf("problems: %v", problems)
	}
	for _, g := range groups {
		if slices.Contains(g.Tests, "TestKV10_longContextFidelity") && (g.Label == heavySmallGroup || len(g.Tests) != 1) {
			t.Errorf("KV10 is in group %q with %v; want a group of its own", g.Label, g.Tests)
		}
	}
	if got := heavyGroupEnv()["GOINFER_KV10_DEPTHS"]; got != "3900,65536" {
		t.Errorf("heavy processes set GOINFER_KV10_DEPTHS=%q, want 3900,65536", got)
	}
}

// A test that fits no group is named, never silently dropped or guessed into one.
func TestHeavyPartition_aTestThatFitsNoGroupIsAProblem(t *testing.T) {
	cases := []struct {
		name    string
		refs    map[string][]string
		entries []string
		want    string
	}{
		{"not in source", map[string][]string{}, nil, "not found in cuda"},
		{"unknown gguf", map[string][]string{"TestX": {"llama-70b-q4.gguf"}}, nil, "no heavyFamilies row"},
		{"unknown models path", map[string][]string{"TestX": {"$HOME/models/mixtral-8x7b"}}, nil, "no heavyFamilies row"},
		{"unknown models entry", map[string][]string{"TestX": {"mixtral-8x7b"}}, []string{"mixtral-8x7b"}, "no heavyFamilies row"},
		{"two large families", map[string][]string{"TestX": {"gemma4-26b-int4.giw", "qwen2.5-7b-instruct-q4_k_m.gguf"}}, nil, "2 large families"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			groups, problems := heavyPartition([]string{"TestX"}, c.refs, c.entries)
			if len(problems) != 1 || !strings.Contains(problems[0], c.want) {
				t.Fatalf("problems %v, want one naming %q", problems, c.want)
			}
			for _, g := range groups {
				if slices.Contains(g.Tests, "TestX") {
					t.Errorf("TestX was put in %s anyway", g.Label)
				}
			}
		})
	}
	// Prose that happens to contain a models path is not a checkpoint.
	if _, problems := heavyPartition([]string{"TestX"}, map[string][]string{"TestX": {"POST /admin/models/unload breaks it"}}, nil); len(problems) > 0 {
		t.Errorf("a message literal was read as a checkpoint: %v", problems)
	}
}

// The derivation follows helpers, package-level variables, methods and asset keys, and reads only the files that build
// under the tags.
func TestTestReachableLiterals_followsWhatATestCanReach(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("a_test.go", `//go:build cuda

package cuda

import "testing"

var bigModel = "gemma4-26b-int4.giw"

type loader struct{}

func (loader) load() string { return "qwen3-4b" }

func helper() string { return deeper() }
func deeper() string  { return bigModel }

func TestViaHelper(t *testing.T) { _ = helper() }
func TestViaMethod(t *testing.T) { _ = loader{}.load() }
func TestViaAsset(t *testing.T)  { _ = "GOINFER_X" }
func TestPlain(t *testing.T)     {}
`)
	write("b_test.go", `//go:build cuda && realckpt

package cuda

import "testing"

func TestRealckptOnly(t *testing.T) { _ = "qwen2.5-7b-instruct-q4_k_m.gguf" }
`)
	write("c_darwin_test.go", `package cuda

import "testing"

func TestDarwinOnly(t *testing.T) {}
`)
	assets := filepath.Join(dir, "assets.json")
	if err := os.WriteFile(assets, []byte(`{"assets":[{"env":"GOINFER_X","candidates":["$MODELS/glm-ocr"]}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	refs, err := testReachableLiterals(dir, []string{"cuda"}, assets)
	if err != nil {
		t.Fatal(err)
	}
	for test, lit := range map[string]string{"TestViaHelper": "gemma4-26b-int4.giw", "TestViaMethod": "qwen3-4b", "TestViaAsset": "$MODELS/glm-ocr"} {
		if !slices.Contains(refs[test], lit) {
			t.Errorf("%s reaches %v, want %q among them", test, refs[test], lit)
		}
	}
	if _, ok := refs["TestPlain"]; !ok {
		t.Error("TestPlain is missing: a test that reaches no literal must still be found")
	}
	for _, n := range []string{"TestRealckptOnly", "TestDarwinOnly"} {
		if _, ok := refs[n]; ok {
			t.Errorf("%s does not build under the tags on linux but was read", n)
		}
	}
}

// Every test in cuda/ fits a group under the table. This is the check a new test with an unclassified checkpoint fails,
// on any machine and in CI, before the gate ever runs it.
func TestHeavyPartition_cudaTreeFitsTheTable(t *testing.T) {
	root, err := repoRoot()
	if err != nil {
		t.Skip(err)
	}
	refs, err := testReachableLiterals(filepath.Join(root, "backend", "cuda"), []string{"cuda", "goinfer_testhooks"}, filepath.Join(root, "testdata", "assets.json"))
	if err != nil {
		t.Fatal(err)
	}
	var listed []string
	for k := range refs {
		listed = append(listed, k)
	}
	slices.Sort(listed)
	if len(listed) < 100 {
		t.Fatalf("found %d tests in cuda/; the derivation is reading the wrong files", len(listed))
	}
	groups, problems := heavyPartition(listed, refs, nil)
	for _, p := range problems {
		t.Error(p)
	}
	byLabel := map[string]int{}
	byGroup := map[string][]string{}
	for _, g := range groups {
		byLabel[g.Label] = len(g.Tests)
		byGroup[g.Label] = g.Tests
	}
	for _, l := range []string{"gemma4-26b", "7b-8b", heavySmallGroup} {
		if byLabel[l] == 0 {
			t.Errorf("group %s is empty; got %v", l, byLabel)
		}
	}
	if got := byGroup["nemotron-h"]; !slices.Equal(got, []string{"TestNemotronResidentCUDA_tier1", "TestP1r_nemotronRangeSurvey"}) {
		t.Errorf("group nemotron-h = %v, want [TestNemotronResidentCUDA_tier1 TestP1r_nemotronRangeSurvey] so absent checkpoints grade as an all-skip group", got)
	}
	if got := byGroup["kv10"]; !slices.Equal(got, []string{"TestKV10_longContextFidelity"}) {
		t.Errorf("group kv10 = %v, want [TestKV10_longContextFidelity] alone, out of the shared group", got)
	}
}

func TestParseProcStat(t *testing.T) {
	name, ppid, ok := parseProcStat("4242 (we(ird) name.test) S 17 4242 4242 0 -1")
	if !ok || name != "we(ird) name.test" || ppid != 17 {
		t.Errorf("parseProcStat: %q %d %v", name, ppid, ok)
	}
}

// The sampler finds a .test binary that is a grandchild of the process it is given, as `go test`'s binary is of `go`.
func TestProcSampler_findsTheTestBinaryUnderItsParent(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("reads /proc")
	}
	// The grandchild is this test binary, already named "<pkg>.test", re-run into TestProcSampler_helperSleeps. A copy of a system
	// binary under a .test name is not a stand-in: a multi-call coreutils (uutils, busybox) picks its utility from argv[0] and exits.
	bin, err := os.Executable()
	if err != nil {
		t.Skip(err)
	}
	if !strings.HasSuffix(filepath.Base(bin), ".test") {
		t.Skipf("test binary %s is not named .test", bin)
	}
	old := procSamplePeriod
	procSamplePeriod = 20 * time.Millisecond
	t.Cleanup(func() { procSamplePeriod = old })
	cmd := exec.Command("sh", "-c", `"$0" -test.run='^TestProcSampler_helperSleeps$' -procsampler.helper; true`, bin)
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	stop := startProcSampler(cmd.Process.Pid, func() (string, time.Duration, bool) { return "TestFake/sub", 0, true })
	_ = cmd.Wait()
	p := stop()
	if p.Samples == 0 || p.HWMKB == 0 || p.PID == cmd.Process.Pid {
		t.Fatalf("sampled %+v; want the grandchild's VmHWM", p)
	}
	if p.ByTest["TestFake"] == 0 || len(p.ByTest) != 1 {
		t.Errorf("attribution %v; want the top-level test only", p.ByTest)
	}
}

var procSamplerHelper = flag.Bool("procsampler.helper", false, "run TestProcSampler_helperSleeps as the sampler test's grandchild")

// TestProcSampler_helperSleeps is the grandchild TestProcSampler_findsTheTestBinaryUnderItsParent samples: it lives a second.
func TestProcSampler_helperSleeps(t *testing.T) {
	if !*procSamplerHelper {
		t.Skip("only as TestProcSampler_findsTheTestBinaryUnderItsParent's grandchild")
	}
	time.Sleep(time.Second)
}

// The shared group runs only tests that reach a heavy check (Francis, 2026-11-23): a test with none runs in the parity cell,
// which runs the whole package, so running it in small too pays for it twice. A drain- or fresh-process-marked test stays,
// since the tier reconciles its skip token there; a test in a family group is not affected.
func TestHeavyMainGroups_aTestWithNoHeavyCheckRunsOnlyInTheParityCell(t *testing.T) {
	refs := map[string][]string{
		"TestTinyNoHeavy":  {"../../testdata/llama-tiny"},
		"TestTinyHeavy":    {"../../testdata/llama-tiny", "GOINFER_HEAVY_TESTS"},
		"TestRequireHeavy": {"qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", heavyGateMark},
		"TestDrainNoHeavy": {"../../testdata/llama-tiny"},
		"TestIsoNoHeavy":   nil,
		"TestBigNoHeavy":   {"gemma4-26b-int4.giw"},
		"TestNothingAtAll": nil,
	}
	var listed []string
	for k := range refs {
		listed = append(listed, k)
	}
	slices.Sort(listed)
	groups, parityOnly, problems := heavyMainGroups(listed, refs, nil, []string{"TestDrainNoHeavy", "TestIsoNoHeavy"})
	if len(problems) > 0 {
		t.Fatalf("problems: %v", problems)
	}
	owner := map[string]string{}
	for _, g := range groups {
		re := regexp.MustCompile(g.Run)
		for _, n := range g.Tests {
			owner[n] = g.Label
		}
		for _, n := range listed {
			if re.MatchString(n) != slices.Contains(g.Tests, n) {
				t.Errorf("%s's pattern matches %s: %v, but owning it: %v", g.Label, n, re.MatchString(n), slices.Contains(g.Tests, n))
			}
		}
	}
	want := map[string]string{"TestTinyHeavy": "small", "TestRequireHeavy": "small", "TestDrainNoHeavy": "small",
		"TestIsoNoHeavy": "small", "TestBigNoHeavy": "gemma4-26b"}
	for n, g := range want {
		if owner[n] != g {
			t.Errorf("%s: in %q, want %q", n, owner[n], g)
		}
	}
	if w := []string{"TestNothingAtAll", "TestTinyNoHeavy"}; !slices.Equal(parityOnly, w) {
		t.Errorf("parity-only %v, want %v", parityOnly, w)
	}
	for _, n := range parityOnly {
		if owner[n] != "" {
			t.Errorf("%s is parity-only and also in %s", n, owner[n])
		}
	}
}

// A call to testgate.RequireHeavy is a heavy check as much as the variable's name in a literal, through a helper too.
func TestReachableLiterals_aRequireHeavyCallIsAHeavyCheck(t *testing.T) {
	dir := t.TempDir()
	src := `package cuda

import (
	"testing"

	"github.com/townsendmerino/goinfer/internal/testgate"
)

func needsModel(t *testing.T) { testgate.RequireHeavy(t) }

func TestDirect(t *testing.T) { testgate.RequireHeavy(t) }
func TestHelper(t *testing.T) { needsModel(t) }
func TestNone(t *testing.T)   { _ = "../../testdata/llama-tiny" }
`
	if err := os.WriteFile(filepath.Join(dir, "x_test.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	refs, err := testReachableLiterals(dir, nil, filepath.Join(dir, "no-assets.json"))
	if err != nil {
		t.Fatal(err)
	}
	for n, want := range map[string]bool{"TestDirect": true, "TestHelper": true, "TestNone": false} {
		if got := reachesHeavyCheck(refs[n]); got != want {
			t.Errorf("%s: reaches a heavy check %v, want %v (literals %q)", n, got, want, refs[n])
		}
	}
}
