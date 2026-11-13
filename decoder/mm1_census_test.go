package decoder

import (
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/townsendmerino/goinfer/internal/runtime/mem/probe"
)

// The decision census (MM1, docs/completed/task-subsystems-2026-10.md, "MM1's decision census"): every memory decision the fit code
// makes, over every family fixture, option cell and machine, recorded so that the code that replaces it (mem.Solve and its callers)
// can be held to the same answers and the same lines, byte for byte. The golden holds one SHA-256 prefix per group (a decision on
// one family with one option set, over every machine reading); -census.dump writes the full text, to diff a red run against its base.

var (
	censusUpdate = flag.Bool("census.update", false, "rewrite this host class's golden, testdata/census/<class>.txt.gz")
	censusDump   = flag.String("census.dump", "", "write the full census, one group per line, to this file")
)

// censusGoldenDir holds one golden per host class (censusGoldenPath), so a PR that moves one class changes only that file.
const censusGoldenDir = "../testdata/census"

// censusMachine is one testdata/machines entry.
type censusMachine struct {
	Name           string `json:"name"`
	PhysicalBytes  int64  `json:"physical_bytes"`
	AvailableBytes int64  `json:"available_bytes"`
	CUDAFreeBytes  *int64 `json:"cuda_free_bytes"`
	MetalBudget    *int64 `json:"metal_budget_bytes"`
}

func censusMachines(t *testing.T) []censusMachine {
	t.Helper()
	paths, err := filepath.Glob("../testdata/machines/*.json")
	if err != nil || len(paths) == 0 {
		t.Fatalf("no machine zoo: %v", err)
	}
	var ms []censusMachine
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		var m censusMachine
		if err := json.Unmarshal(b, &m); err != nil || m.Name == "" {
			t.Fatalf("%s: %v", p, err)
		}
		ms = append(ms, m)
	}
	sort.Slice(ms, func(i, j int) bool { return ms[i].Name < ms[j].Name })
	return ms
}

// census is the recorded decisions: group key to the group's canonical text, and the outcomes reached per decision.
type census struct {
	mu       sync.Mutex
	groups   map[string]string
	outcomes map[string]map[string]bool
}

func (c *census) add(key string, lines []string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, dup := c.groups[key]; dup {
		panic("census: duplicate group " + key)
	}
	c.groups[key] = strings.Join(lines, "\n")
}

func (c *census) reached(decision, outcome string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.outcomes[decision] == nil {
		c.outcomes[decision] = map[string]bool{}
	}
	c.outcomes[decision][outcome] = true
}

// hashes is the per-group SHA-256 prefix the golden stores.
func (c *census) hashes() map[string]string {
	out := make(map[string]string, len(c.groups))
	for k, v := range c.groups {
		h := sha256.Sum256([]byte(v))
		out[k] = hex.EncodeToString(h[:8])
	}
	return out
}

// total is one hash over every group, in key order.
func (c *census) total() string {
	keys := make([]string, 0, len(c.groups))
	for k := range c.groups {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	h := sha256.New()
	for _, k := range keys {
		fmt.Fprintf(h, "%s\x00%s\x00", k, c.groups[k])
	}
	return hex.EncodeToString(h.Sum(nil))
}

// censusHostClass is the architecture, the bytes per element the loader charges for each weight mode on this CPU (the int4
// repacks), and, where WebGPU is not Vulkan, that: what makes a plan differ between hosts. darwin/arm64 and linux/arm64 share
// the rest, but only Vulkan's plan prices gpu-allocator's blocks (webgpuVulkan).
func censusHostClass() string {
	var parts []string
	for _, q := range []string{"f32", "int8", "int4", "int4mix", "int8int8", "q4k", "bf16", "f16"} {
		m, _ := parseQuant(q)
		parts = append(parts, fmt.Sprintf("%s=%.6f", q, quantBytesPerElem(m)))
	}
	if !webgpuVulkan {
		parts = append(parts, "webgpu=metal")
	}
	return runtime.GOARCH + " " + strings.Join(parts, " ")
}

// changePoints returns the readings in [lo, hi] at which outcome changes, each as the lowest reading of the new outcome, at most
// max of them, by bisection. outcome must be a function of the reading alone.
func changePoints(lo, hi int64, max int, outcome func(int64) string) []int64 {
	var out []int64
	var walk func(a, b int64, oa, ob string)
	walk = func(a, b int64, oa, ob string) {
		if oa == ob || len(out) >= max {
			return
		}
		if b-a <= 1 {
			out = append(out, b)
			return
		}
		m := a + (b-a)/2
		om := outcome(m)
		walk(a, m, oa, om)
		walk(m, b, om, ob)
	}
	walk(lo, hi, outcome(lo), outcome(hi))
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// changePointsBelow is changePoints over [lo, hi] when the outcome at hi already equals the outcome at the top of every reading
// (censusTop), so no change point lies above hi; otherwise over [lo, censusTop]. hi is a cell's own bound, a multiple of its need,
// which keeps the search short.
func changePointsBelow(lo, hi int64, max int, outcome func(int64) string) []int64 {
	if hi <= lo || hi >= censusTop || outcome(hi) != outcome(censusTop) {
		hi = censusTop
	}
	return changePoints(lo, hi, max, outcome)
}

// edgeReadings is the zoo's readings plus, for each change point, the point and one byte under it, deduplicated and sorted.
func edgeReadings(zoo []int64, edges []int64) []int64 {
	set := map[int64]bool{}
	for _, v := range zoo {
		set[v] = true
	}
	for _, e := range edges {
		set[e], set[e-1] = true, true
	}
	out := make([]int64, 0, len(set))
	for v := range set {
		if v >= 0 {
			out = append(out, v)
		}
	}
	slices.Sort(out)
	return out
}

// censusFamily is one tracked checkpoint: its path, and whether it is a directory.
type censusFamily struct {
	name, path string
	dir        bool
}

// censusFamilies is censusFixtures, each of which must be on disk: a missing one is a red run, not a smaller census.
func censusFamilies(t *testing.T) []censusFamily {
	t.Helper()
	var fams []censusFamily
	for _, p := range censusFixtures {
		files, err := censusFixtureFiles(p)
		if err != nil {
			t.Fatalf("census fixture %v", err)
		}
		for _, f := range files {
			if _, err := os.Stat(f); err != nil {
				t.Fatalf("census fixture %s: %v", p, err)
			}
		}
		fams = append(fams, censusFamily{name: censusName(p), path: p, dir: !strings.HasSuffix(p, ".gguf")})
	}
	sort.Slice(fams, func(i, j int) bool { return fams[i].name < fams[j].name })
	return fams
}

// censusName is a fixture's name in a group key: its path from the repository root, with forward slashes on every OS.
func censusName(p string) string { return strings.TrimPrefix(filepath.ToSlash(p), "../") }

// linkOrCopy hard-links src at dst (no privilege needed on Windows, unlike a symlink), and copies it where the filesystem
// refuses the link.
func linkOrCopy(src, dst string) error {
	if err := os.Link(src, dst); err == nil {
		return nil
	}
	b, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, b, 0o644)
}

// longVariant copies a checkpoint directory into a temp dir with its window raised to 131,072 positions, so the auto-pin and floor
// branches are reachable on a tiny model. The weights are hard-linked where the filesystem allows it, else copied. The copy keeps the directory's own name, which the
// fit guard's lines print.
func longVariant(t *testing.T, dir string) string {
	t.Helper()
	out := filepath.Join(t.TempDir(), filepath.Base(dir))
	if err := os.Mkdir(out, 0o755); err != nil {
		t.Fatal(err)
	}
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range ents {
		if e.IsDir() {
			continue // a downloader's cache folder (.cache) holds no weights and can be neither hard-linked nor read as a file
		}
		src, _ := filepath.Abs(filepath.Join(dir, e.Name()))
		if e.Name() != "config.json" {
			if err := linkOrCopy(src, filepath.Join(out, e.Name())); err != nil {
				t.Fatal(err)
			}
			continue
		}
		b, err := os.ReadFile(src)
		if err != nil {
			t.Fatal(err)
		}
		var raw map[string]any
		if err := json.Unmarshal(b, &raw); err != nil {
			t.Fatal(err)
		}
		raw["max_position_embeddings"] = 131072
		if tc, ok := raw["text_config"].(map[string]any); ok {
			tc["max_position_embeddings"] = 131072
		}
		if b, err = json.Marshal(raw); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(out, "config.json"), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

func censusSetAvail(fake *probe.Fake, phys, avail int64) {
	fake.Set(func(f *probe.Fake) { f.PhysicalBytes, f.AvailableBytes = phys, avail })
	resetAvailProbeCache()
}

// d1Outcome names a resident-guard outcome, for coverage and for the change-point search.
func d1Outcome(f fitCheck) string {
	pin, decline := f.decide()
	switch {
	case decline != nil && f.pinned:
		return "refuses a pin"
	case decline != nil:
		return "refuses at the floor"
	case pin > 0:
		return "auto-pins"
	case f.noGuard || !f.known():
		return "proceeds unpriced"
	case f.ratio() >= fitWarnRatio:
		return "warns"
	default:
		return "fits"
	}
}

// d1Line is everything the resident guard says of f: its decision, its numbers, and the line guardFit prints or the refusal.
func d1Line(f fitCheck) string {
	pin, decline := f.decide()
	line := ""
	switch {
	case decline != nil:
		line = "refusal: " + decline.Error() + fmt.Sprintf(" [streamable %v]", decline.DenseStreamable)
	case pin > 0:
		line = "prints: " + f.capNote(pin)
	case f.known() && !f.noGuard && f.ratio() >= fitWarnRatio:
		line = "prints: " + f.warning()
	}
	return fmt.Sprintf("avail %d: %s pin %d known %v need %d budget %d ratio %.6f | %s | arithmetic: %s",
		f.availBytes, d1Outcome(f), pin, f.known(), f.need(), f.budget(), f.ratio(), line, f.arithmetic())
}

// TestDecisionCensus records the census and holds it to the golden for this host class (G-MM1-C1 to C3).
func TestDecisionCensus(t *testing.T) {
	c1 := buildCensus(t)
	c2 := buildCensus(t)
	if c1.total() != c2.total() {
		var diff []string
		for k, v := range c1.groups {
			if w := c2.groups[k]; w != v && len(diff) < 3 {
				diff = append(diff, fmt.Sprintf("%s\n  first:  %.600s\n  second: %.600s", k, v, w))
			}
		}
		t.Fatalf("two runs in one process differ (G-MM1-C2): %s, %s\n%s", c1.total(), c2.total(), strings.Join(diff, "\n"))
	}
	censusTotalHere = c1.total()
	checkCensusCoverage(t, c1)
	class := censusHostClass()
	t.Logf("host class %q; %d groups; census %s", class, len(c1.groups), c1.total())
	if *censusDump != "" {
		writeCensusDump(t, c1, *censusDump)
	}
	if *censusUpdate {
		writeCensusGolden(t, censusGoldenPath(), class, censusGoldenClass{Total: c1.total(), Groups: c1.hashes()})
		t.Logf("wrote %s for host class %q", censusGoldenPath(), class)
		return
	}
	want, recorded, found := readCensusGolden(t, censusGoldenPath())
	skip, err := checkCensusGoldenClass(found, recorded, class, censusGoldenPath())
	if err != nil {
		t.Fatal(err)
	}
	if skip {
		t.Skipf("host class %q has no census golden (%s): a capture run (census %s). Record it with -census.update on this host", class, censusGoldenPath(), c1.total())
	}
	if want.Total == c1.total() {
		return
	}
	got := c1.hashes()
	var diffs []string
	for k, h := range want.Groups {
		switch g, ok := got[k]; {
		case !ok:
			diffs = append(diffs, "missing: "+k)
		case g != h:
			diffs = append(diffs, "changed: "+k)
		}
	}
	for k := range got {
		if _, ok := want.Groups[k]; !ok {
			diffs = append(diffs, "added: "+k)
		}
	}
	sort.Strings(diffs)
	t.Errorf("the census differs from the golden in %d group(s); -census.dump writes the full text to diff against the base:\n  %s",
		len(diffs), strings.Join(diffs[:min(len(diffs), 40)], "\n  "))
}

// censusTotalHere is the total of the census TestDecisionCensus built in this process, once it has. A whole-package run has both tests in
// one process, so the second need not pay for a third build; a census is minutes on CI's slower hosts.
var censusTotalHere string

// TestDecisionCensus_secondProcess (G-MM1-C2) runs the census in a fresh process and requires the same total as this one.
func TestDecisionCensus_secondProcess(t *testing.T) {
	if os.Getenv("GOINFER_CENSUS_CHILD") == "1" {
		fmt.Printf("census-total %s\n", buildCensus(t).total())
		return
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	here := censusTotalHere
	if here == "" {
		here = buildCensus(t).total()
	}
	cmd := newCensusChild(exe)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("child: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "census-total "+here) {
		t.Errorf("a second process computed another census (want %s):\n%s", here, lastLines(string(out), 5))
	}
}

func lastLines(s string, n int) string {
	ls := strings.Split(strings.TrimSpace(s), "\n")
	return strings.Join(ls[max(0, len(ls)-n):], "\n")
}

// censusGoldenClass is one host class's golden: a SHA-256 prefix per group, and one hash over every group.
type censusGoldenClass struct {
	Total  string
	Groups map[string]string
}

// censusGoldenPath is this host's golden: testdata/census/<GOARCH>.txt.gz, with "-metal" where the host plans WebGPU as Metal (darwin).
// The file's first line names the full host class (censusHostClass), so a host whose quantization sizes differ from the recorded
// class's reads no golden instead of a wrong one.
func censusGoldenPath() string { return censusGoldenPathFor(runtime.GOARCH, !webgpuVulkan) }

func censusGoldenPathFor(arch string, metal bool) string {
	if metal {
		arch += "-metal"
	}
	return filepath.Join(censusGoldenDir, arch+".txt.gz")
}

// The golden is text, one group a line in key order: "# class <host class>", "# total <hash>", then "<key>\t<hash>". A key holds
// neither a tab nor a newline (writeCensusGolden checks).
func readCensusGolden(t *testing.T, path string) (c censusGoldenClass, recordedClass string, found bool) {
	t.Helper()
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return censusGoldenClass{}, "", false
	}
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(zr)
	if err != nil {
		t.Fatal(err)
	}
	return parseCensusGolden(t, path, string(b))
}

func parseCensusGolden(t *testing.T, path, text string) (censusGoldenClass, string, bool) {
	t.Helper()
	c, class, err := decodeCensusGolden(text)
	if err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	return c, class, true
}

// decodeCensusGolden reads the golden's text. The early return on a matching total would hide a corrupt body, so a missing header, a
// line with no tab, and a key repeated or out of order are errors.
func decodeCensusGolden(text string) (censusGoldenClass, string, error) {
	lines := strings.Split(strings.TrimSuffix(text, "\n"), "\n")
	if len(lines) < 2 || !strings.HasPrefix(lines[0], "# class ") || !strings.HasPrefix(lines[1], "# total ") {
		return censusGoldenClass{}, "", errors.New("no class and total header")
	}
	c := censusGoldenClass{Total: strings.TrimPrefix(lines[1], "# total "), Groups: make(map[string]string, len(lines))}
	prev := ""
	for _, l := range lines[2:] {
		k, h, ok := strings.Cut(l, "\t")
		if !ok {
			return censusGoldenClass{}, "", fmt.Errorf("a line with no tab: %q", l)
		}
		if prev != "" && k <= prev {
			return censusGoldenClass{}, "", fmt.Errorf("group %q is out of order or repeated after %q", k, prev)
		}
		prev = k
		c.Groups[k] = h
	}
	return c, strings.TrimPrefix(lines[0], "# class "), nil
}

// checkCensusGoldenClass decides what this host's golden file means. No file: an unrecorded host class, a capture run (skip). A file
// recorded for another class string than this host's (a changed quantization size, say) is an error naming both, not a skip: a skip is
// not a pass, and the golden would otherwise stop guarding this host unseen.
func checkCensusGoldenClass(found bool, recorded, class, path string) (skip bool, err error) {
	if !found {
		return true, nil
	}
	if recorded != class {
		return false, fmt.Errorf("%s was recorded for host class %q, but this host is %q: re-record it with -census.update (or fix the class string if the quantization sizes changed by mistake)", path, recorded, class)
	}
	return false, nil
}

func writeCensusGolden(t *testing.T, path, class string, c censusGoldenClass) {
	t.Helper()
	keys := make([]string, 0, len(c.Groups))
	for k := range c.Groups {
		if strings.ContainsAny(k, "\t\n") {
			t.Fatalf("a group key holds a tab or a newline: %q", k)
		}
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var sb strings.Builder
	sb.WriteString("# class " + class + "\n# total " + c.Total + "\n")
	for _, k := range keys {
		sb.WriteString(k + "\t" + c.Groups[k] + "\n")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	zw, _ := gzip.NewWriterLevel(f, gzip.BestCompression)
	zw.ModTime = zeroTime // a byte-identical file for identical content
	if _, err := zw.Write([]byte(sb.String())); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func writeCensusDump(t *testing.T, c *census, path string) {
	t.Helper()
	keys := make([]string, 0, len(c.groups))
	for k := range c.groups {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var sb strings.Builder
	for _, k := range keys {
		sb.WriteString("## " + k + "\n" + c.groups[k] + "\n")
	}
	if err := os.WriteFile(path, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}

// checkCensusCoverage is G-MM1-C1: every outcome of every decision reached at least once.
func checkCensusCoverage(t *testing.T, c *census) {
	t.Helper()
	want := map[string][]string{
		"D1": {"proceeds unpriced", "fits", "warns", "auto-pins", "refuses a pin", "refuses at the floor", "refuses streamable", "cuda build peak"},
		"D2": {"proceeds unpriced", "fits", "auto-pins", "refuses a pin", "refuses at the floor"},
		"D3": {"cpu resident", "cpu weight-paged", "cuda resident", "cuda decline", "metal resident", "metal decline", "webgpu resident", "webgpu decline", "shrinks the context"},
		"D4": {"off or unknown", "fits at the candidate", "shrinks", "floor"},
		"D5": {"admitted", "refused", "unknown proceeds"},
		"D6": {"explicit", "auto from available", "unknown"},
		"D7": {"cpu", "cuda", "metal", "webgpu"},
	}
	for d, outs := range want {
		for _, o := range outs {
			if !c.outcomes[d][o] {
				t.Errorf("G-MM1-C1: %s never reached %q", d, o)
			}
		}
	}
}

// TestCensusGolden_roundTripAndClassGuard holds the per-class golden's encoding: a write reads back to the same groups, the lines are in
// key order, and a golden recorded for another class string than this host's fails naming both instead of skipping.
func TestCensusGolden_roundTripAndClassGuard(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x", "g.txt.gz")
	in := censusGoldenClass{Total: "abc", Groups: map[string]string{"D2 b | cpu": "02", "D1 a | cpu": "01", "D1 a | cuda": "03"}}
	writeCensusGolden(t, path, "amd64 f32=4", in)
	out, recorded, ok := readCensusGolden(t, path)
	if !ok || recorded != "amd64 f32=4" || out.Total != "abc" || len(out.Groups) != 3 || out.Groups["D1 a | cuda"] != "03" {
		t.Fatalf("round trip: %v %+v", ok, out)
	}
	f, _ := os.Open(path)
	zr, _ := gzip.NewReader(f)
	b, _ := io.ReadAll(zr)
	f.Close()
	want := "# class amd64 f32=4\n# total abc\nD1 a | cpu\t01\nD1 a | cuda\t03\nD2 b | cpu\t02\n"
	if string(b) != want {
		t.Errorf("encoding:\n%s\nwant:\n%s", b, want)
	}
	for name, text := range map[string]string{
		"repeated":     "# class c\n# total t\nA\t1\nA\t2\n",
		"out of order": "# class c\n# total t\nB\t1\nA\t2\n",
		"no tab":       "# class c\n# total t\nA1\n",
		"no header":    "A\t1\n",
	} {
		if _, _, err := decodeCensusGolden(text); err == nil {
			t.Errorf("a %s golden decoded without an error", name)
		}
	}
	// The decision on a file: absent is a skip; present for this class passes; present for another class string is an error that
	// names both (a skip would let the golden stop guarding this host).
	if skip, err := checkCensusGoldenClass(false, "", "amd64 f32=4", path); !skip || err != nil {
		t.Errorf("no file: skip %v, err %v; want a skip", skip, err)
	}
	if skip, err := checkCensusGoldenClass(true, "amd64 f32=4", "amd64 f32=4", path); skip || err != nil {
		t.Errorf("a matching file: skip %v, err %v; want neither", skip, err)
	}
	skip, err := checkCensusGoldenClass(true, "amd64 f32=4", "amd64 f32=4 int4=1.125", path)
	if skip || err == nil || !strings.Contains(err.Error(), "amd64 f32=4 int4=1.125") || !strings.Contains(err.Error(), `"amd64 f32=4"`) {
		t.Errorf("a file for another class string: skip %v, err %v; want an error naming both", skip, err)
	}
}
