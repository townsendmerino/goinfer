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

	"github.com/townsendmerino/goinfer/internal/mem/probe"
)

// The decision census (MM1, docs/tasks/task-subsystems-2026-10.md, "MM1's decision census"): every memory decision the fit code
// makes, over every family fixture, option cell and machine, recorded so that the code that replaces it (mem.Solve and its callers)
// can be held to the same answers and the same lines, byte for byte. The golden holds one SHA-256 prefix per group (a decision on
// one family with one option set, over every machine reading); -census.dump writes the full text, to diff a red run against its base.

var (
	censusUpdate = flag.Bool("census.update", false, "rewrite testdata/mm1_census.json.gz for this host class")
	censusDump   = flag.String("census.dump", "", "write the full census, one group per line, to this file")
)

const censusGolden = "../testdata/mm1_census.json.gz"

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

// censusHostClass is the architecture and the bytes per element the loader charges for each weight mode on this CPU: what makes a
// weight figure differ between hosts (the int4 repacks).
func censusHostClass() string {
	var parts []string
	for _, q := range []string{"f32", "int8", "int4", "int4mix", "int8int8", "q4k", "bf16", "f16"} {
		m, _ := parseQuant(q)
		parts = append(parts, fmt.Sprintf("%s=%.6f", q, quantBytesPerElem(m)))
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

// censusFamilies lists every tracked tiny checkpoint: directories with a config.json and weights, and the tiny .gguf files.
func censusFamilies(t *testing.T) []censusFamily {
	t.Helper()
	var fams []censusFamily
	for _, root := range []string{"../testdata", "testdata", "selftestdata", "../internal/vision/testdata"} {
		ents, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, e := range ents {
			p := filepath.Join(root, e.Name())
			switch {
			case e.IsDir():
				ws, _ := filepath.Glob(filepath.Join(p, "*.safetensors"))
				if _, err := os.Stat(filepath.Join(p, "config.json")); err != nil || len(ws) == 0 {
					continue
				}
				fams = append(fams, censusFamily{name: strings.TrimPrefix(p, "../"), path: p, dir: true})
			case strings.HasSuffix(e.Name(), ".gguf"):
				fams = append(fams, censusFamily{name: strings.TrimPrefix(p, "../"), path: p})
			}
		}
	}
	split := filepath.Join("..", "testdata", "gguf-split", "glm-tiny-00001-of-00004.gguf")
	if _, err := os.Stat(split); err == nil {
		fams = append(fams, censusFamily{name: "testdata/gguf-split/glm-tiny-00001-of-00004.gguf", path: split})
	}
	sort.Slice(fams, func(i, j int) bool { return fams[i].name < fams[j].name })
	return fams
}

// longVariant copies a checkpoint directory into a temp dir with its window raised to 131,072 positions, so the auto-pin and floor
// branches are reachable on a tiny model. The weights are linked, not copied. The copy keeps the directory's own name, which the
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
		src, _ := filepath.Abs(filepath.Join(dir, e.Name()))
		if e.Name() != "config.json" {
			if err := os.Symlink(src, filepath.Join(out, e.Name())); err != nil {
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
	checkCensusCoverage(t, c1)
	class := censusHostClass()
	t.Logf("host class %q; %d groups; census %s", class, len(c1.groups), c1.total())
	if *censusDump != "" {
		writeCensusDump(t, c1, *censusDump)
	}
	golden := readCensusGolden(t)
	if *censusUpdate {
		golden.Classes[class] = censusGoldenClass{Total: c1.total(), Groups: c1.hashes()}
		writeCensusGolden(t, golden)
		t.Logf("wrote %s for host class %q", censusGolden, class)
		return
	}
	want, ok := golden.Classes[class]
	if !ok {
		t.Skipf("host class %q has no census entry: a capture run (census %s). Record it with -census.update on this host", class, c1.total())
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
	here := buildCensus(t).total()
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

type censusGoldenClass struct {
	Total  string            `json:"total"`
	Groups map[string]string `json:"groups"`
}

type censusGoldenFile struct {
	About   string                       `json:"_about"`
	Classes map[string]censusGoldenClass `json:"classes"`
}

func readCensusGolden(t *testing.T) censusGoldenFile {
	t.Helper()
	g := censusGoldenFile{Classes: map[string]censusGoldenClass{}}
	f, err := os.Open(censusGolden)
	if errors.Is(err, os.ErrNotExist) {
		return g
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
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	if g.Classes == nil {
		g.Classes = map[string]censusGoldenClass{}
	}
	return g
}

func writeCensusGolden(t *testing.T, g censusGoldenFile) {
	t.Helper()
	g.About = "MM1's decision census (docs/tasks/task-subsystems-2026-10.md, \"MM1's decision census\"): per host class, a SHA-256 prefix of " +
		"each group's full text, and one hash over every group. Written by go test ./decoder -run TestDecisionCensus -census.update."
	b, err := json.MarshalIndent(g, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(censusGolden)
	if err != nil {
		t.Fatal(err)
	}
	zw, _ := gzip.NewWriterLevel(f, gzip.BestCompression)
	zw.ModTime = zeroTime // a byte-identical file for identical content
	if _, err := zw.Write(append(b, '\n')); err != nil {
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
