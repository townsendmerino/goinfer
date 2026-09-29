package site

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The fit rule exists twice: FitFor (Go) renders the machines we have measured, and otherFit in assets/site.js answers
// for "something else". They must never disagree. This runs the script's function in Node over a grid of sizes,
// memories, GPUs and family flags, and compares every answer with FitFor's.
func TestFitRuleJSMatchesGo(t *testing.T) {
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node is not installed")
	}
	src, err := os.ReadFile(filepath.Join("assets", "site.js"))
	if err != nil {
		t.Fatal(err)
	}
	s := string(src)
	i := strings.Index(s, "function otherFit(")
	j := strings.Index(s, "function span(")
	if i < 0 || j < i {
		t.Fatal("otherFit is not where the test expects it in site.js")
	}
	fn := s[i:j]

	type in struct {
		Bytes    int64
		Moe, GPU bool
		Mem      float64
		Kind     string
	}
	var grid []in
	for _, gb := range []float64{0.3, 1, 2.4, 4.7, 7.4, 9, 11, 12.1, 14.4, 20, 26, 35, 45, 70, 120} {
		for _, mem := range []float64{8, 16, 32, 62, 64} {
			for _, kind := range []string{"apple", "nv8", "nv16", "none"} {
				for _, moe := range []bool{false, true} {
					for _, gpu := range []bool{false, true} {
						grid = append(grid, in{int64(gb * 1e9), moe, gpu, mem, kind})
					}
				}
			}
		}
	}
	data, _ := json.Marshal(grid)
	script := fn + fmt.Sprintf(`
const grid = %s;
console.log(JSON.stringify(grid.map(g => otherFit(g.Bytes, g.Moe, g.GPU, g.Mem, g.Kind))));`, data)
	cmd := exec.Command(node, "-e", script)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("node: %v", err)
	}
	var got [][2]string
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("node printed %q: %v", out, err)
	}
	if len(got) != len(grid) {
		t.Fatalf("node answered %d of %d", len(got), len(grid))
	}
	for k, g := range grid {
		want := FitFor(g.Bytes, g.Moe, g.GPU, g.Mem, g.Kind)
		if got[k][0] != want.Verdict || got[k][1] != want.Note {
			t.Fatalf("%+v: JS says %v, Go says %+v", g, got[k], want)
		}
	}
	t.Logf("%d cases agree", len(grid))
}

func TestSiteJSParses(t *testing.T) {
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node is not installed")
	}
	if out, err := exec.Command(node, "--check", filepath.Join("assets", "site.js")).CombinedOutput(); err != nil {
		t.Fatalf("site.js does not parse: %v\n%s", err, out)
	}
}
