package decoder

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// D8A-G (docs/tasks/task-memory-campaign-2026-10.md, §4 MEM13, "D8a-build gates"): under Backend "cpu" the int4 layer
// tensors keep one row4 copy (quantizeLayerWM), and every CPU path computes what the canonical-plus-row4 load computes.

// d8aFixtures is every tiny fixture directory with a config.json.
func d8aFixtures(t *testing.T) []string {
	t.Helper()
	var dirs []string
	for _, root := range []string{"../testdata", "testdata"} {
		ents, err := os.ReadDir(root)
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range ents {
			d := filepath.Join(root, e.Name())
			if e.IsDir() && strings.HasSuffix(e.Name(), "-tiny") {
				if _, err := os.Stat(filepath.Join(d, "config.json")); err == nil {
					dirs = append(dirs, d)
				}
			}
		}
	}
	return dirs
}

// d8aInt4 is every int4 matrix of a loaded model the CPU reads: Weights.matmulWeights and Gemma 4's expert stacks.
func d8aInt4(w *Weights) []*linalg.WeightMat {
	var out []*linalg.WeightMat
	add := func(m *linalg.WeightMat) {
		if m.IsInt4() {
			out = append(out, m)
		}
	}
	for _, m := range w.matmulWeights() {
		add(m)
	}
	for i := range w.Layers {
		if g := w.Layers[i].gemma4moe; g != nil {
			for e := range g.expertsGateUp {
				add(&g.expertsGateUp[e])
				add(&g.expertsDown[e])
			}
		}
	}
	return out
}

type d8aPair struct {
	dir      string
	cpu, ref *Model
	moe      bool
}

// d8aLoad loads every fixture Load reads as a text model twice at int4: Backend "cpu" and Backend "" (canonical bytes
// kept, so canonical plus row4 on arm64).
func d8aLoad(t *testing.T) []d8aPair {
	t.Helper()
	if runtime.GOARCH != "arm64" || !linalg.Int4Row4Usable(4, 32, 32) {
		t.Skip("the row4 layout is arm64 with DotProd only; D8a changes nothing elsewhere")
	}
	var out []d8aPair
	for _, d := range d8aFixtures(t) {
		ref, rerr := Load(d, Options{Quant: "int4"})
		cpu, cerr := Load(d, Options{Backend: "cpu", Quant: "int4"})
		switch {
		case rerr != nil && cerr != nil:
			continue // not a model Load reads as text
		case rerr != nil || cerr != nil:
			t.Fatalf("%s: Backend \"\" err %v, Backend \"cpu\" err %v", d, rerr, cerr)
		}
		t.Cleanup(func() { ref.Close(); cpu.Close() })
		moe := false
		for i := range cpu.w.Layers {
			moe = moe || len(cpu.w.Layers[i].Experts) > 0 || cpu.w.Layers[i].gemma4moe != nil
		}
		out = append(out, d8aPair{dir: d, cpu: cpu, ref: ref, moe: moe})
	}
	nMoE := 0
	for _, p := range out {
		if p.moe {
			nMoE++
		}
	}
	fmt.Fprintf(os.Stderr, "D8A: %d families load, %d MoE\n", len(out), nMoE)
	if len(out) < 20 || nMoE < 4 {
		t.Fatalf("VOID: %d families loaded (%d MoE); the gate needs at least 20 and 4", len(out), nMoE)
	}
	return out
}

// TestD8a_cpuLoadKeepsOneInt4Copy is D8A-G1.
func TestD8a_cpuLoadKeepsOneInt4Copy(t *testing.T) {
	var changed, saved int64
	for _, p := range d8aLoad(t) {
		var cpuBytes, refBytes int64
		for _, m := range d8aInt4(p.cpu.w) {
			cpuBytes += wmBytes(m)
			if linalg.Int4Row4Usable(m.Rows(), m.Cols(), 32) && m.Int4Layout() != "row4" {
				t.Errorf("%s: a %dx%d int4 tensor is %q under Backend \"cpu\", want row4 alone", p.dir, m.Rows(), m.Cols(), m.Int4Layout())
			}
		}
		for _, m := range d8aInt4(p.ref.w) {
			refBytes += wmBytes(m)
		}
		l0c, l0r := &p.cpu.w.Layers[0], &p.ref.w.Layers[0]
		for _, pair := range [][2]*linalg.WeightMat{{&l0c.OProj, &l0r.OProj}, {&l0c.DownProj, &l0r.DownProj}} {
			if pair[0].IsInt4() && pair[1].Int4Layout() == "canonical+row4" && pair[0].Int4Layout() == "row4" {
				changed++
			}
		}
		saved += refBytes - cpuBytes
		fmt.Fprintf(os.Stderr, "D8A-G1 %s: int4 bytes %d (Backend \"\") -> %d (Backend \"cpu\")\n", p.dir, refBytes, cpuBytes)
	}
	if changed == 0 || saved <= 0 {
		t.Fatalf("no o_proj/down_proj went from canonical+row4 to row4 (changed %d, bytes saved %d): the gate checked nothing", changed, saved)
	}
}

// TestD8a_cpuLoadComputesTheSame is D8A-G2: the logits of a prompt token by token, and greedy tokens through Generate's
// batched CPU prefill, byte-equal between Backend "cpu" and Backend "".
func TestD8a_cpuLoadComputesTheSame(t *testing.T) {
	prompt := []int{3, 14, 15, 9, 26, 5, 35, 8}
	logits := func(m *Model) []float32 {
		cache := m.NewCache(len(prompt))
		var all []float32
		for _, id := range prompt {
			lg, _, err := m.ForwardCapture(id, cache, nil)
			if err != nil {
				t.Fatal(err)
			}
			all = append(all, lg...)
		}
		return all
	}
	gen := func(m *Model) []int {
		out, g := m.Generate(context.Background(), prompt, 8, SamplingParams{})
		var ids []int
		for id := range out {
			ids = append(ids, id)
		}
		if err := g.Err(); err != nil {
			t.Fatal(err)
		}
		return ids
	}
	for _, p := range d8aLoad(t) {
		if a, b := logits(p.cpu), logits(p.ref); !slices.Equal(a, b) {
			t.Errorf("%s: token-by-token logits differ between Backend \"cpu\" and Backend \"\"", p.dir)
		}
		if a, b := gen(p.cpu), gen(p.ref); !slices.Equal(a, b) {
			t.Errorf("%s: greedy tokens differ: cpu %v, ref %v", p.dir, a, b)
		}
	}
}

// TestD8a_writerRefusesARepackedOnlyTensor is D8A-G3.
func TestD8a_writerRefusesARepackedOnlyTensor(t *testing.T) {
	if runtime.GOARCH != "arm64" || !linalg.Int4Row4Usable(4, 32, 32) {
		t.Skip("a repacked-only tensor is built only on arm64 with DotProd")
	}
	m, err := Load("../testdata/llama-tiny", Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	b, err := SerializeWeights(m.Weights(), "llama-tiny")
	if err == nil || !strings.Contains(err.Error(), "row4 layout") {
		t.Fatalf("SerializeWeights of a Backend \"cpu\" int4 model: %d bytes, err %v; want the refusal naming the row4 layout", len(b), err)
	}
	if len(b) != 0 {
		t.Fatalf("the refusal returned %d bytes", len(b))
	}
}
