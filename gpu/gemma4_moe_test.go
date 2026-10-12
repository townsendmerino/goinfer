//go:build gpu && goinfer_testhooks

package gpu

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// The gates of Gemma 4's parallel dense and MoE feed-forward on WebGPU (docs/tasks/task-multimodal-support-2026-10.md,
// S12.2), on the tiny enable_moe_block fixtures, int4 on both sides.

// g4MoEFixtures are the tiny fixtures; the first is committed, the other two exist only where they were generated.
var g4MoEFixtures = []string{"../testdata/gemma4-moe-kv-tiny", "../testdata/gemma4-moe-tiny", "../testdata/gemma4-moe-unified-tiny"}

// g4MoEPrompt is the fixed ten-token prompt of the gate.
var g4MoEPrompt = []int{1, 7, 42, 100, 5, 200, 13, 88, 3, 150}

// g4MoERead is one fixture's reading: the lowest cosine of a layer's output residual at position 0 against the CPU's,
// and over the prompt the argmax agreement and the lowest logit cosine.
type g4MoERead struct {
	layers, moeLayers  int
	lowestLayer        float64
	exact, n, hardGaps int
	worstTie           float64
	minCos             float64
	nan                bool
}

func (r g4MoERead) pass() bool {
	return !r.nan && r.lowestLayer >= 0.9999 && r.hardGaps == 0 && r.minCos >= 0.995
}

// runG4MoE builds the WebGPU resident for one fixture (with the planted defect d, 0 for none) and reads it against the
// CPU. ok is false when the fixture or an adapter is missing.
func runG4MoE(t *testing.T, dir string, d int) (res g4MoERead, ok bool) {
	t.Helper()
	if _, err := os.Stat(dir); err != nil {
		return res, false
	}
	if c, err := New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	} else {
		c.Close()
	}
	t.Setenv("GOINFER_GPU_CAPTURE", "1")
	g4MoEDefectForTest = d
	defer func() { g4MoEDefectForTest = 0 }()
	mg, err := decoder.Load(dir, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("%s: load (webgpu int4): %v", dir, err)
	}
	defer mg.Close()
	rf := mg.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("%s: webgpu declined the fixture", dir)
	}
	rd, isRD := rf.(*residentDecoder)
	if !isRD {
		t.Fatalf("the resident is a %T, want *residentDecoder", rf)
	}
	for _, l := range rd.rm.layers {
		if l.g4moe != nil {
			res.moeLayers++
		}
	}
	res.layers = len(rd.rm.layers)
	if res.moeLayers == 0 || !rd.rm.g4moe {
		t.Fatalf("%s: the resident has no Gemma 4 MoE layer: the block is not what ran", dir)
	}
	mc, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("%s: load (cpu int4): %v", dir, err)
	}
	defer mc.Close()

	res.minCos, res.lowestLayer, res.n = 1, 1, len(g4MoEPrompt)
	cache := mc.NewCache(len(g4MoEPrompt))
	decoder.SetGemma4HiddenCaptureForTest(true)
	defer decoder.SetGemma4HiddenCaptureForTest(false)
	for i, tok := range g4MoEPrompt {
		cpu, err := mc.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu pos %d: %v", i, err)
		}
		l, err := rf.Forward(mg.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("webgpu pos %d: %v", i, err)
		}
		gpu := append([]float32(nil), l...)
		for _, v := range gpu {
			if v != v {
				res.nan = true
			}
		}
		if i == 0 { // the residual after each layer, against the CPU's capture [post-embed, after layer 0, ...]
			hidden := decoder.Gemma4HiddenCaptureForTest()
			decoder.SetGemma4HiddenCaptureForTest(false)
			_, _, mlp, err := rd.runner.ReadCapture()
			if err != nil {
				t.Fatalf("ReadCapture: %v", err)
			}
			if len(hidden) != len(mlp)+1 {
				t.Fatalf("%d captured CPU states for %d device layers", len(hidden), len(mlp))
			}
			for L := range mlp {
				c, _ := cosMaxAbs(hidden[L+1], mlp[L])
				if c != c {
					res.nan = true
				}
				res.lowestLayer = math.Min(res.lowestLayer, c)
				if d == 0 {
					t.Logf("  %s after layer %d: cosine %.7f", dir, L, c)
				}
			}
		}
		c, _ := cosMaxAbs(cpu, gpu)
		res.minCos = math.Min(res.minCos, c)
		if ca, ga := argmax(cpu), argmax(gpu); ca == ga {
			res.exact++
		} else {
			gap := (float64(cpu[ca]) - float64(cpu[ga])) / (math.Abs(float64(cpu[ca])) + 1e-30)
			res.worstTie = math.Max(res.worstTie, gap)
			if gap > 0.03 {
				res.hardGaps++
			}
		}
	}
	return res, true
}

// TestGemma4MoE_webgpuResidentParity is G-S12.2a: on each tiny fixture present, the residual after each layer at
// position 0 at cosine >= 0.9999 against the CPU's, and over the prompt the argmax identical or a first difference at
// a near-tie under 3% with the lowest logit cosine >= 0.995.
func TestGemma4MoE_webgpuResidentParity(t *testing.T) {
	ran := 0
	for _, dir := range g4MoEFixtures {
		r, ok := runG4MoE(t, dir, 0)
		if !ok {
			t.Logf("%s: not present, skipped", dir)
			continue
		}
		ran++
		t.Logf("%s: %d layers (%d with the MoE block); lowest layer cosine %.7f; exact argmax %d/%d, gaps over 3%%: %d (worst near-tie %.2f%%); lowest logit cosine %.7f",
			dir, r.layers, r.moeLayers, r.lowestLayer, r.exact, r.n, r.hardGaps, r.worstTie*100, r.minCos)
		if !r.pass() {
			t.Errorf("%s: the block does not match the CPU (NaN %v, lowest layer %.7f, hard gaps %d, lowest logit cosine %.7f)", dir, r.nan, r.lowestLayer, r.hardGaps, r.minCos)
		}
	}
	if ran == 0 {
		t.Fatal("no Gemma 4 MoE fixture was read: testdata/gemma4-moe-kv-tiny is committed and must be there")
	}
}

// TestGemma4MoE_webgpuPlantedDefects is G-S12.2b: each planted defect, alone, must put some fixture's reading outside
// the gate.
func TestGemma4MoE_webgpuPlantedDefects(t *testing.T) {
	names := []string{
		1: "the per-expert scale skipped",
		2: "post_ffn_norm_1 skipped",
		3: "the expert branch fed pre_ffn_norm",
		4: "the joint post_ffn_norm skipped",
		5: "the expert sum not cleared between tokens",
		6: "the gate and up halves exchanged",
		7: "the router fed the dense branch's weighted norm",
	}
	for d := 1; d < len(names); d++ {
		red := false
		for _, dir := range g4MoEFixtures {
			r, ok := runG4MoE(t, dir, d)
			if !ok {
				continue
			}
			t.Logf("planted (%d) %s, %s: lowest layer cosine %.6f, hard gaps %d, lowest logit cosine %.6f", d, names[d], dir, r.lowestLayer, r.hardGaps, r.minCos)
			red = red || !r.pass()
		}
		if !red {
			t.Errorf("planted defect (%d) %s left the gate green on every fixture", d, names[d])
		}
	}
}

// TestGemma4MoE_webgpuPrefill is G-S12.2c's prefill half: a prompt prefilled through the resident and then decoded
// gives the logits the same prompt gives walked a token at a time.
func TestGemma4MoE_webgpuPrefill(t *testing.T) {
	dir := g4MoEFixtures[0]
	if c, err := New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	} else {
		c.Close()
	}
	load := func() (*decoder.Model, decoder.ResidentForward) {
		m, err := decoder.Load(dir, decoder.Options{Backend: "webgpu", Quant: "int4"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		rf := m.ResidentForwardForTest()
		if rf == nil {
			t.Fatal("webgpu declined the fixture")
		}
		return m, rf
	}
	ma, ra := load()
	defer ma.Close()
	mb, rb := load()
	defer mb.Close()
	n := len(g4MoEPrompt)
	var walked []float32
	rows := make([][]float32, n-1)
	for i, tok := range g4MoEPrompt {
		l, err := ra.Forward(ma.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatal(err)
		}
		walked = append(walked[:0], l...)
		if i < n-1 {
			rows[i] = append([]float32(nil), mb.EmbedResidentForTest(tok)...)
		}
	}
	pf, isPF := rb.(decoder.Prefiller)
	if !isPF {
		t.Fatalf("the resident (%T) is not a Prefiller", rb)
	}
	if _, err := pf.PrefillLast(t.Context(), rows, 0); err != nil {
		t.Fatalf("PrefillLast: %v", err)
	}
	l, err := rb.Forward(mb.EmbedResidentForTest(g4MoEPrompt[n-1]), n-1)
	if err != nil {
		t.Fatal(err)
	}
	c, m := cosMaxAbs(walked, l)
	t.Logf("last token's logits, prefilled then decoded against walked: cosine %.9f, largest difference %.3e", c, m)
	if !(c >= 0.999999) {
		t.Errorf("a prefilled prompt's next logits differ from the walked prompt's (cosine %.9f)", c)
	}
}
