//go:build gpu && goinfer_testhooks

package gpu

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

const (
	eModelDir    = "../testdata/gemma4-emodel-tiny"
	eModelGolden = "../testdata/gemma4_emodel_tiny_golden.json"
)

// eModelPrompt is the golden's prompt followed by its greedy continuation: 18 positions, so the KV-shared layers
// attend over history well past the sliding window (4).
func eModelPrompt(t *testing.T) []int {
	t.Helper()
	raw, err := os.ReadFile(eModelGolden)
	if err != nil {
		t.Skipf("no golden (%v): run scripts/pin_gemma4_emodel_tiny.py", err)
	}
	var g struct {
		PromptIDs       []int `json:"prompt_ids"`
		ContinuationIDs []int `json:"continuation_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return append(append([]int(nil), g.PromptIDs...), g.ContinuationIDs...)
}

// eModelG1 is what G-S12.1a grades (docs/tasks/task-multimodal-support-2026-10.md, S12): the WebGPU resident against
// the CPU, int4 on both sides, teacher-forced over eModelPrompt, with the CPU's own int4-against-float32 agreement
// as the envelope.
type eModelG1 struct {
	n, exact, gaps3  int
	worstTie         float64
	meanGPU, meanCPU float64
	minGPU           float64
	firstHardPos     int
	gpu              [][]float32
}

func runEModelG1(t *testing.T) eModelG1 {
	t.Helper()
	if _, err := os.Stat(eModelDir); err != nil {
		t.Skipf("no fixture (%s): run scripts/pin_gemma4_emodel_tiny.py", eModelDir)
	}
	if _, err := New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	prompt := eModelPrompt(t)
	mg, err := decoder.Load(eModelDir, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (webgpu int4): %v", err)
	}
	defer mg.Close()
	rf := mg.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("webgpu declined the tiny E-model")
	}
	rd, ok := rf.(*residentDecoder)
	if !ok {
		t.Fatalf("the resident is a %T, want *residentDecoder", rf)
	}
	shared := 0
	for i, l := range rd.rm.layers {
		if !l.kvShared {
			continue
		}
		shared++
		// The registered defect (5), a shared layer storing its own K and V, cannot be planted here: such a layer is built
		// with no K or V projection, so the plan has nothing to store. This is that statement, checked.
		if l.k != nil || l.v != nil {
			t.Fatalf("shared layer %d carries a K or V projection", i)
		}
	}
	if rd.rm.pleDim != 32 || shared != 2 || rd.rm.layers[4].gate.nRows() != 512 || rd.rm.layers[0].gate.nRows() != 256 {
		t.Fatalf("the resident did not take the E-model shape: pleDim %d, %d shared layers, FFN widths %d and %d",
			rd.rm.pleDim, shared, rd.rm.layers[0].gate.nRows(), rd.rm.layers[4].gate.nRows())
	}
	mc4, err := decoder.Load(eModelDir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu int4): %v", err)
	}
	defer mc4.Close()
	mcF, err := decoder.Load(eModelDir, decoder.Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("load (cpu f32): %v", err)
	}
	defer mcF.Close()

	c4, cF := mc4.NewCache(len(prompt)), mcF.NewCache(len(prompt))
	res := eModelG1{n: len(prompt), minGPU: 1, firstHardPos: -1}
	var sumG, sumC float64
	for i, tok := range prompt {
		cpu4, err := mc4.ForwardForTest(tok, c4)
		if err != nil {
			t.Fatalf("cpu int4 pos %d: %v", i, err)
		}
		cpuF, err := mcF.ForwardForTest(tok, cF)
		if err != nil {
			t.Fatalf("cpu f32 pos %d: %v", i, err)
		}
		l, err := rf.Forward(mg.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("webgpu pos %d: %v", i, err)
		}
		gpu := append([]float32(nil), l...)
		res.gpu = append(res.gpu, gpu)
		cg, _ := cosMaxAbs(cpu4, gpu)
		cc, _ := cosMaxAbs(cpuF, cpu4)
		sumG += cg
		sumC += cc
		res.minGPU = math.Min(res.minGPU, cg)
		ca, ga := argmax(cpu4), argmax(gpu)
		if ca == ga {
			res.exact++
		} else {
			// The near-tie rule: benign when the CPU's top logit and its logit for the index WebGPU chose are within 3%.
			gap := (float64(cpu4[ca]) - float64(cpu4[ga])) / (math.Abs(float64(cpu4[ca])) + 1e-30)
			res.worstTie = math.Max(res.worstTie, gap)
			if gap > 0.03 {
				res.gaps3++
				if res.firstHardPos < 0 {
					res.firstHardPos = i
				}
			}
		}
		t.Logf("  pos %2d WebGPU-vs-CPUint4 %.6f | CPUint4-vs-f32 %.6f  argmax cpu=%d webgpu=%d", i, cg, cc, ca, ga)
	}
	res.meanGPU, res.meanCPU = sumG/float64(len(prompt)), sumC/float64(len(prompt))
	t.Logf("E-model G1: exact argmax %d/%d, gaps over 3%%: %d, worst near-tie %.2f%% | mean WebGPU-vs-CPUint4 %.6f (min %.6f), mean CPUint4-vs-f32 %.6f",
		res.exact, res.n, res.gaps3, res.worstTie*100, res.meanGPU, res.minGPU, res.meanCPU)
	return res
}

// pass is the registered verdict: every argmax identical or a near-tie, and the mean WebGPU-against-CPU(int4) cosine
// at least the mean CPU(int4)-against-CPU(float32) cosine.
func (g eModelG1) pass() (bool, string) {
	if g.gaps3 > 0 {
		return false, "argmax diverges by more than 3% (not a near-tie)"
	}
	if g.meanGPU < g.meanCPU {
		return false, "mean WebGPU-vs-CPUint4 cosine below the CPU's own int4-vs-f32 envelope"
	}
	return true, ""
}

// TestGemma4EModel_webgpuResidentParity is G-S12.1a: the tiny E-model's per-layer embeddings, KV-shared layers and
// double-wide FFNs running resident on WebGPU against the CPU.
func TestGemma4EModel_webgpuResidentParity(t *testing.T) {
	g := runEModelG1(t)
	if ok, why := g.pass(); !ok {
		t.Errorf("G-S12.1a FAIL: %s (first hard divergence at pos %d)", why, g.firstHardPos)
	}
}

// TestGemma4EModel_webgpuPlantedDefects is G-S12.1b: each planted defect, alone, must turn G-S12.1a red. One that stays
// green means the fixture or the bar cannot see it. The registered list's (5) is absent because it cannot be built on
// this backend (runEModelG1 checks why).
func TestGemma4EModel_webgpuPlantedDefects(t *testing.T) {
	defects := []struct {
		name string
		set  func(bool)
	}{
		{"(1) PLE branch skipped", func(on bool) { g4SkipPLEForTest = on }},
		{"(2) PLE token-identity term zeroed", decoder.SetGemma4PLEDropTokenForTest},
		{"(3) shared-KV source one owning layer off", func(on bool) { g4KVSrcOffForTest = on }},
		{"(4) one FFN width for the model", func(on bool) { g4OneFFNWidthForTest = on }},
		{"(6) layer scalar dropped", func(on bool) { g4DropLayerScalarForTest = on }},
		{"(7) v_norm dropped", func(on bool) { g4DropVNormForTest = on }},
		{"(8) PLE inputs one layer off", func(on bool) { g4PLEOffByOneForTest = on }},
	}
	for _, d := range defects {
		t.Run(d.name, func(t *testing.T) {
			d.set(true)
			defer d.set(false)
			g := runEModelG1(t)
			ok, why := g.pass()
			t.Logf("G-S12.1b %s: G-S12.1a %s %s", d.name, map[bool]string{true: "GREEN", false: "RED"}[ok], why)
			if ok {
				t.Errorf("planted defect %s left the gate green", d.name)
			}
		})
	}
}

// TestGemma4EModel_webgpuRowLength is G-S12.1c: every resident entry refuses a row that is not Model.ResidentEmbedLen
// floats, the hidden state alone included.
func TestGemma4EModel_webgpuRowLength(t *testing.T) {
	if _, err := os.Stat(eModelDir); err != nil {
		t.Skipf("no fixture (%s)", eModelDir)
	}
	if _, err := New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	mg, err := decoder.Load(eModelDir, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer mg.Close()
	rd, ok := mg.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		t.Fatal("webgpu declined the tiny E-model")
	}
	row := mg.EmbedResidentForTest(1)
	if len(row) != mg.ResidentEmbedLen() || len(row) != 256+6*32 {
		t.Fatalf("the row is %d floats, ResidentEmbedLen %d, want 256 + 6·32", len(row), mg.ResidentEmbedLen())
	}
	for _, short := range [][]float32{row[:256], row[:len(row)-1], append(append([]float32(nil), row...), 0)} {
		entries := map[string]func() error{
			"Forward":         func() error { _, err := rd.Forward(short, 0); return err },
			"ForwardMRoPE":    func() error { _, err := rd.ForwardMRoPE(short, 0, 0); return err },
			"ForwardNoLogits": func() error { return rd.ForwardNoLogits(short, 0) },
			"ForwardSample":   func() error { _, err := rd.ForwardSample(short, 0, 1, 1, 0); return err },
		}
		for name, call := range entries {
			if err := call(); err == nil || !strings.Contains(err.Error(), "resident row") {
				t.Errorf("%s took a row of %d floats (want %d): err = %v", name, len(short), len(row), err)
			}
		}
	}
	if _, err := rd.Forward(row, 0); err != nil {
		t.Errorf("Forward refused a full row: %v", err)
	}
}

// TestGemma4EModel_webgpuPrefillAndBatch: the two entries that take several rows at once agree with the one-row
// forward on an E-model. PrefillLast, which an E-model's image and audio turns go through, must return the last
// position's logits of the sequential walk and leave the same cache behind; ForwardN must return each position's.
func TestGemma4EModel_webgpuPrefillAndBatch(t *testing.T) {
	if _, err := os.Stat(eModelDir); err != nil {
		t.Skipf("no fixture (%s)", eModelDir)
	}
	if _, err := New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	prompt := eModelPrompt(t)
	mg, err := decoder.Load(eModelDir, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer mg.Close()
	rd, ok := mg.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		t.Fatal("webgpu declined the tiny E-model")
	}
	rows := make([][]float32, len(prompt))
	for i, tok := range prompt {
		rows[i] = mg.EmbedResidentForTest(tok)
	}
	// The reference: one row at a time.
	seq := make([][]float32, len(rows))
	for i, row := range rows {
		l, err := rd.Forward(row, i)
		if err != nil {
			t.Fatalf("sequential pos %d: %v", i, err)
		}
		seq[i] = append([]float32(nil), l...)
	}
	same := func(what string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: %d logits, want %d", what, len(got), len(want))
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				c, m := cosMaxAbs(got, want)
				t.Errorf("%s differs from the sequential forward (logit %d: %v against %v; cosine %.8f, max |diff| %.4g)", what, i, got[i], want[i], c, m)
				return
			}
		}
	}
	const n = 12
	rd.Reset()
	last, err := rd.PrefillLast(t.Context(), rows[:n], 0)
	if err != nil {
		t.Fatalf("PrefillLast: %v", err)
	}
	same("PrefillLast's logits", last, seq[n-1])
	// The cache it left: the next positions, one row at a time, must match too.
	for i := n; i < len(rows); i++ {
		l, err := rd.Forward(rows[i], i)
		if err != nil {
			t.Fatalf("after PrefillLast, pos %d: %v", i, err)
		}
		same("the forward after PrefillLast", l, seq[i])
	}
	// ForwardN over positions n..n+3 after a sequential prefix.
	rd.Reset()
	for i := 0; i < n; i++ {
		if _, err := rd.Forward(rows[i], i); err != nil {
			t.Fatal(err)
		}
	}
	batch, err := rd.ForwardN(rows[n:n+4], n)
	if err != nil {
		t.Fatalf("ForwardN: %v", err)
	}
	for i, l := range batch {
		same("ForwardN's row", l, seq[n+i])
	}
}
