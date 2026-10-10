package whisper

import (
	"archive/zip"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"path/filepath"
	"slices"
	"testing"
)

// G-S14f1 of docs/tasks/task-multimodal-support-2026-10.md: the decoder against transformers' own WhisperDecoder on the tiny random-weight Whisper (testdata/whisper-tiny-rand), given the
// fixture's encoder output for each of its two clips. Goldens: scripts/pin_whisper_tiny.py (golden.zip, the encoder output) and scripts/pin_whisper_decoder_tiny.py (dec_golden.zip).

func tinyDir() string { return filepath.Join("..", "..", "testdata", "whisper-tiny-rand") }

func readZip(t *testing.T, name string) map[string][]byte {
	t.Helper()
	zr, err := zip.OpenReader(filepath.Join(tinyDir(), name))
	if err != nil {
		t.Skipf("no golden %s: %v", name, err)
	}
	defer zr.Close()
	out := map[string][]byte{}
	for _, f := range zr.File {
		rc, err := f.Open()
		if err != nil {
			t.Fatal(err)
		}
		b, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			t.Fatal(err)
		}
		out[f.Name] = b
	}
	return out
}

func f32s(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func cos(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	return ab / math.Sqrt(aa*bb)
}

type tinyCase struct {
	name   string
	enc    []float32
	logits []float32
	greedy []int
}

func tinyCases(t *testing.T) (ids []int, start int, cases []tinyCase) {
	t.Helper()
	enc, dec := readZip(t, "golden.zip"), readZip(t, "dec_golden.zip")
	var meta struct {
		IDs   []int `json:"ids"`
		Start int   `json:"start"`
	}
	if err := json.Unmarshal(dec["meta.json"], &meta); err != nil {
		t.Fatal(err)
	}
	for _, n := range []string{"clip6s5", "clip25s"} {
		var g []int
		if err := json.Unmarshal(dec[n+".greedy.json"], &g); err != nil {
			t.Fatal(err)
		}
		cases = append(cases, tinyCase{n, f32s(enc[n+".enc.f32"]), f32s(dec[n+".logits.f32"]), g})
	}
	return meta.IDs, meta.Start, cases
}

func loadTiny(t *testing.T, defect int) *Decoder {
	t.Helper()
	d, err := Load(tinyDir())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	d.defect = defect
	return d
}

// run reports, for one defect setting, how the decoder differs from transformers: the worst per-position cosine and max |diff| of the teacher-forced logits, the largest difference between the
// incremental (one token at a time) logits and the full forward's, and whether the greedy ids equal transformers'.
func run(t *testing.T, defect int) (worstCos, maxAbs, incrDiff float64, greedyOK bool) {
	t.Helper()
	ids, start, cases := tinyCases(t)
	d := loadTiny(t, defect)
	V := d.Cfg.Vocab
	worstCos, greedyOK = 1, true
	for _, c := range cases {
		st, err := d.NewState(c.enc)
		if err != nil {
			t.Fatal(err)
		}
		full, err := st.Forward(ids, true)
		if err != nil {
			t.Fatal(err)
		}
		for i := range full {
			ref := c.logits[i*V : (i+1)*V]
			worstCos = math.Min(worstCos, cos(full[i], ref))
			for j := range ref {
				maxAbs = math.Max(maxAbs, math.Abs(float64(full[i][j]-ref[j])))
			}
		}
		st2, _ := d.NewState(c.enc)
		for i, id := range ids {
			one, err := st2.Forward([]int{id}, true)
			if err != nil {
				t.Fatal(err)
			}
			for j := range one[0] {
				incrDiff = math.Max(incrDiff, math.Abs(float64(one[0][j]-full[i][j])))
			}
		}
		// greedy from the decoder start token, no processors
		st3, _ := d.NewState(c.enc)
		in, got := []int{start}, []int{}
		for range len(c.greedy) {
			l, err := st3.Forward(in, false)
			if err != nil {
				t.Fatal(err)
			}
			best := 0
			for j, v := range l[0] {
				if v > l[0][best] {
					best = j
				}
			}
			got, in = append(got, best), []int{best}
		}
		greedyOK = greedyOK && slices.Equal(got, c.greedy)
	}
	return
}

func TestDecoder_tinyMatchesTransformers(t *testing.T) {
	cos, abs, incr, ok := run(t, defectNone)
	t.Logf("teacher-forced logits: worst per-position cosine %.9f, max |diff| %.3e; incremental against full forward: max |diff| %.3e; greedy ids equal transformers': %v", cos, abs, incr, ok)
	if cos < 0.99999 || abs > 2e-3 || incr > 1e-5 || !ok {
		t.Errorf("the decoder differs from transformers (cosine %.9f, max |diff| %.3e, incremental %.3e, greedy %v)", cos, abs, incr, ok)
	}
}

// TestDecoder_plantedDefectsAreRed: each planted defect must fail the gate above, or the gate could not see that mistake. The k_proj bias is not among them: a bias on the keys adds the same
// number to every score of a query row and softmax is shift-invariant, so no test can see it (the encoder gate recorded the same).
func TestDecoder_plantedDefectsAreRed(t *testing.T) {
	for name, defect := range map[string]int{"query unscaled": defectNoScale, "positions shifted": defectPosShift, "cross K/V from the decoder state": defectCrossFromX,
		"no final layer norm": defectNoFinalNorm, "non-causal self-attention": defectNonCausal, "head untied": defectUntiedHead} {
		cos, abs, incr, ok := run(t, defect)
		red := cos < 0.99999 || abs > 2e-3 || incr > 1e-5 || !ok
		t.Logf("%-34s cosine %.6f max |diff| %.3e incremental %.3e greedy %v -> red %v", name, cos, abs, incr, ok, red)
		if !red {
			t.Errorf("%s: the gate did not go red", name)
		}
	}
}
