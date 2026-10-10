package decoder

import (
	"compress/gzip"
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"
)

// layerDumpPositions are the token positions of the 1441-token window golden whose per-layer residuals are dumped. The sliding
// window is 1024, so a token at position p attends [p-1023, p]: positions <= 1023 see their whole history (the window cannot
// have changed anything for them) and positions >= 1024 lose their oldest tokens on the 21 sliding layers. The split is what lets
// ONE run separate "the window path" from "long-context drift" (docs/tasks/task-mellum21-2026-10.md, follow-up A).
var layerDumpPositions = []int{10, 300, 600, 900, 1000, 1030, 1100, 1300, 1440}

// TestMellum2_layerDump writes the goinfer side of the per-layer comparison against Hugging Face for the window golden: the residual
// stream AFTER each of the 28 layers (pre-final-norm, ForwardCapture's contract) at layerDumpPositions, through the SAME sequential
// int8int8 path TestMellum2_windowParity runs (runLayers per token, ForwardCapture at the dumped positions, which is byte-identical to
// forward). It also recomputes the window golden's sample-256 logit cosine from this run's own final logits, so the dump is proved to be
// of the run that measured the number under study. Only runs with GOINFER_MELLUM_GOLDEN_PREFIX set (it writes beside that pair).
// The reader is scripts/diff_mellum_layers.py; this test asserts nothing about the model, only that the dump was made from the same
// forward (argmax equals the golden's, for the default quant) and writes a file.
func TestMellum2_layerDump(t *testing.T) {
	prefix := os.Getenv("GOINFER_MELLUM_GOLDEN_PREFIX")
	if prefix == "" {
		t.Skip("set GOINFER_MELLUM_GOLDEN_PREFIX (it names where the dump is written and which golden's ids are run)")
	}
	requireHeavyModel(t)
	raw, err := os.ReadFile(mellum2GoldenPath("window"))
	if err != nil {
		t.Skipf("no window golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if len(g.IDs) <= 1024 {
		t.Fatalf("golden has %d ids; the window split needs more than 1024", len(g.IDs))
	}
	path := assetPath(t, "GOINFER_MELLUM_CKPT")
	// The default is the window gate's own path. GOINFER_MELLUM_DUMP_QUANT picks another quant ("" for f32, "int8" for weight-only) to
	// separate activation and weight quantization from the window path (follow-up A2); such a dump is written under its own name.
	quant := "int8int8"
	if q, ok := os.LookupEnv("GOINFER_MELLUM_DUMP_QUANT"); ok {
		quant = q
	}
	m, err := Load(path, Options{Quant: quant})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "mellum" {
		t.Fatalf("arch = %q, want mellum", m.w.arch.Name)
	}
	n := len(g.IDs)
	L := m.w.arch.NumLayers
	layers := make([]int, L)
	for i := range layers {
		layers[i] = i
	}
	want := map[int]int{}
	for i, p := range layerDumpPositions {
		if p >= n {
			t.Fatalf("position %d is past the %d-token golden", p, n)
		}
		want[p] = i
	}
	// resid[layer][posIdx] = the residual after that layer at that position.
	resid := make([][][]float32, L)
	for l := range resid {
		resid[l] = make([][]float32, len(layerDumpPositions))
	}
	prog := newProgress(t, t.Name(), n)
	prog.Phase("prefill with capture")
	cache := m.NewCache(n)
	var logits []float32
	for i, id := range g.IDs {
		pi, capture := want[i]
		switch {
		case i == n-1:
			lg, hid, err := m.ForwardCapture(id, cache, layers)
			if err != nil {
				t.Fatalf("ForwardCapture(last): %v", err)
			}
			logits = lg
			for l := range hid {
				resid[l][pi] = append([]float32(nil), hid[l]...)
			}
		case capture:
			_, hid, err := m.ForwardCapture(id, cache, layers)
			if err != nil {
				t.Fatalf("ForwardCapture(%d): %v", i, err)
			}
			for l := range hid {
				resid[l][pi] = append([]float32(nil), hid[l]...)
			}
		default:
			if _, err := m.runLayers(id, cache); err != nil {
				t.Fatalf("runLayers(%d): %v", i, err)
			}
		}
		prog.Step(1)
	}
	if len(logits) != g.Vocab {
		t.Fatalf("got %d logits, want vocab %d", len(logits), g.Vocab)
	}
	got := argmax(logits)
	var dot, na, nb float64
	for _, kv := range g.Sample {
		a, b := float64(logits[int(kv[0])]), kv[1]
		dot += a * b
		na += a * a
		nb += b * b
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	t.Logf("this run: argmax %d (golden %d), sample-256 logit cosine %.5f", got, g.Argmax, cos)
	suffix := "_layers_goinfer.json.gz"
	if quant != "int8int8" {
		// another quant may legitimately pick another argmax: it is reported, and the reader (scripts/diff_mellum_layers.py) reads the quant from the dump
		suffix = "_layers_goinfer_" + map[bool]string{true: "f32", false: quant}[quant == ""] + ".json.gz"
		if got != g.Argmax {
			t.Logf("quant %q: argmax %d differs from the int8int8 golden's %d (reported, not an error)", quant, got, g.Argmax)
		}
	} else if got != g.Argmax {
		t.Errorf("argmax %d != golden %d: this is not the run the window gate measured", got, g.Argmax)
	}
	out := strings.TrimSuffix(mellum2GoldenPath("window"), "_golden.json") + suffix
	f, err := os.Create(out)
	if err != nil {
		t.Fatal(err)
	}
	zw := gzip.NewWriter(f)
	rec := map[string]any{
		"positions": layerDumpPositions, "n_ids": n, "layers": L, "hidden": m.w.arch.HiddenDim,
		"quant": map[bool]string{true: "f32", false: quant}[quant == ""], "argmax": got, "golden_argmax": g.Argmax, "sample256_logit_cosine": cos,
		"resid": resid, // [layer][position index][hidden]
	}
	if err := json.NewEncoder(zw).Encode(rec); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %s", out)
}
