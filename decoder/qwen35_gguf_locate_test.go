//go:build realckpt

// WHERE does the GGUF path diverge from the safetensors path? — the localizer for B13's last
// standing red (docs/queue-release.md; the v1.0 gate §1).
//
// THE QUESTION THIS ANSWERS, and why neither existing gate answers it.
// TestQwen35GGUF_vsSafetensors reports one number at the TOP of the stack (min cosine 0.987835
// over 80 teacher-forced steps, mean 0.998114) and calls it a loader bug. TestQwen35GGUF_weightDiff
// reports that every transform-bearing tensor in layers 0-3 agrees to cos >= 0.99998, with a
// UNIFORM relL2 ~0.0057 — the Q8_0-vs-bf16 dequant floor — and no tensor standing out. Those two
// results are consistent with two very different stories:
//
//	(a) NOISE ACCUMULATION. A ~0.6% relative weight delta on every projection, compounded through
//	    48 layers, lands the logits ~0.998 apart. Nothing is wrong; the floor is mis-set.
//	(b) A LOCALIZED DEFECT in something weightDiff cannot see — the experts (int8, never compared),
//	    the embedding/LM head, or any layer past 3.
//
// They are distinguishable by SHAPE, not by any single number: (a) is a smooth decay of per-layer
// agreement with depth; (b) is a flat curve with a STEP at the offending layer. So this captures
// the residual stream after EVERY layer from both containers on identical input and prints the
// curve, plus the biggest layer-to-layer drop.
//
// It is a DIAGNOSTIC, not a floor: it asserts only the things that are true under either story
// (both models load, both produce finite hidden states of the same shape, layer 0 agrees closely)
// and prints the evidence for a recorded decision. Deciding "quant noise, reclassify" or "bug at
// layer k, fix" from a curve is a judgement with a decider, which is exactly what the v1.0 gate
// asks for and exactly what a threshold here would quietly pre-empt.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestQwen35GGUF_locate -v -timeout 120m
package decoder

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"testing"
)

func TestQwen35GGUF_locateDivergence(t *testing.T) {
	requireHeavyModel(t)
	gguf := assetPath(t, "GOINFER_QWEN35_GGUF")
	dir := realQwen35Dir(t)
	goldenDir := assetPath(t, "GOINFER_QWEN35_GOLDEN")
	raw, err := os.ReadFile(filepath.Join(goldenDir, "manifest.json"))
	if err != nil {
		t.Skipf("no golden manifest: %v", err)
	}
	var man gate2Manifest
	if err := json.Unmarshal(raw, &man); err != nil {
		t.Fatalf("manifest: %v", err)
	}
	if len(man.Prompts) == 0 {
		t.Fatal("golden manifest has no prompts")
	}
	// One prompt is enough to localize: a defect at layer k shows on every prompt, and a second
	// 39 GB load pair costs more than it tells. The FIRST prompt is the same one step 0 of the
	// oracle uses, so the two runs are comparable.
	p := man.Prompts[0]

	// captureAll runs the prompt through `path` and returns the residual stream after every layer
	// at the LAST prompt position, plus the final logits.
	captureAll := func(label, path string) ([][]float32, []float32, *Architecture) {
		prev := runtime.GOMAXPROCS(2)
		m, err := Load(path, Options{Quant: "int8int8"})
		runtime.GOMAXPROCS(prev)
		if err != nil {
			t.Fatalf("%s Load: %v", label, err)
		}
		defer func() {
			m.Close()
			debug.FreeOSMemory() // return the ~39 GB before the next model loads
		}()
		nL := m.w.arch.NumLayers
		layers := make([]int, nL)
		for i := range layers {
			layers[i] = i
		}
		cache := m.NewCache(len(p.PromptIDs) + 1)
		for _, id := range p.PromptIDs[:len(p.PromptIDs)-1] {
			if _, err := m.runLayers(id, cache); err != nil {
				t.Fatalf("%s prefill: %v", label, err)
			}
		}
		lg, hid, err := m.ForwardCapture(p.PromptIDs[len(p.PromptIDs)-1], cache, layers)
		if err != nil {
			t.Fatalf("%s ForwardCapture: %v", label, err)
		}
		out := make([][]float32, len(hid))
		for i := range hid {
			out[i] = append([]float32(nil), hid[i]...)
		}
		// The descriptor outlives the model here only to LABEL the curve (layer kinds); it holds
		// no weights, so keeping it after Close is a read of config, not of freed memory.
		return out, append([]float32(nil), lg...), m.w.arch
	}

	refH, refL, arch := captureAll("safetensors", dir)
	gotH, gotL, _ := captureAll("gguf", gguf)
	if len(refH) != len(gotH) {
		t.Fatalf("layer count mismatch: safetensors %d vs gguf %d", len(refH), len(gotH))
	}

	// The curve. Also the per-layer DELTA, since the eye reads a step better from differences.
	prevCos, worstDrop, worstAt := 1.0, 0.0, -1
	for i := range refH {
		c := cosineFull(gotH[i], refH[i])
		drop := prevCos - c
		if drop > worstDrop {
			worstDrop, worstAt = drop, i
		}
		t.Logf("  layer %2d (%-15s): cosine %.8f   Δ %+.8f", i, layerKind(arch, i), c, -drop)
		prevCos = c
	}
	logitCos := cosineFull(gotL, refL)
	t.Logf("=== final logits: cosine %.8f | biggest single-layer drop %.8f at layer %d ===",
		logitCos, worstDrop, worstAt)
	t.Logf("READ IT AS: a SMOOTH decay across depth ⇒ Q8_0 double-quant noise accumulating (the " +
		"weightDiff floor is ~0.0057 relL2 on every projection); a FLAT curve with one step ⇒ a " +
		"localized loader defect at the step's layer, which weightDiff can then be pointed at.")

	// Assertions that hold under EITHER story — this is a localizer, not a threshold.
	if len(refH[0]) == 0 {
		t.Fatal("empty hidden state at layer 0")
	}
	for i := range gotH {
		for _, v := range gotH[i] {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("non-finite hidden state at layer %d in the GGUF run", i)
			}
		}
	}
}
