package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// runMellum2Golden loads the unquantized Mellum2 (int8int8, the serve default)
// and gates one forwardGolden: argmax exact + sample-256 cosine vs the HF bf16
// oracle. Skips cleanly without the golden or the ~24 GB checkpoint.
func runMellum2Golden(t *testing.T, goldenPath string, cosFloor float64, emit bool) {
	t.Helper()
	requireHeavyModel(t)
	if testing.Short() {
		t.Skip("slow: 12B forward on the naive backend")
	}
	raw, err := os.ReadFile(goldenPath)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden at %s — run scripts/pin_mellum2.py", goldenPath)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	prog := newProgress(t, t.Name(), len(g.IDs)-1)
	prog.Phase("load 12B checkpoint")
	path := assetPath(t, "GOINFER_MELLUM_CKPT") // the checkpoint under test, registered in testdata/assets.json (the P18 gate reads the same one); default $MODELS/mellum2-unq is Mellum2 2.0
	m, err := Load(path, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "mellum" {
		t.Fatalf("arch = %q, want mellum", m.w.arch.Name)
	}

	cache := m.NewCache(len(g.IDs))
	prog.Phase("prefill")
	for _, id := range g.IDs[:len(g.IDs)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			t.Fatalf("runLayers: %v", err)
		}
		prog.Step(1)
	}
	prog.Phase("final forward")
	logits, err := m.forward(g.IDs[len(g.IDs)-1], cache)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if len(logits) != g.Vocab {
		t.Fatalf("got %d logits, want vocab %d", len(logits), g.Vocab)
	}

	got := argmax(logits)
	t.Logf("argmax: got %d (logit %.4f) | want %d (golden logit %.4f)", got, logits[got], g.Argmax, logits[g.Argmax])
	if got != g.Argmax {
		t.Errorf("argmax = %d, want %d", got, g.Argmax)
	}

	var dot, na, nb float64
	for _, kv := range g.Sample {
		a, b := float64(logits[int(kv[0])]), kv[1]
		dot += a * b
		na += a * a
		nb += b * b
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	t.Logf("sample-256 cosine (int8int8 vs bf16) = %.5f (floor %.2f)", cos, cosFloor)
	if cos < cosFloor {
		t.Errorf("sample cosine %.5f < %.2f", cos, cosFloor)
	}
	// Record the validated metrics (no-op unless GOINFER_MANIFEST_EMIT; skipped if any check above failed). int8int8
	// (serve default) vs HF bf16 is a real-model-oracle row; argmax is exact when green. The method string is the T3
	// vocabulary name, checked by emitParityRow. Emit only from the forward gate (not the window gate) to avoid a double
	// row.
	if emit && os.Getenv("GOINFER_MELLUM_GOLDEN_PREFIX") == "" { // a pair pinned from another checkpoint never writes the 2.0 family's manifest row
		emitParityRow(t, "mellum", "real-model-oracle", "HF bf16 (Mellum2-12B-A2.5B-Instruct)", 100.0, cos, cos)
	}
}

// TestMellum2_logitParity gates the Mellum2 forward (MoE 64/top-8, 3:1 sliding/full interleave, YaRN-on-full RoPE,
// QK-norm) against the HF bf16 oracle on a chat-templated prompt; argmax is the first answer token (50195
// "Paris"), so this also gates coherence. int8int8 (the serve default) vs bf16, sample-256 cosine floor 0.98 (the
// Gemma 4 12B reference).
func TestMellum2_logitParity(t *testing.T) {
	runMellum2Golden(t, mellum2GoldenPath("forward"), 0.98, true)
}

// TestMellum2_windowParity pins the sliding-window EVICTION path on the real checkpoint: a 1441-token prompt (> the
// 1024 window), so the local layers attend only within the window while the YaRN'd full layers see everything. The
// next-token logits must still match the HF bf16 oracle. The synthetic unit-level proof is
// TestMellum2_slidingWindowEviction.
func TestMellum2_windowParity(t *testing.T) {
	runMellum2Golden(t, mellum2GoldenPath("window"), 0.98, false)
}

// mellum2GoldenPath is the golden pair the two parity tests read. Default: the 2.0 pair (testdata/mellum2_{forward,window}_golden.json). GOINFER_MELLUM_GOLDEN_PREFIX
// (e.g. ../testdata/mellum21) selects another checkpoint's pair, pinned by scripts/pin_mellum2.py from that checkpoint's own Hugging Face weights; with it set the tests never write the
// 2.0 family's parity-manifest row.
func mellum2GoldenPath(kind string) string {
	prefix := "../testdata/mellum2"
	if v := os.Getenv("GOINFER_MELLUM_GOLDEN_PREFIX"); v != "" {
		prefix = v
	}
	return prefix + "_" + kind + "_golden.json"
}
