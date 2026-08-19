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
	path := os.Getenv("HOME") + "/models/mellum2-unq"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no Mellum2 checkpoint (%v)", err)
	}
	m, err := Load(path, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "mellum" {
		t.Fatalf("arch = %q, want mellum", m.w.arch.Name)
	}

	cache := m.NewCache(len(g.IDs))
	for _, id := range g.IDs[:len(g.IDs)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			t.Fatalf("runLayers: %v", err)
		}
	}
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
	// Record the validated metrics (no-op unless GOINFER_MANIFEST_EMIT; skipped if any check
	// above failed). int8int8 (serve default) vs HF bf16 → real-model-oracle; argmax exact when green.
	// The method string is the T3 vocabulary name, checked by emitParityRow: "real-oracle" was
	// written here for months and reached the manifest verbatim (B15).
	// Emit only from the forward gate (not the window gate) to avoid a double row.
	if emit {
		emitParityRow(t, "mellum", "real-model-oracle", "HF bf16 (Mellum2-12B-A2.5B-Instruct)", 100.0, cos, cos)
	}
}

// TestMellum2_logitParity gates the Mellum2 forward (MoE 64/top-8, 3:1
// sliding/full interleave, YaRN-on-full RoPE, QK-norm) against the HF bf16
// oracle on a chat-templated prompt — argmax is the first answer token (50195
// "Paris"), so this also gates coherence. int8int8 (the serve default) vs bf16:
// measured sample-256 cosine 0.99955 (floor 0.98, the Gemma 4 12B reference).
func TestMellum2_logitParity(t *testing.T) {
	runMellum2Golden(t, "../testdata/mellum2_forward_golden.json", 0.98, true)
}

// TestMellum2_windowParity pins the sliding-window EVICTION path on the real
// checkpoint: a 1441-token prompt (> the 1024 window), so the local layers attend
// only within the window while the YaRN'd full layers see everything. The
// next-token logits still match the HF bf16 oracle (measured cosine 0.99636).
// (Inc3 real-model proof; the synthetic unit-level proof is
// TestMellum2_slidingWindowEviction.)
func TestMellum2_windowParity(t *testing.T) {
	runMellum2Golden(t, "../testdata/mellum2_window_golden.json", 0.98, false)
}
