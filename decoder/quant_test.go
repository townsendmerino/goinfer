package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// M8 int8 accuracy. Loads the checkpoint with per-row int8 weight quant and
// checks the forward stays faithful to the f32 reference: the argmax (predicted
// token) is unchanged and the full logit vector keeps a high cosine (a looser
// bar than the f32 parity gate — quantization perturbs the logits, it doesn't
// reproduce them). Also confirms the f32 weights were actually freed.
func TestQuantInt8_accuracy(t *testing.T) {
	raw, err := os.ReadFile(gemmaForwardGoldenPath)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no forward golden at %s — regenerate with scripts/pin_gemma_forward.py", gemmaForwardGoldenPath)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(gemmaModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s", gemmaModelDir)
	}

	m, err := Load(gemmaModelDir, Options{Quant: "int8"})
	if err != nil {
		t.Fatalf("Load int8: %v", err)
	}
	// The quantization must have replaced f32 with int8 (the memory win).
	if m.w.Embed.f32 != nil || m.w.Embed.q8 == nil {
		t.Fatalf("Embed not quantized: f32=%v q8=%v", m.w.Embed.f32 != nil, m.w.Embed.q8 != nil)
	}

	// Retained weight footprint: int8 codes + f32 scales vs the f32 originals.
	var q8Bytes, f32Bytes int
	for _, wm := range m.w.matmulWeights() {
		n := wm.rows * wm.cols
		q8Bytes += n + 4*wm.rows // int8 codes + per-row f32 scale
		f32Bytes += 4 * n
	}
	ratio := float64(f32Bytes) / float64(q8Bytes)
	t.Logf("matmul weight memory: int8 %.0f MB vs f32 %.0f MB (%.2fx smaller)",
		float64(q8Bytes)/1e6, float64(f32Bytes)/1e6, ratio)
	if ratio < 3.5 {
		t.Errorf("int8 memory ratio %.2fx, want ≥ 3.5x", ratio)
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

	// Argmax (the predicted next token) must survive quantization.
	if got := argmax(logits); got != g.Argmax {
		t.Errorf("int8 argmax = %d, want %d (f32) — quantization flipped the top token", got, g.Argmax)
	}

	// High cosine vs the f32 reference dump, when present (looser than 1−1e-4).
	if cos := quantCosine(t, logits); !math.IsNaN(cos) {
		if cos < 0.999 {
			t.Errorf("int8 vs f32 cosine = %.6f, want ≥ 0.999", cos)
		}
		t.Logf("int8: argmax=%d (want %d) | cosine vs f32 = %.6f", argmax(logits), g.Argmax, cos)
	}
}

// M8 int4 structure + footprint on the 270M. int4 is a big-model tool — on a
// model this small it is lossy enough to shift the top token, so the strict
// accuracy gate (argmax + cosine) lives in TestGGUF_int4_resident on TinyLlama
// (1.1B, cosine ~0.994). Here we check the wiring: projections are int4, the
// logit-critical embedding stays int8, f32 is freed, the footprint shrinks, and
// the forward still runs and stays loosely correlated with f32.
func TestQuantInt4_accuracy(t *testing.T) {
	raw, err := os.ReadFile(gemmaForwardGoldenPath)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no forward golden at %s — regenerate with scripts/pin_gemma_forward.py", gemmaForwardGoldenPath)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(gemmaModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s", gemmaModelDir)
	}

	m, err := Load(gemmaModelDir, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load int4: %v", err)
	}
	// The projections must be int4 (the bulk of the win); the embedding/tied head
	// stays int8 (logit-critical — the embedding policy). Both must have freed f32.
	gate := &m.w.Layers[0].GateProj
	if gate.f32 != nil || gate.q4 == nil {
		t.Fatalf("GateProj not int4: f32=%v q4=%v", gate.f32 != nil, gate.q4 != nil)
	}
	if m.w.Embed.f32 != nil || m.w.Embed.q8 == nil {
		t.Fatalf("Embed not int8 (embedding policy): f32=%v q8=%v q4=%v",
			m.w.Embed.f32 != nil, m.w.Embed.q8 != nil, m.w.Embed.q4 != nil)
	}

	// Retained footprint vs the f32 originals: int4 = packed nibbles + per-group
	// scales; int8 (embed/head) = codes + per-row scale. The ratio is lower on
	// this 270M than a typical model because its 262k-row embedding (kept int8)
	// dwarfs the int4 projections; on a normal model the projections dominate
	// and the ratio approaches int4's ~6.4×.
	var qBytes, f32Bytes int
	for _, wm := range m.w.matmulWeights() {
		f32Bytes += 4 * wm.rows * wm.cols
		switch {
		case wm.q4 != nil:
			nGroups := (wm.cols + int4GroupSize - 1) / int4GroupSize
			qBytes += wm.rows*((wm.cols+1)/2) + 4*wm.rows*nGroups
		case wm.q8 != nil:
			qBytes += wm.rows*wm.cols + 4*wm.rows
		}
	}
	ratio := float64(f32Bytes) / float64(qBytes)
	t.Logf("matmul weight memory: quant %.0f MB vs f32 %.0f MB (%.2fx smaller)",
		float64(qBytes)/1e6, float64(f32Bytes)/1e6, ratio)
	if ratio < 4.0 {
		t.Errorf("int4 memory ratio %.2fx, want ≥ 4.0x", ratio)
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
	// Loose correlation only — int4 on a 270M legitimately moves the argmax (see
	// the doc comment); the strict gate is TestGGUF_int4_resident.
	if cos := quantCosine(t, logits); !math.IsNaN(cos) {
		if cos < 0.95 {
			t.Errorf("int4 vs f32 cosine = %.6f, want ≥ 0.95", cos)
		}
		t.Logf("int4 (270M): argmax=%d (f32 %d) | cosine vs f32 = %.6f", argmax(logits), g.Argmax, cos)
	}
}

// M8 W8A8 (full int8×int8) accuracy. Like the weight-only int8 test, but the
// activations are also quantized to int8 each matmul (the integer kernel), so it
// is lossier — the argmax must still hold and the cosine clear a looser bar than
// weight-only int8 (which keeps f32 activations). Confirms the int8 weights are
// stored and the W8A8 matmul path is selected.
func TestQuantInt8I8_accuracy(t *testing.T) {
	raw, err := os.ReadFile(gemmaForwardGoldenPath)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no forward golden at %s — regenerate with scripts/pin_gemma_forward.py", gemmaForwardGoldenPath)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(gemmaModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s", gemmaModelDir)
	}

	m, err := Load(gemmaModelDir, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load int8int8: %v", err)
	}
	// int8 weights stored, and the W8A8 matmul path selected.
	if gate := &m.w.Layers[0].GateProj; gate.q8 == nil || !gate.w8a8 || gate.f32 != nil {
		t.Fatalf("GateProj not W8A8 (q8=%v w8a8=%v f32=%v)", gate.q8 != nil, gate.w8a8, gate.f32 != nil)
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
	if got := argmax(logits); got != g.Argmax {
		t.Errorf("W8A8 argmax = %d, want %d (f32) — activation quant flipped the top token", got, g.Argmax)
	}
	if cos := quantCosine(t, logits); !math.IsNaN(cos) {
		if cos < 0.99 {
			t.Errorf("W8A8 vs f32 cosine = %.6f, want ≥ 0.99", cos)
		}
		t.Logf("int8int8 (W8A8): argmax=%d (want %d) | cosine vs f32 = %.6f", argmax(logits), g.Argmax, cos)
	}
}

// quantCosine returns cosine(int8-logits, f32-reference) from the per-machine
// full dump, or NaN if absent.
func quantCosine(t *testing.T, logits []float32) float64 {
	raw, err := os.ReadFile(gemmaForwardFullPath)
	if err != nil {
		return math.NaN()
	}
	var full struct {
		Logits []float64 `json:"logits"`
	}
	if err := json.Unmarshal(raw, &full); err != nil {
		t.Fatalf("parse full dump: %v", err)
	}
	var dot, na, nb float64
	for i, v := range logits {
		a := float64(v)
		b := full.Logits[i]
		dot += a * b
		na += a * a
		nb += b * b
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
