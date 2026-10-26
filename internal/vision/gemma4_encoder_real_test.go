package vision

import (
	"compress/gzip"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"
)

// gemma4RealCkptDir is the real google/gemma-4-E2B-it checkpoint, pulled to
// ~/models per CLAUDE.md's model-storage rule (never read a correctness gate's
// checkpoint from /srv/models — pulled properly here, not archive-read).
func gemma4RealCkptDir(t *testing.T) string {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skipf("no home dir: %v", err)
	}
	dir := filepath.Join(home, "models", "gemma-4-E2B-unq")
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("real gemma-4-E2B-it checkpoint not found at %s (models-pull gemma-4-E2B-unq first): %v", dir, err)
	}
	return dir
}

func readMaybeGzipJSON(t *testing.T, path string, v any) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("golden %s not found (run scripts/pin_gemma4_vision_real.py in goinfer first): %v", path, err)
	}
	defer f.Close()
	var r io.Reader = f
	if filepath.Ext(path) == ".gz" {
		gz, err := gzip.NewReader(f)
		if err != nil {
			t.Fatalf("gunzip %s: %v", path, err)
		}
		defer gz.Close()
		r = gz
	}
	if err := json.NewDecoder(r).Decode(v); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
}

// TestGemma4Encoder_realCheckpointParity is the P7 Phase A T3 gate: the REAL
// google/gemma-4-E2B-it vision_tower + embed_vision weights (not a tiny random
// model — scripts/pin_gemma4_vision_real.py), on fixed synthetic patches,
// matched f32 precision on both sides (this session's own "matched precision"
// discipline — avoids conflating a real wiring bug with ordinary quantization
// noise). A real checkpoint can catch what a tiny golden cannot: e.g. a real
// clip-bound distribution, real weight-scale interactions across 16 layers, or
// a reference-library defect (CLAUDE.md's InternLM2 lesson) — none of which a
// hand-built tiny model exercises.
func TestGemma4Encoder_realCheckpointParity(t *testing.T) {
	ckptDir := gemma4RealCkptDir(t)
	goldenPath := gemma4Testdata(t, "gemma4_vision_real_golden.json.gz")

	var g gemma4Golden
	readMaybeGzipJSON(t, goldenPath, &g)

	enc, err := LoadGemma4Encoder(ckptDir, false)
	if err != nil {
		t.Fatalf("LoadGemma4Encoder: %v", err)
	}
	if enc.Cfg.HiddenSize != g.Config.HiddenSize || enc.Cfg.NumHiddenLayers != g.Config.NumHiddenLayers ||
		enc.Cfg.NumAttentionHeads != g.Config.NumAttentionHeads || enc.Cfg.HeadDim != g.Config.HeadDim {
		t.Fatalf("loaded config mismatch: got %+v, want hidden=%d layers=%d heads=%d headDim=%d",
			enc.Cfg, g.Config.HiddenSize, g.Config.NumHiddenLayers, g.Config.NumAttentionHeads, g.Config.HeadDim)
	}
	if !enc.Cfg.UseClippedLinears {
		t.Fatalf("UseClippedLinears = false, want true (the real E2B checkpoint sets this — the clamp path must be exercised, not silently skipped)")
	}
	if enc.TextHiddenSize != g.TextHiddenSize {
		t.Fatalf("TextHiddenSize = %d, want %d", enc.TextHiddenSize, g.TextHiddenSize)
	}

	positionIDs := make([][2]int, g.NumPatches)
	for i := range positionIDs {
		positionIDs[i] = [2]int{g.PositionIDs[i*2], g.PositionIDs[i*2+1]}
	}

	got, err := enc.Forward(g.Patches, positionIDs)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if len(got) != len(g.Projected) {
		t.Fatalf("output len %d, want %d (shape %v)", len(got), len(g.Projected), g.ProjectedShape)
	}
	cos := cosineSim(got, g.Projected)
	var maxDiff float32
	for i := range got {
		d := got[i] - g.Projected[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	// Matched-precision f32-vs-f32 real weights: this repo's usual near-1.0 bar
	// (not the wider int8/quant-noise floor other gates use).
	if cos < 0.9999 {
		n := min(8, len(got))
		t.Fatalf("cosine = %.9f (want >= 0.9999), max|diff| = %g\ngot[:8]  = %v\nwant[:8] = %v", cos, maxDiff, got[:n], g.Projected[:n])
	}
	t.Logf("gemma4 vision REAL checkpoint (E2B-it) parity: cosine = %.9f, max|diff| = %g", cos, maxDiff)
}

// gemma4Real26BCkptDir is the real google/gemma-4-26b-a4b-it checkpoint — the
// standardize=True geometry TestGemma4Encoder_realCheckpointParity's E2B fixture
// does not exercise at all (hidden 1152/27 layers/head_dim 72 vs E2B's smaller
// tower, and use_clipped_linears=False vs E2B's True).
func gemma4Real26BCkptDir(t *testing.T) string {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skipf("no home dir: %v", err)
	}
	dir := filepath.Join(home, "models", "gemma-4-26b-a4b-it")
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors.index.json")); err != nil {
		t.Skipf("real gemma-4-26b-a4b-it checkpoint not found at %s: %v", dir, err)
	}
	return dir
}

// TestGemma4Encoder_realCheckpointParity_26B is TestGemma4Encoder_realCheckpointParity's twin
// for the standardize=True geometry: this is what caught goinfer's own real-checkpoint gate
// failing outright (LoadGemma4Encoder used to hard-refuse standardize=true — "not implemented"
// — before this fix added the missing `(x - std_bias) * std_scale` affine,
// Gemma4VisionModel.forward's own placement, immediately after the pooler's root-hidden_size
// scaling and before the embedder's RMSNorm+projection). Matched-precision f32-vs-f32, same bar
// as the E2B gate.
func TestGemma4Encoder_realCheckpointParity_26B(t *testing.T) {
	ckptDir := gemma4Real26BCkptDir(t)
	goldenPath := gemma4Testdata(t, "gemma4_vision_real_26b_golden.json.gz")

	var g gemma4Golden
	readMaybeGzipJSON(t, goldenPath, &g)

	enc, err := LoadGemma4Encoder(ckptDir, false)
	if err != nil {
		t.Fatalf("LoadGemma4Encoder: %v", err)
	}
	if enc.Cfg.HiddenSize != g.Config.HiddenSize || enc.Cfg.NumHiddenLayers != g.Config.NumHiddenLayers ||
		enc.Cfg.NumAttentionHeads != g.Config.NumAttentionHeads || enc.Cfg.HeadDim != g.Config.HeadDim {
		t.Fatalf("loaded config mismatch: got %+v, want hidden=%d layers=%d heads=%d headDim=%d",
			enc.Cfg, g.Config.HiddenSize, g.Config.NumHiddenLayers, g.Config.NumAttentionHeads, g.Config.HeadDim)
	}
	if !enc.Cfg.Standardize {
		t.Fatalf("Standardize = false, want true (the real 26B-A4B checkpoint sets this — the fixed affine path must be exercised, not silently skipped)")
	}
	if enc.Cfg.UseClippedLinears {
		t.Fatalf("UseClippedLinears = true, want false (the real 26B-A4B checkpoint has this off — the unclipped ±inf-bound path must be exercised here, distinct from the E2B gate)")
	}
	if enc.TextHiddenSize != g.TextHiddenSize {
		t.Fatalf("TextHiddenSize = %d, want %d", enc.TextHiddenSize, g.TextHiddenSize)
	}

	positionIDs := make([][2]int, g.NumPatches)
	for i := range positionIDs {
		positionIDs[i] = [2]int{g.PositionIDs[i*2], g.PositionIDs[i*2+1]}
	}

	got, err := enc.Forward(g.Patches, positionIDs)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if len(got) != len(g.Projected) {
		t.Fatalf("output len %d, want %d (shape %v)", len(got), len(g.Projected), g.ProjectedShape)
	}
	cos := cosineSim(got, g.Projected)
	var maxDiff float32
	for i := range got {
		d := got[i] - g.Projected[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	if cos < 0.9999 {
		n := min(8, len(got))
		t.Fatalf("cosine = %.9f (want >= 0.9999), max|diff| = %g\ngot[:8]  = %v\nwant[:8] = %v", cos, maxDiff, got[:n], g.Projected[:n])
	}
	t.Logf("gemma4 vision REAL checkpoint (26B-A4B, standardize=true) parity: cosine = %.9f, max|diff| = %g", cos, maxDiff)
}
