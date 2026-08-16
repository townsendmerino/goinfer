package decoder

import (
	"os"
	"path/filepath"
	"testing"
)

// The gate for the hidden-state capture seam on the families that run their OWN runLayers
// (P10 / docs/spec/08). It exists because the failure mode here is quiet: a capture placed a
// few lines too early copies the residual after ATTENTION instead of after the whole layer,
// which has the right shape, the right dtype, plausible magnitudes, and is simply the wrong
// tensor. Every shape assertion passes and the drafter just accepts less, which reads as "block
// drafting doesn't work well for this family" rather than as a bug.
//
// So this does not check shapes. It checks the one identity that pins PLACEMENT without needing
// a reference dump: runLayersX returns the residual after the last layer, so capturing layer
// NumLayers-1 must reproduce that tensor EXACTLY. Bitwise — these are the same additions in the
// same order, so anything but equality means the copy is taken somewhere else.
//
// Two further properties, both cheap and both real bugs I would otherwise ship:
//   - capture must not PERTURB: logits with capture on must be bit-identical to logits with it
//     off, since the seam is documented read-only and a drafter that changes the target's output
//     is not a drafter.
//   - distinct layers must yield DISTINCT tensors: the natural bug in the copy loop is to append
//     into a shared backing array, which makes every requested layer alias the last one.
func TestCaptureSeam_ownRunLayers(t *testing.T) {
	cases := []struct {
		family string
		dir    string
	}{
		{"qwen3_5_moe", "../testdata/qwen35-tiny"},
		{"gemma4", "../testdata/gemma4-moe-tiny"},
	}
	for _, tc := range cases {
		t.Run(tc.family, func(t *testing.T) {
			// stat the WEIGHTS, not the directory: fixture metadata is committed while
			// model.safetensors is gitignored, so a dir-only guard passes on a checkout
			// that cannot actually run this.
			if _, err := os.Stat(filepath.Join(tc.dir, "model.safetensors")); err != nil {
				t.Skipf("no fixture weights at %s (%v)", tc.dir, err)
			}
			m, err := Load(tc.dir, Options{})
			if err != nil {
				t.Fatalf("Load(%s): %v", tc.dir, err)
			}
			defer m.Close()
			nL := m.w.arch.NumLayers
			if nL < 2 {
				t.Fatalf("fixture has %d layers; this gate needs >= 2 to tell layers apart", nL)
			}
			const tok = 1

			// --- control: no capture ---
			plain := m.NewCache(8)
			wantLogits, err := m.forward(tok, plain)
			if err != nil {
				t.Fatalf("forward (control): %v", err)
			}
			wantLogits = append([]float32(nil), wantLogits...)

			// --- capture first and last layer ---
			cache := m.NewCache(8)
			cache.captureLayers = []int{0, nL - 1}
			cache.captured = make([][]float32, 2)
			gotLogits, err := m.forward(tok, cache)
			if err != nil {
				t.Fatalf("forward (capture): %v", err)
			}

			// 1. read-only: the seam must not change the model's output.
			if len(gotLogits) != len(wantLogits) {
				t.Fatalf("logit length changed under capture: %d vs %d", len(gotLogits), len(wantLogits))
			}
			for i := range wantLogits {
				if gotLogits[i] != wantLogits[i] {
					t.Fatalf("capture PERTURBED the forward at logit %d: %v != %v (the seam is documented read-only)",
						i, gotLogits[i], wantLogits[i])
				}
			}

			first, last := cache.captured[0], cache.captured[1]
			hidden := m.w.arch.HiddenDim
			for i, c := range [][]float32{first, last} {
				if len(c) != hidden {
					t.Fatalf("captured[%d] has length %d, want hidden=%d", i, len(c), hidden)
				}
			}

			// 2. PLACEMENT: re-run the family's runLayers with the same capture and require
			// its return value to equal the last capture bitwise. This is the assertion the
			// whole test exists for.
			ref := m.NewCache(8)
			ref.captureLayers = []int{nL - 1}
			ref.captured = make([][]float32, 1)
			h, err := m.runLayers(tok, ref)
			if err != nil {
				t.Fatalf("runLayers: %v", err)
			}
			if len(ref.captured[0]) != len(h) {
				t.Fatalf("last capture len %d != runLayers return len %d", len(ref.captured[0]), len(h))
			}
			for i := range h {
				if ref.captured[0][i] != h[i] {
					t.Fatalf("capture of layer %d differs from runLayers' return at %d: %v != %v\n"+
						"the copy is NOT taken after the full layer — check it sits after the LAST "+
						"residual add in the loop body, not after attention",
						nL-1, i, ref.captured[0][i], h[i])
				}
			}

			// 3. distinct layers, distinct tensors (catches a shared backing array).
			same := true
			for i := range first {
				if first[i] != last[i] {
					same = false
					break
				}
			}
			if same {
				t.Errorf("layer 0 and layer %d captured IDENTICAL tensors — the rows alias", nL-1)
			}
			t.Logf("%s: %d layers, capture placement exact, forward bit-identical under capture",
				tc.family, nL)
		})
	}
}
