//go:build darwin

package metal

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGemmaSublayer_MetalContribution is the Metal half of the cross-box per-sublayer trace
// (docs/prompts/gemma-metal-signflip-bisect.md, Fork 1). The CUDA box built ForwardSubCapture and
// gave f32-truth attention/MLP contributions for the two confirmed Metal sign-flips (1723, 227) at
// the last layers. This produces MY Metal attention contribution (o-proj → post-attn-norm, before
// the residual add) and MLP contribution (down → post-MLP-norm, before the add) at the same
// channels/layers, so overlaying the two names the sublayer where Metal's contribution crosses
// zero — i.e. which kernel flips the sign.
//
// f32 truth from the CUDA box (real bf16, same byte-identical checkpoint) — the sign to match:
//
//	1723  L31 attn -58.66 mlp -43.55 | L32 attn +19.42 mlp +49.94 | L33 attn -66.54 mlp  -6.86
//	 227  L31 attn -81.88 mlp -13.44 | L32 attn -11.20 mlp -50.95 | L33 attn -30.18 mlp -110.19
//
// Where Metal's attn or mlp contribution goes POSITIVE against a negative truth is the culprit.
func TestGemmaSublayer_MetalContribution(t *testing.T) {
	if testing.Short() {
		t.Skip("loads a real model")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	path := os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	m8, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load int8: %v", err)
	}
	r, err := BuildResident(m8)
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	defer r.Close()
	if !r.sandwich {
		t.Fatal("gemma3 resident is not sandwich — features not declared? this test needs the sandwich path")
	}
	seed := seedPrompt(t, path, probeText)
	pos := len(seed) - 1

	// Walk pos 0..pos-1 so Metal's KV holds real history, then capture the sublayer contributions
	// at the probe position.
	for i := 0; i < pos; i++ {
		r.forwardTrunkForTest(m8.EmbedResidentForTest(seed[i]), i, r.nL)
	}
	attn, mlp := r.forwardSubCaptureForTest(m8.EmbedResidentForTest(seed[pos]), pos)
	if attn == nil {
		t.Fatal("forwardSubCaptureForTest returned nil (not sandwich?)")
	}

	// Self-check the seam: emb + Σ_l (attn[l] + mlp[l]) must equal the full-trunk residual, or the
	// captured contributions don't mean what the label says (the CUDA box self-checked theirs too).
	emb := m8.EmbedResidentForTest(seed[pos])
	full := r.forwardTrunkForTest(emb, pos, r.nL)
	recon := append([]float32(nil), emb...)
	for l := 0; l < r.nL; l++ {
		for i := range recon {
			recon[i] += attn[l][i] + mlp[l][i]
		}
	}
	var worst float64
	for i := range full {
		if d := math.Abs(float64(full[i] - recon[i])); d > worst {
			worst = d
		}
	}
	t.Logf("seam self-check: max|full − (emb+Σ contributions)| = %.4g (should be ~0)", worst)
	if worst > 1 {
		t.Errorf("sublayer contributions do NOT reconstruct the residual (worst %.4g) — the capture is mislabeled", worst)
	}

	// Local f32-truth per-sublayer on the SAME byte-identical checkpoint: Quant:int8 (weight-only)
	// = MatmulBTQ8, int8 weight × f32 ACTIVATION → no activation crush, so its sublayer signs are
	// ground truth. Avoids overlaying the CUDA box's hand-transcribed bf16 numbers (which are a
	// slightly different f32 than my Q4_K_M-dequant); this is apples-to-apples on my file.
	m8w, e8w := decoder.Load(path, decoder.Options{Quant: "int8"})
	if e8w != nil {
		t.Fatalf("load int8-weight f32-truth: %v", e8w)
	}
	_, nL, _, nKV, hd, _, _ := m8w.Dims()
	cache := decoder.NewKVCache(nL, nKV, hd, 0, 1024)
	for i := 0; i < pos; i++ {
		if _, err := m8w.ForwardForTest(seed[i], cache); err != nil {
			t.Fatalf("cpu walk: %v", err)
		}
	}
	tAttn, tMLP, serr := m8w.ForwardSubCapture(seed[pos], cache)
	if serr != nil {
		t.Skipf("ForwardSubCapture: %v", serr)
	}

	for _, ch := range []int{1723, 227} {
		t.Logf("channel %d — Metal vs f32-truth per-sublayer (attn=o-proj+postAttnNorm, mlp=down+postMLPNorm):", ch)
		for l := 27; l < nL; l++ {
			ma, mm := attn[l][ch], mlp[l][ch]
			ta, tm := tAttn[l][ch], tMLP[l][ch]
			flag := ""
			if (ma < 0) != (ta < 0) && math.Abs(float64(ta)) > 2 {
				flag += "  ATTN-FLIP"
			}
			if (mm < 0) != (tm < 0) && math.Abs(float64(tm)) > 2 {
				flag += "  MLP-FLIP"
			}
			t.Logf("  L%2d | attn metal=%+9.2f truth=%+9.2f | mlp metal=%+9.2f truth=%+9.2f%s",
				l, ma, ta, mm, tm, flag)
		}
	}
}
