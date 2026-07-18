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
	attn, mlp, _, ctx, cqDeq := r.forwardSubCaptureForTest(m8.EmbedResidentForTest(seed[pos]), pos)
	if attn == nil {
		t.Fatal("forwardSubCaptureForTest returned nil (not sandwich?)")
	}

	// Candidate 1 — is Metal's context int8-quant (quant_vec/pQv) faithful? Quantize Metal's OWN
	// f32 context with the CPU's absmax-int8 (aikit's QuantizeRowInt8 math) and compare to Metal's
	// dequantized int8 context. Match ⇒ quant_vec is correct; combined with the exonerated o-proj
	// GEMV, the o-proj output is then a correct function of Metal's context, so the divergence is
	// the CONTEXT ITSELF (candidate 2) — a precise cross-box handoff. Checked per layer over 27-33.
	var worstQuant float64
	worstQL := -1
	for l := 27; l < r.nL; l++ {
		c := ctx[l]
		var amax float32
		for _, v := range c {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}
		sc := amax / 127
		for i := range c {
			q := math.Round(float64(c[i] / sc))
			if q > 127 {
				q = 127
			} else if q < -127 {
				q = -127
			}
			cpuDeq := float32(q) * sc
			if d := math.Abs(float64(cpuDeq - cqDeq[l][i])); d > worstQuant {
				worstQuant, worstQL = d, l
			}
		}
	}
	t.Logf("quant_vec fidelity: worst |Metal-int8-ctx − CPU-int8-ctx| = %.4g (layer %d) — ~0 means pQv is faithful, bug is the CONTEXT", worstQuant, worstQL)

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
	tAttn, tMLP, tCtx, serr := m8w.ForwardSubCapture(seed[pos], cache)
	if serr != nil {
		t.Skipf("ForwardSubCapture: %v", serr)
	}

	// Backend-independent int4 context reference (== CUDA-int4 per the CUDA box: their CUDA-int4
	// and CPU-int4 contexts agree to ~3 decimals). This is THE discriminator for candidate 2 vs 3:
	// Metal's r.ctx (attn output, pre-o-proj) overlaid on goinfer's int4 context.
	//   Metal ≈ CPU-int4 ctx ⇒ Metal's attention matches goinfer ⇒ the o-proj divergence is WEIGHTS
	//     (candidate 3, resurrect the f32-source weight compare).
	//   Metal diverges from CPU-int4 ctx ⇒ Metal has its own ATTENTION-path bug (candidate 2).
	m4, e4 := decoder.Load(path, decoder.Options{Quant: "int4"})
	if e4 != nil {
		t.Fatalf("load int4 ctx reference: %v", e4)
	}
	c4 := decoder.NewKVCache(nL, nKV, hd, 0, 1024)
	for i := 0; i < pos; i++ {
		if _, err := m4.ForwardForTest(seed[i], c4); err != nil {
			t.Fatalf("int4 walk: %v", err)
		}
	}
	_, _, i4Ctx, e4c := m4.ForwardSubCapture(seed[pos], c4)
	if e4c != nil {
		t.Skipf("int4 ForwardSubCapture: %v", e4c)
	}

	// The CUDA box's top-magnitude qDim indices (f32 → CUDA-int4) at L31-33, to overlay Metal's.
	cudaIdx := map[int][]int{31: {1287, 824, 568, 1031}, 32: {770, 814, 794, 952}, 33: {1903, 1647, 1910, 1884}}
	cosCtx := func(a, b []float32) float64 {
		var dot, na, nb float64
		for i := range a {
			dot += float64(a[i]) * float64(b[i])
			na += float64(a[i]) * float64(a[i])
			nb += float64(b[i]) * float64(b[i])
		}
		return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
	}
	// WHERE does Metal's context first break? Per-layer cos(Metal, int4-ref) across the whole
	// trunk: a break from layer 0 = a per-layer attention-op bug (RoPE base / window / QKV); a
	// clean early trunk that degrades late = accumulated drift feeding attention.
	t.Logf("=== per-layer cos(Metal ctx, int4-ref ctx) — where the attention context first breaks ===")
	for l := 0; l < nL; l++ {
		tag := "global"
		if m4.LayerIsLocalResident(l) {
			tag = "local "
		}
		t.Logf("  L%2d [%s]: cos(Metal,int4-ref)=%.4f", l, tag, cosCtx(ctx[l], i4Ctx[l]))
	}

	t.Logf("=== PRE-O-PROJ CONTEXT: Metal vs int4-ref (==CUDA-int4) vs f32-truth ===")
	for l := 31; l < nL; l++ {
		mc, i4c, fc := ctx[l], i4Ctx[l], tCtx[l]
		t.Logf("L%2d | cos(Metal,f32)=%.4f  cos(int4-ref,f32)=%.4f  cos(Metal,int4-ref)=%.4f",
			l, cosCtx(mc, fc), cosCtx(i4c, fc), cosCtx(mc, i4c))
		for _, idx := range cudaIdx[l] {
			if idx < len(mc) {
				t.Logf("     qDim %4d: metal=%+8.3f  int4-ref=%+8.3f  f32-truth=%+8.3f", idx, mc[idx], i4c[idx], fc[idx])
			}
		}
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
