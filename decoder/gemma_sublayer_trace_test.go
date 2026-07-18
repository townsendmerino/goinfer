package decoder

import (
	"os"
	"testing"
)

// TestGemmaSublayerTrace answers the Metal bisect's Fork-1 question: at the first layer a
// flagged channel's residual sign flips under int4, is the culprit the ATTENTION contribution
// (o-proj out) or the MLP contribution (down out)?
//
// It taps the per-sublayer contributions (ForwardSubCapture) at pos 5 of "The capital of France
// is" and reconstructs the running residual per channel from the scaled embedding, so the first
// layer where int4's running sign departs from bf16-f32's is exact — and it reports which
// sublayer (attn or mlp) carried the residual across zero.
//
// Ground truth is the REAL bf16 safetensors (f32), not a Q4_K_M-dequant, so it is authoritative
// even at the near-zero 1698-class channels. int4 is the byte-identical Q4_K_M gguf the Metal box
// runs (sha 882e8d2d), so this is the shared-source reference the cross-backend oracle needs; the
// CUDA resident path tracks this CPU-int4 by the committed parity gate, and the Metal W4A8 path
// tracks it up to the measured-negligible bf16->int8->int4 double-quant.
func TestGemmaSublayerTrace(t *testing.T) {
	bf16 := os.ExpandEnv("$HOME/models/gemma-3-4b-it")
	gguf := os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf")
	if _, e := os.Stat(bf16); e != nil {
		t.Skipf("no bf16 checkpoint at %s", bf16)
	}
	if _, e := os.Stat(gguf); e != nil {
		t.Skipf("no gguf at %s", gguf)
	}
	prompt := []int{2, 669, 5279, 529, 7001, 563} // "<bos> The capital of France is"
	const probe = 5
	channels := []int{1723, 227, 1698, 443}

	// running[c] = scaledEmbed[c] + sum over layers<=l of (attn_contrib + mlp_contrib), captured
	// after each sublayer. Returns two [nL][len(channels)] tables: residual-after-attn and
	// residual-after-mlp, plus the raw contributions.
	type trace struct {
		attnContrib, mlpContrib [][]float32 // [layer][channel]
		afterAttn, afterMLP     [][]float32 // [layer][channel] running residual
	}
	run := func(path, quant string) trace {
		m, err := Load(path, Options{Quant: quant})
		if err != nil {
			t.Fatalf("load %s (%s): %v", path, quant, err)
		}
		defer m.Close()
		nL := m.w.arch.NumLayers
		cache := m.NewCache(len(prompt) + 1)
		var attn, mlp [][]float32
		var scaledEmbed []float32
		for i, tok := range prompt {
			if i == probe {
				a, mm, _, _, err := m.ForwardSubCapture(tok, cache)
				if err != nil {
					t.Fatalf("ForwardSubCapture: %v", err)
				}
				attn, mlp = a, mm
				scaledEmbed = append([]float32(nil), m.EmbedResidentForTest(tok)...)
			} else if _, err := m.ForwardForTest(tok, cache); err != nil {
				t.Fatalf("warm pos %d: %v", i, err)
			}
		}
		tr := trace{
			attnContrib: make([][]float32, nL), mlpContrib: make([][]float32, nL),
			afterAttn: make([][]float32, nL), afterMLP: make([][]float32, nL),
		}
		run := make([]float32, len(scaledEmbed))
		copy(run, scaledEmbed)
		for l := 0; l < nL; l++ {
			ac := make([]float32, len(channels))
			mc := make([]float32, len(channels))
			aa := make([]float32, len(channels))
			am := make([]float32, len(channels))
			for j, c := range channels {
				ac[j] = attn[l][c]
			}
			for i := range run {
				run[i] += attn[l][i]
			}
			for j, c := range channels {
				aa[j] = run[c]
				mc[j] = mlp[l][c]
			}
			for i := range run {
				run[i] += mlp[l][i]
			}
			for j, c := range channels {
				am[j] = run[c]
			}
			tr.attnContrib[l], tr.mlpContrib[l], tr.afterAttn[l], tr.afterMLP[l] = ac, mc, aa, am
		}
		return tr
	}

	tF := run(bf16, "")     // f32 ground truth (real bf16)
	tQ := run(gguf, "int4") // Q4_K_M int4 (the shared source with the Metal box)
	nL := len(tF.afterMLP)

	sgn := func(x float32) int {
		if x > 0 {
			return 1
		}
		if x < 0 {
			return -1
		}
		return 0
	}
	// A MEANINGFUL flip needs the f32 value to actually have magnitude: these channels straddle
	// zero as sub-1 noise for the first ~28 layers, so a "sign difference" there is meaningless.
	// Only count a divergence once |f32 residual| clears this floor.
	const meaningful = 10.0
	absf := func(x float32) float64 {
		if x < 0 {
			return float64(-x)
		}
		return float64(x)
	}
	for j, c := range channels {
		flipL, flipWhere := -1, ""
		for l := 0; l < nL && flipL < 0; l++ {
			if absf(tF.afterAttn[l][j]) > meaningful && sgn(tQ.afterAttn[l][j]) != sgn(tF.afterAttn[l][j]) {
				flipL, flipWhere = l, "ATTENTION"
			} else if absf(tF.afterMLP[l][j]) > meaningful && sgn(tQ.afterMLP[l][j]) != sgn(tF.afterMLP[l][j]) {
				flipL, flipWhere = l, "MLP"
			}
		}
		fF, fQ := tF.afterMLP[nL-1][j], tQ.afterMLP[nL-1][j]
		t.Logf("=== channel %d ===", c)
		if flipL >= 0 {
			t.Logf("  int4 MEANINGFULLY flips f32's sign: first at layer %d, %s sublayer", flipL, flipWhere)
		} else {
			infl := 0.0
			if fF != 0 {
				infl = (float64(fQ)/float64(fF) - 1) * 100
			}
			t.Logf("  NO meaningful flip on this box's int4 — final sign held; magnitude inflated %+.1f%%", infl)
		}
		t.Logf("  final residual: f32(truth)=%.3f  int4(Q4_K_M)=%.3f", fF, fQ)
		// Show the late layers, where these channels actually acquire magnitude (they are near-zero
		// noise before ~layer 28). This is the region a real divergence lives in, and the window the
		// Metal box overlays its own attn/mlp contributions onto to find where Metal departs.
		lo := max(0, nL-7)
		t.Logf("  %-3s | %10s %10s %11s | %10s %10s %11s", "L",
			"f32.attnC", "f32.mlpC", "f32.resid", "i4.attnC", "i4.mlpC", "i4.resid")
		for l := lo; l < nL; l++ {
			t.Logf("  %-3d| %10.3f %10.3f %11.3f | %10.3f %10.3f %11.3f", l,
				tF.attnContrib[l][j], tF.mlpContrib[l][j], tF.afterMLP[l][j],
				tQ.attnContrib[l][j], tQ.mlpContrib[l][j], tQ.afterMLP[l][j])
		}
	}
}
