package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// Gemma 1 and Gemma 2 against HF f32 (eager) on seeded tiny checkpoints (scripts/pin_gemma_tiny.py), built so every
// family feature moves the output: random norm weights under the (1+w) RMSNorm, the sqrt(hidden) embedding scale,
// head_dim != hidden/heads; for Gemma 1 MQA; for Gemma 2 the sandwich norms, query_pre_attn_scalar != head_dim, a
// sliding window of 4 under a 12-token prompt, and both softcaps (the pin script refuses a fixture where turning either
// off does not move the logits). Each family is checked twice: the per-token forward (attendQuery) and the batched
// prefill (attendBatchedHeads), since the attention softcap is applied at each scoring site separately.
func TestGemma12_forwardParity(t *testing.T) {
	for _, tc := range []struct {
		name, arch, file string
		sandw            bool
		softcap          bool
	}{{"gemma1", "gemma", "", false, false}, {"gemma2", "gemma2", "", true, true},
		{"gemma1", "gemma", ".gguf", false, false}, {"gemma2-hd", "gemma2", ".gguf", true, true}} {
		t.Run(tc.name+tc.file, func(t *testing.T) {
			dir, gold, full := "../testdata/"+tc.name+"-tiny"+tc.file, "../testdata/"+tc.name+"_forward_golden.json", "../testdata/"+tc.name+"_forward_full.json"
			raw, err := os.ReadFile(gold)
			if errors.Is(err, fs.ErrNotExist) {
				t.Skipf("no golden at %s — scripts/pin_gemma_tiny.py", gold)
			}
			if err != nil {
				t.Fatal(err)
			}
			var g forwardGolden
			if err := json.Unmarshal(raw, &g); err != nil {
				t.Fatal(err)
			}
			m, err := Load(dir, Options{})
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			defer m.Close()
			a := m.w.arch
			if tc.file != "" { // the GGUF load must derive what the HF config states
				st, err := Load("../testdata/"+tc.name+"-tiny", Options{})
				if err != nil {
					t.Fatal(err)
				}
				b := st.w.arch
				st.Close()
				if a.AttnScale != b.AttnScale || a.SlidingWindow != b.SlidingWindow || a.AttnLogitSoftcap != b.AttnLogitSoftcap ||
					a.FinalLogitSoftcap != b.FinalLogitSoftcap || a.NumKVHeads != b.NumKVHeads || a.HeadDim != b.HeadDim {
					t.Fatalf("GGUF arch %+v differs from the safetensors one %+v", a, b)
				}
				for i := range a.NumLayers {
					if a.isGlobalLayer(i) != b.isGlobalLayer(i) {
						t.Fatalf("layer %d: GGUF global %v, safetensors %v", i, a.isGlobalLayer(i), b.isGlobalLayer(i))
					}
				}
			}
			if a.Name != tc.arch || !a.RMSAddOne || a.EmbedScale != 8 || !a.TiedLMHead || a.QKNorm {
				t.Fatalf("resolved %s: RMSAddOne %v EmbedScale %v Tied %v QKNorm %v", a.Name, a.RMSAddOne, a.EmbedScale, a.TiedLMHead, a.QKNorm)
			}
			if (a.NormPlacement == NormSandwich4) != tc.sandw || (a.AttnLogitSoftcap == 5 && a.FinalLogitSoftcap == 3) != tc.softcap {
				t.Fatalf("%s: placement %v softcaps %v/%v", a.Name, a.NormPlacement, a.AttnLogitSoftcap, a.FinalLogitSoftcap)
			}
			check := func(path string, logits []float32) {
				t.Helper()
				if got := argmax(logits); got != g.Argmax {
					t.Errorf("%s: argmax %d, want %d", path, got, g.Argmax)
				}
				var maxd float64
				for _, kv := range append(append([][2]float64(nil), g.TopK...), g.Sample...) {
					maxd = math.Max(maxd, math.Abs(float64(logits[int(kv[0])])-kv[1]))
				}
				cos := fullCosine(t, logits, full)
				t.Logf("%s %s: max |Δlogit| %.2e, cosine %.8f", tc.name, path, maxd, cos)
				if maxd > 1e-3 || cos < 0.99999 {
					t.Errorf("%s: max |Δlogit| %.3g, cosine %.8f against HF", path, maxd, cos)
				}
			}
			cache := m.NewCache(len(g.IDs))
			for _, id := range g.IDs[:len(g.IDs)-1] {
				if _, err := m.runLayers(id, cache); err != nil {
					t.Fatal(err)
				}
			}
			lg, err := m.forward(g.IDs[len(g.IDs)-1], cache)
			if err != nil {
				t.Fatal(err)
			}
			check("per-token", lg)
			if !m.canBatchN(len(g.IDs)) {
				t.Fatalf("%s: the batched prefill does not apply", tc.name)
			}
			bl, err := m.prefillLogits(context.Background(), g.IDs, m.NewCache(len(g.IDs)))
			if err != nil {
				t.Fatal(err)
			}
			check("batched", bl)
			if tc.file != "" {
				return
			}
			if tc.name == "gemma2" {
				emitParityRow(t, "gemma2", "tiny-golden", "HF f32 eager (gemma2-tiny seeded fixture, sandwich norms + sliding 4 + softcaps 5/3), safetensors and GGUF", 100.0, fullCosine(t, lg, full), fullCosine(t, lg, full))
			} else {
				emitParityRow(t, "gemma", "tiny-golden", "HF f32 eager (gemma1-tiny seeded fixture, MQA + (1+w) RMSNorm), safetensors and GGUF", 100.0, fullCosine(t, lg, full), fullCosine(t, lg, full))
			}
		})
	}
}
