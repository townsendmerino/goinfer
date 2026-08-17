//go:build realckpt

// realLogitOracle — the shared body of a T3 `real-model-oracle` gate: load a released
// checkpoint at int8 and match its last-token logits against an HF bf16 reference
// (argmax + cosine + greedy continuation), then record the measured row.
//
// It exists because the two Mamba-2 hybrid families (granitemoehybrid, nemotron_h) both
// needed one and the body is identical apart from the architecture name — deepseekRealGate
// is the same shape but carries MLA-specific assertions, so it could not be reused. Each
// caller keeps its own structural assertions (layer-kind split, multipliers, n_groups);
// this only covers the numerics.
//
// The int8-vs-bf16 shape is the one deepseek_v2/v3 and mellum already use: an f32 forward
// of a 7-9B model does not co-reside with the bf16 reference on this box, and goinfer has
// no bf16-resident mode. W8A8 is high-fidelity but not bit-exact, hence a 0.99 cosine bar
// rather than the 0.9999 the sub-2B f32 gates (llama, qwen2) can hold.
package decoder

import (
	"encoding/json"
	"os"
	"testing"
)

// realLogitOracle loads ckpt at int8, replays the golden's prompt ids, and gates
// argmax + cosine + greedy continuation against the HF bf16 reference.
//
// The parity row is emitted here, not by the caller, so that a family cannot record a
// row without having run these checks — emitParityRow is a no-op if any of them failed.
func realLogitOracle(t *testing.T, ckpt, golden, wantArch, family, reference string) {
	realLogitOracleQuant(t, ckpt, golden, wantArch, family, reference, "int8int8")
}

// realLogitOracleQuant is realLogitOracle with the load precision named. Most families use
// int8int8 (weights AND activations int8), which is what the resident GPU paths run. A family
// whose routing is too sensitive for int8 ACTIVATIONS passes its own quant — see
// nemotron3nano_real_test.go, where 6-of-128 sparse routing costs 0.978 at int8int8 and 0.9977
// with f32 activations, on a forward that is otherwise correct.
func realLogitOracleQuant(t *testing.T, ckpt, golden, wantArch, family, reference, quant string) {
	t.Helper()
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Skipf("no golden (%v) — run the pin script", err)
	}
	var g struct {
		PromptIDs       []int     `json:"prompt_ids"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
		NNew            int       `json:"n_new"`
		ContinuationIDs []int     `json:"continuation_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	m, err := Load(ckpt, Options{Quant: quant})
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()
	if m.w.arch.Name != wantArch {
		t.Fatalf("arch = %q, want %s", m.w.arch.Name, wantArch)
	}

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("%s parity: argmax got=%d want=%d | logit cosine=%.6f", family, gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.99 { // int8 W8A8 vs bf16 — same bar as the deepseek real gates
		t.Errorf("last-logit cosine %.6f < 0.99", cos)
	}

	got := make([]int, 0, g.NNew)
	for range g.NNew {
		id := argmax(logits)
		got = append(got, id)
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("continuation forward: %v", err)
		}
	}
	t.Logf("%s continuation got=%v want=%v", family, got, g.ContinuationIDs)
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d", i, got[i], g.ContinuationIDs[i])
			break
		}
	}
	emitParityRow(t, family, "real-model-oracle", reference, 100.0, float64(cos), float64(cos))
}
