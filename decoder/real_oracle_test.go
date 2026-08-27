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

	// The load is the opaque part: for the qwen3_next 80B it streams 162.7GB of bf16 off disk and
	// quantizes it, taking ~10 minutes during which the test emits nothing at all. Counting is not
	// possible from out here, so the phase name plus the io= field carries it.
	prog := newProgress(t, t.Name(), len(g.PromptIDs)+g.NNew)
	prog.Phase("load " + quant + " (streams + quantizes the full checkpoint)")
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
	prog.Phase("prefill the golden prompt")
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
		prog.Step(1)
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("%s parity: argmax got=%d want=%d | logit cosine=%.6f", family, gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	floor := oracleCosFloor(t, quant)
	if cos < floor {
		t.Errorf("last-logit cosine %.6f < %.4g (the %s bar)", cos, floor, quant)
	}

	got := make([]int, 0, g.NNew)
	prog.Phase("greedy continuation")
	for range g.NNew {
		id := argmax(logits)
		got = append(got, id)
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("continuation forward: %v", err)
		}
		prog.Step(1)
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

// oracleCosFloor is the last-logit cosine bar for a T3 real-model oracle, BY PRECISION.
//
// There was one bar for a long time, 0.99, and its comment said what it was: "int8 W8A8 vs bf16 —
// same bar as the deepseek real gates". That was fine while every real oracle was int8. It stopped
// being fine when qwen3_next arrived at int4 — not by choice but by capacity, since 80B at int8 is
// ~80 GB against 62 GB of RAM — and was measured against a bar calibrated on a population it is
// not in. int4 is a coarser grid than int8; holding both to one number is the same category of
// error as holding gpt2's int4 goldens to the tiny fixtures' absolute gate.
//
// An UNREGISTERED precision is a hard failure rather than a default. Silently inheriting a bar
// belonging to some other precision is exactly how the int4 gate ended up judged by an int8 number,
// and a default would let the next precision repeat it without anyone deciding anything.
func oracleCosFloor(t *testing.T, quant string) float64 {
	t.Helper()
	switch quant {
	case "int8int8", "int8":
		// Calibrated on the deepseek real gates. int8 weights with f32 or int8 activations.
		return 0.99
	case "int4":
		// Pre-registered in docs/queue-correctness.md G5 BEFORE the qwen3_next checkpoint had
		// finished downloading and before any number existed — which is what makes it usable.
		// It is not the int8 bar relaxed after a near miss; it is the bar written down in advance
		// for the precision the run was always going to use.
		return 0.98
	default:
		t.Fatalf("no oracle cosine bar registered for quant %q — decide one deliberately in "+
			"oracleCosFloor rather than letting it inherit another precision's", quant)
		return 0
	}
}
