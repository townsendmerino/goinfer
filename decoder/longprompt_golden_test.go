package decoder

import (
	"context"
	"runtime"
	"testing"
)

// longPromptIDs is a deterministic ≥512-token prompt. The CONTENT does not matter; the LENGTH
// does — it must sit above fastAttnMinPrompt so this golden exercises the f32 prefill path that
// every other forward golden misses.
func longPromptIDs(n int) []int {
	p := make([]int, n)
	for i := range p {
		p[i] = 700 + (i*7919)%9000 // fixed sequence, no RNG, no model-specific vocabulary
	}
	return p
}

// longPromptFastWant is the greedy continuation of longPromptIDs(768) on the bench model through the DEFAULT
// (f32, fused-schedule) prefill path. Regenerate ONLY when a numerics change to that path is intentional and
// reviewed, the same contract parityWant carries for the exact path.
//
// It is KEYED BY GOARCH on purpose: arm64 and amd64 diverge at the FIRST generated token on the same checkpoint,
// which the exact kernel does not do (parityWant is one list for both). gc fuses x*y+z into FMA on arm64 and not
// on amd64 (docs/parity-coverage-policy.md, "CPU reference is arch-scoped"), and f32 attention has no f64
// accumulator to absorb the difference. A single shared list would be a permanently red gate on one runner.
//
// The prompt is synthetic nonsense (700 + (i*7919)%9000), so a degenerate continuation is not read as a quality
// signal: the golden pins REPRODUCIBILITY, not quality, and these continuations sit on knife-edges (a
// tie-flip reading, not a defect one). Pre-fusion lists and the evidence:
// docs/code-notes/decoder.md#longPromptFastWant.
var longPromptFastWant = map[string][]int{
	"arm64": {13, 715, 522, 2599, 397, 522, 1551, 397, 522, 2599, 397, 522, 1551, 397, 522, 2599},
	"amd64": {11, 714, 279, 1196, 374, 537, 2952, 311, 1490, 279, 1196, 594, 3139, 13, 576, 1196},
}

// TestLongPromptFast_forwardParity is the golden for the f32 fast-attention prefill path, the shipped default:
// every other forward golden uses a prompt SHORTER than fastAttnMinPrompt and takes the exact kernel, so a
// numerics change to the default path was invisible to them (scripts/refresh_parity_hashes.sh would call it a
// "non-numeric core refresh").
//
// THE NAME ENDS IN _forwardParity DELIBERATELY: that is what the refresh script's selector matches
// ((_forwardParity|_logitParity|_textParity)$|^TestGGUF_.*_parity$), and a gate the refresh script does not run
// does not protect it.
func TestLongPromptFast_forwardParity(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	const K = 768 // > fastAttnMinPrompt (512), so this is the f32 path
	if K < fastAttnMinPrompt {
		t.Fatalf("K=%d is below the floor %d — this golden would silently test the exact path", K, fastAttnMinPrompt)
	}
	if !m.canBatchN(K) {
		t.Skip("model has no batched prefill")
	}
	prompt := longPromptIDs(K)

	// Default path (no env override): this is what a user gets. t.Setenv pins the DEFAULT-on value for both knobs
	// that can perturb this golden's float sequence, not only GOINFER_CPU_FAST_ATTENTION: an ambient
	// GOINFER_FUSED_ATTENTION=0 (fusedattn.go's documented opt-out) would switch the run to the materialized
	// arithmetic and fail the golden, which was recorded fused.
	if len(forcedFallbacks()) > 0 {
		forcedLongPromptCloseness(t, m, prompt)
		return
	}
	setKnob(t, m, knobCPUFastAttention, "1")
	setKnob(t, m, knobFusedAttention, "1")
	out, gen := m.Generate(context.Background(), prompt, 16, SamplingParams{Temperature: 0})
	var got []int
	for id := range out {
		got = append(got, id)
	}
	if gen.Err() != nil {
		t.Fatalf("generate: %v", gen.Err())
	}
	want, ok := longPromptFastWant[runtime.GOARCH]
	if !ok {
		t.Skipf("no f32 golden recorded for GOARCH=%s — record one rather than comparing against another arch's", runtime.GOARCH)
	}
	for i := range want {
		if i >= len(got) || got[i] != want[i] {
			t.Fatalf("f32 prefill continuation drifted at %d (GOARCH=%s):\n got %v\nwant %v", i, runtime.GOARCH, got, want)
		}
	}
	// NON-VACUITY: prove this golden is on the f32 side. If the floor moved or the default were reverted, the
	// assertions above would still pass while testing the exact kernel.
	setKnob(t, m, knobCPUFastAttention, "0")
	exOut, _ := m.Generate(context.Background(), prompt, 16, SamplingParams{Temperature: 0})
	var exact []int
	for id := range exOut {
		exact = append(exact, id)
	}
	same := len(exact) == len(got)
	for i := range exact {
		if i < len(got) && exact[i] != got[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("the default and the exact kernel produced identical output at K=768 — this golden " +
			"is NOT covering the f32 path (floor raised above 768, or the default reverted?)")
	}
}

// forcedLongPromptCloseness replaces TestLongPromptFast_forwardParity's token-exact comparison under a forced
// CPU fallback. The golden pins one realization of the f32 fast-attention path (a documented, accepted
// divergence from the exact path), and the top logits sit close enough that a different realization reorders
// the continuation. The draw-independent question: how far is the fast path from the EXACT path on this build?
// Figures: docs/code-notes/decoder.md#forcedLongPromptCloseness.
func forcedLongPromptCloseness(t *testing.T, m *Model, prompt []int) {
	t.Helper()
	_, nL, _, nKV, hd, _, _ := m.Dims()
	logits := func(fast string) []float32 {
		setKnob(t, m, knobCPUFastAttention, fast)
		setKnob(t, m, knobFusedAttention, "1")
		cache := NewKVCache(nL, nKV, hd, 0, len(prompt)+8, nil)
		lg, err := m.prefillLogits(context.Background(), prompt, cache)
		if err != nil {
			t.Fatalf("prefillLogits(fast=%s): %v", fast, err)
		}
		return append([]float32(nil), lg...)
	}
	exact, fast := logits("0"), logits("1")
	if d := cosineDistance(fast, exact); d > forcedLogitCloseness {
		t.Errorf("forced %v: f32 fast attention against the exact path, 1-cosine = %.3e at %d tokens, want <= %.0e — the forced fast path drifted", forcedFallbacks(), d, len(prompt), forcedLogitCloseness)
	} else {
		t.Logf("forced %v: f32 fast attention against the exact path, 1-cosine = %.3e at %d tokens (bound %.0e)", forcedFallbacks(), d, len(prompt), forcedLogitCloseness)
	}
}
