package decoder

import (
	"context"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestNgramSpecMLA_parity flips the MLA foundation gate (00-core §6 / experiments §0):
// n-gram speculative decode on a DeepSeek-V2 (MLA latent-KV) model must be
// token-identical to plain greedy. MLA's rollback is KVCache.TruncateTo reslicing the
// per-layer mlaLatent store — already correct, and already allowed by
// specRollbackSafe — so this proves it rather than building anything. MLA is excluded
// from canBatchN (the verify runs sequential forwardN), which exercises the
// truncate-rollback after each round on a non-batched family.
//
// Gated on the local DeepSeek-V2-Lite GGUF (GOINFER_DEEPSEEK_GGUF); skips without it.
func TestNgramSpecMLA_parity(t *testing.T) {
	requireHeavyModel(t)
	gguf := assetPath(t, "GOINFER_DEEPSEEK_GGUF")
	m, err := Load(gguf, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load(%s): %v", gguf, err)
	}
	defer m.Close()
	if m.w.arch.mla == nil {
		t.Fatalf("arch %q has no MLA", m.w.arch.Name)
	}
	if !m.specRollbackSafe() {
		t.Fatal("MLA must be spec-rollback-safe (latent KV is truncatable)")
	}
	tk, err := tokenizer.LoadGGUF(gguf)
	if err != nil {
		t.Fatalf("LoadGGUF tokenizer: %v", err)
	}
	// Repetition-heavy prompt so the n-gram drafter actually fires (acceptance > 0)
	// and the verify/rollback runs over real accepted+rejected blocks.
	ids, err := tk.Encode("List: alpha, beta, gamma, delta. Repeat the list: alpha, beta, gamma,", true)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	const n = 32
	ctx := context.Background()
	greedy := SamplingParams{Temperature: 0}

	prog := newProgress(t, t.Name(), 0)
	prog.Phase("reference greedy (32 tok)")
	ref := collectTokens(first(m.Generate(ctx, ids, n, greedy)))
	prog.Phase("n-gram speculative (32 tok)")
	ch, g, err := m.GenerateNgramSpeculative(ctx, ids, n, &NgramDrafter{}, 8, greedy)
	if err != nil {
		t.Fatalf("GenerateNgramSpeculative: %v", err)
	}
	got := collectTokens(ch)
	if g.Err() != nil {
		t.Fatalf("spec stream err: %v", g.Err())
	}
	if !slices.Equal(got, ref) {
		t.Fatalf("MLA n-gram spec != greedy (rollback bug)\n got %v\n ref %v", got, ref)
	}
	t.Logf("MLA spec parity OK: %d tok, acceptance %.3f, %.2f tok/round", len(got), g.Spec.AcceptanceRate(), g.Spec.TokensPerRound())
}
