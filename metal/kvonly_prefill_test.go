//go:build darwin

package metal

import (
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestForwardNoLogits_byteIdenticalKV gates M-01 (audit-metal-2026-09-12.md): ForwardNoLogits
// must write EXACTLY the K/V that Forward would at the same position, so a decode continuing
// from the last prompt token is byte-identical whether prefill used the full-logits Forward loop
// (every prompt token pays the LM head) or the KV-only skip (ForwardNoLogits on every token but
// the last). Mirrors cuda/kvonly_prefill_test.go's TestKVOnlyPrefill_byteIdentical_tiny, at the
// logits level rather than via full Generate (the tiny synthetic fixture here has no tokenizer).
func TestForwardNoLogits_byteIdenticalKV(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	w := genTinyWeights(rand.New(rand.NewSource(7)))
	dir := t.TempDir()
	writeDense(t, dir, w)

	load := func() *metalResident {
		m, err := decoder.Load(dir, decoder.Options{Quant: "int8int8"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		r, err := buildResident(m)
		if err != nil {
			t.Fatalf("build resident: %v", err)
		}
		return &metalResident{r: r, hidden: r.H}
	}

	const n = 12 // multi-token prompt so several tokens take the KV-only path
	embs := make([][]float32, n)
	for i := range embs {
		embs[i] = make([]float32, tmHidden)
		for j := range embs[i] {
			embs[i][j] = float32(i*7+j) * 0.01 // non-zero: a zeroed row can't distinguish assembly bugs
		}
	}

	// Baseline: full-logits Forward for every position.
	full := load()
	var fullLogits []float32
	for i, e := range embs {
		l, err := full.Forward(e, i)
		if err != nil {
			t.Fatalf("full Forward(%d): %v", i, err)
		}
		fullLogits = l
	}

	// KV-only: ForwardNoLogits for every position but the last, then Forward for the last.
	kvOnly := load()
	for i := 0; i < n-1; i++ {
		if err := kvOnly.ForwardNoLogits(embs[i], i); err != nil {
			t.Fatalf("ForwardNoLogits(%d): %v", i, err)
		}
	}
	kvLogits, err := kvOnly.Forward(embs[n-1], n-1)
	if err != nil {
		t.Fatalf("final Forward: %v", err)
	}

	if len(fullLogits) != len(kvLogits) {
		t.Fatalf("logits length differs: full %d vs kv-only %d", len(fullLogits), len(kvLogits))
	}
	for i := range fullLogits {
		if fullLogits[i] != kvLogits[i] {
			t.Fatalf("logit[%d] NOT byte-identical: full=%.8f kv-only=%.8f — ForwardNoLogits wrote "+
				"different K/V than full-logits Forward", i, fullLogits[i], kvLogits[i])
		}
	}
	t.Logf("ForwardNoLogits == full-logits Forward: %d logits byte-identical", len(fullLogits))
}

// TestForwardNoLogits_pagedMoEFallback gates the C-02 constraint M-01's fix noted:
// forwardHiddenNoHead's trunk encoder has no paged-MoE branch (only Forward → forwardLogitsPaged/
// forwardLogitsMoEPaged does), so ForwardNoLogits must fall back to the full head-bearing Forward
// on a paged resident instead of taking the trunk-only path — same byte-identity requirement as
// the dense case above, on the arch the bare fix would have silently corrupted.
func TestForwardNoLogits_pagedMoEFallback(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const ckpt = "../decoder/testdata/qwen3_5_moe-tiny" // nE=4, topK=2 (config.json)
	if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
		t.Skipf("no qwen3_5_moe fixture at %s (run scripts/pin_qwen3_5_forward.py --moe): %v", ckpt, err)
	}
	t.Setenv("GOINFER_METAL_MOE_SLOTS", "3") // < nE=4, >= topK=2 → forces mo.paged

	load := func() *metalResident {
		m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		r, err := buildResident(m)
		if err != nil {
			t.Fatalf("build resident: %v", err)
		}
		if r.moe == nil || !r.moe.paged {
			t.Fatal("fixture did not build a paged MoE resident — GOINFER_METAL_MOE_SLOTS not honored")
		}
		return &metalResident{r: r, hidden: r.H}
	}

	full := load()
	const n = 6
	embs := make([][]float32, n)
	for i := range embs {
		embs[i] = make([]float32, full.hidden)
		for j := range embs[i] {
			embs[i][j] = float32(i*5+j) * 0.01
		}
	}

	var fullLogits []float32
	for i, e := range embs {
		l, err := full.Forward(e, i)
		if err != nil {
			t.Fatalf("full Forward(%d): %v", i, err)
		}
		fullLogits = l
	}

	kvOnly := load()
	for i := 0; i < n-1; i++ {
		if err := kvOnly.ForwardNoLogits(embs[i], i); err != nil {
			t.Fatalf("ForwardNoLogits(%d): %v", i, err)
		}
	}
	kvLogits, err := kvOnly.Forward(embs[n-1], n-1)
	if err != nil {
		t.Fatalf("final Forward: %v", err)
	}

	for i := range fullLogits {
		if fullLogits[i] != kvLogits[i] {
			t.Fatalf("logit[%d] NOT byte-identical on paged MoE: full=%.8f kv-only=%.8f",
				i, fullLogits[i], kvLogits[i])
		}
	}
	t.Logf("paged-MoE ForwardNoLogits fallback == full-logits Forward: %d logits byte-identical", len(fullLogits))
}

// TestForwardNoLogits_pipelineTransitionParity verifies that transitioning across
// ForwardNoLogits (noHead) and Forward (withHead) modes across multiple conversational
// turns remains 100% byte-identical to running the reference full-logits forward.
// This tests:
// 1. Prefill (noHead) -> Prompt End (withHead) -> Decode (withHead)
// 2. Fresh Turn Reset (pos=0) -> Prefill (noHead) -> Prompt End -> Decode
// 3. Executor pool draining and command buffer re-encoding on mode switches.
func TestForwardNoLogits_pipelineTransitionParity(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	w := genTinyWeights(rand.New(rand.NewSource(42)))
	dir := t.TempDir()
	writeDense(t, dir, w)

	load := func() *metalResident {
		m, err := decoder.Load(dir, decoder.Options{Quant: "int8int8"})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		r, err := buildResident(m)
		if err != nil {
			t.Fatalf("build resident: %v", err)
		}
		return &metalResident{r: r, hidden: r.H}
	}

	full := load()
	pipe := load()

	// Run two turns:
	turns := []struct {
		prefillLen int
		decodeLen  int
	}{
		{prefillLen: 6, decodeLen: 4},
		{prefillLen: 8, decodeLen: 3},
	}

	rnd := rand.New(rand.NewSource(99))
	for turnIdx, turn := range turns {
		total := turn.prefillLen + turn.decodeLen
		embs := make([][]float32, total)
		for i := range embs {
			embs[i] = make([]float32, tmHidden)
			for j := range embs[i] {
				embs[i][j] = (rnd.Float32() - 0.5) * 0.1
			}
		}

		// Full reference: all tokens use Forward
		fullLogits := make([][]float32, total)
		for i, e := range embs {
			l, err := full.Forward(e, i)
			if err != nil {
				t.Fatalf("turn %d full Forward(%d): %v", turnIdx, i, err)
			}
			fullLogits[i] = append([]float32(nil), l...)
		}

		// Pipelined: prefill tokens use ForwardNoLogits, then Forward for prompt end & decode
		pipeLogits := make([][]float32, total)
		for i := 0; i < turn.prefillLen-1; i++ {
			if err := pipe.ForwardNoLogits(embs[i], i); err != nil {
				t.Fatalf("turn %d pipe ForwardNoLogits(%d): %v", turnIdx, i, err)
			}
		}
		for i := turn.prefillLen - 1; i < total; i++ {
			l, err := pipe.Forward(embs[i], i)
			if err != nil {
				t.Fatalf("turn %d pipe Forward(%d): %v", turnIdx, i, err)
			}
			pipeLogits[i] = append([]float32(nil), l...)
		}

		// Verify that all positions that produced logits match byte-identically
		for i := turn.prefillLen - 1; i < total; i++ {
			f := fullLogits[i]
			p := pipeLogits[i]
			if len(f) != len(p) {
				t.Fatalf("turn %d pos %d: len mismatch %d vs %d", turnIdx, i, len(f), len(p))
			}
			for j := range f {
				if f[j] != p[j] {
					t.Fatalf("turn %d pos %d logit[%d] NOT byte-identical: full=%.8f pipe=%.8f",
						turnIdx, i, j, f[j], p[j])
				}
			}
		}
	}
	t.Logf("pipeline transition parity: all turns and decode steps 100%% byte-identical")
}

func BenchmarkForwardNoLogits_SyncVsPipe(b *testing.B) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		b.Skipf("no metal device: %v", err)
	}
	w := genTinyWeights(rand.New(rand.NewSource(17)))
	dir := b.TempDir()
	writeDense(&testing.T{}, dir, w)

	load := func() *resident {
		m, err := decoder.Load(dir, decoder.Options{Quant: "int8int8"})
		if err != nil {
			b.Fatalf("load: %v", err)
		}
		r, err := buildResident(m)
		if err != nil {
			b.Fatalf("build resident: %v", err)
		}
		return r
	}

	const nTokens = 32
	embs := make([][]float32, nTokens)
	for i := range embs {
		embs[i] = make([]float32, tmHidden)
		for j := range embs[i] {
			embs[i][j] = float32(i*13+j) * 0.01
		}
	}

	b.Run("Sync", func(b *testing.B) {
		r := load()
		b.ResetTimer()
		for it := 0; it < b.N; it++ {
			for i := 0; i < nTokens; i++ {
				if _, err := r.forwardHiddenNoHead(embs[i], i, false); err != nil {
					b.Fatalf("sync pos %d: %v", i, err)
				}
			}
		}
	})

	b.Run("Pipelined", func(b *testing.B) {
		r := load()
		b.ResetTimer()
		for it := 0; it < b.N; it++ {
			for i := 0; i < nTokens; i++ {
				r.ForwardEmbNoLogitsPipe(embs[i], i)
			}
		}
		r.stopExec()
	})
}
