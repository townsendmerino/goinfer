package decoder

import (
	"context"
	"os"
	"runtime"
	"slices"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// loadInt4Model loads GOINFER_PREQUANT_GGUF as group-wise int4 — the W4A8 path,
// the int4 analog of loadBenchModel's int8int8. A plain q4_k_m GGUF works: the
// loader dequantizes it and re-quantizes to goinfer's group-wise int4, so no
// prebuilt .giw bundle (or torch) is needed. Fresh model per call (no cache).
func loadInt4Model(tb testing.TB) *Model {
	// The path resolves through the asset registry's candidate list like the other readers of this variable, so a
	// bare `go test ./decoder` runs this gate instead of skipping when GOINFER_PREQUANT_GGUF is unset.
	path := assetPath(tb, "GOINFER_PREQUANT_GGUF")
	m, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		tb.Fatalf("load int4: %v", err)
	}
	return m
}

// parityWantInt4ByArch pins the 0.5B's int4 (W4A8) greedy continuation of parityPrompt, one list per host
// architecture, the int4 analog of parityWant (int8int8). It guards the WHOLE int4 forward, including the
// prefill→W4A8 switch (decoder/weightmat.go), against silent numerics drift: greedy + fixed prompt is fully
// deterministic. The first 5 ids match parityWant; they diverge after as int4's coarser weights bend the
// path.
//
// Regenerate ONLY when an int4 numerics change is intentional and parity-reviewed. The decision to re-capture
// rather than call a drift a regression is the human decision docs/parity-coverage-policy.md requires: score
// old and new lists by how many leading ids match an f32 forward, and do not revert to hold the gate green if
// the new list is the more faithful one (that would pin the less correct path).
//
// The CPU int4 path is bit-identical within an architecture but not across them (arm64 fuses multiply-adds
// that amd64 rounds separately; docs/parity-coverage-policy.md, "arch-scoped"), so each architecture has its
// own golden, and an architecture with no entry is a capture run (the test logs the list and skips).
// Re-capture history and the evidence for each: docs/code-notes/decoder.md#parityWantInt4ByArch.
var parityWantInt4ByArch = map[string][]int{
	"arm64": {4710, 73594, 12669, 198, 750, 1438, 4136, 3932, 262, 671, 1096, 374, 264, 6573, 315, 2038, 429, 3880, 311, 387, 10865, 198, 262, 1494},
	"amd64": {4710, 73594, 12669, 198, 750, 1438, 4136, 3932, 262, 671, 1096, 374, 264, 5878, 369, 279, 5042, 2038, 198, 262, 1494, 271, 8960, 4136},
	// Captured under Intel SDE, because no VNNI machine is on hand. The VNNI kernels are integer arithmetic,
	// which SDE executes exactly, so the list is what a real VNNI CPU computes. A real VNNI host (Ice Lake and
	// newer) is the check on that claim, and the first one to run this test with the asset is owed a look. It
	// parts from the AVX2 list at id 22 (333, not 8960). Evidence: docs/measurements/sde-2026-10-04/README.md.
	"amd64-vnni": {4710, 73594, 12669, 198, 750, 1438, 4136, 3932, 262, 671, 1096, 374, 264, 5878, 369, 279, 5042, 2038, 198, 262, 1494, 271, 333, 1304},
}

// int4GoldenKey is the key into parityWantInt4ByArch for THIS host: the architecture, plus "-vnni" on an
// amd64 host whose dispatcher is using the AVX-512 VNNI W4A8 kernels, which compute a different quantised
// result from the AVX2 path (docs/measurements/sde-2026-10-04/README.md). It reads ACTIVE kernels, not
// detected ones, so a host whose self-test stepped VNNI down to AVX2 compares against the AVX2 list. Call it
// AFTER the first Load (the CPU self-test runs there).
func int4GoldenKey() string {
	k := linalg.ActiveKernels()
	if k.Arch == "amd64" && slices.Contains(k.Active, "avx512vnni") {
		return "amd64-vnni"
	}
	return runtime.GOARCH
}

// TestDecodeParityInt4 greedily continues parityPrompt at int4 and checks the token ids against
// parityWantInt4ByArch. The prompt is prefilled (batched forwardLayersN → W4A8) then decoded (W4A8), so a
// regression in either the prefill switch or the decode kernel moves the continuation and trips here.
// Skips without the model asset.
func TestDecodeParityInt4(t *testing.T) {
	m := loadInt4Model(t)
	defer m.Close()
	if len(forcedFallbacks()) > 0 {
		forcedInt4Closeness(t, m)
		return
	}

	const n = 24
	out, gen := m.Generate(context.Background(), parityPrompt, n, SamplingParams{Temperature: 0})
	var got []int
	for tok := range out {
		got = append(got, tok)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	parityWantInt4 := parityWantInt4ByArch[int4GoldenKey()]
	if len(parityWantInt4) == 0 {
		t.Logf("CAPTURE parityWantInt4 = %#v", got)
		t.Skipf("parityWantInt4 has no %s entry — capture run", int4GoldenKey())
	}
	if len(got) != len(parityWantInt4) {
		t.Fatalf("got %d tokens, want %d: %#v", len(got), len(parityWantInt4), got)
	}
	for i := range got {
		if got[i] != parityWantInt4[i] {
			t.Fatalf("int4 parity drift at %d: got %d want %d\n got=%#v\nwant=%#v",
				i, got[i], parityWantInt4[i], got, parityWantInt4)
		}
	}
}

// BenchmarkPrefillInt4 times an end-to-end int4 prefill via the batched M=K path (forwardLayersN → W4A8),
// the prefill→W4A8 payoff. To compare against the old Q4 split, check out the parent's decoder/weightmat.go
// and rerun:
//
//	M=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
//	GOINFER_PREQUANT_GGUF=$M GOINFER_PREFILL_LEN=2048 \
//	  go test ./decoder -run '^$' -bench BenchmarkPrefillInt4 -benchtime 8x
func BenchmarkPrefillInt4(b *testing.B) {
	m := loadInt4Model(b)
	defer m.Close()
	linalg.SetParallelThreshold(DefaultDecodeParallelThreshold)

	L := 2048
	if s := os.Getenv("GOINFER_PREFILL_LEN"); s != "" {
		if v, err := atoiPositive(s); err == nil {
			L = v
		}
	}
	b.Logf("canBatchN(%d)=%v", L, m.canBatchN(L))

	vocab := m.w.arch.VocabSize
	prompt := make([]int, L)
	for i := range prompt {
		prompt[i] = (i*131 + 7) % vocab
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache := m.NewCache(L + 4)
		if _, err := m.prefillLogits(context.Background(), prompt, cache); err != nil {
			b.Fatalf("prefill: %v", err)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(L)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

// atoiPositive parses a positive decimal length (small helper to avoid pulling
// strconv into the test for one parse).
func atoiPositive(s string) (int, error) {
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0, os.ErrInvalid
		}
		n = n*10 + int(c-'0')
	}
	if n == 0 {
		return 0, os.ErrInvalid
	}
	return n, nil
}

// forcedInt4Closeness replaces TestDecodeParityInt4's token-exact comparison under a forced CPU fallback.
// The golden pins the default path's greedy continuation, whose 4th token is a near-tie that a different
// rounding draw flips, so the forced run asks the draw-independent question: is the int4 forward still as
// close to the int8int8 forward, for the same ids, as it should be? It checks the prompt and the golden's
// own 4-token prefix.
//
// It is a COARSE sanity bound: the int4-to-int8int8 distance is dominated by int4's own quantization noise,
// so a doubled int4GroupSize (32 -> 64) stays GREEN here. Do not read this check passing as evidence the
// int4 grouping is right; that mutation is caught by TestInt4_forwardParity's forced centered-cosine floor.
// Figures: docs/code-notes/decoder.md#forcedInt4Closeness.
func forcedInt4Closeness(t *testing.T, m *Model) {
	t.Helper()
	ref, err := loadBenchModel() // int8int8 by default: the reference the int4 path is measured against
	if err != nil {
		t.Skipf("no int8int8 reference model (%v)", err)
	}
	prefix := append(append([]int{}, parityPrompt...), parityWantInt4ByArch["amd64"][:4]...)
	for name, ids := range map[string][]int{"prompt": parityPrompt, "golden 4-token prefix": prefix} {
		a, err := m.PromptLogitsMany(context.Background(), [][]int{ids})
		if err != nil {
			t.Fatal(err)
		}
		b, err := ref.PromptLogitsMany(context.Background(), [][]int{ids})
		if err != nil {
			t.Fatal(err)
		}
		if d := cosineDistance(a[0], b[0]); d > forcedLogitCloseness {
			t.Errorf("forced %v, %s: int4 against int8int8 logits 1-cosine = %.3e, want <= %.0e — the forced int4 forward drifted", forcedFallbacks(), name, d, forcedLogitCloseness)
		} else {
			t.Logf("forced %v, %s: int4 against int8int8 logits 1-cosine = %.3e (bound %.0e)", forcedFallbacks(), name, d, forcedLogitCloseness)
		}
	}
}
