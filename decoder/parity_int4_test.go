package decoder

import (
	"context"
	"os"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// loadInt4Model loads GOINFER_PREQUANT_GGUF as group-wise int4 — the W4A8 path,
// the int4 analog of loadBenchModel's int8int8. A plain q4_k_m GGUF works: the
// loader dequantizes it and re-quantizes to goinfer's group-wise int4, so no
// prebuilt .giw bundle (or torch) is needed. Fresh model per call (no cache).
func loadInt4Model(tb testing.TB) *Model {
	// BEHAVIOUR CHANGE, deliberate. This site previously skipped whenever GOINFER_PREQUANT_GGUF was
	// UNSET -- alone among the four readers of that variable, the other three fell back to
	// ../testdata. So the same box ran TestSerializeWeightsTo_matchesBuffer and skipped
	// TestDecodeParityInt4 with nothing in either output naming the difference. Both now resolve
	// through the registry's candidate list. Under the sweep nothing changes (the preflight exports
	// the variable either way); under a bare `go test ./decoder`, this gate now RUNS where it used
	// to skip.
	path := assetPath(tb, "GOINFER_PREQUANT_GGUF")
	m, err := Load(path, Options{Quant: "int4"})
	if err != nil {
		tb.Fatalf("load int4: %v", err)
	}
	return m
}

// parityWantInt4 pins the 0.5B's int4 (W4A8) greedy continuation of parityPrompt
// — the int4 analog of parityWant (which pins int8int8). It guards the WHOLE int4
// forward, including the prefill→W4A8 switch (decoder/weightmat.go), against
// silent numerics drift: greedy + fixed prompt → fully deterministic. The first
// 5 ids match parityWant (the prompt's strongest continuation survives 4-bit);
// they diverge after as int4's coarser weights bend the path. Regenerate ONLY
// when an int4 numerics change is intentional and parity-reviewed (set this to
// nil for a capture run — the test logs the new list and skips).
//
// Re-captured 2026-06-12 after the dense-attention fix (causalAttention routes
// decode through the f64 attendBatchedHeads, batched-identical to prefill — see
// attention.go). The PRIOR golden was recorded under the old decode≠prefill bug
// (scalar f32 attendQuery, aikit 1.3.0) and diverged from int8int8 already at id
// 4 (474 vs parityWant's 750) — i.e. it pinned the buggy output. The fixed int4
// decode now MATCHES int8int8 through id 4 (both 750) and tracks it one token
// further before int4 lossiness bends off — the more-correct continuation.
var parityWantInt4 = []int{4710, 73594, 12669, 198, 750, 11047, 10784, 15890, 24337, 982, 262, 2790, 15890, 284, 220, 15, 198, 262, 369, 1509, 304, 3589, 510, 286}

// TestDecodeParityInt4 greedily continues parityPrompt at int4 and checks the
// token ids against parityWantInt4. The prompt is prefilled (batched
// forwardLayersN → W4A8) then decoded (W4A8), so a regression in either the
// prefill switch or the decode kernel moves the continuation and trips here.
// Skips without the model asset.
func TestDecodeParityInt4(t *testing.T) {
	m := loadInt4Model(t)
	defer m.Close()

	const n = 24
	out, gen := m.Generate(context.Background(), parityPrompt, n, SamplingParams{Temperature: 0})
	var got []int
	for tok := range out {
		got = append(got, tok)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	if len(parityWantInt4) == 0 {
		t.Logf("CAPTURE parityWantInt4 = %#v", got)
		t.Skip("parityWantInt4 empty — capture run")
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

// BenchmarkPrefillInt4 times an end-to-end int4 prefill via the batched M=K path
// (forwardLayersN → W4A8) to record the prefill→W4A8 payoff. Compare HEAD (W4A8
// at every M) against the parent commit (M>1 used the dequant-bound MatmulBTQ4):
//
//	M=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
//	GOINFER_PREQUANT_GGUF=$M GOINFER_PREFILL_LEN=2048 \
//	  go test ./decoder -run '^$' -bench BenchmarkPrefillInt4 -benchtime 8x
//	git checkout HEAD~1 -- decoder/weightmat.go   # old Q4 split
//	GOINFER_PREQUANT_GGUF=$M GOINFER_PREFILL_LEN=2048 \
//	  go test ./decoder -run '^$' -bench BenchmarkPrefillInt4 -benchtime 8x
//	git checkout HEAD -- decoder/weightmat.go      # restore
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
		if _, err := m.prefillLogits(prompt, cache); err != nil {
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
