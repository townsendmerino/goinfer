package decoder

import (
	"context"
	"os"
	"runtime"
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
//
// RE-CAPTURED AGAIN 2026-08-15, and the same way: the golden was pinning the WORSE
// of two int4 paths. Attributed by bisect to `7deb368` (2026-06-14, aikit 1.7.3 →
// 1.8.1) — a bump whose message is entirely about the Qwen2.5-VL vision encoder and
// which asserts "no regression in the … decoder paths". It carried two aikit linalg
// commits that are not about vision at all: `36ce824` "fold W4A8 weight scales
// in-register" and `52890f5` wiring that kernel on NEON. Folding the scales changes
// the W4A8 accumulation, which moves a greedy continuation.
//
// EVIDENCE FOR RE-CAPTURING RATHER THAN CALLING IT A REGRESSION — this is the
// human decision `parity-coverage-policy.md` requires before a first-run value may
// be banked, and it is a measurement, not a judgement call. Same prompt, same 0.5B,
// 24 greedy tokens, scored by how many leading ids match an f32 forward:
//
//	int8int8 (unchanged, its own gate green)   19/24
//	int4 THIS golden                           11/24
//	int4 the 2026-06-12 golden below it         5/24
//
// The kernel change made int4 twice as faithful to f32; it also tracks int8int8 for
// 11 ids where the old golden bent away at 5. Reverting to hold the gate green would
// pin the less correct path — the identical mistake the 2026-06-12 note describes.
//
// WHY IT WENT UNSEEN FOR TWO MONTHS, which is the part worth fixing elsewhere: this
// gate was SKIPPING when the kernel landed (it alone required GOINFER_PREQUANT_GGUF
// to be set, while its three sibling readers fell back to ../testdata — see
// loadInt4Model above). It only began running when the asset registry gave it the
// fallback, and it went red on its first execution. A dependency bump moved a
// numerics path with the one gate that watches it dark.
//
// RE-CAPTURED 2026-09-28 for L1's binary16 int4 group scales (aikit v1.50.0, and v1.50.1's
// bit-identical arm64 fix; docs/tasks/task-cpu-decode-peer-gap-2026-09.md). That change moves int4
// numerics by design, and was owner-approved on 2026-09-27. This golden was not re-baselined with it:
// only TestInt4_forwardParity was, and this one went red on arm64 at the merge (3cd62e6d passes,
// 5c85f7c0 fails, drift at id 13). It was found the next day by a whole-package run. The evidence:
//   - The new list is exactly what the PRE-L1 build (3cd62e6d) produces with GOINFER_INT4_F16_SCALES=1,
//     all 24 ids identical, so the drift is the intended f16 rounding and nothing else.
//   - Scored against an f32 forward, as above: the old golden 11/24, THIS golden 11/24, int8int8 19/24.
//     Both int4 paths leave f32 at the same id and differ from each other only after it, so this one is
//     no less faithful.
//
// PER ARCHITECTURE since 2026-09-30. The 2026-09-28 re-capture was made on arm64, and the binary16 scales move
// arm64's path only: the v0.20.0 parity sweep on nobara-pc (amd64, 0ca36756) produced the PRE-L1 list exactly, all 24
// ids, and so failed against the arm64 one at id 13. The CPU int4 path is bit-identical within an architecture but
// not across them (arm64 fuses multiply-adds that amd64 rounds separately; docs/parity-coverage-policy.md, "arch-
// scoped"), so one list cannot hold both. Each is its own architecture's golden, and per the evidence above the two are
// equally faithful to f32 (11/24 each). An architecture with no entry is a capture run.
var parityWantInt4ByArch = map[string][]int{
	"arm64": {4710, 73594, 12669, 198, 750, 1438, 4136, 3932, 262, 671, 1096, 374, 264, 6573, 315, 2038, 429, 3880, 311, 387, 10865, 198, 262, 1494},
	"amd64": {4710, 73594, 12669, 198, 750, 1438, 4136, 3932, 262, 671, 1096, 374, 264, 5878, 369, 279, 5042, 2038, 198, 262, 1494, 271, 8960, 4136},
}

var parityWantInt4 = parityWantInt4ByArch[runtime.GOARCH]

// TestDecodeParityInt4 greedily continues parityPrompt at int4 and checks the
// token ids against parityWantInt4. The prompt is prefilled (batched
// forwardLayersN → W4A8) then decoded (W4A8), so a regression in either the
// prefill switch or the decode kernel moves the continuation and trips here.
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
	if len(parityWantInt4) == 0 {
		t.Logf("CAPTURE parityWantInt4 = %#v", got)
		t.Skipf("parityWantInt4 has no %s entry — capture run", runtime.GOARCH)
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

// forcedInt4Closeness replaces TestDecodeParityInt4's token-exact comparison under a forced CPU fallback (H1.3). The golden pins the default path's greedy continuation, whose
// 4th token is a 0.15 near-tie that a different rounding draw flips (measured: the pure-Go path reads 474 where the AVX2 golden reads 750), so the forced run asks the question that
// does not depend on the draw: is the int4 forward still as close to the int8int8 forward, for the same ids, as it should be? The prompt and the golden's own 4-token prefix are checked
// (measured 1 - cosine: 1.64e-2 and 7.98e-3 pure Go, 1.75e-2 and 8.09e-3 AVX2).
//
// HOW SENSITIVE THIS IS, measured: it is a COARSE sanity bound. The int4-to-int8int8 distance is dominated by int4's own quantization noise, so the int4GroupSize 32 -> 64 mutation
// moves it only from 1.64e-2 to 1.92e-2 and this check stays GREEN; 32 -> 128 reads 4.1e-2 and fails it. That mutation is caught by TestInt4_forwardParity's forced centered-cosine
// floor (cosines 0.13 to 0.99 against 0.99). Do not read this check passing as evidence the int4 grouping is right.
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
