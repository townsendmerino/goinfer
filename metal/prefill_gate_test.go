//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestPrefillGate is docs/task-prefill-gap.md §3's gate: may Metal's batched (f16-MMA) PrefillLast
// become the DEFAULT prompt-ingestion path, in place of today's shipped sequential Forward-per-
// token prefill? It runs both paths on the SAME model, SAME prompts, and scores the batched path
// against the sequential one as the oracle — sequential decode is what every other gate in this
// tree (the 3%-near-tie rule, `benchmarks.md` §B2) already trusts as bit-identical-enough.
//
// Two checks are HARD gates (a failure means Metal's fast prefill is "not quality-neutral as
// built" and stays opt-in, per the doc's own framing — that is a valid, useful outcome, not a
// bug in this test):
//
//  1. Seed-logit argmax agreement, at the prompt's last position, under the tree's existing
//     3%-near-tie rule (decoder.NearTieArgmaxForTest / decoder.NearTieHardFailPct) — the same bar
//     cuda/realforward_test.go and gpu/kv_i8_parity_test.go already hold CUDA decode to against
//     CPU.
//
//  2. Teacher-forced top-1 agreement over 64 continuation positions past the prompt
//     (decoder.TeacherForcedTop1AgreementForTest), which measures the batched path's KV without
//     the cascade a free-running greedy comparison carries (one early flip and everything after
//     it diverges for a reason that has nothing to do with per-position quality).
//
//     The doc's own bar for check 2 is "≥ the CUDA-decode-vs-CPU figure on the same model" — a
//     number that does not exist anywhere in this tree yet (confirmed by exploration; teacher-
//     forced top-1 agreement is new). Until a real cross-backend figure is measured, this test
//     gates check 2 the same way as check 1: a disagreement is a hard fail only when it is NOT a
//     3%-near-tie by the tree's own established rule (i.e. every continuation position is scored
//     with decoder.NearTieArgmaxForTest against the SEQUENTIAL PATH'S OWN logits at that
//     position, not just compared by raw token equality). The raw agreement rate is reported
//     alongside so this substitution is visible and the real bar can replace it once measured.
//
// Two more are REPORTED, not gating, per the doc's own table:
//   - Seed-distribution KL divergence vs the exact path (decoder.KLDivergenceForTest).
//   - Greedy stream divergence rate — NOT re-measured here. It is already measured and gated by
//     TestMetalPrefillDivergenceRate (54% — metal/backend.go's PrefillLast decline comment,
//     §A2-Metal); re-running it would duplicate that test for a number this gate only reports.
//
// LONG-RUNNING: with GOINFER_HEAVY_TESTS=1 this loads two real checkpoints (S ~1.5B, D7 ~7B) and
// runs each through prefillGateProseFiles × the K depths below, sequential-decode dominated at
// the deepest K (today's shipped ~40-60 tok/s on S means K=3900 alone is over a minute per
// prompt). Progress is printed to stdout per prompt and on a 20s ticker inside the sequential
// pass so a long run is legible rather than silent, per this repo's long-test convention.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./metal/ -run TestPrefillGate -v -timeout 4h
func TestPrefillGate(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads two real checkpoints, S and D7)")
	}
	if testing.Short() {
		t.Skip("long-running gate: skipped in -short")
	}
	t.Setenv("GOINFER_METAL_BATCHED_PREFILL", "1") // the FAST arm; the exact arm calls Forward directly

	models := []struct {
		name        string
		pathEnv     string
		defaultPath string
	}{
		{"S", "GOINFER_METAL_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
		{"D7", "GOINFER_METAL_MODEL_D7", "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf"},
	}
	depths := []int{256, 1024, 3900}
	const continuationN = 64

	overallHardFail := false
	for _, mc := range models {
		path := os.Getenv(mc.pathEnv)
		if path == "" {
			path = os.ExpandEnv(mc.defaultPath)
		}
		if _, err := os.Stat(path); err != nil {
			t.Logf("model %s: no fixture at %s (set %s) — skipping this arm", mc.name, path, mc.pathEnv)
			continue
		}
		runPrefillGateModel(t, mc.name, path, depths, continuationN, &overallHardFail)
	}
	if overallHardFail {
		t.Fatalf("§3 GATE FAILED on at least one (model, K) cell — see log above for which; " +
			"the default stays exact there (docs/task-prefill-gap.md §3, §7)")
	}
}

func runPrefillGateModel(t *testing.T, modelName, path string, depths []int, continuationN int, overallHardFail *bool) {
	t.Helper()
	fmt.Printf("\n=== PREFILL GATE — model=%s path=%s ===\n", modelName, path)
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("model %s: load: %v", modelName, err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Skipf("model %s: metal resident not built for this model", modelName)
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("model %s: load tokenizer: %v", modelName, err)
	}

	prompts := make([][]int, 0, len(prefillGateProseFiles))
	for _, f := range prefillGateProseFiles {
		ids := prefillGateProseIDs(t, tk, f, depths[len(depths)-1])
		prompts = append(prompts, ids)
	}

	for _, K := range depths {
		var (
			seedHardFails, contHardFails   int
			worstSeedGap, worstContGap     float64
			sumAgreement, sumSeedKL, sumKL float64
			n                              int
		)
		t0 := time.Now()
		for pi, ids := range prompts {
			if len(ids) < K {
				t.Fatalf("model %s prompt %d: only %d tokens, need >= %d", modelName, pi, len(ids), K)
			}
			res := runPrefillGateCell(t, rf, m, ids[:K], continuationN, K)
			n++
			if res.seedHardFail {
				seedHardFails++
			}
			if res.seedGapPct > worstSeedGap {
				worstSeedGap = res.seedGapPct
			}
			if res.contHardFails > 0 {
				contHardFails += res.contHardFails
			}
			if res.worstContGap > worstContGap {
				worstContGap = res.worstContGap
			}
			sumAgreement += res.agreementRate
			sumSeedKL += res.seedKL
			sumKL += res.meanContKL
			fmt.Printf("[gate] %s K=%d prompt %2d/%2d seed=%s(gap=%.3f%%) contAgree=%.1f%% contHardFails=%d/%d firstDiv=%d KLseed=%.4f KLcont=%.4f elapsed=%s\n",
				modelName, K, pi+1, len(prompts),
				map[bool]string{true: "AGREE", false: "FLIP"}[res.seedAgree], res.seedGapPct*100,
				res.agreementRate*100, res.contHardFails, continuationN, res.firstDivergence,
				res.seedKL, res.meanContKL, time.Since(t0).Round(time.Second))
		}
		meanAgreement := sumAgreement / float64(n)
		meanSeedKL := sumSeedKL / float64(n)
		meanKL := sumKL / float64(n)
		gateFail := seedHardFails > 0 || contHardFails > 0
		if gateFail {
			*overallHardFail = true
		}
		fmt.Printf("=== %s K=%d SUMMARY (n=%d prompts): seedHardFails=%d/%d (worst gap %.3f%%) "+
			"contHardFails=%d/%d (worst gap %.3f%%) meanTeacherForcedAgreement=%.1f%% "+
			"meanSeedKL=%.4f meanContKL=%.4f — §3 GATE %s\n",
			modelName, K, n, seedHardFails, n, worstSeedGap*100, contHardFails, n*continuationN,
			worstContGap*100, meanAgreement*100, meanSeedKL, meanKL,
			map[bool]string{true: "FAILED", false: "PASSED"}[gateFail])
		t.Logf("%s K=%d: seedHardFails=%d contHardFails=%d meanAgreement=%.1f%% meanSeedKL=%.4f meanContKL=%.4f gate=%s",
			modelName, K, seedHardFails, contHardFails, meanAgreement*100, meanSeedKL, meanKL,
			map[bool]string{true: "FAILED", false: "PASSED"}[gateFail])
	}
	fmt.Printf("[gate] %s: greedy stream divergence — NOT re-measured here; see TestMetalPrefillDivergenceRate "+
		"(54%%, metal/backend.go PrefillLast decline comment, §A2-Metal). Reported, not gating (§3).\n", modelName)
}

type prefillGateCellResult struct {
	seedAgree       bool
	seedGapPct      float64
	seedHardFail    bool
	seedKL          float64
	agreementRate   float64
	firstDivergence int
	contHardFails   int
	worstContGap    float64
	meanContKL      float64
}

// runPrefillGateCell runs the exact (sequential Forward) path and the fast (batched PrefillLast)
// path over the same K-token prompt on the same resident model, then teacher-forces continuationN
// positions past the prompt through both — the exact path free-running (its own greedy argmax
// becomes the reference continuation), the fast path fed the REFERENCE's tokens at each step (so
// a divergence at position i cannot cascade into position i+1, isolating each position's own
// quality). The two paths share one resident KV store, overwritten in place per call — the same
// arrangement metal/prefill_ttft_test.go already relies on ("only the timing is compared, not the
// resulting cache content"); here we run the exact pass, capture everything we need from it, THEN
// overwrite with the fast pass, so nothing from one path survives into the other's reads.
func runPrefillGateCell(t *testing.T, rf *metalResident, m *decoder.Model, ids []int, continuationN, K int) prefillGateCellResult {
	t.Helper()
	ctx := context.Background()
	embs := make([][]float32, K)
	for i, id := range ids {
		embs[i] = m.EmbedResidentForTest(id)
	}

	// EXACT — sequential Forward per token, today's shipped default.
	//
	// Forward's return is a REUSED buffer (metal/model.go ForwardEmbPipe: "Returns logits[V]
	// (reused buffer; consume before the next call)") — every capture below is cloned
	// immediately. Storing the raw slice instead silently aliases whatever the LAST Forward call
	// in the whole cell wrote, which is exactly the bug this comment is here to stop someone
	// from reintroducing: it first shipped that way, and the seed and every continuation position
	// all came back reading the same final buffer, producing a self-contradictory result (a
	// "42% seed gap" while position 0 of the continuation — which SHOULD be the same comparison —
	// quietly agreed).
	lastLog := time.Now()
	var exactSeed []float32
	for i := 0; i < K; i++ {
		lg, err := rf.Forward(embs[i], i)
		if err != nil {
			t.Fatalf("exact Forward pos=%d: %v", i, err)
		}
		exactSeed = cloneF32(lg)
		if time.Since(lastLog) > 20*time.Second {
			fmt.Printf("[gate]   ... exact prefill K=%d at pos %d/%d\n", K, i+1, K)
			lastLog = time.Now()
		}
	}
	refTokens := make([]int, continuationN)
	exactLogits := make([][]float32, continuationN)
	pos := K - 1
	cur := exactSeed
	for i := 0; i < continuationN; i++ {
		refTokens[i] = prefillGateArgmax(cur)
		exactLogits[i] = cur
		if i == continuationN-1 {
			break
		}
		pos++
		lg, err := rf.Forward(m.EmbedResidentForTest(refTokens[i]), pos)
		if err != nil {
			t.Fatalf("exact continuation Forward pos=%d: %v", pos, err)
		}
		cur = cloneF32(lg)
	}

	// FAST — one batched PrefillLast over the same K embeddings, THEN teacher-forced continuation
	// fed the reference's own tokens (not the fast path's own predictions) at each step. Cloned
	// for the same reason as the exact pass above.
	fastSeed, err := rf.PrefillLast(ctx, embs, 0)
	if err != nil {
		t.Fatalf("fast PrefillLast: %v", err)
	}
	fastSeed = cloneF32(fastSeed)
	candLogits := make([][]float32, continuationN)
	candLogits[0] = fastSeed
	pos = K - 1
	for i := 1; i < continuationN; i++ {
		pos++
		lg, err := rf.Forward(m.EmbedResidentForTest(refTokens[i-1]), pos)
		if err != nil {
			t.Fatalf("fast continuation Forward pos=%d: %v", pos, err)
		}
		candLogits[i] = cloneF32(lg)
	}

	seedAgree, seedGapPct, seedHardFail := decoder.NearTieArgmaxForTest(exactSeed, fastSeed)
	seedKL := decoder.KLDivergenceForTest(exactSeed, fastSeed)
	agreementRate, firstDivergence := decoder.TeacherForcedTop1AgreementForTest(candLogits, refTokens)

	contHardFails := 0
	worstContGap := 0.0
	var klSum float64
	for i := range candLogits {
		_, gapPct, hardFail := decoder.NearTieArgmaxForTest(exactLogits[i], candLogits[i])
		if hardFail {
			contHardFails++
		}
		if gapPct > worstContGap {
			worstContGap = gapPct
		}
		klSum += decoder.KLDivergenceForTest(exactLogits[i], candLogits[i])
	}

	return prefillGateCellResult{
		seedAgree: seedAgree, seedGapPct: seedGapPct, seedHardFail: seedHardFail, seedKL: seedKL,
		agreementRate: agreementRate, firstDivergence: firstDivergence,
		contHardFails: contHardFails, worstContGap: worstContGap,
		meanContKL: klSum / float64(len(candLogits)),
	}
}

// cloneF32 copies a logits slice at capture time. Forward/PrefillLast return REUSED buffers
// (see runPrefillGateCell) — every value this test keeps past the next backend call must be a
// copy, not the returned slice itself.
func cloneF32(v []float32) []float32 { return append([]float32(nil), v...) }

func prefillGateArgmax(v []float32) int {
	bi, bv := 0, v[0]
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}

// prefillGateProseFiles are real prose read at run time — not scripts/prompts.json's word-
// repetition filler, which docs/task-prefill-gap.md §0 rules out for anything content-dependent
// ("the fidelity gate (§3) uses prose"). Ten distinct real technical documents from this repo,
// chosen only for being real, sizeable (each encodes to well over 3900 tokens on its own, so no
// prompt needs repeating to reach the deepest K), and stable — not for their content, the same
// reasoning metal/spec_prefill_regression_test.go's readRepoCorpus gives for reading real
// repository source instead of a short hand-written corpus.
var prefillGateProseFiles = []string{
	"../docs/audit-2026-09-02.md",
	"../docs/QUEUE.md",
	"../docs/queue-engineering.md",
	"../docs/ollama-chase.md",
	"../docs/benchmarks.md",
	"../docs/parity-coverage-policy.md",
	"../docs/task-w4a8-neon-bandwidth.md",
	"../docs/legacy-benchmarks.md",
	"../docs/task-zeno-compare.md",
	"../docs/queue-release.md",
}

// prefillGateProseIDs reads f, encodes it with tk, and returns at least minTokens ids (repeating
// the same real content if one file is somehow too short for a future larger K, rather than
// padding with filler — a repeated real paragraph is still content-dependent, unlike
// scripts/prompts.json's "the the the").
func prefillGateProseIDs(t *testing.T, tk *tokenizer.Tokenizer, f string, minTokens int) []int {
	t.Helper()
	raw, err := os.ReadFile(f)
	if err != nil {
		t.Fatalf("read prose seed %s: %v", f, err)
	}
	text := string(raw)
	ids, err := tk.Encode(text, true)
	if err != nil {
		t.Fatalf("encode prose seed %s: %v", f, err)
	}
	for len(ids) < minTokens {
		more, err := tk.Encode(text, false)
		if err != nil {
			t.Fatalf("encode prose seed %s: %v", f, err)
		}
		ids = append(ids, more...)
	}
	return ids
}
