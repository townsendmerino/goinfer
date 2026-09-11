//go:build darwin

package metal

import (
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMoEPrefillMeasure_batchedVsSequential is P-15's real-checkpoint measurement
// (audit-2026-09-10): Metal's batched f16-MMA prefill (metal/prefill.go's PrefillLast) has been
// default-ON above the 512-token floor for MoE since 306b16b, but §3.2's decision set that
// justified default-ON was dense-only — nobody has actually timed the batched path against the
// sequential per-token loop on a real, generic (non-Gemma-4) MoE. This times both, directly on
// the raw resident (bypassing metalResident's floor/env-var wrapper, which only decides which
// path a caller reaches — it does not change either kernel path's own cost), at K ∈ {512, 1024,
// 2048}.
//
// Manual/one-off by design (real 28.6GB checkpoint, several minutes of GPU time) — gated behind
// GOINFER_MOE_PREFILL_CKPT rather than GOINFER_HEAVY_TESTS' usual asset registry, since this is a
// measurement script (docs/measurements/), not a correctness gate:
//
//	GOINFER_MOE_PREFILL_CKPT=~/models/qwen15-moe-a27b \
//	  go test -tags metal ./metal/ -run TestMoEPrefillMeasure_batchedVsSequential -v -timeout 30m
func TestMoEPrefillMeasure_batchedVsSequential(t *testing.T) {
	ckpt := os.Getenv("GOINFER_MOE_PREFILL_CKPT")
	if ckpt == "" {
		t.Skip("GOINFER_MOE_PREFILL_CKPT not set — this is a manual measurement, not a CI gate")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}

	fmt.Fprintf(os.Stderr, "[moe-prefill-measure] loading %s (int4)...\n", ckpt)
	tLoad := time.Now()
	m, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	fmt.Fprintf(os.Stderr, "[moe-prefill-measure] loaded in %s\n", time.Since(tLoad).Round(time.Millisecond))

	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("build resident: %v", err)
	}
	if r.moe == nil {
		t.Fatal("no generic MoE on this resident — checkpoint is not what this measurement needs")
	}
	if r.moe.paged {
		t.Fatal("resident came up paged (GOINFER_METAL_MOE_SLOTS set?) — this measurement needs the " +
			"all-experts-resident path PrefillLast is actually for")
	}
	if !r.prefillOK {
		t.Fatal("prefillOK is false — this checkpoint declines the batched path entirely, nothing to measure")
	}
	fmt.Fprintf(os.Stderr, "[moe-prefill-measure] resident: %d layers, %d experts, top-%d, hidden %d\n",
		len(r.layers), r.moe.nE, r.moe.k, r.H)

	rng := rand.New(rand.NewSource(1234))
	vocab := r.V

	for _, K := range []int{512, 1024, 2048} {
		fmt.Fprintf(os.Stderr, "[moe-prefill-measure] K=%d: building embeddings...\n", K)
		ids := make([]int, K)
		embs := make([][]float32, K)
		for i := range ids {
			ids[i] = rng.Intn(vocab)
			e := make([]float32, r.H)
			r.embed.Row(ids[i], e)
			if r.embedScale > 1 {
				for j := range e {
					e[j] *= r.embedScale
				}
			}
			embs[i] = e
		}

		// Sequential reference: one token at a time, the exact per-token decode loop a caller
		// falls back to when PrefillLast declines. Heartbeat every ~256 tokens since a slow K=2048
		// run can otherwise look identical to a hang for several minutes.
		fmt.Fprintf(os.Stderr, "[moe-prefill-measure] K=%d: sequential pass starting %s...\n",
			K, time.Now().Format(time.TimeOnly))
		tSeq := time.Now()
		for i, id := range ids {
			r.Forward(id, i)
			if i > 0 && i%256 == 0 {
				fmt.Fprintf(os.Stderr, "[moe-prefill-measure] K=%d: sequential %d/%d, elapsed %s\n",
					K, i, K, time.Since(tSeq).Round(time.Millisecond))
			}
		}
		seqDur := time.Since(tSeq)
		fmt.Fprintf(os.Stderr, "[moe-prefill-measure] K=%d: sequential done in %s (%.2f ms/token)\n",
			K, seqDur.Round(time.Millisecond), float64(seqDur.Microseconds())/1000/float64(K))

		// Batched: the SAME positions (0..K-1), overwriting what the sequential pass just wrote —
		// same pattern metal/prefill_moe_parity_test.go uses. Raw resident method, no floor/env
		// gating (that wrapper is metalResident.PrefillLast, a policy decision layered on top).
		fmt.Fprintf(os.Stderr, "[moe-prefill-measure] K=%d: batched pass starting %s...\n",
			K, time.Now().Format(time.TimeOnly))
		tBatch := time.Now()
		r.PrefillLast(embs, 0)
		batchDur := time.Since(tBatch)
		fmt.Fprintf(os.Stderr, "[moe-prefill-measure] K=%d: batched done in %s (%.2f ms/token)\n",
			K, batchDur.Round(time.Millisecond), float64(batchDur.Microseconds())/1000/float64(K))

		speedup := float64(seqDur) / float64(batchDur)
		t.Logf("K=%d: sequential=%s batched=%s speedup=%.2fx", K, seqDur.Round(time.Millisecond),
			batchDur.Round(time.Millisecond), speedup)
		fmt.Fprintf(os.Stderr, "[moe-prefill-measure] K=%d RESULT: sequential=%s batched=%s speedup=%.2fx\n\n",
			K, seqDur.Round(time.Millisecond), batchDur.Round(time.Millisecond), speedup)
	}
}
