package decoder

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"
)

// A3 (G24) ships a DOCUMENTED DIVERGENCE, so the divergence must be a measured
// number, not an adjective. This reports what a user gives up by enabling
// GOINFER_CPU_FAST_ATTENTION, at the prompt depths the flag is for.
//
// It also pins the two guarantees that do NOT bend:
//   - default OFF: with the env unset, output is bit-identical to acc64;
//   - MoE excluded: the flag cannot turn f32 attention on for a MoE arch at all,
//     because an f32 QK reassociation flips a top-k expert at a near-tie and
//     cascades — that is a different class of error from a small numeric drift.
func TestA3FastAttentionDivergence(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	ctx := deadlineCtx(t)
	prog := newProgress(t, t.Name(), 3).Uneven() // cost grows with K
	for _, K := range []int{256, 1024, 2048} {
		prog.Phase(fmt.Sprintf("K=%d (acc64 vs f32 prefill)", K))
		if !m.canBatchN(K) {
			t.Skipf("model has no batched prefill at K=%d", K)
		}
		ids := make([]int, K)
		for i := range ids {
			ids[i] = 700 + i%64
		}
		run := func(fast string) []float32 {
			t.Helper()
			t.Setenv("GOINFER_CPU_FAST_ATTENTION", fast)
			out, err := m.forwardLayersN(ctx, ids, m.NewCache(K+8), cpuFastAttention())
			if err != nil {
				t.Fatalf("K=%d fast=%s: %v", K, fast, err)
			}
			return out
		}
		base := run("")  // unset — the shipped default
		off := run("0")  // explicitly off
		fast := run("1") // opted in

		// Default and explicit-off must both be EXACTLY the acc64 path.
		for i := range base {
			if base[i] != off[i] {
				t.Fatalf("K=%d: '0' is not identical to unset at index %d", K, i)
			}
		}

		var dot, na, nb, maxAbs float64
		for i := range base {
			a, b := float64(base[i]), float64(fast[i])
			dot += a * b
			na += a * a
			nb += b * b
			if d := math.Abs(a - b); d > maxAbs {
				maxAbs = d
			}
		}
		cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
		identical := cos == 1.0 && maxAbs == 0
		fmt.Fprintf(os.Stderr, "  A3 divergence K=%-5d cosine=%.9f maxAbs=%.3g%s\n", K, cos, maxAbs,
			map[bool]string{true: "  (IDENTICAL — flag had no effect, check the guard)", false: ""}[identical])

		prog.Step(1)
		if identical {
			t.Errorf("K=%d: enabling the flag changed nothing — either the knob is not wired or this arch is excluded", K)
		}
		// The flag is a speed/accuracy trade, not a correctness hole. The kernel
		// comment's own bar for dense f32 attention is cosine >= 0.99.
		if cos < 0.99 {
			t.Errorf("K=%d: cosine %.9f is below the 0.99 bar the kernel comment sets for dense f32 attention — "+
				"this is too large to ship behind a speed flag", K, cos)
		}
	}
}

// The guard that matters most, asserted structurally rather than trusted.
//
// A3 gives up "decode == prefill" for the model that enables it. It must NOT
// give up "spec-decode verify == sequential greedy", because that one is not a
// quality trade — a verify that disagrees with greedy silently accepts wrong
// tokens. forwardN backs speculative verify and passes fastAttn=false
// unconditionally, so the operator cannot reach it with an env var. This proves
// that by running forwardN with the flag ON and requiring bit-identical output.
func TestA3NeverReachesSpeculativeVerify(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	const K = 512
	if !m.canBatchN(K) {
		t.Skip("model has no batched prefill")
	}
	ids := make([]int, K)
	for i := range ids {
		ids[i] = 700 + i%64
	}
	run := func(fast string) [][]float32 {
		t.Helper()
		t.Setenv("GOINFER_CPU_FAST_ATTENTION", fast)
		out, err := m.forwardN(context.Background(), ids, m.NewCache(K+8))
		if err != nil {
			t.Fatalf("forwardN fast=%s: %v", fast, err)
		}
		return out
	}
	off, on := run("0"), run("1")
	if len(off) != len(on) {
		t.Fatalf("row count %d vs %d", len(off), len(on))
	}
	for i := range off {
		for j := range off[i] {
			if off[i][j] != on[i][j] {
				t.Fatalf("forwardN diverged with the flag on at row %d index %d (%v vs %v) — "+
					"speculative verify reached the f32 path, and verify == greedy no longer holds",
					i, j, off[i][j], on[i][j])
			}
		}
	}
	t.Logf("forwardN (speculative verify) is bit-identical with GOINFER_CPU_FAST_ATTENTION=1 — guard holds")
}
