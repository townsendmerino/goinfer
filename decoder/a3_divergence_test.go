package decoder

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"
)

// A3 ships a documented divergence, so the divergence must be a measured number: this reports what
// enabling GOINFER_CPU_FAST_ATTENTION (f32 prefill attention) costs at the prompt depths the flag is for.
//
// It loads the DENSE bench checkpoint, so every assertion is about dense:
//   - unset is bit-identical to "0" (acc64) below fastAttnMinPrompt and to "1" (f32) at or above it;
//   - "0" and "1" differ above the floor and agree below it;
//   - the f32 divergence clears the kernel comment's cosine >= 0.99 bar.
//
// MoE is not asserted here (cpuFastAttention does not exclude it); the MoE measurements are
// a3_moe_exclusion_test.go, a3_moe_routeflip_test.go and a3_moe_tokenlevel_test.go.
// History and the MoE numbers: docs/code-notes/decoder.md#TestA3FastAttentionDivergence.
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
			setKnob(t, m, knobCPUFastAttention, fast)
			out, err := m.forwardLayersN(ctx, ids, m.NewCache(K+8), cpuFastAttention())
			if err != nil {
				t.Fatalf("K=%d fast=%s: %v", K, fast, err)
			}
			return out
		}
		base := run("")  // unset — the shipped default
		off := run("0")  // explicitly off
		fast := run("1") // opted in

		// What "unset" must equal depends on K: below fastAttnMinPrompt the f32 path is floored off and unset is
		// exactly acc64; at or above it unset is the f32 path. Asserting both halves pins the floor itself, not just
		// the default: a floor of 0 or MaxInt fails here instead of silently changing what every short request returns.
		want, wantName := off, `"0" (acc64)`
		if K >= fastAttnMinPrompt {
			want, wantName = fast, `"1" (f32)`
		}
		for i := range base {
			if base[i] != want[i] {
				t.Fatalf("K=%d (floor %d): unset is not identical to %s at index %d",
					K, fastAttnMinPrompt, wantName, i)
			}
		}
		// The two explicit settings must still differ above the floor, or the flag has stopped doing anything
		// and every divergence number here is noise.
		if K >= fastAttnMinPrompt {
			same := true
			for i := range off {
				if off[i] != fast[i] {
					same = false
					break
				}
			}
			if same {
				t.Fatalf("K=%d: '0' and '1' produced identical output — the f32 path is not running", K)
			}
		}

		// Compare off vs fast, not base vs fast: above the floor unset IS the f32 path, so base-vs-fast would be
		// fast-vs-fast (cosine 1.0) and trip the "flag had no effect" guard below.
		var dot, na, nb, maxAbs float64
		for i := range off {
			a, b := float64(off[i]), float64(fast[i])
			dot += a * b
			na += a * a
			nb += b * b
			if d := math.Abs(a - b); d > maxAbs {
				maxAbs = d
			}
		}
		cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
		// Bit-identity is maxAbs == 0, not cos == 1.0: cos is a float64 quotient of sums and for two
		// bit-identical vectors lands near 1.0 without being required to equal it.
		identical := maxAbs == 0
		fmt.Fprintf(os.Stderr, "  A3 divergence K=%-5d cosine=%.9f maxAbs=%.3g%s\n", K, cos, maxAbs,
			map[bool]string{true: "  (IDENTICAL — flag had no effect, check the guard)", false: ""}[identical])

		prog.Step(1)
		// Below the floor the two settings are supposed to agree (the floor working, not the knob unwired).
		if identical && K >= fastAttnMinPrompt {
			t.Errorf("K=%d: enabling the flag changed nothing — either the knob is not wired or this arch is excluded", K)
		}
		if !identical && K < fastAttnMinPrompt {
			t.Errorf("K=%d is below the floor (%d) but the settings differ — the floor is not being applied", K, fastAttnMinPrompt)
		}
		// The flag is a speed/accuracy trade, not a correctness hole: the kernel comment's bar for dense f32
		// attention is cosine >= 0.99.
		if cos < 0.99 && K >= fastAttnMinPrompt {
			t.Errorf("K=%d: cosine %.9f is below the 0.99 bar the kernel comment sets for dense f32 attention — "+
				"this is too large to ship behind a speed flag", K, cos)
		}
	}
}

// The guard that matters most, asserted structurally: A3 gives up "decode == prefill" for the model
// that enables it, but must not give up "spec-decode verify == sequential greedy" (a verify that
// disagrees with greedy silently accepts wrong tokens). forwardN backs speculative verify and passes
// fastAttn=false unconditionally, so no env var reaches it. This runs forwardN with the flag ON and
// requires bit-identical output.
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
		setKnob(t, m, knobCPUFastAttention, fast)
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
