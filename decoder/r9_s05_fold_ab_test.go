package decoder

import (
	"context"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestR9_s05FoldAB is the end-to-end arm of aikit S-05 (docs/task-simd-audit.md: the
// -8 centering folded into the SDOT accumulator of the M=1 W4A8 decode kernel,
// bit-identical). The kernel's single-core win is measured in aikit's own harness; this
// is the separate token-level number the decision rule asks for — one loaded 1.5B, depth
// 128, 24 greedy tokens, the fold flipped IN-PROCESS via linalg.SetW4A8RowFold, ABBA
// pairs, paired ratio with a win count (the same shape as TestR9_cpuTuningAB). The
// per-component split names where any change lands: the matmul terms (q/k/v, o,
// gate+up, down, LM head) are the only ones the kernel touches.
//
// Pre-registered band (docs/measurements/s05-centering-fold-2026-09-22.md): expected
// ~0 — step 0's split puts the 1.5B's MLP matmuls at 54% of the read ceiling and ~40% of
// the kernel's hot rate per worker, i.e. fan-out-bound (S-02), where a faster kernel is
// mostly hidden. ≥3% paired on 3/3 pairs = reaches the token; 1.5–3% = ambiguous, parked;
// <1.5% = does not reach the token, as predicted. The ship decision is the single-core
// harness either way; this number is reported beside it, not gated on.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./decoder/ -run TestR9_s05FoldAB -v -timeout 20m
func TestR9_s05FoldAB(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real checkpoint on CPU)")
	}
	path := os.Getenv("GOINFER_CPU_MODEL")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	prevTiming := decodeTiming
	decodeTiming = true
	defer func() { decodeTiming = prevTiming }()
	prevFold := linalg.W4A8RowFold()
	defer linalg.SetW4A8RowFold(prevFold)

	m, err := Load(path, Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	ids, err := tk.Encode("Continue this text. "+strings.Repeat(" the", 128), true)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(ids) > 128 {
		ids = ids[:128]
	}
	type obs struct{ fwd, matmul float64 }
	gen := func() obs {
		out, g := m.Generate(context.Background(), ids, 24, SamplingParams{Temperature: 0})
		for range out {
		}
		if g.err != nil {
			t.Fatalf("generate: %v", g.err)
		}
		s := lastDecodeSplit
		return obs{s.fwd, s.qkv + s.o + s.gu + s.down + s.head}
	}
	gen()      // warm-up
	pairs := 3 // the pre-registered run; GOINFER_AB_PAIRS=6 for the replicates
	if v, err := strconv.Atoi(os.Getenv("GOINFER_AB_PAIRS")); err == nil && v > 0 {
		pairs = v
	}
	var sumOn, sumOff obs
	wins := 0
	for p := 0; p < pairs; p++ {
		var on, off obs
		if p%2 == 0 {
			linalg.SetW4A8RowFold(true)
			on = gen()
			linalg.SetW4A8RowFold(false)
			off = gen()
		} else {
			linalg.SetW4A8RowFold(false)
			off = gen()
			linalg.SetW4A8RowFold(true)
			on = gen()
		}
		sumOn.fwd, sumOn.matmul = sumOn.fwd+on.fwd, sumOn.matmul+on.matmul
		sumOff.fwd, sumOff.matmul = sumOff.fwd+off.fwd, sumOff.matmul+off.matmul
		if on.fwd < off.fwd {
			wins++
		}
		t.Logf("pair %d: fold ON %.2f ms/tok (matmul terms %.2f) | OFF %.2f ms/tok (matmul terms %.2f) | ON is %.3fx faster on the token, %.3fx on the matmul terms",
			p, on.fwd, on.matmul, off.fwd, off.matmul, off.fwd/on.fwd, off.matmul/on.matmul)
	}
	n := float64(pairs)
	t.Logf("S-05 fold end-to-end, 1.5B depth 128: paired ON %.2f -> OFF %.2f ms/tok = OFF/ON %.3fx (fold wins %d/%d); matmul terms %.2f -> %.2f = %.3fx",
		sumOn.fwd/n, sumOff.fwd/n, sumOff.fwd/sumOn.fwd, wins, pairs, sumOn.matmul/n, sumOff.matmul/n, sumOff.matmul/sumOn.matmul)
}
