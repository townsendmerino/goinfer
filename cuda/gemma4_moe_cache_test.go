//go:build cuda && goinfer_testhooks

package cuda

import (
	"errors"
	"io/fs"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// loadG4MoECache loads a gemma4-MoE fixture as a CUDA resident int4 runner with the C′ expert cache
// on or off. Skips if the fixture/GPU is absent.
func loadG4MoECache(t *testing.T, dir string, cache bool) (*decoder.Model, decoder.ResidentForward) {
	t.Helper()
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s)", dir)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	if cache {
		t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")
	} else {
		t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "")
	}
	mc, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cache=%v): %v", cache, err)
	}
	rf := mc.ResidentForwardForTest()
	if rf == nil {
		mc.Close()
		t.Fatalf("cuda resident DECLINED gemma4 MoE (cache=%v)", cache)
	}
	return mc, rf
}

// cacheBitExact runs the prompt through the C′ cache path and the fully-resident path and asserts
// BIT-IDENTICAL logits at every position. This is C′'s core gate AND gate #1 for the whole design:
// it proves the UNMODIFIED gemv_w4a8_moe reads a small nSlots-deep, slot-remapped DEVICE stack
// exactly like the full one — i.e. the slot-id trick needs NO moe.ptx change. Unlike A′ zero-copy
// (host-mapped read, wrong at width), C′ reads device memory, which holdalive already proved exact.
func cacheBitExact(t *testing.T, dir string) {
	prompt := []int{1, 7, 42, 100, 5, 200, 13, 88, 3, 71, 128, 9, 250, 17, 60, 200}
	run := func(cache bool) [][]float32 {
		mc, rf := loadG4MoECache(t, dir, cache)
		defer mc.Close()
		out := make([][]float32, len(prompt))
		for i, tok := range prompt {
			l, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
			if err != nil {
				t.Fatalf("forward pos %d (cache=%v): %v", i, cache, err)
			}
			out[i] = append([]float32(nil), l...)
		}
		return out
	}
	resident := run(false)
	cached := run(true)
	for i := range resident {
		for j := range resident[i] {
			if resident[i][j] != cached[i][j] {
				t.Fatalf("pos %d logit %d: cached %.9g != resident %.9g — the slot-stacked DEVICE read is not "+
					"bit-identical to the full stack (C′ broken, or slot-id trick needs a kernel change)", i, j, cached[i][j], resident[i][j])
			}
		}
	}
	t.Logf("C′ VRAM expert cache (%d pos, %s): DMA'd-into-slots == fully-resident, BIT-IDENTICAL — unmodified kernel, no moe.ptx change", len(prompt), dir)
}

// TestGemma4MoE_cacheExpertsBitExact_tiny is the committed-fixture gate (always runs on a GPU).
func TestGemma4MoE_cacheExpertsBitExact_tiny(t *testing.T) {
	cacheBitExact(t, "../testdata/gemma4-moe-tiny")
}

// TestGemma4MoE_cacheReuse_tiny is the C′ step-2 gate: with nSlots BETWEEN topK and nE, the LRU
// cache reuses cached experts across tokens AND evicts — and the result must still be BIT-IDENTICAL
// to fully-resident. A reuse that returned a stale slot, or an eviction that freed a slot still
// needed this token, would diverge here. tiny is topK=2 of nE=4, so 3 slots exercises both paths.
func TestGemma4MoE_cacheReuse_tiny(t *testing.T) {
	t.Setenv("GOINFER_MOE_CACHE_SLOTS", "3")
	cacheBitExact(t, "../testdata/gemma4-moe-tiny")
}

// TestGemma4MoE_cacheExpertsBitExact_scaled runs the same gate at the WIDTH that broke A′ zero-copy
// (bigk/scaled): the correctness proof that matters for B′. Gated on a scaled fixture + heavy tests.
func TestGemma4MoE_cacheExpertsBitExact_scaled(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset")
	}
	dir := os.Getenv("GOINFER_MOE_SCALED_FIXTURE")
	if dir == "" {
		t.Skip("GOINFER_MOE_SCALED_FIXTURE unset")
	}
	cacheBitExact(t, dir)
}
