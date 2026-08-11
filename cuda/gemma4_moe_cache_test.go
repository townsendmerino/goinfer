//go:build cuda && goinfer_testhooks

package cuda

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// loadG4MoECache loads a gemma4-MoE fixture as a CUDA resident int4 runner with the C′ expert cache
// on or off. Skips if the fixture/GPU is absent.
func loadG4MoECache(t *testing.T, dir string, cache bool) (*decoder.Model, decoder.ResidentForward) {
	t.Helper()
	// Stat the WEIGHTS, not the directory. Fixture weights are gitignored but their config JSONs
	// are not, so a stray `git add` of the directory makes it "exist" in CI with no model in it —
	// and a dir-only guard then skips nothing and Fatalfs on the load instead.
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture weights (%s/model.safetensors) — run scripts/pin_gemma4_moe_scaled.py", dir)
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
		// The cache=false arm needs the WHOLE expert stack in VRAM, which is precisely what the
		// cache exists to avoid. Pointing this gate at a model that does not fit (the real 26B:
		// ~11.4 GB of experts on an 8 GB card) therefore fails STRUCTURALLY, not numerically —
		// it burned 307 s to reach an OOM the runtime already knew about. Say so here rather
		// than leaving a bare decline that reads like a parity failure.
		t.Fatalf("cuda resident DECLINED gemma4 MoE (cache=%v).\n"+
			"If cache=false: this gate needs BOTH arms resident, so the fixture's ENTIRE expert stack "+
			"must fit VRAM — it is the control arm, not the streaming one. A model that only runs WITH "+
			"the cache (the real 26B) can never satisfy it; use testdata/gemma4-moe-scaled, which is "+
			"sized so both arms fit (see scripts/pin_gemma4_moe_scaled.py). The runtime prints the "+
			"decline reason unconditionally — check stderr above for the actual cause.", cache)
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

// TestGemma4MoE_cacheExpertsBitExact_scaled runs the same gate at the WIDTH that broke A′ zero-copy:
// the correctness proof that matters for B′, and the one this track never actually had.
//
// It had never run. GOINFER_MOE_SCALED_FIXTURE named no fixture that existed (the only MoE fixtures
// were the three tiny ones), so it skipped from the day it was written; and aimed at the real 26B it
// fails structurally, because the cache=false control arm cannot be resident on a card the model
// does not fit. So C′ — the path the shipped 26B result runs on — was gated only at toy width
// (2 layers, hidden 256, 4 experts), which is the SAME class of evidence A′ had when A′ was wrong.
//
// testdata/gemma4-moe-scaled resolves both problems. It keeps hidden=2816 and moe_inter=704 — the
// REAL per-expert row geometry, the dimension A′ was actually sensitive to — and shrinks only the
// axes the A′ post-mortem excludes (128→32 experts, 30→4 layers). Its full int4 expert stack is
// ~428 MB, so BOTH arms are resident simultaneously-satisfiable on an 8 GB card with wide margin,
// which is what makes the control arm meaningful rather than impossible.
//
// Defaults to that fixture; GOINFER_MOE_SCALED_FIXTURE still overrides for a one-off.
func TestGemma4MoE_cacheExpertsBitExact_scaled(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset")
	}
	dir := os.Getenv("GOINFER_MOE_SCALED_FIXTURE")
	if dir == "" {
		dir = "../testdata/gemma4-moe-scaled"
	}
	cacheBitExact(t, dir)
}

// TestGemma4MoE_cacheReuse_scaled is the same gate with cross-token slot reuse AND eviction active
// at real width: nSlots=12 sits between topK=8 and nE=32, so the LRU both hits and evicts. Step-2's
// reuse path is what the 26B actually decodes on (38 slots of 128), and until now it too was only
// gated at nE=4.
func TestGemma4MoE_cacheReuse_scaled(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset")
	}
	dir := os.Getenv("GOINFER_MOE_SCALED_FIXTURE")
	if dir == "" {
		dir = "../testdata/gemma4-moe-scaled"
	}
	t.Setenv("GOINFER_MOE_CACHE_SLOTS", "12")
	cacheBitExact(t, dir)
}
