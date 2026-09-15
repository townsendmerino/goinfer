package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"sync/atomic"
	"testing"
)

// gemma4VLBidirImageGolden mirrors the inline struct in gemma4_vl_bidir_test.go's
// TestGemma4VLBidir_imageParity — pulled out to a named type so it can be shared
// with this file's resident-decode wiring tests.
type gemma4VLBidirImageGolden struct {
	InputIDs        []int     `json:"input_ids"`
	ImageTokenStart int       `json:"image_token_start"`
	NImageTokens    int       `json:"n_image_tokens"`
	ImageFeatures   []float32 `json:"image_features"`
}

func loadGemma4VLTinySequential(t *testing.T) (*Model, gemma4VLBidirImageGolden) {
	t.Helper()
	const golden = "../testdata/gemma4_vl_tiny_image_golden.json"
	const ckpt = "../testdata/gemma4-vl-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no image golden — run scripts/pin_gemma4_vl_image.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skip("no checkpoint — run scripts/pin_gemma4_vl_tiny.py")
	}
	var g gemma4VLBidirImageGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if got := string(m.w.Cfg.UseBidirectionalAttention); got != "" {
		t.Fatalf("UseBidirectionalAttention = %q, want \"\" (this fixture must stay the sequential/causal, E2B/E4B-class shape)", got)
	}
	return m, g
}

func loadGemma4VLBidirTiny(t *testing.T) (*Model, gemma4VLBidirImageGolden) {
	t.Helper()
	const golden = "../testdata/gemma4_vl_bidir_tiny_image_golden.json"
	const ckpt = "../testdata/gemma4-vl-bidir-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no image golden — run scripts/pin_gemma4_vl_bidir_image.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skip("no checkpoint — run scripts/pin_gemma4_vl_bidir_tiny.py")
	}
	var g gemma4VLBidirImageGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if got := string(m.w.Cfg.UseBidirectionalAttention); got != "vision" {
		t.Fatalf("UseBidirectionalAttention = %q, want \"vision\" (fixture didn't load the shape this test exists to cover)", got)
	}
	return m, g
}

// uploadFailResident wraps fakeResident and forces every UploadKV call to fail —
// simulating a resident backend that declines the bridge (busy, OOM, whatever) so
// GenerateGemma4VL's fallback-to-CPU-decode path can be exercised directly.
type uploadFailResident struct{ *fakeResident }

func (u *uploadFailResident) UploadKV(layer, base int, keys, vals []float32) error {
	_ = u.fakeResident.UploadKV(layer, base, keys, vals) // still record the attempt
	return errors.New("uploadFailResident: forced decline")
}

// TestGenerateGemma4VL_residentDecodeEngagesOnBidirectional is the wiring gate for the
// 26B-A4B/31B resident bridge added to GenerateGemma4VL: given a resident backend and a
// use_bidirectional_attention="vision" checkpoint, a turn must (a) call UploadKV once per
// layer with the CPU bidirectional prefill's K/V, (b) dispatch decode through the resident
// Forward (not m.forward), and (c) commit resIDs afterward — mirroring
// TestGenerateVL_residentDecodeEngagesAndUploadsKV's shape for Gemma 3, but through
// GenerateGemma4VL/prefillLogitsGemma4VLBidirectional instead.
func TestGenerateGemma4VL_residentDecodeEngagesOnBidirectional(t *testing.T) {
	m, g := loadGemma4VLBidirTiny(t)
	rf := &fakeResident{vocab: m.w.arch.VocabSize}
	m.resident = rf
	m.resIDs = []int{9, 9, 9} // stale — must not be mistaken for a reusable prefix

	const maxNew = 4
	const testImgHash = 0xC0FFEE // nonzero — a real hash, so the M-07 block-commit assertion below is meaningful
	features := func() ([]float32, error) { return g.ImageFeatures, nil }
	stream, gen := m.GenerateGemma4VL(context.Background(), g.InputIDs, g.ImageTokenStart, g.NImageTokens, testImgHash, features, maxNew, SamplingParams{Temperature: 0})
	var got []int
	for id := range stream {
		got = append(got, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("GenerateGemma4VL: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("GenerateGemma4VL streamed no tokens")
	}
	// M-07 (docs/audit-2026-09-10.md): the commit must record an image block, or a later turn's
	// residentReuseLen has nothing to key a reuse check on and can walk straight through this
	// image's K/V for any prompt sharing Gemma4's content-independent soft-token ids.
	if len(m.resImgBlocks) != 1 {
		t.Fatalf("resImgBlocks = %v, want exactly 1 entry after a resident-committed image turn", m.resImgBlocks)
	}
	if got := m.resImgBlocks[0]; got.start != g.ImageTokenStart || got.end != g.ImageTokenStart+g.NImageTokens || got.hash != testImgHash {
		t.Errorf("resImgBlocks[0] = %+v, want {start:%d end:%d hash:%d}",
			got, g.ImageTokenStart, g.ImageTokenStart+g.NImageTokens, uint64(testImgHash))
	}
	if nLayers := m.w.arch.NumLayers; len(rf.uploadKVs) != nLayers {
		t.Errorf("UploadKV called %d times, want %d (once per layer)", len(rf.uploadKVs), nLayers)
	}
	for l, c := range rf.uploadKVs {
		if c.layer != l {
			t.Errorf("uploadKVs[%d].layer = %d, want %d — layers must upload in order", l, c.layer, l)
		}
		if c.base != 0 {
			t.Errorf("uploadKVs[%d].base = %d, want 0 — gemma4 never rings, so base is always 0", l, c.base)
		}
		if len(c.keys) == 0 {
			t.Errorf("uploadKVs[%d]: empty keys — CPU bidirectional prefill produced nothing to upload", l)
		}
	}
	if rf.forwards == 0 {
		t.Error("resident Forward was never called — decode silently ran on CPU despite a resident being present")
	}
	want := append(append([]int{}, g.InputIDs...), got...)
	if !equalIntSlices(m.resIDs, want) {
		t.Errorf("resIDs = %v after a resident-touching GenerateGemma4VL, want prompt+generated %v", m.resIDs, want)
	}
	if busy := atomic.LoadInt32(&m.resBusy); busy != 0 {
		t.Errorf("resBusy = %d after GenerateGemma4VL returned, want 0 (released)", busy)
	}
}

// TestGenerateGemma4VL_residentContextCapPublishesBudgetClamped mirrors
// TestGenerateVL_residentContextCapPublishesBudgetClamped (generate_vl_resident_test.go) for
// GenerateGemma4VL's own single clamp site (M-02, docs/audit-2026-09-10.md) — before this fix,
// none of the 7 VL resident clamp sites published Budget/BudgetClamped, so a cap-truncated turn
// silently reported the same finish_reason as an ordinary EOS-terminated one.
func TestGenerateGemma4VL_residentContextCapPublishesBudgetClamped(t *testing.T) {
	m, g := loadGemma4VLBidirTiny(t)
	rf := &fakeResident{vocab: m.w.arch.VocabSize}
	rf.capPos = len(g.InputIDs) + 2
	m.resident = rf

	const maxNew = 5
	features := func() ([]float32, error) { return g.ImageFeatures, nil }
	stream, gen := m.GenerateGemma4VL(context.Background(), g.InputIDs, g.ImageTokenStart, g.NImageTokens, 0, features, maxNew, SamplingParams{Temperature: 0})
	var got []int
	for id := range stream {
		got = append(got, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("GenerateGemma4VL: %v", err)
	}
	if !gen.BudgetClamped {
		t.Error("BudgetClamped = false, want true — the resident cap bound this turn, not the request")
	}
	if gen.Budget != 2 {
		t.Errorf("Budget = %d, want 2 (capPos %d − prompt %d)", gen.Budget, rf.capPos, len(g.InputIDs))
	}
	if len(got) > gen.Budget {
		t.Errorf("streamed %d tokens, more than the published Budget %d", len(got), gen.Budget)
	}
}

// TestGenerateGemma4VL_residentUploadFailureFallsBackToCPU is the decline half: an UploadKV
// error (a resident backend that can't accept this turn's KV) must fall back to the CPU
// decode loop cleanly — no error surfaced to the caller, resIDs left forgotten (not stale,
// not falsely committed) since the resident cache was never actually seeded, and resBusy
// released.
func TestGenerateGemma4VL_residentUploadFailureFallsBackToCPU(t *testing.T) {
	m, g := loadGemma4VLBidirTiny(t)
	rf := &uploadFailResident{fakeResident: &fakeResident{vocab: m.w.arch.VocabSize}}
	m.resident = rf
	m.resIDs = []int{1, 2, 3}

	const maxNew = 4
	features := func() ([]float32, error) { return g.ImageFeatures, nil }
	stream, gen := m.GenerateGemma4VL(context.Background(), g.InputIDs, g.ImageTokenStart, g.NImageTokens, 0, features, maxNew, SamplingParams{Temperature: 0})
	var got []int
	for id := range stream {
		got = append(got, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("GenerateGemma4VL with a forced UploadKV decline: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("GenerateGemma4VL streamed no tokens on the CPU fallback path")
	}
	if rf.fakeResident.forwards != 0 {
		t.Errorf("resident Forward called %d times after a declined upload — decode must run entirely on CPU", rf.fakeResident.forwards)
	}
	if m.resIDs != nil {
		t.Errorf("resIDs = %v after a declined upload, want nil (forgotten) — the resident cache was never actually seeded this turn", m.resIDs)
	}
	if busy := atomic.LoadInt32(&m.resBusy); busy != 0 {
		t.Errorf("resBusy = %d after GenerateGemma4VL returned, want 0 (released)", busy)
	}
}

// TestGenerateGemma4VL_sequentialPathNeverTouchesResident confirms the resident bridge is
// gated on UseBidirectionalAttention, not attempted incidentally: an E2B/E4B-class checkpoint
// (use_bidirectional_attention unset) must never call UploadKV or Forward even when a resident
// backend is attached, since resident CUDA has no cross-layer-KV-sharing/PLE implementation for
// that class at all (a decode-side gap this bridge deliberately does not paper over).
func TestGenerateGemma4VL_sequentialPathNeverTouchesResident(t *testing.T) {
	m, g := loadGemma4VLTinySequential(t)
	rf := &fakeResident{vocab: m.w.arch.VocabSize}
	m.resident = rf

	const maxNew = 4
	features := func() ([]float32, error) { return g.ImageFeatures, nil }
	stream, gen := m.GenerateGemma4VL(context.Background(), g.InputIDs, g.ImageTokenStart, g.NImageTokens, 0, features, maxNew, SamplingParams{Temperature: 0})
	var got []int
	for id := range stream {
		got = append(got, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("GenerateGemma4VL: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("GenerateGemma4VL streamed no tokens")
	}
	if len(rf.uploadKVs) != 0 {
		t.Errorf("UploadKV called %d times for a sequential (non-bidirectional) checkpoint, want 0", len(rf.uploadKVs))
	}
	if rf.forwards != 0 {
		t.Errorf("resident Forward called %d times for a sequential (non-bidirectional) checkpoint, want 0", rf.forwards)
	}
}

// TestResidentCommitIDs_gemma4ImageBlockPreventsCrossImageReuse is M-07's own gate, at the unit
// level (no checkpoint needed — residentCommitIDs/residentReuseLen are pure Model-field
// manipulation, the same primitives GenerateGemma4VL's real commit call now uses). Mirrors
// TestResidentReuseLen_imageBlockAtomicity's style (this file's own package).
//
// Reproduces M-07's exact failure shape: Gemma4's soft-token placeholder ids are
// content-independent (same id/count for the same patch grid), so two DIFFERENT images of the
// same size produce IDENTICAL ids in that span — without a committed residentImageBlock,
// residentReuseLen's prefix scan has nothing to key a boundary check on and falls through to a
// plain id comparison that a different image's identical-shaped placeholder ids satisfy by
// coincidence, silently reusing the first image's K/V for the second.
func TestResidentCommitIDs_gemma4ImageBlockPreventsCrossImageReuse(t *testing.T) {
	m := &Model{}
	// [text 100,101] [image: 3 soft-tokens, all id 900 — content-independent] [text 200,201]
	prompt := []int{100, 101, 900, 900, 900, 200, 201}
	generated := []int{300, 301}
	m.residentCommitIDs(prompt, generated, &residentImageBlock{start: 2, end: 5, hash: 0xC0FFEE}, nil)

	if len(m.resImgBlocks) != 1 {
		t.Fatalf("resImgBlocks = %v, want exactly 1 entry", m.resImgBlocks)
	}
	if got := m.resImgBlocks[0]; got.start != 2 || got.end != 5 || got.hash != 0xC0FFEE {
		t.Errorf("resImgBlocks[0] = %+v, want {start:2 end:5 hash:0xC0FFEE}", got)
	}

	// A SECOND, DIFFERENT image (different hash) at the identical span, same placeholder ids —
	// the exact scenario M-07 describes. Without this pass's fix, the scan would report a full
	// match through the whole shared prefix (id-identical); with the block recorded, it must
	// stop at (or before) the block's start instead of reusing past it.
	secondPrompt := []int{100, 101, 900, 900, 900, 202, 203}
	claims := []residentImageClaim{{Start: 2, Len: 3, Hash: 0xBADC0DE}} // different hash — NOT the same image
	if got := m.residentReuseLen(secondPrompt, claims, nil); got > 2 {
		t.Errorf("residentReuseLen = %d, want <= 2 — a differently-hashed image at the same span "+
			"must not be treated as reusable, but the scan walked past the image boundary", got)
	}
}
