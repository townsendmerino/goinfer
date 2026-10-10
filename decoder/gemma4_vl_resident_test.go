package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
// TestGenerateVL_residentContextCapPublishesBudgetClamped (generate_vl_resident_test.go) for GenerateGemma4VL's own
// single clamp site (M-02, docs/audit-2026-09-10.md): a cap-truncated turn must publish Budget/BudgetClamped, or it
// reports the same finish_reason as an ordinary EOS-terminated one.
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

// TestGenerateGemma4VL_sequentialEModelUploadsOwningLayers is the gitignored gemma4-vl-tiny fixture's check of the S1.8
// contract (docs/tasks/task-multimodal-support-2026-10.md): a sequential (causal) checkpoint of the E-model class (this one
// has KV-shared layers) with a resident attached uploads its CPU prefill's K/V for the layers that own it, never a KV-shared
// one, then decodes resident. CI skips it (the fixture is gitignored); the same contract runs on a committed-path fixture in
// TestGenerateGemma4VL_eModelDecodesResident.
func TestGenerateGemma4VL_sequentialEModelUploadsOwningLayers(t *testing.T) {
	m, g := loadGemma4VLTinySequential(t)
	if m.w.arch.gemma4.SharedKVLayers == 0 && m.w.arch.gemma4.HiddenSizePerLayerInput == 0 {
		t.Fatal("gemma4-vl-tiny is no longer E-model-shaped; this test's premise needs re-reading")
	}
	features := func() ([]float32, error) { return g.ImageFeatures, nil }
	assertEModelResidentDecode(t, m, g.InputIDs, g.ImageTokenStart, g.NImageTokens, features)
}

// TestGenerateGemma4VL_eModelDecodesResident is S1.8's contract on the tiny E-model fixture (PLE, two KV-shared
// layers), with synthetic image features: the image turn's CPU prefill uploads K/V for the four owning layers only and
// decode runs resident (Generation.DecodeResident).
func TestGenerateGemma4VL_eModelDecodesResident(t *testing.T) {
	if _, err := os.Stat(gemma4EModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no tiny checkpoint (%s) — run scripts/pin_gemma4_emodel_tiny.py", gemma4EModelDir)
	}
	m, err := Load(gemma4EModelDir, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids, pos, n, features := syntheticGemma4Image(m)
	assertEModelResidentDecode(t, m, ids, pos, n, features)
}

// TestGenerateGemma4VL_causalNonEModelStaysOnCPU keeps the original test's intent for the class it still covers: a
// causal Gemma 4 that is NOT an E-model decodes an image turn on the CPU even with a resident attached (S1.8 widened
// the gate to E-models only; widening it further is untested on CUDA).
func TestGenerateGemma4VL_causalNonEModelStaysOnCPU(t *testing.T) {
	const dir = "../testdata/gemma4-dense-twogeom-tiny"
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s)", dir)
	}
	m, err := Load(dir, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if g4 := m.w.arch.gemma4; g4.SharedKVLayers != 0 || g4.HiddenSizePerLayerInput != 0 || m.w.Cfg.UseBidirectionalAttention != "" {
		t.Fatal("the two-geometry fixture is no longer a causal non-E Gemma 4")
	}
	rf := &fakeResident{vocab: m.w.arch.VocabSize}
	m.resident = rf
	ids, pos, n, features := syntheticGemma4Image(m)
	stream, gen := m.GenerateGemma4VL(context.Background(), ids, pos, n, 0, features, 4, SamplingParams{Temperature: 0})
	var got []int
	for id := range stream {
		got = append(got, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 {
		t.Fatal("no tokens")
	}
	if len(rf.uploadKVs) != 0 || rf.forwards != 0 || gen.DecodeResident {
		t.Errorf("a causal non-E Gemma 4 touched the resident: %d uploads, %d forwards, DecodeResident=%v", len(rf.uploadKVs), rf.forwards, gen.DecodeResident)
	}
}

// syntheticGemma4Image is a small prompt with a 4-token image block of deterministic features, for the resident-bridge
// contract tests (which check where K/V goes and where decode runs, not what the image says).
func syntheticGemma4Image(m *Model) (ids []int, pos, n int, features func() ([]float32, error)) {
	H := m.w.arch.HiddenDim
	ids = []int{2, 10, 11, 0, 0, 0, 0, 12, 13}
	pos, n = 3, 4
	feats := make([]float32, n*H)
	for i := range feats {
		feats[i] = float32((i*37)%101-50) / 100
	}
	return ids, pos, n, func() ([]float32, error) { return feats, nil }
}

func assertEModelResidentDecode(t *testing.T, m *Model, ids []int, imgPos, imgLen int, features func() ([]float32, error)) {
	t.Helper()
	rf := &fakeResident{vocab: m.w.arch.VocabSize}
	m.resident = rf
	stream, gen := m.GenerateGemma4VL(context.Background(), ids, imgPos, imgLen, 0, features, 4, SamplingParams{Temperature: 0})
	var got []int
	for id := range stream {
		got = append(got, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("GenerateGemma4VL: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("no tokens")
	}
	if !gen.DecodeResident || rf.forwards == 0 {
		t.Errorf("an E-model image turn did not decode resident: DecodeResident=%v, %d resident forwards", gen.DecodeResident, rf.forwards)
	}
	var owning, uploaded []int
	for l := range m.w.arch.NumLayers {
		if m.w.arch.gemma4KVSrcAt(l) == l {
			owning = append(owning, l)
		}
	}
	for _, c := range rf.uploadKVs {
		uploaded = append(uploaded, c.layer)
	}
	if fmt.Sprint(uploaded) != fmt.Sprint(owning) {
		t.Errorf("UploadKV went to layers %v, want exactly the K/V-owning layers %v (never a KV-shared one)", uploaded, owning)
	}
	t.Logf("uploaded layers %v; %d resident forwards for %d tokens", uploaded, rf.forwards, len(got))
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
	m.residentCommitIDs(prompt, generated, []residentImageBlock{{start: 2, end: 5, hash: 0xC0FFEE}}, nil)

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
