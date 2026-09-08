package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"sync"
	"sync/atomic"
	"testing"
)

// gemma3VLGolden mirrors TestGenerateVL_streams' inline struct (generate_vl_test.go) — pulled
// out to a named type so it can be shared with this file's resident-decode tests.
type gemma3VLGolden struct {
	InputIDs        []int     `json:"input_ids"`
	ImageTokenStart int       `json:"image_token_start"`
	MMTokens        int       `json:"mm_tokens_per_image"`
	ImageFeatures   []float32 `json:"image_features"`
	Argmax          int       `json:"argmax"`
}

func loadGemma3VLTiny(t *testing.T) (*Model, gemma3VLGolden) {
	t.Helper()
	const golden = "../testdata/gemma3_vl_tiny_image_golden.json"
	const ckpt = "../testdata/gemma3-vl-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no image golden — run scripts/pin_gemma3_vl_image.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skip("no checkpoint — run scripts/pin_gemma3_vl_tiny.py")
	}
	var g gemma3VLGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m, g
}

// TestGenerateVL_residentDecodeEngagesAndUploadsKV is gap 0's wiring gate for GenerateVL: given
// a resident backend, the turn must (a) call UploadKV once per layer with the CPU prefill's K/V,
// (b) dispatch decode through the resident Forward (not the CPU m.forward), and (c) forget resIDs
// afterward — the resident cache now holds THIS turn's content, so a stale prefix-reuse record
// must not survive (V-11's discipline, now required in the opposite direction: GenerateVL used
// to be forbidden from touching resIDs at all because it never touched resident; now that it
// does, forgetting is mandatory, not forbidden).
func TestGenerateVL_residentDecodeEngagesAndUploadsKV(t *testing.T) {
	m, g := loadGemma3VLTiny(t)
	rf := &fakeResident{vocab: m.w.arch.VocabSize}
	m.resident = rf
	m.resIDs = []int{9, 9, 9} // must NOT survive — see doc comment

	const maxNew = 5
	stream, gen := m.GenerateVL(context.Background(), g.InputIDs, g.ImageFeatures, g.ImageTokenStart, g.MMTokens, maxNew, SamplingParams{Temperature: 0})
	var got []int
	for id := range stream {
		got = append(got, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("GenerateVL: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("GenerateVL streamed no tokens")
	}
	if nLayers := m.w.arch.NumLayers; len(rf.uploadKVs) != nLayers {
		t.Errorf("UploadKV called %d times, want %d (once per layer)", len(rf.uploadKVs), nLayers)
	}
	for l, c := range rf.uploadKVs {
		if c.layer != l {
			t.Errorf("uploadKVs[%d].layer = %d, want %d — layers must upload in order", l, c.layer, l)
		}
		if c.base != 0 {
			t.Errorf("uploadKVs[%d].base = %d, want 0 — gemma3-vl-tiny's short prefill should never wrap a ring", l, c.base)
		}
		if len(c.keys) == 0 {
			t.Errorf("uploadKVs[%d]: empty keys — CPU prefill produced nothing to upload", l)
		}
	}
	if rf.forwards == 0 {
		t.Error("resident Forward was never called — decode silently ran on CPU despite a resident being present")
	}
	if m.resIDs != nil {
		t.Errorf("resIDs = %v after a resident-touching GenerateVL, want nil (forgotten) — the resident "+
			"cache now holds this turn's image content, not whatever resIDs described before", m.resIDs)
	}
	if busy := atomic.LoadInt32(&m.resBusy); busy != 0 {
		t.Errorf("resBusy = %d after GenerateVL returned, want 0 (released)", busy)
	}
}

// TestGenerateVL_residentBusyDeclinesToCPU is the other half: a resident already claimed by a
// concurrent generation must make GenerateVL fall back to the CPU path cleanly — no error, no
// resident calls, the claim and any pre-existing resIDs left exactly as found (this call never
// touched them, so V-11's original rule — a decline must not forget — still applies here).
func TestGenerateVL_residentBusyDeclinesToCPU(t *testing.T) {
	m, g := loadGemma3VLTiny(t)
	rf := &fakeResident{vocab: m.w.arch.VocabSize}
	m.resident = rf
	m.resIDs = []int{1, 2, 3}
	atomic.StoreInt32(&m.resBusy, 1) // simulate another in-flight generation holding the claim

	const maxNew = 5
	stream, gen := m.GenerateVL(context.Background(), g.InputIDs, g.ImageFeatures, g.ImageTokenStart, g.MMTokens, maxNew, SamplingParams{Temperature: 0})
	var got []int
	for id := range stream {
		got = append(got, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("GenerateVL with resBusy held: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("GenerateVL streamed no tokens on the CPU fallback path")
	}
	if rf.forwards != 0 {
		t.Errorf("resident Forward called %d times while resBusy was held — the busy path must never touch the resident", rf.forwards)
	}
	if len(rf.uploadKVs) != 0 {
		t.Errorf("UploadKV called %d times while resBusy was held", len(rf.uploadKVs))
	}
	if !equalIntSlices(m.resIDs, []int{1, 2, 3}) {
		t.Errorf("resIDs = %v after a declined claim, want unchanged [1 2 3] — a losing claim must not "+
			"forget a commit it never touched", m.resIDs)
	}
	if busy := atomic.LoadInt32(&m.resBusy); busy != 1 {
		t.Errorf("resBusy = %d, want 1 (still held by the simulated concurrent caller)", busy)
	}
}

// TestGenerateVL_residentConcurrencyRace is the direct regression guard for V-11's actual bug
// shape (docs/review-2026-09-04.md), now meaningful for the first time since that fix: a
// resident-touching plain Generate and a resident-touching GenerateVL running concurrently on
// the SAME *Model must not corrupt resIDs or double-claim resBusy. Run with -race.
func TestGenerateVL_residentConcurrencyRace(t *testing.T) {
	m, g := loadGemma3VLTiny(t)
	rf := &fakeResident{vocab: m.w.arch.VocabSize}
	m.resident = rf

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		stream, _ := m.Generate(context.Background(), []int{1, 2, 3}, 4, SamplingParams{Temperature: 0})
		for range stream { //nolint:revive // draining is the point
		}
	}()
	go func() {
		defer wg.Done()
		stream, gen := m.GenerateVL(context.Background(), g.InputIDs, g.ImageFeatures, g.ImageTokenStart, g.MMTokens, 4, SamplingParams{Temperature: 0})
		for range stream { //nolint:revive
		}
		if err := gen.Err(); err != nil {
			t.Errorf("GenerateVL under concurrent resident contention: %v", err)
		}
	}()
	wg.Wait()

	if busy := atomic.LoadInt32(&m.resBusy); busy != 0 {
		t.Errorf("resBusy = %d after both generations completed, want 0 — a claim was never released", busy)
	}
}

// qwen25vlGolden mirrors TestQwen25VL_generate's inline struct (qwen25vl_test.go).
type qwen25vlGolden struct {
	InputIDs      []int     `json:"input_ids"`
	ImageToken    int       `json:"image_token_id"`
	ImageStart    int       `json:"image_token_start"`
	NImageTokens  int       `json:"n_image_tokens"`
	GridTHW       [][3]int  `json:"grid_thw"`
	ImageFeatures []float32 `json:"image_features"`
	Continuation  []int     `json:"continuation_ids"`
}

func loadQwen25VLTiny(t *testing.T) (*Model, qwen25vlGolden) {
	t.Helper()
	const golden = "../testdata/qwen25vl_tiny_image_golden.json"
	const ckpt = "../testdata/qwen25vl-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no golden — run scripts/pin_qwen25vl_image.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skip("no checkpoint")
	}
	var g qwen25vlGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m, g
}

// TestGenerateQwenVL_residentDecodeUsesForwardMRoPE is gap 0's wiring gate for GenerateQwenVL:
// with a ResidentMRoPE-capable resident attached, decode past the image block must dispatch
// through ForwardMRoPE (not the base Forward), with ropePos = pos + cache.mropeDelta — the exact
// distinction Forward alone cannot express (decoder/residency.go's ResidentMRoPE doc comment).
func TestGenerateQwenVL_residentDecodeUsesForwardMRoPE(t *testing.T) {
	m, g := loadQwen25VLTiny(t)
	rf := &fakeResident{vocab: m.w.arch.VocabSize}
	m.resident = rf
	m.resIDs = []int{9, 9, 9}

	stream, gen := m.GenerateQwenVL(context.Background(), g.InputIDs, g.ImageFeatures,
		g.ImageStart, g.NImageTokens, g.GridTHW, 2, g.ImageToken, len(g.Continuation), SamplingParams{Temperature: 0})
	var got []int
	for id := range stream {
		got = append(got, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("GenerateQwenVL: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("GenerateQwenVL streamed no tokens")
	}
	if nLayers := m.w.arch.NumLayers; len(rf.uploadKVs) != nLayers {
		t.Errorf("UploadKV called %d times, want %d (once per layer)", len(rf.uploadKVs), nLayers)
	}
	if rf.forwards == 0 {
		t.Error("resident ForwardMRoPE (which calls into Forward's bookkeeping) was never counted — decode silently ran on CPU")
	}
	if m.resIDs != nil {
		t.Errorf("resIDs = %v after a resident-touching GenerateQwenVL, want nil (forgotten)", m.resIDs)
	}
	if busy := atomic.LoadInt32(&m.resBusy); busy != 0 {
		t.Errorf("resBusy = %d after GenerateQwenVL returned, want 0 (released)", busy)
	}
}

// TestGenerateQwenVL_declinesResidentWithoutMRoPE: a resident that satisfies ResidentForward but
// NOT ResidentMRoPE (residentForwardOnly, below — the shape a hypothetical backend without m-RoPE
// support has) must never be claimed for Qwen decode — the claim attempt is gated on the type
// assertion BEFORE calling tryClaimResident, so this must fall back to CPU without ever touching
// resBusy/resIDs.
func TestGenerateQwenVL_declinesResidentWithoutMRoPE(t *testing.T) {
	m, g := loadQwen25VLTiny(t)
	rf := &residentForwardOnly{fakeResident: &fakeResident{vocab: m.w.arch.VocabSize}}
	m.resident = rf
	m.resIDs = []int{4, 5, 6}

	stream, gen := m.GenerateQwenVL(context.Background(), g.InputIDs, g.ImageFeatures,
		g.ImageStart, g.NImageTokens, g.GridTHW, 2, g.ImageToken, len(g.Continuation), SamplingParams{Temperature: 0})
	var got []int
	for id := range stream {
		got = append(got, id)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("GenerateQwenVL with a non-m-RoPE resident: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("GenerateQwenVL streamed no tokens on the CPU fallback path")
	}
	if rf.fakeResident.forwards != 0 {
		t.Errorf("resident Forward called %d times — a resident without ResidentMRoPE must never be claimed for Qwen decode", rf.fakeResident.forwards)
	}
	if !equalIntSlices(m.resIDs, []int{4, 5, 6}) {
		t.Errorf("resIDs = %v, want unchanged [4 5 6] — the claim must never be attempted for a resident this call cannot use", m.resIDs)
	}
}

// residentForwardOnly exposes exactly decoder.ResidentForward from a *fakeResident, without
// ForwardMRoPE — Go interface satisfaction is structural, so a wrapper that does NOT itself
// define ForwardMRoPE, and does not embed fakeResident in a way that promotes it, fails the
// decoder.ResidentMRoPE assertion even though the underlying fake could serve it.
type residentForwardOnly struct{ fakeResident *fakeResident }

func (r *residentForwardOnly) Forward(embedding []float32, pos int) ([]float32, error) {
	return r.fakeResident.Forward(embedding, pos)
}
func (r *residentForwardOnly) ForwardN(embeddings [][]float32, startPos int) ([][]float32, error) {
	return r.fakeResident.ForwardN(embeddings, startPos)
}
func (r *residentForwardOnly) UploadKV(layer, base int, keys, vals []float32) error {
	return r.fakeResident.UploadKV(layer, base, keys, vals)
}
func (r *residentForwardOnly) TruncateTo(pos int) { r.fakeResident.TruncateTo(pos) }
func (r *residentForwardOnly) Reset()             { r.fakeResident.Reset() }
func (r *residentForwardOnly) Close() error       { return r.fakeResident.Close() }
func (r *residentForwardOnly) ContextCap() int    { return r.fakeResident.ContextCap() }
