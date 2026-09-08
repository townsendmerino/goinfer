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

// TestGenerateVL_residentDecodeEngagesAndUploadsKV is gap 0's wiring gate for GenerateVL, extended
// for P9(a): given a resident backend, the turn must (a) call UploadKV once per layer with the CPU
// prefill's K/V, (b) dispatch decode through the resident Forward (not the CPU m.forward), and (c)
// COMMIT resIDs + the image block afterward — a stale resIDs from an unrelated prior generation
// must not survive as a false reuse candidate (m.resIDs starts at a value that cannot satisfy the
// image-block check below), and this turn's own prefix becomes the reuse candidate for the NEXT
// turn if the same image comes back (the entire point of P9(a); GenerateVL used to be forbidden
// from touching resIDs at all, then gap 0 made it unconditionally forget — now it commits).
func TestGenerateVL_residentDecodeEngagesAndUploadsKV(t *testing.T) {
	m, g := loadGemma3VLTiny(t)
	rf := &fakeResident{vocab: m.w.arch.VocabSize}
	m.resident = rf
	m.resIDs = []int{9, 9, 9} // stale — must not be mistaken for a reusable image block

	const maxNew = 5
	features := func() ([]float32, error) { return g.ImageFeatures, nil }
	stream, gen := m.GenerateVL(context.Background(), g.InputIDs, g.ImageTokenStart, g.MMTokens, 0, features, maxNew, SamplingParams{Temperature: 0})
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
	want := append(append([]int{}, g.InputIDs...), got...)
	if !equalIntSlices(m.resIDs, want) {
		t.Errorf("resIDs = %v after a resident-touching GenerateVL, want prompt+generated %v — P9(a) commits "+
			"the prefix so a future resend of the same image can reuse it", m.resIDs, want)
	}
	if len(m.resImgBlocks) != 1 || m.resImgBlocks[0].start != g.ImageTokenStart || m.resImgBlocks[0].end != g.ImageTokenStart+g.MMTokens {
		t.Errorf("resImgBlocks = %v, want one block [%d,%d)", m.resImgBlocks, g.ImageTokenStart, g.ImageTokenStart+g.MMTokens)
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
	features := func() ([]float32, error) { return g.ImageFeatures, nil }
	stream, gen := m.GenerateVL(context.Background(), g.InputIDs, g.ImageTokenStart, g.MMTokens, 0, features, maxNew, SamplingParams{Temperature: 0})
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
		features := func() ([]float32, error) { return g.ImageFeatures, nil }
		stream, gen := m.GenerateVL(context.Background(), g.InputIDs, g.ImageTokenStart, g.MMTokens, 0, features, 4, SamplingParams{Temperature: 0})
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

// TestGenerateQwenVL_residentDecodeUsesForwardMRoPE is gap 0's wiring gate for GenerateQwenVL,
// extended for P9(a): with a ResidentMRoPE-capable resident attached, decode past the image block
// must dispatch through ForwardMRoPE (not the base Forward), with ropePos = pos + mropeDelta — the
// exact distinction Forward alone cannot express (decoder/residency.go's ResidentMRoPE doc
// comment) — and the turn must COMMIT resIDs + the image block afterward (see the Gemma 3 sibling
// test's doc comment for why commit, not forget).
func TestGenerateQwenVL_residentDecodeUsesForwardMRoPE(t *testing.T) {
	m, g := loadQwen25VLTiny(t)
	rf := &fakeResident{vocab: m.w.arch.VocabSize}
	m.resident = rf
	m.resIDs = []int{9, 9, 9}

	features := func() ([]float32, error) { return g.ImageFeatures, nil }
	stream, gen := m.GenerateQwenVL(context.Background(), g.InputIDs, g.ImageStart, g.NImageTokens, 0, features,
		g.GridTHW, 2, g.ImageToken, len(g.Continuation), SamplingParams{Temperature: 0})
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
	want := append(append([]int{}, g.InputIDs...), got...)
	if !equalIntSlices(m.resIDs, want) {
		t.Errorf("resIDs = %v after a resident-touching GenerateQwenVL, want prompt+generated %v (P9(a) commit)", m.resIDs, want)
	}
	if len(m.resImgBlocks) != 1 || m.resImgBlocks[0].start != g.ImageStart || m.resImgBlocks[0].end != g.ImageStart+g.NImageTokens {
		t.Errorf("resImgBlocks = %v, want one block [%d,%d)", m.resImgBlocks, g.ImageStart, g.ImageStart+g.NImageTokens)
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

	features := func() ([]float32, error) { return g.ImageFeatures, nil }
	stream, gen := m.GenerateQwenVL(context.Background(), g.InputIDs, g.ImageStart, g.NImageTokens, 0, features,
		g.GridTHW, 2, g.ImageToken, len(g.Continuation), SamplingParams{Temperature: 0})
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

// TestGenerateVL_imageReuseFastPath_sameImageSkipsTower is P9(a)'s own wiring gate, one level up
// from the atomic-block-aware LCP scan's unit tests (resident_reuse_test.go): driven through the
// REAL GenerateVL entrypoint, a turn that resends the SAME image (same imgHash) as an
// already-committed resident block must reuse it — the tower closure must never run, UploadKV
// must not be called again, and Generation.PrefillReused must report the fast path actually
// fired (not just "no error," which a silently-declined fast path would also produce).
func TestGenerateVL_imageReuseFastPath_sameImageSkipsTower(t *testing.T) {
	m, g := loadGemma3VLTiny(t)
	rf := &fakeResident{vocab: m.w.arch.VocabSize}
	m.resident = rf

	towerCalls := 0
	features := func() ([]float32, error) {
		towerCalls++
		return g.ImageFeatures, nil
	}
	const imgHash = 42
	const maxNew = 3

	// Turn 1: cold — commits the image block to the resident KV.
	stream1, gen1 := m.GenerateVL(context.Background(), g.InputIDs, g.ImageTokenStart, g.MMTokens, imgHash, features, maxNew, SamplingParams{Temperature: 0})
	var gen1IDs []int
	for id := range stream1 {
		gen1IDs = append(gen1IDs, id)
	}
	if err := gen1.Err(); err != nil {
		t.Fatalf("turn 1 GenerateVL: %v", err)
	}
	if towerCalls != 1 {
		t.Fatalf("turn 1: tower called %d times, want 1 (a cold turn must run it)", towerCalls)
	}
	if gen1.PrefillReused != 0 {
		t.Errorf("turn 1: PrefillReused = %d, want 0 (nothing committed yet)", gen1.PrefillReused)
	}
	uploadsAfterTurn1 := len(rf.uploadKVs)
	if uploadsAfterTurn1 == 0 {
		t.Fatal("turn 1 never uploaded KV — test setup broke, turn 2 couldn't reuse anything real")
	}

	// Turn 2: the SAME image resent, as a strict extension of turn 1's committed prefix — the
	// agent-loop shape P9(a) exists for (previous prompt + reply + new user text).
	prompt2 := append(append([]int{}, g.InputIDs...), gen1IDs...)
	prompt2 = append(prompt2, g.InputIDs[len(g.InputIDs)-1]) // one more token, extending past the commit
	stream2, gen2 := m.GenerateVL(context.Background(), prompt2, g.ImageTokenStart, g.MMTokens, imgHash, features, maxNew, SamplingParams{Temperature: 0})
	var got2 []int
	for id := range stream2 {
		got2 = append(got2, id)
	}
	if err := gen2.Err(); err != nil {
		t.Fatalf("turn 2 GenerateVL: %v", err)
	}
	if len(got2) == 0 {
		t.Fatal("turn 2 streamed no tokens")
	}
	if towerCalls != 1 {
		t.Errorf("turn 2: tower called again (now %d total) — the same resent image must skip it entirely", towerCalls)
	}
	if len(rf.uploadKVs) != uploadsAfterTurn1 {
		t.Errorf("turn 2: UploadKV called again (%d total, was %d) — the fast path must reseed via Forward, not re-upload", len(rf.uploadKVs), uploadsAfterTurn1)
	}
	if want := g.ImageTokenStart + g.MMTokens; gen2.PrefillReused < want {
		t.Errorf("turn 2: PrefillReused = %d, want >= %d (the image block plus everything before it)", gen2.PrefillReused, want)
	}
	wantIDs := append(append([]int{}, prompt2...), got2...)
	if !equalIntSlices(m.resIDs, wantIDs) {
		t.Errorf("resIDs after turn 2 = %v, want prompt2+generated2 %v", m.resIDs, wantIDs)
	}
	if len(m.resImgBlocks) != 1 {
		t.Errorf("resImgBlocks after turn 2 = %v, want exactly 1 (re-verified, not duplicated)", m.resImgBlocks)
	}
}

// TestGenerateVL_imageReuseFastPath_differentImageDoesNotSkipTower is the atomicity kill
// condition's decoder-level half (the real-hardware half lives in
// gpu/resident_reuse_parity_test.go): a turn that claims a DIFFERENT image (a different imgHash)
// at the SAME placeholder position an earlier turn committed must NOT take the fast path — the
// tower must run, and the turn must fall through to the ordinary full-prefill path exactly as if
// nothing were resident at all.
func TestGenerateVL_imageReuseFastPath_differentImageDoesNotSkipTower(t *testing.T) {
	m, g := loadGemma3VLTiny(t)
	rf := &fakeResident{vocab: m.w.arch.VocabSize}
	m.resident = rf

	towerCalls := 0
	features := func() ([]float32, error) {
		towerCalls++
		return g.ImageFeatures, nil
	}
	const maxNew = 3

	stream1, gen1 := m.GenerateVL(context.Background(), g.InputIDs, g.ImageTokenStart, g.MMTokens, 42, features, maxNew, SamplingParams{Temperature: 0})
	for range stream1 { //nolint:revive // draining is the point
	}
	if err := gen1.Err(); err != nil {
		t.Fatalf("turn 1 GenerateVL: %v", err)
	}
	if towerCalls != 1 {
		t.Fatalf("turn 1: tower called %d times, want 1", towerCalls)
	}

	// Turn 2: SAME placeholder position, a DIFFERENT claimed image (hash 99, not 42) — the
	// adversarial case. The placeholder token ids themselves are identical either way (that is
	// exactly why the hash exists at all), so g.InputIDs is reused verbatim as the prompt.
	stream2, gen2 := m.GenerateVL(context.Background(), g.InputIDs, g.ImageTokenStart, g.MMTokens, 99, features, maxNew, SamplingParams{Temperature: 0})
	for range stream2 { //nolint:revive
	}
	if err := gen2.Err(); err != nil {
		t.Fatalf("turn 2 GenerateVL: %v", err)
	}
	if towerCalls != 2 {
		t.Errorf("turn 2: tower called %d times total, want 2 — a different image must never skip it", towerCalls)
	}
	if gen2.PrefillReused != 0 {
		t.Errorf("turn 2: PrefillReused = %d, want 0 — a hash mismatch at the block's own start must not report any reuse", gen2.PrefillReused)
	}
}

// TestGenerateQwenVL_imageReuseFastPath_sameImageUsesForwardMRoPE confirms P9(a)'s fast path
// exercises Qwen's m-RoPE-aware reseed (residentPrefillSeedMRoPE), not just the Gemma-shaped
// plain-Forward path TestGenerateVL_imageReuseFastPath_sameImageSkipsTower already covers — the
// ropePos = pos + mropeDelta distinction only ForwardMRoPE can express (decoder/residency.go).
func TestGenerateQwenVL_imageReuseFastPath_sameImageUsesForwardMRoPE(t *testing.T) {
	m, g := loadQwen25VLTiny(t)
	rf := &fakeResident{vocab: m.w.arch.VocabSize}
	m.resident = rf

	towerCalls := 0
	features := func() ([]float32, error) {
		towerCalls++
		return g.ImageFeatures, nil
	}
	const imgHash = 7
	maxNew := len(g.Continuation)

	stream1, gen1 := m.GenerateQwenVL(context.Background(), g.InputIDs, g.ImageStart, g.NImageTokens, imgHash, features,
		g.GridTHW, 2, g.ImageToken, maxNew, SamplingParams{Temperature: 0})
	var gen1IDs []int
	for id := range stream1 {
		gen1IDs = append(gen1IDs, id)
	}
	if err := gen1.Err(); err != nil {
		t.Fatalf("turn 1 GenerateQwenVL: %v", err)
	}
	if towerCalls != 1 {
		t.Fatalf("turn 1: tower called %d times, want 1", towerCalls)
	}
	forwardsAfterTurn1 := rf.forwards

	prompt2 := append(append([]int{}, g.InputIDs...), gen1IDs...)
	prompt2 = append(prompt2, g.InputIDs[len(g.InputIDs)-1])
	stream2, gen2 := m.GenerateQwenVL(context.Background(), prompt2, g.ImageStart, g.NImageTokens, imgHash, features,
		g.GridTHW, 2, g.ImageToken, maxNew, SamplingParams{Temperature: 0})
	var got2 []int
	for id := range stream2 {
		got2 = append(got2, id)
	}
	if err := gen2.Err(); err != nil {
		t.Fatalf("turn 2 GenerateQwenVL: %v", err)
	}
	if len(got2) == 0 {
		t.Fatal("turn 2 streamed no tokens")
	}
	if towerCalls != 1 {
		t.Errorf("turn 2: tower called again (now %d total) — the same resent image must skip it", towerCalls)
	}
	if want := g.ImageStart + g.NImageTokens; gen2.PrefillReused < want {
		t.Errorf("turn 2: PrefillReused = %d, want >= %d", gen2.PrefillReused, want)
	}
	if rf.forwards <= forwardsAfterTurn1 {
		t.Error("turn 2 never called resident Forward (via ForwardMRoPE) — decode did not run at all")
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
