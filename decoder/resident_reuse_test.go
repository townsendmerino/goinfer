package decoder

import "testing"

// TestResidentReuseLen pins the prefix rule, including the two cases that make it SAFE rather
// than merely fast: it never claims the whole prompt (the seed's logits must be recomputed),
// and it stops at the first divergent id rather than resynchronising later — a client that
// edits its last message shifts tokenisation, and everything after the edit is a different
// conversation even where the text looks similar.
func TestResidentReuseLen(t *testing.T) {
	for _, tc := range []struct {
		name   string
		cached []int
		prompt []int
		want   int
	}{
		{"nothing cached", nil, []int{1, 2, 3}, 0},
		{"empty prompt", []int{1, 2, 3}, nil, 0},
		{"exact prefix extension (the agent-turn case)", []int{1, 2, 3}, []int{1, 2, 3, 4, 5}, 3},
		{"cache longer than prompt", []int{1, 2, 3, 4, 5}, []int{1, 2, 3}, 2},
		{"identical — must still leave a seed token", []int{1, 2, 3}, []int{1, 2, 3}, 2},
		{"diverges at 0", []int{9, 2, 3}, []int{1, 2, 3, 4}, 0},
		{"diverges midway", []int{1, 2, 9, 9}, []int{1, 2, 3, 4}, 2},
		{"a later match must NOT resynchronise", []int{1, 9, 3, 4}, []int{1, 2, 3, 4}, 1},
		{"single-token prompt", []int{1}, []int{1}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := &Model{resIDs: tc.cached}
			if got := m.residentReuseLen(tc.prompt, nil, nil); got != tc.want {
				t.Errorf("residentReuseLen(%v | cached %v) = %d, want %d", tc.prompt, tc.cached, got, tc.want)
			}
		})
	}
}

// TestResidentReuseLen_neverClaimsTheSeed is the invariant stated separately because a
// regression here is silent: reusing the whole prompt leaves generateInto with no seed logits
// and nothing to recompute them from.
func TestResidentReuseLen_neverClaimsTheSeed(t *testing.T) {
	for n := 1; n <= 8; n++ {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = i + 1
		}
		m := &Model{resIDs: ids}
		if got := m.residentReuseLen(ids, nil, nil); got >= n {
			t.Errorf("prompt of %d identical tokens reused %d — must leave at least one to prefill", n, got)
		}
	}
}

// recurrentTestModel builds a *Model whose hasRecurrentState() is true (qwen3_5_moe's dispatch
// table entry has Recurrent: true), without loading any real weights — residentReuseLen's id
// arithmetic never touches them.
func recurrentTestModel(resIDs []int) *Model {
	return &Model{w: &Weights{arch: &Architecture{qwen35: &qwen35Params{}}}, resIDs: resIDs}
}

// TestResidentReuseLen_recurrentExactExtensionOnly is audit R-01 Phase 0: a recurrent family's
// state cannot be rewound to an arbitrary earlier position (unlike a positional KV, which the
// generic branch above LCP-matches into), so the ONLY safe reuse is a strict extension of the
// entire committed sequence — anything else, including a partial/diverging match the generic
// branch WOULD reuse, must fall to 0.
func TestResidentReuseLen_recurrentExactExtensionOnly(t *testing.T) {
	for _, tc := range []struct {
		name   string
		cached []int
		prompt []int
		want   int
	}{
		{"nothing cached", nil, []int{1, 2, 3}, 0},
		{"exact prefix extension (the agent-turn case)", []int{1, 2, 3}, []int{1, 2, 3, 4, 5}, 3},
		{"single-token extension", []int{1, 2, 3}, []int{1, 2, 3, 4}, 3},
		{"identical resend — no new token to extend with, must fall to 0 (the qwen3.6-35B-A3B repro)",
			[]int{1, 2, 3}, []int{1, 2, 3}, 0},
		{"shorter resend", []int{1, 2, 3}, []int{1, 2}, 0},
		{"diverges at 0 — the generic branch would still reuse 0 here too, but for a different reason",
			[]int{9, 2, 3}, []int{1, 2, 3, 4}, 0},
		{"diverges midway — the generic branch would reuse 2 here; recurrent must not", []int{1, 2, 9}, []int{1, 2, 3, 4}, 0},
		{"an edited earlier message with a coincidental later match must not resynchronise",
			[]int{1, 9, 3}, []int{1, 2, 3, 4}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := recurrentTestModel(tc.cached)
			if !m.hasRecurrentState() {
				t.Fatal("test setup broke: recurrentTestModel is not recognised as recurrent")
			}
			if got := m.residentReuseLen(tc.prompt, nil, nil); got != tc.want {
				t.Errorf("residentReuseLen(%v | cached %v) = %d, want %d", tc.prompt, tc.cached, got, tc.want)
			}
		})
	}
}

// TestResidentForgetIDs: forgetting must be total, because a PARTIALLY stale id list is worse
// than none — it would match a prefix that the cache no longer holds.
func TestResidentForgetIDs(t *testing.T) {
	m := &Model{}
	m.residentCommitIDs([]int{1, 2}, []int{3, 4}, nil, nil)
	if len(m.resIDs) != 4 {
		t.Fatalf("commit recorded %v, want prompt+generated", m.resIDs)
	}
	m.residentForgetIDs()
	if m.resIDs != nil {
		t.Errorf("forget left %v, want nil", m.resIDs)
	}
	if got := m.residentReuseLen([]int{1, 2, 3, 4, 5}, nil, nil); got != 0 {
		t.Errorf("after forgetting, reuse must be 0, got %d", got)
	}
}

// TestResidentCommitIDs_copies: the recorded list must not alias the caller's slices, or a
// later append by the caller silently rewrites what we believe the cache holds.
func TestResidentCommitIDs_copies(t *testing.T) {
	prompt := []int{1, 2, 3}
	generated := []int{4, 5}
	m := &Model{}
	m.residentCommitIDs(prompt, generated, nil, nil)
	prompt[0], generated[0] = 99, 99
	if m.resIDs[0] == 99 || m.resIDs[3] == 99 {
		t.Errorf("resIDs aliases the caller's slice: %v", m.resIDs)
	}
}

// TestResidentReuseLen_imageBlockAtomicity is P9(a)'s kill condition at the unit level (the
// gpu-package real-hardware version is TestResidentPrefixReuse_tokenIdentical, extended
// separately): an image block is either reused WHOLE or not at all, never a partial position
// inside it — because every position inside the block attended every OTHER position under a
// bidirectional mask, so there is no such thing as "half the block's KV, causally consistent
// with the other half."
func TestResidentReuseLen_imageBlockAtomicity(t *testing.T) {
	// cached: [text 0,1] [image 2,3,4 hash=7] [text 5,6]
	cached := []int{100, 101, 900, 900, 900, 200, 201}
	block := residentImageBlock{start: 2, end: 5, hash: 7}

	for _, tc := range []struct {
		name   string
		prompt []int
		imgs   []residentImageClaim
		want   int
	}{
		{
			"same image (hash+len match), extension past it — jumps the WHOLE block atomically",
			[]int{100, 101, 900, 900, 900, 200, 201, 300},
			[]residentImageClaim{{Start: 2, Len: 3, Hash: 7}},
			7, // past the block AND the trailing matched text — one past cache's own [0,7)
		},
		{
			"same image, prompt ends exactly at the block's end — capped at len(prompt)-1, which lands INSIDE the block: must stop at the block's start, not partway through it",
			[]int{100, 101, 900, 900, 900},
			[]residentImageClaim{{Start: 2, Len: 3, Hash: 7}},
			2,
		},
		{
			"different image at the same position (hash mismatch) — must stop exactly at the block's start, never one position into it",
			[]int{100, 101, 900, 900, 900, 200, 201, 300},
			[]residentImageClaim{{Start: 2, Len: 3, Hash: 999}},
			2,
		},
		{
			"different image, shorter — must still stop exactly at the block's start",
			[]int{100, 101, 900, 900, 200, 201, 300},
			[]residentImageClaim{{Start: 2, Len: 2, Hash: 7}},
			2,
		},
		{
			"no claim at all where a block was recorded (plain text resent over an old image slot) — stop at the block's start",
			[]int{100, 101, 900, 900, 900, 200, 201, 300},
			nil,
			2,
		},
		{
			"text-only prefix ending BEFORE the block starts — ordinary LCP, block irrelevant",
			[]int{100, 101, 999},
			nil,
			2,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := &Model{resIDs: cached, resImgBlocks: []residentImageBlock{block}}
			if got := m.residentReuseLen(tc.prompt, tc.imgs, nil); got != tc.want {
				t.Errorf("residentReuseLen(%v, %v) = %d, want %d", tc.prompt, tc.imgs, got, tc.want)
			}
		})
	}
}

// TestResidentReuseLen_recurrentRefusesImageClaims: no recurrent-family VL architecture exists
// today, and the rewind-free recurrent rule has no notion of an image block's atomicity — a
// claim there must refuse (fall to 0) rather than silently mis-serve it.
func TestResidentReuseLen_recurrentRefusesImageClaims(t *testing.T) {
	m := recurrentTestModel([]int{1, 2, 3})
	claim := []residentImageClaim{{Start: 1, Len: 1, Hash: 7}}
	if got := m.residentReuseLen([]int{1, 2, 3, 4}, claim, nil); got != 0 {
		t.Errorf("recurrent family with an image claim: residentReuseLen = %d, want 0 (refuse)", got)
	}
}

// TestResidentCommitIDs_imageBlocksAppendOnlyValid: an older recorded block that ends before a
// newly committed one's start stays valid (positions are append-only); this is a structural
// sanity check on residentCommitIDs' bookkeeping, not a reuse-correctness test by itself.
func TestResidentCommitIDs_imageBlocksAppendOnlyValid(t *testing.T) {
	m := &Model{resImgBlocks: []residentImageBlock{{start: 2, end: 5, hash: 7}}}
	m.residentCommitIDs([]int{0, 0, 0, 0, 0, 0, 0}, nil, &residentImageBlock{start: 8, end: 11, hash: 42}, nil)
	if len(m.resImgBlocks) != 2 {
		t.Fatalf("resImgBlocks = %v, want 2 entries (old block kept, new block added)", m.resImgBlocks)
	}
	if m.resImgBlocks[0] != (residentImageBlock{start: 2, end: 5, hash: 7}) {
		t.Errorf("old block not preserved: %v", m.resImgBlocks[0])
	}
	if m.resImgBlocks[1] != (residentImageBlock{start: 8, end: 11, hash: 42}) {
		t.Errorf("new block not recorded: %v", m.resImgBlocks[1])
	}
}

// TestResidentForgetIDs_clearsImageBlocksToo: forgetting must be total (see
// TestResidentForgetIDs's own doc comment) — a stale image-block list is exactly as dangerous
// as a stale resIDs, for the same reason.
func TestResidentForgetIDs_clearsImageBlocksToo(t *testing.T) {
	m := &Model{resImgBlocks: []residentImageBlock{{start: 2, end: 5, hash: 7}}}
	m.residentCommitIDs([]int{1, 2}, []int{3, 4}, nil, nil)
	m.residentForgetIDs()
	if m.resImgBlocks != nil {
		t.Errorf("resImgBlocks = %v after forget, want nil", m.resImgBlocks)
	}
}

// TestResidentReuseLen_adapterMustMatch pins audit C-02 (2026-09-10): the resident KV is keyed on
// WHICH WEIGHTS built it, not only on token ids. Any targeted projection changes the residual
// stream, so every later layer's K/V differs under an adapter — an identical id prefix computed
// under different weights is someone else's context, the "confidently wrong, no error" failure
// resident_reuse.go names as the entire risk. The same-adapter row matters as much as the refusals:
// forgetting on every adapter change would also be correct, and would quietly throw away the
// agent-loop win for every fine-tune served off one base.
func TestResidentReuseLen_adapterMustMatch(t *testing.T) {
	a, b := &loraRuntime{name: "a"}, &loraRuntime{name: "b"}
	prompt := []int{1, 2, 3, 4, 5}
	for _, tc := range []struct {
		name          string
		built, asking *loraRuntime
		want          int
	}{
		{"base then base reuses", nil, nil, 3},
		{"adapter then SAME adapter reuses", a, a, 3},
		{"adapter then base must NOT reuse", a, nil, 0},
		{"base then adapter must NOT reuse", nil, a, 0},
		{"adapter then a DIFFERENT adapter must NOT reuse", a, b, 0},
		{"same NAME but a reloaded runtime must NOT reuse", a, &loraRuntime{name: "a"}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := &Model{}
			m.residentCommitIDs([]int{1, 2, 3}, nil, nil, tc.built)
			if got := m.residentReuseLen(prompt, nil, tc.asking); got != tc.want {
				t.Errorf("reuse = %d, want %d", got, tc.want)
			}
		})
	}
}

// TestResidentForgetIDs_clearsAdapterToo: a forgotten record must not leave a stale adapter
// identity behind for the next commit's absence to be compared against.
func TestResidentForgetIDs_clearsAdapterToo(t *testing.T) {
	m := &Model{}
	m.residentCommitIDs([]int{1, 2}, []int{3}, nil, &loraRuntime{name: "a"})
	m.residentForgetIDs()
	if m.resIDs != nil || m.resIDsLora != nil {
		t.Errorf("after forget: resIDs=%v resIDsLora=%v, want both nil", m.resIDs, m.resIDsLora)
	}
}
