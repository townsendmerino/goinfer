package decoder

import "testing"

// SP2's decoder-side tier-0 gates (docs/tasks/task-forward-architecture-2026-10.md §6a, "SP2 on nobara"): admission, the
// arena's price, and the loop's fold accounting.

type recurrentSpecStub struct {
	rows    int
	prepErr error
}

func (s recurrentSpecStub) PrepareRecurrentSpec(int) error                  { return s.prepErr }
func (s recurrentSpecStub) RecurrentSpecRows() int                          { return s.rows }
func (recurrentSpecStub) VerifyLastNArgmax([][]float32, int) ([]int, error) { panic("not reached") }
func (recurrentSpecStub) FoldVerified(int) error                            { panic("not reached") }
func (recurrentSpecStub) FlushFold() error                                  { panic("not reached") }

func deltaNetArch(layers int) *Architecture {
	return &Architecture{NumLayers: layers, HiddenDim: 64,
		qwen35:        &qwen35Params{ConvKernel: 4, KeyHeadDim: 128, ValueHeadDim: 128, NumKeyHeads: 16, NumValueHeads: 32},
		layerIsLinear: func(i int) bool { return i%4 != 3 }}
}

// TestSpecRollbackSafeVia_admitsOnlyTheFoldPath: a recurrent model is admitted only by a loop on the fold path, within the rows
// the resident reports, and only when every recurrent layer is Gated DeltaNet; the truncate-path predicate still refuses it.
func TestSpecRollbackSafeVia_admitsOnlyTheFoldPath(t *testing.T) {
	mk := func(a *Architecture) *Model { return &Model{w: &Weights{arch: a}} }
	dn := deltaNetArch(8)
	mamba := &Architecture{NumLayers: 1, granite: &graniteParams{}, layerIsMamba: func(int) bool { return true }}
	windowedDN := deltaNetArch(8)
	windowedDN.SlidingWindow = 512
	for _, c := range []struct {
		name  string
		arch  *Architecture
		fold  ResidentRecurrentSpec
		width int
		want  bool
	}{
		{"DeltaNet, truncate path", dn, nil, 0, false},
		{"DeltaNet, fold path, width 4 of 16", dn, recurrentSpecStub{rows: 16}, 4, true},
		{"DeltaNet, fold path, width 16 of 16", dn, recurrentSpecStub{rows: 16}, 16, true},
		{"DeltaNet, fold path, width 17 of 16", dn, recurrentSpecStub{rows: 16}, 17, false},
		{"DeltaNet, a resident that kept no rows", dn, recurrentSpecStub{rows: 0}, 4, false},
		{"Mamba-2, a resident claiming rows", mamba, recurrentSpecStub{rows: 16}, 4, false},
		{"DeltaNet with a sliding window", windowedDN, recurrentSpecStub{rows: 16}, 4, false},
		{"dense, fold path given", &Architecture{}, recurrentSpecStub{rows: 16}, 4, true},
	} {
		if got := mk(c.arch).specRollbackSafeVia(c.fold, c.width); got != c.want {
			t.Errorf("%s: specRollbackSafeVia = %v, want %v", c.name, got, c.want)
		}
	}
	if mk(dn).specRollbackSafe() {
		t.Error("specRollbackSafe() admits a DeltaNet model: the truncate-path loops must keep refusing it")
	}
}

// TestRecurrentSpecArenaBytes_formula pins G3's price on Qwen3.5-9B's geometry: 24 DeltaNet layers of 32, 16 key and 32 value heads
// of 128, 16 rows in two banks, rounded up to 2 MiB.
func TestRecurrentSpecArenaBytes_formula(t *testing.T) {
	a := deltaNetArch(32)
	// per row and layer: conv in and out 8,192 each, q and k 2,048 each, two gates per value head; plus the fold's output once.
	// Two banks: a verify writes one while the previous verify's committed rows fold from the other.
	floats := int64(2*24*16*(2*8192+2*2048+2*32) + 16*4096)
	want := (floats*4 + (2<<20 - 1)) / (2 << 20) * (2 << 20)
	if got := RecurrentSpecArenaBytes(a, 16); got != want || want != 62<<20 {
		t.Errorf("RecurrentSpecArenaBytes = %d, want %d (62 MiB)", got, want)
	}
	if got := recurrentSpecReserve(a, false); got != 0 {
		t.Errorf("no drafter: reserve %d, want 0", got)
	}
	if got := recurrentSpecReserve(a, true); got != want {
		t.Errorf("drafter: reserve %d, want %d", got, want)
	}
	if got := RecurrentSpecArenaBytes(&Architecture{NumLayers: 1, granite: &graniteParams{}, layerIsMamba: func(int) bool { return true }}, 16); got != 0 {
		t.Errorf("Mamba-2: %d, want 0", got)
	}
	if got := RecurrentSpecArenaBytes(&Architecture{NumLayers: 4}, 16); got != 0 {
		t.Errorf("dense: %d, want 0", got)
	}
	wide := deltaNetArch(32)
	wide.qwen35.KeyHeadDim = 256
	if got := RecurrentSpecArenaBytes(wide, 16); got != 0 {
		t.Errorf("key heads of 256, past the scan twins: %d, want 0", got)
	}
	windowed := deltaNetArch(32)
	windowed.SlidingWindow = 4096
	if got := RecurrentSpecArenaBytes(windowed, 16); got != 0 || recurrentSpecReserve(windowed, true) != 0 {
		t.Errorf("a sliding window, which the path refuses: priced %d, want 0", got)
	}
}

// prepHost is a drafter host with a fold capability whose attach must not be reached: prepareRecurrentSpec runs before it. It
// counts the preparations asked of it.
type prepHost struct {
	recurrentSpecStub
	*blockspecWriteHost
	prepared *int
}

func (h prepHost) PrepareRecurrentSpec(rows int) error {
	*h.prepared++
	return h.recurrentSpecStub.PrepareRecurrentSpec(rows)
}

// TestPrepareRecurrentSpec_refusesBeforeAttach: NewBlockSpec readies the fold path before the drafter's weights go to the
// device, so a recurrent model the path refuses has nothing uploaded. The refusals: no fold capability, the resident declining
// to prepare, no rows kept, and an architecture the path does not serve (Mamba-2, a sliding window, heads past 128), which is
// refused before the resident is asked to allocate anything.
func TestPrepareRecurrentSpec_refusesBeforeAttach(t *testing.T) {
	mk := func(a *Architecture) *Model { return &Model{w: &Weights{arch: a}} }
	dw := blockspecStubWeights{blockSize: 16}
	dn := deltaNetArch(8)
	mamba := &Architecture{NumLayers: 1, granite: &graniteParams{}, layerIsMamba: func(int) bool { return true }}
	windowed := deltaNetArch(8)
	windowed.SlidingWindow = 4096
	wide := deltaNetArch(8)
	wide.qwen35.ValueHeadDim = 256
	for _, c := range []struct {
		name      string
		arch      *Architecture
		rows      int
		prepErr   error
		noFold    bool
		ok        bool
		wantPreps int
	}{
		{"no fold capability", dn, 0, nil, true, false, 0},
		{"the resident declines to prepare", dn, 16, errBlockSpecUnsupported, false, false, 1},
		{"no rows kept", dn, 0, nil, false, false, 1},
		{"Mamba-2", mamba, 16, nil, false, false, 0},
		{"DeltaNet with a sliding window", windowed, 16, nil, false, false, 0},
		{"DeltaNet with value heads of 256", wide, 16, nil, false, false, 0},
		{"DeltaNet with a prepared path", dn, 16, nil, false, true, 1},
		{"dense: nothing to prepare", &Architecture{}, 16, nil, false, true, 0},
		{"a prepared path narrower than the drafter's block", dn, 8, nil, false, false, 1},
	} {
		preps := 0
		var host ResidentDrafterHost = prepHost{recurrentSpecStub{rows: c.rows, prepErr: c.prepErr}, &blockspecWriteHost{}, &preps}
		if c.noFold {
			host = &blockspecWriteHost{}
		}
		if err := mk(c.arch).prepareRecurrentSpec(host, dw); (err == nil) != c.ok {
			t.Errorf("%s: err = %v, want ok %v", c.name, err, c.ok)
		}
		if preps != c.wantPreps {
			t.Errorf("%s: the resident was asked to prepare %d times, want %d", c.name, preps, c.wantPreps)
		}
	}
	// A block of one position verifies no draft: refused before the resident is asked for anything.
	preps := 0
	one := prepHost{recurrentSpecStub{rows: 16}, &blockspecWriteHost{}, &preps}
	if err := mk(dn).prepareRecurrentSpec(one, blockspecStubWeights{blockSize: 1}); err == nil || preps != 0 {
		t.Errorf("a one-position block: err %v after %d preparations, want a refusal before any", err, preps)
	}
}

// foldHost is a drafter host with the fold path: the seed answers 7, every verify answers script, and it records each verify's
// rows and each fold.
type foldHost struct {
	script  []int
	folds   []int
	verify  []int
	flushes int // FlushFold calls that found rows marked
	marked  bool
}

func (h *foldHost) AttachBlockDrafter(BlockDrafterWeights) (ResidentBlockDrafter, error) {
	panic("not reached")
}
func (h *foldHost) PrefillLastNArgmax(embs [][]float32, startPos int) ([]int, error) {
	ids := make([]int, len(embs))
	ids[len(ids)-1] = 7
	return ids, nil
}
func (h *foldHost) SetBatchedCapture([]int) error  { return nil }
func (h *foldHost) BatchedCapture() [][]float32    { return nil }
func (h *foldHost) RecurrentSpecRows() int         { return 16 }
func (h *foldHost) PrepareRecurrentSpec(int) error { return nil }
func (h *foldHost) VerifyLastNArgmax(embs [][]float32, startPos int) ([]int, error) {
	h.verify = append(h.verify, len(embs))
	h.marked = false // a verify folds what was marked
	return h.script[:len(embs)], nil
}
func (h *foldHost) FoldVerified(n int) error {
	h.folds = append(h.folds, n)
	h.marked = true
	return nil
}
func (h *foldHost) FlushFold() error {
	if h.marked {
		h.flushes++
		h.marked = false
	}
	return nil
}

// scriptDrafter drafts the same ids every round.
type scriptDrafter struct{ ids []int }

func (scriptDrafter) FuseContext(rows [][]float32) ([][]float32, error) { return rows, nil }
func (scriptDrafter) ExtendContext([][]float32) error                   { return nil }
func (scriptDrafter) ContextLen() int                                   { return 0 }
func (scriptDrafter) TruncateContext(int)                               {}
func (scriptDrafter) DraftBlock(in [][]float32) ([][]float32, error)    { return in, nil }
func (d scriptDrafter) DraftTokens(trunk [][]float32) ([]int, error)    { return d.ids[:len(trunk)], nil }

// TestBlockSpecGenerate_foldsTheEmittedRows: on a recurrent model the loop verifies through the fold path and folds exactly the
// rows it emits. A stop token inside a fully accepted burst emits fewer rows than were accepted; the fold, the position and the
// commit must all end at the last emitted token, or the recurrent state runs past the ids recorded for the next turn.
func TestBlockSpecGenerate_foldsTheEmittedRows(t *testing.T) {
	m, _ := loadWithFakeResident(t)
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible")
	}
	// Make the fixture's first layer a Gated-DeltaNet one so the loop takes the fold path; nothing below runs a forward.
	a := *m.w.arch
	a.qwen35 = &qwen35Params{ConvKernel: 4, KeyHeadDim: 16, ValueHeadDim: 16, NumKeyHeads: 2, NumValueHeads: 4}
	a.layerIsLinear = func(i int) bool { return i == 0 }
	m.w.arch = &a
	if !m.hasRecurrentState() {
		t.Fatal("setup: the model does not report recurrent state")
	}
	prompt := []int{1, 2, 3}

	t.Run("partial accept", func(t *testing.T) {
		// Drafts 5 6 4 against a target answering 5 6 9 8: two accepted, the target's 9 next. Rounds repeat until max tokens.
		host := &foldHost{script: []int{5, 6, 9, 8}}
		s := &BlockSpec{m: m, host: host, rd: scriptDrafter{ids: []int{5, 6, 4}}, dw: blockspecCommitStubWeights{blockspecStubWeights{blockSize: 8}}}
		out, _, err := s.generate(prompt, BlockSpecOptions{VerifyWidth: 4, MaxTokens: 7}, nil)
		if err != nil {
			t.Fatalf("generate: %v", err)
		}
		if len(host.folds) == 0 || host.folds[0] != 3 {
			t.Errorf("folds %v, want the first to be 3 rows (the anchor and two accepted drafts)", host.folds)
		}
		if want := append(append([]int{}, prompt...), out...); !equalIntSlices(m.resIDs, want) {
			t.Errorf("resIDs = %v, want prompt+out %v", m.resIDs, want)
		}
	})
	t.Run("stop inside the burst", func(t *testing.T) {
		// Drafts 5 99 6, all accepted (the target answers 5 99 6 8), and 99 is a stop: the round emits only 5.
		host := &foldHost{script: []int{5, 99, 6, 8}}
		s := &BlockSpec{m: m, host: host, rd: scriptDrafter{ids: []int{5, 99, 6}}, dw: blockspecCommitStubWeights{blockspecStubWeights{blockSize: 8}}}
		out, _, err := s.generate(prompt, BlockSpecOptions{VerifyWidth: 4, MaxTokens: 32, StopIDs: []int{99}}, nil)
		if err != nil {
			t.Fatalf("generate: %v", err)
		}
		if !equalIntSlices(out, []int{7, 5}) {
			t.Fatalf("out = %v, want [7 5]: the seed's anchor, then the round's tokens before the stop", out)
		}
		if !equalIntSlices(host.folds, []int{2}) {
			t.Errorf("folds %v, want [2]: the anchor and the one emitted draft, not the three accepted", host.folds)
		}
		if want := []int{1, 2, 3, 7, 5}; !equalIntSlices(m.resIDs, want) {
			t.Errorf("resIDs = %v, want %v", m.resIDs, want)
		}
		if host.flushes != 1 {
			t.Errorf("%d flushes of marked rows, want 1: the marked rows must fold before the commit reads the state", host.flushes)
		}
	})
}

// captureHost is a dense drafter host whose batched capture is one tap holding each row's absolute position, so a test can read
// which rows reached the drafter and in what order. Every row answers 7.
type captureHost struct {
	hidden, rows, start int
}

func (h *captureHost) AttachBlockDrafter(BlockDrafterWeights) (ResidentBlockDrafter, error) {
	panic("not reached")
}
func (h *captureHost) PrefillLastNArgmax(embs [][]float32, startPos int) ([]int, error) {
	h.rows, h.start = len(embs), startPos
	ids := make([]int, len(embs))
	for i := range ids {
		ids[i] = 7
	}
	return ids, nil
}
func (h *captureHost) SetBatchedCapture([]int) error { return nil }
func (h *captureHost) BatchedCapture() [][]float32 {
	tap := make([]float32, h.rows*h.hidden)
	for i := range h.rows {
		for j := range h.hidden {
			tap[i*h.hidden+j] = float32(h.start + i)
		}
	}
	return [][]float32{tap}
}

// fuseRecorder is a drafter that records each FuseContext call's rows and the positions ExtendContext received, in order.
type fuseRecorder struct {
	calls []int
	ctx   []int
}

func (d *fuseRecorder) FuseContext(rows [][]float32) ([][]float32, error) {
	d.calls = append(d.calls, len(rows))
	return rows, nil
}
func (d *fuseRecorder) ExtendContext(f [][]float32) error {
	for _, r := range f {
		d.ctx = append(d.ctx, int(r[0]))
	}
	return nil
}
func (d *fuseRecorder) ContextLen() int                                { return len(d.ctx) }
func (d *fuseRecorder) TruncateContext(n int)                          { d.ctx = d.ctx[:min(n, len(d.ctx))] }
func (d *fuseRecorder) DraftBlock(in [][]float32) ([][]float32, error) { return in, nil }
func (d *fuseRecorder) DraftTokens(t [][]float32) ([]int, error)       { return make([]int, len(t)), nil }

// TestBlockSpecGenerate_fusesALongSeedInPieces: a seed of 1,100 rows reaches the drafter in pieces of blockSpecFuseRows (512, 512,
// 76), and the drafter's context holds every position once, in order, then the trailing token's.
func TestBlockSpecGenerate_fusesALongSeedInPieces(t *testing.T) {
	m, _ := loadWithFakeResident(t)
	if !m.ResidentActive() {
		t.Skip("fixture is not resident-eligible")
	}
	prompt := make([]int, 1100)
	for i := range prompt {
		prompt[i] = 1 + i%50
	}
	host := &captureHost{hidden: m.w.arch.HiddenDim}
	d := &fuseRecorder{}
	s := &BlockSpec{m: m, host: host, rd: d, dw: blockspecCommitStubWeights{blockspecStubWeights{blockSize: 8}}}
	if _, _, err := s.generate(prompt, BlockSpecOptions{VerifyWidth: 4, MaxTokens: 1}, nil); err != nil {
		t.Fatalf("generate: %v", err)
	}
	if want := []int{512, 512, 76, 1}; !equalIntSlices(d.calls, want) {
		t.Errorf("FuseContext calls of %v rows, want %v (the seed in pieces, then the trailing token)", d.calls, want)
	}
	if len(d.ctx) != len(prompt)+1 {
		t.Fatalf("the drafter's context holds %d positions, want %d", len(d.ctx), len(prompt)+1)
	}
	for i, p := range d.ctx {
		if p != i {
			t.Fatalf("context position %d holds row %d: the pieces were joined out of order", i, p)
		}
	}
}
