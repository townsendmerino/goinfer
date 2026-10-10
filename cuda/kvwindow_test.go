//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"path/filepath"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// windowedKVTestSlack is the slack the windowed-KV tests run with: the fixture's window is 16 and its context 512, so the production slack
// (512) would never engage the feature; 24 makes a compaction every ~24 tokens.
const windowedKVTestSlack = 24

func withWindowedKVSlack(t *testing.T, slack int) {
	t.Helper()
	old := kvWindowSlack
	kvWindowSlack = slack
	t.Cleanup(func() { kvWindowSlack = old })
}

// windowFixtures: an all-sliding-window model, and two with global layers beside the windowed ones (gemma3's 1-in-2 and cohere2's 3-in-4), so a
// mixed stack proves each layer reads its own view.
var windowFixtures = []string{"mistral-tiny-window", "gemma3-vl-tiny", "cohere2-tiny"}

func eachWindowFixture(t *testing.T, f func(t *testing.T, fixture string)) {
	for _, fx := range windowFixtures {
		t.Run(fx, func(t *testing.T) { f(t, fx) })
	}
}

func loadWindowedPair(t *testing.T, fixture string, slots int) (full, win *cudaResident, mf, mw *decoder.Model) {
	t.Helper()
	requireCUDADevice(t)
	withWindowedKVSlack(t, windowedKVTestSlack)
	path := filepath.Join("..", "testdata", fixture)
	load := func(on bool) (*decoder.Model, *cudaResident) {
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentKVSlots: slots, ResidentWindowedKV: on})
		if err != nil {
			t.Fatalf("load (windowed=%v): %v", on, err)
		}
		t.Cleanup(func() { m.Close() })
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			t.Fatalf("not CUDA-resident (windowed=%v): %s", on, m.ResidentDecline())
		}
		return m, r
	}
	mf, full = load(false)
	mw, win = load(true)
	if full.kvWin {
		t.Fatal("the control resident is windowed: Options.ResidentWindowedKV leaked")
	}
	if !win.kvWin {
		t.Fatalf("windowed KV did not engage (ctxCap %d, window, slack %d)", win.ctxCap, kvWindowSlack)
	}
	return
}

func sameBits(a, b []float32) (int, bool) {
	if len(a) != len(b) {
		return -1, false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return i, false
		}
	}
	return 0, true
}

// TestWindowedKV_view is G-W0: a Buffer view at a NEGATIVE byte offset is what a windowed layer hands every kernel, so a kernel that addresses
// it by absolute position must read and write the physical buffer at position-base, and a compaction (copy of the live tail to physical slot 0)
// must be exact.
func TestWindowedKV_view(t *testing.T) {
	requireCUDADevice(t)
	m, err := decoder.Load(filepath.Join("..", "testdata", "llama-tiny"), decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("not CUDA-resident: %s", m.ResidentDecline())
	}
	const n, base, total = 8, 5, 64
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i + 1)
	}
	sentinel := make([]float32, total)
	for i := range sentinel {
		sentinel[i] = -7
	}
	read := func(b Buffer) []float32 {
		out := make([]float32, total)
		if err := r.do(func() error {
			if e := r.stream.Sync(); e != nil {
				return e
			}
			return gpu.Download(b, out)
		}); err != nil {
			t.Fatal(err)
		}
		return out
	}
	var s, d Buffer
	if err := r.do(func() error {
		s, d = r.af(n), r.af(total)
		if e := gpu.Upload(s, src); e != nil {
			return e
		}
		if e := gpu.Upload(d, sentinel); e != nil {
			return e
		}
		// kv_store writes dst[pos*n+i] = src[i]; a view shifted back by base positions makes pos=base land on physical slot 0.
		view := d.At(-base * n * 4)
		return r.launch(r.kvCopy, g1cfg(n, 256), Arg(s), Arg(view), gpu.ArgValue(int32(base)), gpu.ArgValue(int32(n)))
	}); err != nil {
		t.Fatal(err)
	}
	got := read(d)
	for i := range got {
		want := float32(-7)
		if i < n {
			want = src[i]
		}
		if got[i] != want {
			t.Fatalf("slot %d = %v, want %v: the shifted view wrote the wrong bytes (all: %v)", i, got[i], want, got)
		}
	}
	// Compaction shape: fill 0..total-1 with its own index, move [20, 20+n) to slot 0 through a scratch, and read it back.
	idx := make([]float32, total)
	for i := range idx {
		idx[i] = float32(i)
	}
	var scratch Buffer
	if err := r.do(func() error {
		scratch = r.af(n)
		if e := gpu.Upload(d, idx); e != nil {
			return e
		}
		if e := r.copyF32(d.At(20*4), scratch, n); e != nil {
			return e
		}
		return r.copyF32(scratch, d, n)
	}); err != nil {
		t.Fatal(err)
	}
	got = read(d)
	for i := 0; i < n; i++ {
		if got[i] != float32(20+i) {
			t.Fatalf("after compaction slot %d = %v, want %v", i, got[i], 20+i)
		}
	}
	for i := n; i < total; i++ {
		if got[i] != float32(i) {
			t.Fatalf("compaction touched slot %d: %v", i, got[i])
		}
	}
}

// TestWindowedKV_allocation checks the saving the feature exists for, and that the resident prices what it allocates (G-W4's cuda half; the
// decoder half is TestWindowedKV_planPrices).
func TestWindowedKV_allocation(t *testing.T) {
	full, win, _, _ := loadWindowedPair(t, "mistral-tiny-window", 1)
	for l := range win.layers {
		if !win.layers[l].kvWin {
			t.Fatalf("layer %d of an all-sliding-window model is not windowed", l)
		}
		wantPos := win.kvWindow + kvWindowSlack
		if got := win.kc[l].Len() / win.layers[l].kvDim; got != wantPos {
			t.Fatalf("layer %d K holds %d positions, want window+slack = %d", l, got, wantPos)
		}
		if full.kc[l].Len() <= win.kc[l].Len() {
			t.Fatalf("layer %d: the windowed cache (%d floats) is not smaller than the full one (%d)", l, win.kc[l].Len(), full.kc[l].Len())
		}
	}
	var held int64
	for l := range win.layers {
		held += int64(win.kc[l].Len()+win.vc[l].Len()) * 4
	}
	if got := kvBytesForCap(win.ctxCap, win.layers); got != held {
		t.Fatalf("kvBytesForCap = %d, but the allocation is %d bytes", got, held)
	}
	if win.graphs {
		t.Fatal("CUDA graphs are admitted under windowed KV: a captured graph would replay a stale shifted pointer")
	}
}

// decodeSteps feeds the same teacher-forced token stream to both residents from position start for n tokens and fails on the first logit bit
// that differs. Returns the compactions the windowed one went through (its final base).
func decodeSteps(t *testing.T, full, win *cudaResident, m *decoder.Model, start, n int) {
	t.Helper()
	_, _, _, _, _, _, vocab := m.Dims()
	for i := 0; i < n; i++ {
		pos := start + i
		emb := m.EmbedResidentForTest((pos*29 + 5) % vocab)
		a, err := full.Forward(emb, pos)
		if err != nil {
			t.Fatalf("full forward at %d: %v", pos, err)
		}
		b, err := win.Forward(emb, pos)
		if err != nil {
			t.Fatalf("windowed forward at %d: %v", pos, err)
		}
		if j, ok := sameBits(a, b); !ok {
			t.Fatalf("position %d: windowed logits differ from full-KV logits at index %d (%v vs %v), base %d", pos, j, b[max(j, 0)], a[max(j, 0)], win.kvBase)
		}
	}
}

// TestWindowedKV_decodeIdentity is G-W1 for decode: 400 teacher-forced single-token steps, 25x past the window and through ~16
// compactions, byte-identical logits against the full-KV resident at every step.
func TestWindowedKV_decodeIdentity(t *testing.T) { eachWindowFixture(t, windowedDecodeIdentity) }

func windowedDecodeIdentity(t *testing.T, fx string) {
	full, win, _, mw := loadWindowedPair(t, fx, 1)
	decodeSteps(t, full, win, mw, 0, 400)
	if win.kvBase == 0 {
		t.Fatal("no compaction ran in 400 steps: the test did not exercise the path it names")
	}
	t.Logf("final base %d after 400 steps (window %d, slack %d)", win.kvBase, win.kvWindow, kvWindowSlack)
}

// TestWindowedKV_prefillIdentity is G-W1 for the batched prefill: a prompt crossing the window in chunks (the chunk is capped to the slack+1, so
// the pass count is the compaction count), then decode from it, byte-identical to the full-KV resident, including an odd prompt length.
func TestWindowedKV_prefillIdentity(t *testing.T) { eachWindowFixture(t, windowedPrefillIdentity) }

func windowedPrefillIdentity(t *testing.T, fx string) {
	for _, plen := range []int{20, 41, 97, 250} {
		full, win, _, mw := loadWindowedPair(t, fx, 1)
		_, _, _, _, _, _, vocab := mw.Dims()
		prompt := make([][]float32, plen)
		for i := range prompt {
			prompt[i] = mw.EmbedResidentForTest((i*37 + 3) % vocab)
		}
		a, err := full.PrefillLast(context.Background(), prompt, 0)
		if err != nil {
			t.Fatalf("plen %d full prefill: %v", plen, err)
		}
		b, err := win.PrefillLast(context.Background(), prompt, 0)
		if err != nil {
			t.Fatalf("plen %d windowed prefill: %v", plen, err)
		}
		if j, ok := sameBits(a, b); !ok {
			t.Fatalf("plen %d: prefill logits differ at %d (base %d)", plen, j, win.kvBase)
		}
		decodeSteps(t, full, win, mw, plen, 60)
		t.Logf("plen %d: identical through 60 decode steps, final base %d", plen, win.kvBase)
	}
}

// TestWindowedKV_verifyIdentity is G-W1 for speculative verify: ForwardN at a run of widths, rolled back by a partial accept, byte-identical to the
// full-KV resident (a width past the slack declines to the sequential loop instead of failing).
func TestWindowedKV_verifyIdentity(t *testing.T) { eachWindowFixture(t, windowedVerifyIdentity) }

func windowedVerifyIdentity(t *testing.T, fx string) {
	full, win, _, mw := loadWindowedPair(t, fx, 1)
	_, _, _, _, _, _, vocab := mw.Dims()
	pos := 0
	for step := 0; pos < 380; step++ {
		k := 1 + (step*5)%9 // 1..9 rows
		rows := make([][]float32, k)
		for i := range rows {
			rows[i] = mw.EmbedResidentForTest(((pos+i)*31 + 7) % vocab)
		}
		a, err := full.ForwardN(rows, pos)
		if err != nil {
			t.Fatalf("full ForwardN at %d: %v", pos, err)
		}
		b, err := win.ForwardN(rows, pos)
		if err != nil {
			t.Fatalf("windowed ForwardN at %d: %v", pos, err)
		}
		for i := range a {
			if j, ok := sameBits(a[i], b[i]); !ok {
				t.Fatalf("ForwardN at %d row %d: logits differ at %d (base %d)", pos, i, j, win.kvBase)
			}
		}
		pos += max(1, k-step%3) // accept all but 0-2 rows: the rejected rows' KV is overwritten by the next pass
	}
	// a pass wider than the slack cannot fit: it must decline to the sequential loop and still match
	wide := make([][]float32, kvWindowSlack+10)
	for i := range wide {
		wide[i] = mw.EmbedResidentForTest((i*13 + 1) % vocab)
	}
	if win.kvRoomFor(pos, len(wide)) {
		t.Fatalf("a %d-row pass fits the windowed KV with slack %d", len(wide), kvWindowSlack)
	}
	// The reference is the full-KV resident's own SEQUENTIAL decode of the same rows: a pass wider than the slack runs as the sequential loop,
	// and a family whose batched pass is not bit-identical to decode (cohere2) would differ from a batched reference for reasons that are not
	// the window's.
	a := make([][]float32, len(wide))
	for i := range wide {
		l, err := full.Forward(wide[i], pos+i)
		if err != nil {
			t.Fatal(err)
		}
		a[i] = append([]float32(nil), l...) // Forward returns the resident's reused host buffer

	}
	b, err := win.ForwardN(wide, pos)
	if err != nil {
		t.Fatalf("windowed wide ForwardN: %v", err)
	}
	for i := range a {
		if j, ok := sameBits(a[i], b[i]); !ok {
			t.Fatalf("wide ForwardN row %d differs at %d", i, j)
		}
	}
}

// TestWindowedKV_reuseFloor pins the prefix-reuse contract: a position whose window has been compacted away is an ERROR, never a stale read, and
// ReusableKV tells the caller what it may reuse.
func TestWindowedKV_reuseFloor(t *testing.T) { eachWindowFixture(t, windowedReuseFloor) }

func windowedReuseFloor(t *testing.T, fx string) {
	_, win, _, mw := loadWindowedPair(t, fx, 1)
	_, _, _, _, _, _, vocab := mw.Dims()
	emb := func(i int) []float32 { return mw.EmbedResidentForTest((i*29 + 5) % vocab) }
	for pos := 0; pos < 200; pos++ {
		if _, err := win.Forward(emb(pos), pos); err != nil {
			t.Fatal(err)
		}
	}
	base := win.kvBase
	if base == 0 {
		t.Fatal("no compaction ran: nothing to test")
	}
	if got := win.ReusableKV(200); got != 200 {
		t.Fatalf("ReusableKV(200) = %d, want 200 (the window of position 200 is held)", got)
	}
	if got := win.ReusableKV(base + win.kvWindow - 2); got != 0 {
		t.Fatalf("ReusableKV(%d) = %d, want 0: its window starts below the held base %d", base+win.kvWindow-2, got, base)
	}
	if _, err := win.Forward(emb(base), base); err == nil {
		t.Fatalf("a forward at position %d (history below base %d) succeeded: a stale read would have been silent", base, base)
	}
	// a rollback inside the slack is fine, and a fresh sequence restarts the window
	if back := base + win.kvWindow - 1; true {
		if _, err := win.Forward(emb(back), back); err != nil {
			t.Fatalf("rollback to %d (the first position whose window is still held) failed: %v", back, err)
		}
	}
	if _, err := win.Forward(emb(0), 0); err != nil {
		t.Fatalf("restart at 0 failed: %v", err)
	}
	if win.kvBase != 0 {
		t.Fatalf("base %d after restarting at position 0", win.kvBase)
	}
}

// TestWindowedKV_slotsIdentity is G-W1 for MC1: three KV slots, each its own base, interleaved at different rates (a switch is a pointer swap, and the
// base must swap with it), byte-identical to the full-KV resident under the same schedule.
func TestWindowedKV_slotsIdentity(t *testing.T) { eachWindowFixture(t, windowedSlotsIdentity) }

func windowedSlotsIdentity(t *testing.T, fx string) {
	full, win, _, mw := loadWindowedPair(t, fx, 3)
	if win.KVSlots() != 3 || full.KVSlots() != 3 {
		t.Fatalf("slots granted: windowed %d, full %d, want 3", win.KVSlots(), full.KVSlots())
	}
	_, _, _, _, _, _, vocab := mw.Dims()
	pace := []int{7, 3, 11}
	pos := make([]int, 3)
	for round := 0; pos[0] < 300; round++ {
		for s := range 3 {
			if err := full.UseKVSlot(s); err != nil {
				t.Fatal(err)
			}
			if err := win.UseKVSlot(s); err != nil {
				t.Fatal(err)
			}
			// binding alone (no forward yet) must expose the slot's own base: ReusableKV answers from it
			if win.kvBase != win.kvBases[s] {
				t.Fatalf("round %d: bound slot %d but kvBase %d != its base %d", round, s, win.kvBase, win.kvBases[s])
			}
			for range pace[s] {
				emb := mw.EmbedResidentForTest((pos[s]*29 + s*13 + 5) % vocab)
				a, err := full.Forward(emb, pos[s])
				if err != nil {
					t.Fatal(err)
				}
				b, err := win.Forward(emb, pos[s])
				if err != nil {
					t.Fatalf("slot %d pos %d: %v", s, pos[s], err)
				}
				if j, ok := sameBits(a, b); !ok {
					t.Fatalf("round %d slot %d pos %d: logits differ at %d (bases %v)", round, s, pos[s], j, win.kvBases)
				}
				pos[s]++
			}
		}
	}
	if win.kvBases[0] == 0 || win.kvBases[1] == 0 || win.kvBases[0] == win.kvBases[1] {
		t.Fatalf("slot bases %v: the slots did not each compact on their own schedule", win.kvBases)
	}
}

// TestWindowedKV_stepIdentity is G-W1 for MC3: StepBatch rows, each on its own slot at its own depth with its own base, byte-identical to the
// full-KV resident's step.
func TestWindowedKV_stepIdentity(t *testing.T) { eachWindowFixture(t, windowedStepIdentity) }

func windowedStepIdentity(t *testing.T, fx string) {
	full, win, _, mw := loadWindowedPair(t, fx, 4)
	if lo, hi := win.BatchStepRange(); lo != 2 || hi != 4 {
		t.Skipf("BatchStepRange = (%d, %d): the step does not engage on %s", lo, hi, fx)
	}
	_, _, _, _, _, _, vocab := mw.Dims()
	depths := []int{9, 70, 130, 33}
	pos := make([]int, 4)
	for b := range 4 {
		prompt := make([][]float32, depths[b])
		for i := range prompt {
			prompt[i] = mw.EmbedResidentForTest((i*37 + b*11 + 3) % vocab)
		}
		for _, r := range []*cudaResident{full, win} {
			if err := r.UseKVSlot(b); err != nil {
				t.Fatal(err)
			}
			if _, err := r.PrefillLast(context.Background(), prompt, 0); err != nil {
				t.Fatalf("prefill slot %d: %v", b, err)
			}
		}
		pos[b] = depths[b]
	}
	for step := range 120 {
		B := 2 + step%3 // 2..4 rows
		seqs := make([]decoder.ResidentBatchSeq, B)
		for b := range B {
			seqs[b] = decoder.ResidentBatchSeq{Slot: b, Pos: pos[b], Emb: mw.EmbedResidentForTest((step*17 + b*5 + 1) % vocab)}
		}
		a, err := full.StepBatch(seqs)
		if err != nil {
			t.Fatalf("full step %d: %v", step, err)
		}
		bOut, err := win.StepBatch(seqs)
		if err != nil {
			t.Fatalf("windowed step %d: %v", step, err)
		}
		for b := range B {
			if j, ok := sameBits(a[b].Logits, bOut[b].Logits); !ok {
				t.Fatalf("step %d row %d (slot %d pos %d): logits differ at %d (bases %v)", step, b, b, pos[b], j, win.kvBases)
			}
			pos[b]++
		}
	}
	if win.kvBases[0] == 0 {
		t.Fatalf("bases %v: no row compacted", win.kvBases)
	}
}

// TestWindowedKV_uploadKV is G-W1 for the CPU-prefill bridge: the full resident's K and V for the first n positions are uploaded (every layer, as
// the bridge does) into the windowed one, which keeps the last window-1 positions of its windowed layers; decode from there is byte-identical.
func TestWindowedKV_uploadKV(t *testing.T) { eachWindowFixture(t, windowedUploadKV) }

func windowedUploadKV(t *testing.T, fx string) {
	full, win, _, mw := loadWindowedPair(t, fx, 1)
	const n = 100
	_, _, _, _, _, _, vocab := mw.Dims()
	for pos := range n { // only the full resident decodes: the windowed one is fresh and gets its history from the upload alone
		if _, err := full.Forward(mw.EmbedResidentForTest((pos*29+5)%vocab), pos); err != nil {
			t.Fatal(err)
		}
	}
	for l := range full.layers {
		Ly := &full.layers[l]
		if Ly.kvShared || Ly.isDeltaNet || Ly.isMLA {
			continue
		}
		keys, vals := make([]float32, n*Ly.kvDim), make([]float32, n*Ly.kvDim)
		if err := full.do(func() error {
			if e := full.stream.Sync(); e != nil {
				return e
			}
			if e := gpu.Download(full.kc[l], keys); e != nil {
				return e
			}
			return gpu.Download(full.vc[l], vals)
		}); err != nil {
			t.Fatal(err)
		}
		if err := win.UploadKV(l, 0, keys, vals); err != nil {
			t.Fatalf("UploadKV layer %d: %v", l, err)
		}
	}
	if win.kvBase != n-(win.kvWindow-1) {
		t.Fatalf("base after the upload = %d, want %d (the last window-1 of %d positions)", win.kvBase, n-(win.kvWindow-1), n)
	}
	decodeSteps(t, full, win, mw, n, 150)
}

// TestWindowedKV_agentTurns drives the real entry point: Model.Generate over a conversation of continuing turns (each prompt is the last one, its
// reply and a short suffix, so the decoder reuses the committed prefix), then an EDITED turn that shares only a short lead with the history. The
// windowed load must emit the same ids as the full-KV one on every turn, reuse the whole committed prefix on a continuation, and fall back to a cold
// prefill (PrefillReused 0) on the edit, whose short lead's window the slot no longer holds.
func TestWindowedKV_agentTurns(t *testing.T) { eachWindowFixture(t, windowedAgentTurns) }

func windowedAgentTurns(t *testing.T, fx string) {
	_, _, _, mw := loadWindowedPair(t, fx, 1)
	_, _, _, _, _, _, vocab := mw.Dims()
	path := filepath.Join("..", "testdata", fx)
	type turn struct {
		ids    []int
		reused int
	}
	play := func(on bool) []turn {
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4", ResidentWindowedKV: on})
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		prompt := make([]int, 40)
		for i := range prompt {
			prompt[i] = (i*31 + 7) % vocab
		}
		var out []turn
		gen := func(p []int) []int {
			ch, g := m.Generate(context.Background(), p, 30, decoder.SamplingParams{})
			var ids []int
			for id := range ch {
				ids = append(ids, id)
			}
			if err := g.Err(); err != nil {
				t.Fatalf("windowed=%v: %v", on, err)
			}
			out = append(out, turn{ids, g.PrefillReused})
			return ids
		}
		for tn := range 8 {
			ids := gen(prompt)
			prompt = append(append(append([]int(nil), prompt...), ids...), (tn*7+3)%vocab, (tn*11+5)%vocab)
		}
		edited := append(append([]int(nil), prompt[:20]...), (3*vocab/4)%vocab, 9, 8, 7)
		gen(edited)
		return out
	}
	want, got := play(false), play(true)
	for i := range want {
		if len(want[i].ids) != len(got[i].ids) {
			t.Fatalf("turn %d: %d ids windowed, %d full", i, len(got[i].ids), len(want[i].ids))
		}
		for j := range want[i].ids {
			if want[i].ids[j] != got[i].ids[j] {
				t.Fatalf("turn %d: id %d = %d windowed, %d full", i, j, got[i].ids[j], want[i].ids[j])
			}
		}
		t.Logf("turn %d: reused %d windowed, %d full", i, got[i].reused, want[i].reused)
	}
	last := len(want) - 1
	if want[last].reused == 0 {
		t.Fatalf("the full-KV control reused nothing on the edited turn: the test does not exercise the floor")
	}
	if got[last].reused != 0 {
		t.Fatalf("the windowed load reused %d tokens of an edited turn whose window it no longer holds", got[last].reused)
	}
	for i := 1; i < last; i++ {
		if got[i].reused != want[i].reused {
			t.Errorf("turn %d: a continuation reused %d windowed but %d full", i, got[i].reused, want[i].reused)
		}
	}
}
