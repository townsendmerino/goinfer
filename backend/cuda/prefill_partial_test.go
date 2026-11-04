//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"errors"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

func withPrefillDeclinePass(t *testing.T, pass int) {
	t.Helper()
	old := prefillDeclinePassForTest
	prefillDeclinePassForTest = pass
	t.Cleanup(func() { prefillDeclinePassForTest = old })
}

// TestPrefillPartial_windowedResumesAfterTheCommittedPasses: a chunked prefill whose third pass declines returns a decoder.PrefillPartial at
// the end of the two passes it committed, by which point the windowed slot's base has moved past the call's start (a restart there is
// refused). Continuing from Next gives the full-KV resident's logits under the same schedule, bit for bit.
func TestPrefillPartial_windowedResumesAfterTheCommittedPasses(t *testing.T) {
	eachWindowFixture(t, windowedPrefillPartial)
}

func windowedPrefillPartial(t *testing.T, fx string) {
	full, win, _, mw := loadWindowedPair(t, fx, 1)
	_, _, _, _, _, _, vocab := mw.Dims()
	emb := func(lo, hi int) [][]float32 {
		rows := make([][]float32, 0, hi-lo)
		for i := lo; i < hi; i++ {
			rows = append(rows, mw.EmbedResidentForTest((i*37+3)%vocab))
		}
		return rows
	}
	ctx := context.Background()
	const start = 40
	chunk := win.prefillPassRows()
	full.prefillChunkCap.Store(int64(chunk)) // the control takes the same passes, so the planted decline fires at the same row
	if full.prefillPassRows() != chunk {
		t.Fatalf("the full-KV control's passes are %d rows, the windowed resident's %d", full.prefillPassRows(), chunk)
	}
	end := start + 4*chunk
	var want, got []float32
	for _, r := range []*cudaResident{full, win} {
		if _, err := r.PrefillLast(ctx, emb(0, start), 0); err != nil {
			t.Fatalf("the first %d rows: %v", start, err)
		}
		withPrefillDeclinePass(t, 3)
		_, err := r.PrefillLast(ctx, emb(start, end), start)
		prefillDeclinePassForTest = 0
		var p *decoder.PrefillPartial
		if !errors.As(err, &p) {
			t.Fatalf("windowed=%v: a third-pass decline returned %v, want a decoder.PrefillPartial", r.kvWin, err)
		}
		if !errors.Is(err, errPrefillDeclined) {
			t.Errorf("windowed=%v: the PrefillPartial does not unwrap to errPrefillDeclined: %v", r.kvWin, err)
		}
		if p.Next != start+2*chunk {
			t.Fatalf("windowed=%v: Next = %d, want %d (two passes of %d committed)", r.kvWin, p.Next, start+2*chunk, chunk)
		}
		if r.kvWin && r.ReusableKV(start) != 0 {
			t.Fatalf("the window still holds position %d's history (base %d): the test does not exercise a moved window", start, r.kvBases[r.kvSlot])
		}
		lg, err := r.PrefillLast(ctx, emb(p.Next, end), p.Next)
		if err != nil {
			t.Fatalf("windowed=%v: resuming at %d: %v", r.kvWin, p.Next, err)
		}
		if r.kvWin {
			got = append([]float32(nil), lg...)
		} else {
			want = append([]float32(nil), lg...)
		}
	}
	if j, ok := sameBits(want, got); !ok {
		t.Fatalf("the resumed windowed prefill's logits differ from the full-KV resident's at %d", j)
	}
	if _, err := win.PrefillLast(ctx, emb(start, end), start); err == nil {
		t.Fatalf("a restart at %d, below the window's base %d, succeeded: a stale read would have been silent", start, win.kvBases[win.kvSlot])
	}
}

// TestPrefillPartial_windowedGenerationAfterADecline is the C1 shape through Model.Generate: a continuation that reuses the slot's prefix and
// prefills a suffix of several passes, the third of which declines. The windowed load must finish the turn (the decoder resumes after the
// committed passes) and emit the full-KV load's ids, having reused the prefix.
func TestPrefillPartial_windowedGenerationAfterADecline(t *testing.T) {
	eachWindowFixture(t, windowedGenerationAfterDecline)
}

func windowedGenerationAfterDecline(t *testing.T, fx string) {
	_, win, mf, mw := loadWindowedPair(t, fx, 1)
	_, _, _, _, _, _, vocab := mw.Dims()
	chunk := win.prefillPassRows()
	gen := func(m *decoder.Model, prompt []int) ([]int, int) {
		ch, g := m.Generate(context.Background(), prompt, 6, decoder.SamplingParams{})
		var ids []int
		for id := range ch {
			ids = append(ids, id)
		}
		if err := g.Err(); err != nil {
			t.Fatalf("%v", err)
		}
		return ids, g.PrefillReused
	}
	play := func(m *decoder.Model) ([]int, int) {
		a := make([]int, 60)
		for i := range a {
			a[i] = (i*31 + 7) % vocab
		}
		reply, _ := gen(m, a)
		b := append(append([]int(nil), a...), reply...)
		for i := range 4 * chunk {
			b = append(b, (i*13+11)%vocab)
		}
		withPrefillDeclinePass(t, 3)
		defer func() { prefillDeclinePassForTest = 0 }()
		return gen(m, b)
	}
	want, wantReused := play(mf)
	got, gotReused := play(mw)
	if wantReused == 0 || gotReused == 0 {
		t.Fatalf("the continuation reused %d tokens full, %d windowed: the test does not exercise a reused prefix", wantReused, gotReused)
	}
	if len(got) != len(want) {
		t.Fatalf("%d ids windowed, %d full", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("id %d = %d windowed, %d full", i, got[i], want[i])
		}
	}
}
