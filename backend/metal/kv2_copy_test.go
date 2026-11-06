//go:build darwin

package metal

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime/debug"
	"slices"
	"testing"
	"time"

	gpu "github.com/townsendmerino/goinfer/backend/metal/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
)

// kv2Embs is n deterministic prompt embeddings, distinct per seed.
func kv2Embs(r *resident, n int, seed uint32) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		seed ^= seed << 13
		seed ^= seed >> 17
		seed ^= seed << 5
		out[i] = mc3Emb(r, int(seed%20000))
	}
	return out
}

// kv2Slot is every byte of slot s's K, V and (int8) scale rows [0, rows), layer by layer, gathered block by block through the
// slot's table, so it reads a sequence's rows whatever blocks hold them.
func kv2Slot(t *testing.T, r *resident, s, rows int) [][]byte {
	t.Helper()
	var out [][]byte
	tab := r.kvTab.U32s()
	gather := func(base Buffer, rowBytes int) []byte {
		block := decoder.KVBlockRows * rowBytes
		h := make([]byte, 0, kvBlocksFor(rows)*block)
		for b := range kvBlocksFor(rows) {
			part := make([]byte, block)
			if err := gpu.Download(base.At(int(tab[s*r.kvTabBlocks+b])*block), part); err != nil {
				t.Fatal(err)
			}
			h = append(h, part...)
		}
		return h[:rows*rowBytes]
	}
	elem := 2
	if r.kvI8 {
		elem = 1
	}
	for l := range r.layers {
		L := &r.layers[l]
		if L.kvShared || L.geom == nil || r.kvSlotBytes[l] == 0 {
			continue
		}
		row := L.geom.kvDim * elem
		out = append(out, gather(r.kc[l], row), gather(r.vc[l], row))
		if r.kvI8 {
			out = append(out, gather(r.ks[l], L.geom.nKV*4), gather(r.vs[l], L.geom.nKV*4))
		}
	}
	return out
}

// kv2RowsEqual reports the first buffer and row where a and b differ inside [lo, hi) (rowBytes per buffer), or -1.
func kv2RowsEqual(a, b [][]byte, rowBytes []int, lo, hi int) (buf, row int) {
	for i := range a {
		rb := rowBytes[i]
		for p := lo; p < hi; p++ {
			if !slices.Equal(a[i][p*rb:(p+1)*rb], b[i][p*rb:(p+1)*rb]) {
				return i, p
			}
		}
	}
	return -1, -1
}

func kv2RowBytes(r *resident) []int {
	elem := 2
	if r.kvI8 {
		elem = 1
	}
	var out []int
	for l := range r.layers {
		L := &r.layers[l]
		if L.kvShared || L.geom == nil || r.kvSlotBytes[l] == 0 {
			continue
		}
		out = append(out, L.geom.kvDim*elem, L.geom.kvDim*elem)
		if r.kvI8 {
			out = append(out, L.geom.nKV*4, L.geom.nKV*4)
		}
	}
	return out
}

// kv2Resident is the MC3 fixture's resident with three KV slots, at f16 KV or int8 KV.
func kv2Resident(t *testing.T, i8 bool) (*metalResident, *resident) {
	t.Helper()
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	o := decoder.Options{Quant: mc3TestQuant, ResidentContext: 2048, ResidentKVSlots: 3}
	if i8 {
		o.KVPrecision = "i8"
	}
	m, err := decoder.Load(mc3FixtureWriter(t, 4096), o)
	if err != nil {
		t.Fatalf("load the MC3 fixture: %v", err)
	}
	r, err := buildResident(m)
	if err != nil {
		m.Close()
		t.Fatalf("build the resident: %v", err)
	}
	t.Cleanup(func() { r.Close(); m.Close(); debug.FreeOSMemory() })
	if len(r.kvSlotBufs) != 3 || r.kvI8 != i8 {
		t.Skipf("the fixture built %d slots, int8 KV %v; the test needs 3, int8 %v", len(r.kvSlotBufs), r.kvI8, i8)
	}
	return &metalResident{r: r, hidden: r.H}, r
}

// TestKV2_copyKVExact is G2 on Metal (docs/measurements/kv2-2026-10-14.md): after CopyKV(src, dst, start, end) dst's
// rows in the range equal src's byte for byte, dst's other rows and every other slot are unchanged, dst's sequence ends where src's
// does, and a range past the cache is declined with nothing written. Each slot's rows are read through its table, and the scrambled arm hands blocks
// out reversed and interleaved across slots, so the copy must move a sequence's blocks, not a slot's byte range.
func TestKV2_copyKVExact(t *testing.T) {
	for _, c := range []struct {
		i8, scramble bool
	}{{false, false}, {true, false}, {false, true}, {true, true}} {
		i8 := c.i8
		name := map[bool]string{false: "f16", true: "i8"}[i8] + map[bool]string{false: "", true: "/scrambled"}[c.scramble]
		t.Run(name, func(t *testing.T) {
			kvScrambleForTest = c.scramble
			a, r := kv2Resident(t, i8)
			kvScrambleForTest = false
			const rows = 1024
			for s, n := range []int{1000, 1000, 300} {
				if err := r.useKVSlot(s); err != nil {
					t.Fatal(err)
				}
				embs := kv2Embs(r, n, uint32(100+s))
				if _, err := a.PrefillLast(context.Background(), embs, 0); err != nil {
					// int8 KV has no batched prefill (it writes f16 K/V): fill the slot as decode does
					for p, e := range embs {
						if _, err := a.Forward(e, p); err != nil {
							t.Fatalf("slot %d position %d: %v", s, p, err)
						}
					}
				}
			}
			rb := kv2RowBytes(r)
			for _, tc := range []struct{ src, dst, start, end int }{{0, 1, 0, 64}, {0, 1, 0, 1024}, {1, 2, 256, 1024}} {
				before := [][][]byte{kv2Slot(t, r, 0, rows), kv2Slot(t, r, 1, rows), kv2Slot(t, r, 2, rows)}
				if err := a.CopyKV(tc.src, tc.dst, tc.start, tc.end); err != nil {
					t.Fatalf("CopyKV%v: %v", tc, err)
				}
				after := [][][]byte{kv2Slot(t, r, 0, rows), kv2Slot(t, r, 1, rows), kv2Slot(t, r, 2, rows)}
				if b, p := kv2RowsEqual(after[tc.dst], before[tc.src], rb, tc.start, tc.end); b >= 0 {
					t.Errorf("CopyKV%v: dst buffer %d row %d is not src's", tc, b, p)
				}
				if b, p := kv2RowsEqual(after[tc.dst], before[tc.dst], rb, 0, tc.start); b >= 0 {
					t.Errorf("CopyKV%v: dst buffer %d row %d below the range changed", tc, b, p)
				}
				if b, p := kv2RowsEqual(after[tc.dst], before[tc.dst], rb, tc.end, rows); b >= 0 {
					t.Errorf("CopyKV%v: dst buffer %d row %d above the range changed", tc, b, p)
				}
				for s := range 3 {
					if s != tc.dst {
						if b, p := kv2RowsEqual(after[s], before[s], rb, 0, rows); b >= 0 {
							t.Errorf("CopyKV%v: slot %d buffer %d row %d changed", tc, s, b, p)
						}
					}
				}
			}
			// A copy hands dst the source's sequence, no further: slot 0 wrote 1,000 rows, so 1,024 rows of whole blocks reach 1,000.
			if got, past := a.ReusableKVSlot(2, 1000), a.ReusableKVSlot(2, 1001); got != 1000 || past != 0 {
				t.Errorf("after the copies slot 2 answers %d for 1,000 rows and %d for 1,001; want 1000 and 0", got, past)
			}
			before := kv2Slot(t, r, 2, rows)
			var d *decoder.KVCopyDecline
			if err := a.CopyKV(0, 2, 0, 4096); !errors.As(err, &d) {
				t.Errorf("a range past the cache: %v, want a KVCopyDecline", err)
			}
			if b, p := kv2RowsEqual(kv2Slot(t, r, 2, rows), before, rb, 0, rows); b >= 0 {
				t.Errorf("a declined copy wrote buffer %d row %d", b, p)
			}
		})
	}
}

// TestKV2_copyThenSuffixEqualsWhole is G0's split arm and G3's device check on Metal: a prompt's first 960 rows copied
// from a slot that prefilled a longer prompt sharing them, then the rest prefilled from 960, gives the KV and the last
// logits of the prompt prefilled whole, bit for bit, both for a 40-row suffix and for a 5-row one (Metal's short
// continuation is exact, PrefillTailExact). The control changes one token before the split and must differ.
// chainSkipLogged keeps TestKVPages_shareByReference's note that the chain did not run to one line.
var chainSkipLogged bool

// TestKVPages_shareByReference is KV5-5's G3b and G1a's second slot on Metal: slot 0 prefills a 640-row preamble and slots 1 and 2
// take it by CopyKV, by reference, so the pool holds it once (its own count, kvBlocksInUse, is the preamble's 10 blocks after the
// copies, against 30 when each slot prefills it).
// Each slot then prefills its own suffix from 640 and decodes. Then slots 0 and 1 rewind into blocks the others still share, by every
// writer in turn (a prefill, the step, ForwardBatch, the verify, UploadKV, the greedy chain across a block edge, a decode), each a copy
// on write. Every logit, and every slot's K/V over its own history read through its table after each write, equals the reference: a
// fresh resident where each slot prefilled the preamble itself, at f16 and int8 KV, on in-order and scrambled tables. With kvOwn
// planted off, a rewind lands in a shared block and another slot's K/V moves.
func TestKVPages_shareByReference(t *testing.T) {
	const pre, suffix = 640, 70
	for _, c := range []struct{ i8, scramble bool }{{false, false}, {true, false}, {false, true}, {true, true}} {
		t.Run(map[bool]string{false: "f16", true: "i8"}[c.i8]+map[bool]string{false: "", true: "/scrambled"}[c.scramble], func(t *testing.T) {
			run := func(share, plant bool) ([][]float32, [][][]byte, int) {
				kvScrambleForTest, kvOwnOffForTest = c.scramble, plant
				defer func() { kvScrambleForTest, kvOwnOffForTest = false, false }()
				a, r := kv2Resident(t, c.i8)
				var out [][]float32
				add := func(l []float32, err error) {
					t.Helper()
					if err != nil {
						t.Fatal(err)
					}
					out = append(out, slices.Clone(l))
				}
				bind := func(s int) {
					t.Helper()
					if err := r.useKVSlot(s); err != nil {
						t.Fatal(err)
					}
				}
				preamble := kv2Embs(r, pre, 7)
				before := r.kvBlocksInUse()
				for s := range 3 {
					bind(s)
					if s == 0 || !share {
						add(a.PrefillLast(context.Background(), preamble, 0))
					} else {
						if err := a.CopyKV(0, s, 0, pre); err != nil {
							t.Fatal(err)
						}
					}
				}
				held := r.kvBlocksInUse() - before
				for s := range 3 {
					bind(s)
					add(a.PrefillLast(context.Background(), kv2Embs(r, suffix, uint32(20+s)), pre))
					add(a.Forward(kv2Embs(r, 1, uint32(30+s))[0], pre+suffix))
				}
				// Rewinds: a write at p ends a sequence at p (a copy on write keeps the rows below it), so each slot that writes again
				// does so further back, and decodes right after its last write. Each write lands in a block the other slots still share.
				// After each, every slot's K/V over its valid history so far (n), gathered through its table, is kept: a write into a
				// block another slot still shares changes that slot's rows whether or not its attention weighs them.
				n := []int{pre + suffix + 1, pre + suffix + 1, pre + suffix + 1}
				var kv [][][]byte
				snap := func(s, to int) {
					n[s] = to
					for q := range 3 {
						kv = append(kv, kv2Slot(t, r, q, n[q]))
					}
				}
				bind(0) // slot 0, the source: a prefill from 600 (block 9), the step at 500 (block 7), ForwardBatch at 400 (block 6)
				add(a.PrefillLast(context.Background(), kv2Embs(r, 50, 41), 600))
				snap(0, 650)
				rows, _, err := r.forwardMulti([]batchSeq{{slot: 0, pos: 500, emb: kv2Embs(r, 1, 61)[0]}})
				if err != nil {
					t.Fatal(err)
				}
				out = append(out, slices.Clone(rows[0]))
				snap(0, 501)
				br, err := r.ForwardBatch(kv2Embs(r, 2, 63), 400)
				if err != nil {
					t.Fatal(err)
				}
				out = append(out, slices.Clone(br[1]))
				snap(0, 402)
				add(a.Forward(kv2Embs(r, 1, 64)[0], 402))
				snap(0, 403)
				bind(1) // slot 1: the verify at 340 (block 5), UploadKV at 280 (block 4)
				vr, err := a.ForwardN(kv2Embs(r, 4, 62), 340)
				if err != nil {
					t.Fatal(err)
				}
				out = append(out, slices.Clone(vr[3]))
				snap(1, 344)
				for l := range r.layers {
					if g := r.layers[l].geom; g != nil && !r.layers[l].kvShared {
						up := make([]float32, g.kvDim)
						for i := range up {
							up[i] = float32(i%7) * 0.25
						}
						if err := a.UploadKV(l, 280, up, up); err != nil {
							t.Fatal(err)
						}
					}
				}
				snap(1, 281)
				add(a.Forward(kv2Embs(r, 1, 65)[0], 281))
				snap(1, 282)
				// slot 2: the greedy chain from 190, where the build has one: it runs past position 192, out of block 2 into block 3,
				// both still shared by all three slots, so it must own a block it has not reached when it opens. Then a decode at 100
				// (block 1, still shared).
				bind(2)
				if why := r.greedyChainWhyNot(); why == "" {
					if err := r.chainStart(5, 190); err != nil {
						t.Fatal(err)
					}
					for range 4 {
						if _, err := r.chainNext(); err != nil {
							t.Fatal(err)
						}
					}
					r.stopChain()
					snap(2, 191) // the chain's first forward wrote position 190; how far it ran past it is not this check's
				} else if !chainSkipLogged {
					chainSkipLogged = true
					t.Logf("no greedy chain on this resident (%s): its write site is not exercised here", why)
				}
				add(a.Forward(kv2Embs(r, 1, 66)[0], 100))
				snap(2, 101)
				return out, kv, held
			}
			differ := func(a, b [][]float32) int {
				n := 0
				for i := range a {
					for j := range a[i] {
						if math.Float32bits(a[i][j]) != math.Float32bits(b[i][j]) {
							n++
						}
					}
				}
				return n
			}
			kvDiffer := func(a, b [][][]byte) int {
				n := 0
				for s := range a {
					for i := range a[s] {
						if !slices.Equal(a[s][i], b[s][i]) {
							n++
						}
					}
				}
				return n
			}
			ref, refKV, refHeld := run(false, false)
			got, gotKV, held := run(true, false)
			if want := pre / decoder.KVBlockRows; held != want || refHeld != 3*want {
				t.Errorf("the preamble takes %d blocks shared (%d unshared); want %d, held once (%d)", held, refHeld, want, 3*want)
			}
			// The reference prefilled the preamble on slots 1 and 2 as well, so its rows are the shared run's minus those two prefills.
			ref = append(ref[:1], ref[3:]...)
			if n := differ(ref, got); n != 0 {
				t.Errorf("%d logits differ between the shared preamble and each slot's own", n)
			}
			if n := kvDiffer(refKV, gotKV); n != 0 {
				t.Errorf("%d per-layer K/V buffers differ, over each slot's own history, between the shared preamble and each slot's own", n)
			}
			// The K/V check is the one a missing copy on write must trip: this fixture's attention is too peaked for every leaked row
			// to move a logit.
			if _, plantedKV, _ := run(true, true); kvDiffer(refKV, plantedKV) == 0 {
				t.Error("with kvOwn off, the rewinds into shared blocks moved no other slot's K/V: the test cannot see a missing copy on write")
			}
		})
	}
}

// TestKVPages_G3bRealPreamble is KV5-5's G3b on a real checkpoint (GOINFER_METAL_MC3=1, the 1.5B by default; tier 1, under the
// exclusive lock): four conversations share a 16,384-token preamble, prefilled once and taken by CopyKV, by reference. The pool holds it
// once (its own count, kvBlocksInUse, stays at the preamble's 256 blocks across the three copies, and each conversation then takes only
// the blocks its own tail writes), and each conversation's 16 greedy tokens after
// its own 64-token suffix equal that conversation run alone on a slot of its own, token for token. Both arms prefill in the same
// 1,024-row passes, so the comparison is of sharing alone.
func TestKVPages_G3bRealPreamble(t *testing.T) {
	const pre, suffix, gen, pass = 16384, 64, 16, 1024
	start := time.Now()
	progress := func(what string) {
		fmt.Fprintf(os.Stderr, "[G3b] %s (elapsed %s)\n", what, time.Since(start).Round(time.Second))
	}
	_, r := mc3LoadCheckpoint(t, 4, pre+suffix+gen+64)
	a := &metalResident{r: r, hidden: r.H}
	prefill := func(embs [][]float32, at int) []float32 {
		t.Helper()
		var l []float32
		for c := 0; c < len(embs); c += pass {
			var err error
			if l, err = a.PrefillLast(context.Background(), embs[c:min(c+pass, len(embs))], at+c); err != nil {
				t.Fatal(err)
			}
		}
		return slices.Clone(l)
	}
	greedy := func(l []float32, at int) []int {
		t.Helper()
		var toks []int
		for i := range gen {
			tok := argmaxF32(l)
			toks = append(toks, tok)
			var err error
			if l, err = a.Forward(mc3Emb(r, tok), at+i); err != nil {
				t.Fatal(err)
			}
		}
		return toks
	}
	bind := func(s int) {
		t.Helper()
		if err := r.useKVSlot(s); err != nil {
			t.Fatal(err)
		}
	}
	preamble := kv2Embs(r, pre, 7)
	suffixes := make([][][]float32, 4)
	for s := range suffixes {
		suffixes[s] = kv2Embs(r, suffix, uint32(100+s))
	}
	// Each conversation alone, on slot 0 (a fresh sequence each time: the prefill starts at position 0).
	alone := make([][]int, 4)
	for s := range 4 {
		bind(0)
		prefill(preamble, 0)
		alone[s] = greedy(prefill(suffixes[s], pre), pre+suffix)
		progress(fmt.Sprintf("conversation %d alone", s))
	}
	// The four together: slot 0 prefills the preamble, slots 1 to 3 take it by reference.
	bind(0)
	prefill(preamble, 0)
	before := r.kvBlocksInUse()
	for s := 1; s < 4; s++ {
		if err := a.CopyKV(0, s, 0, pre); err != nil {
			t.Fatal(err)
		}
	}
	shared := r.kvBlocksInUse()
	progress(fmt.Sprintf("preamble shared: %d blocks in use before the copies, %d after", before, shared))
	for s := range 4 {
		bind(s)
		got := greedy(prefill(suffixes[s], pre), pre+suffix)
		if !slices.Equal(got, alone[s]) {
			t.Errorf("conversation %d: %v with the preamble shared, %v alone", s, got, alone[s])
		}
	}
	tail := kvBlocksFor(pre+suffix+gen) - pre/decoder.KVBlockRows
	if want, end := pre/decoder.KVBlockRows, r.kvBlocksInUse(); before != want || shared != want || end != want+4*tail {
		t.Errorf("blocks in use: %d before the copies, %d after, %d at the end; want %d, %d, %d: the preamble is not held once",
			before, shared, end, want, want, want+4*tail)
	}
	progress("done")
}

func TestKV2_copyThenSuffixEqualsWhole(t *testing.T) {
	a := mc3PrefillResident(t, 2, 2048)
	r := a.r
	if len(r.kvSlotBufs) < 2 {
		t.Skip("needs two slots")
	}
	shared := kv2Embs(r, 990, 7)
	src := append(slices.Clone(shared), kv2Embs(r, 10, 99)...) // a longer prompt that holds the shared rows
	for _, n := range []int{1000, 965} {
		prompt := append(slices.Clone(shared[:960]), kv2Embs(r, n-960, 31)...)
		if err := r.useKVSlot(1); err != nil {
			t.Fatal(err)
		}
		whole, err := a.PrefillLast(context.Background(), prompt, 0)
		if err != nil {
			t.Fatal(err)
		}
		whole = slices.Clone(whole)
		want := kv2Slot(t, r, 1, n)
		if err := r.useKVSlot(0); err != nil {
			t.Fatal(err)
		}
		if _, err := a.PrefillLast(context.Background(), src, 0); err != nil {
			t.Fatal(err)
		}
		if err := a.CopyKV(0, 1, 0, 960); err != nil {
			t.Fatal(err)
		}
		if err := r.useKVSlot(1); err != nil {
			t.Fatal(err)
		}
		got, err := a.PrefillLast(context.Background(), prompt[960:], 960)
		if err != nil {
			t.Fatal(err)
		}
		if b, p := kv2RowsEqual(kv2Slot(t, r, 1, n), want, kv2RowBytes(r), 0, n); b >= 0 {
			t.Errorf("%d tokens: copied + suffix KV differs from whole at buffer %d row %d", n, b, p)
		}
		for i := range whole {
			if math.Float32bits(whole[i]) != math.Float32bits(got[i]) {
				t.Errorf("%d tokens: last logits differ at %d (%g vs %g)", n, i, got[i], whole[i])
				break
			}
		}
	}
	// Control: the source holds a different token at row 500, so the copied rows are not the prompt's, and the result
	// must differ from that row on.
	pert := slices.Clone(src)
	pert[500] = mc3Emb(r, 4321)
	prompt := append(slices.Clone(shared[:960]), kv2Embs(r, 40, 31)...)
	if err := r.useKVSlot(1); err != nil {
		t.Fatal(err)
	}
	if _, err := a.PrefillLast(context.Background(), prompt, 0); err != nil {
		t.Fatal(err)
	}
	want := kv2Slot(t, r, 1, 1000)
	if err := r.useKVSlot(0); err != nil {
		t.Fatal(err)
	}
	if _, err := a.PrefillLast(context.Background(), pert, 0); err != nil {
		t.Fatal(err)
	}
	if err := a.CopyKV(0, 1, 0, 960); err != nil {
		t.Fatal(err)
	}
	if err := r.useKVSlot(1); err != nil {
		t.Fatal(err)
	}
	if _, err := a.PrefillLast(context.Background(), prompt[960:], 960); err != nil {
		t.Fatal(err)
	}
	if b, p := kv2RowsEqual(kv2Slot(t, r, 1, 1000), want, kv2RowBytes(r), 0, 1000); b < 0 || p != 500 {
		t.Errorf("control: first difference at buffer %d row %d, want row 500", b, p)
	}
}

// TestKV2_copyCost is G7's in-process copy cost on Metal (reported, not graded): CopyKV of 18,176 rows (a header-off
// Claude Code preamble's whole blocks) between two slots of a real checkpoint, five times, the median printed. The
// checkpoint is GOINFER_METAL_MC3_MODEL (default the 1.5B).
//
//	GOINFER_METAL_MC3=1 go test -run TestKV2_copyCost -v ./metal/
func TestKV2_copyCost(t *testing.T) {
	_, r := mc3LoadCheckpoint(t, 2, 18432)
	if len(r.kvSlotBufs) < 2 {
		t.Skip("one slot")
	}
	a := &metalResident{r: r, hidden: r.H}
	const rows = 18176
	var ms []float64
	for range 5 {
		t0 := time.Now()
		if err := a.CopyKV(0, 1, 0, rows); err != nil {
			t.Fatal(err)
		}
		ms = append(ms, float64(time.Since(t0).Microseconds())/1000)
	}
	slices.Sort(ms)
	var bytes int
	for _, b := range kv2RowBytes(r) {
		bytes += b * rows
	}
	fmt.Fprintf(os.Stderr, "[kv2-g7 metal] CopyKV of %d rows (%.0f MB): median %.2f ms of %v\n", rows, float64(bytes)/1e6, ms[2], ms)
}
