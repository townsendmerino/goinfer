//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"unsafe"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/prequant"
)

// TestGemma4PagingPread_matchesNonPaged is TestMoEPagingPread_matchesByteCopy's twin for the Gemma 4 pager, the one
// M26 runs. TestGemma4Paging_bitExact loads the safetensors fixture, so its scales live on the heap and stagePread is
// never wired. This transcodes the fixture to a metal-target .giw, as M26's bundle is, and requires exact logit
// equality between non-paged, paged byte-copy (GOINFER_MOE_PREAD=0) and paged pread.
//
// C-P01 (docs/audit-metal-2026-09-30.md): the pager stages each expert's f16 scales from the WeightMat instead of a
// build-time heap cache. The test asserts what makes that a memory saving: every expert's scales lie inside the .giw
// mapping, and the pager's stage function hands out those same bytes, so no heap copy of them exists. It also
// asserts that pread ran, nibbles and scales both, or the pread arm would compare the byte-copy path against itself.
func TestGemma4PagingPread_matchesNonPaged(t *testing.T) {
	const ckpt = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
		t.Skipf("no fixture (%s): %v", ckpt, err)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	giw := filepath.Join(t.TempDir(), "gemma4-moe-tiny.metal.giw")
	if err := prequant.Transcode(context.Background(), ckpt, giw, "int4", false, decoder.GIWTargetMetal); err != nil {
		t.Fatalf("transcode fixture to a metal .giw: %v", err)
	}

	type out struct {
		logits      [][]float32
		preads      int
		scalePreads int64 // stages whose scales were pread too (C-P01)
		stages      int
		scalesTotal int
		scalesInMap int
		stagedInMap int // of the pools' staged scale slices, how many are the mapping's bytes
		stagedTotal int
	}
	run := func(t *testing.T, slots, pread string) out {
		if slots == "" {
			os.Unsetenv("GOINFER_METAL_MOE_SLOTS")
		} else {
			t.Setenv("GOINFER_METAL_MOE_SLOTS", slots)
		}
		if pread == "" {
			os.Unsetenv("GOINFER_MOE_PREAD")
		} else {
			t.Setenv("GOINFER_MOE_PREAD", pread)
		}
		m, err := decoder.Load(giw, decoder.Options{Backend: "metal", Quant: "int4"})
		if err != nil {
			t.Fatalf("load (slots=%q pread=%q): %v", slots, pread, err)
		}
		defer m.Close()
		mr, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			t.Fatalf("not metal-resident (slots=%q pread=%q): %s", slots, pread, m.ResidentDecline())
		}
		if slots != "" && (mr.r.g4moe == nil || !mr.r.g4moe.paged) {
			t.Fatalf("slots=%s: expected a paged Gemma 4 MoE build", slots)
		}
		var o out
		mr.Reset()
		for i, tok := range twoGeomPrompt {
			lg, err := mr.Forward(m.EmbedResidentForTest(tok), i)
			if err != nil {
				t.Fatalf("forward[%d] (slots=%q pread=%q): %v", i, slots, pread, err)
			}
			o.logits = append(o.logits, append([]float32(nil), lg...))
		}
		inMap := func(s []uint16) bool {
			_, ok := m.MmapByteOffset(unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), 2*len(s)))
			return ok
		}
		for l := range mr.r.layers {
			if gl := mr.r.layers[l].g4moe; gl != nil && gl.pool != nil {
				o.preads += gl.pool.preads
				o.scalePreads += gl.pool.scalePreads.Load()
				o.stages += gl.pool.stages
				for e := range 2 { // what the pager stages from: the mapping, not a heap copy of it
					_, guS, _, dS := gl.pool.stage(e)
					for _, sc := range [][]uint16{guS, dS} {
						o.stagedTotal++
						if inMap(sc) {
							o.stagedInMap++
						}
					}
				}
			}
			if b, ok := m.Gemma4MoEResidentLayer(l); ok {
				for _, w := range append(append(b.ExpertsGateUp[:0:0], b.ExpertsGateUp...), b.ExpertsDown...) {
					o.scalesTotal++
					if inMap(w.Int4ScalesF16()) {
						o.scalesInMap++
					}
				}
			}
		}
		return o
	}

	base := run(t, "", "")
	if base.scalesTotal == 0 || base.scalesInMap != base.scalesTotal {
		t.Fatalf("%d of %d expert scale arrays lie in the .giw mapping: C-P01 stages scales from the WeightMat, so any "+
			"outside it are a heap copy after all", base.scalesInMap, base.scalesTotal)
	}
	eq := func(t *testing.T, got [][]float32, label string) {
		t.Helper()
		for i := range base.logits {
			for j := range base.logits[i] {
				if got[i][j] != base.logits[i][j] {
					t.Fatalf("%s tok %d elem %d: %v != non-paged %v", label, i, j, got[i][j], base.logits[i][j])
				}
			}
		}
	}
	for _, slots := range []int{2, 3} {
		s := strconv.Itoa(slots)
		t.Run("slots="+s+"/bytecopy", func(t *testing.T) {
			o := run(t, s, "0")
			if o.preads != 0 {
				t.Fatalf("GOINFER_MOE_PREAD=0 still took the pread path (%d preads)", o.preads)
			}
			if o.stages == 0 {
				t.Fatal("no expert was staged: the pool never paged")
			}
			eq(t, o.logits, "bytecopy")
			t.Logf("slots=%s byte-copy: %d tokens exact vs non-paged, %d stages", s, len(base.logits), o.stages)
		})
		t.Run("slots="+s+"/pread", func(t *testing.T) {
			o := run(t, s, "")
			if o.preads == 0 {
				t.Fatal("pread never ran (0 preads): the arm compared the byte-copy path against itself")
			}
			if o.scalePreads != int64(o.preads) {
				t.Fatalf("%d of %d pread stages read their scales from the file: on a metal .giw every one should (C-P01)", o.scalePreads, o.preads)
			}
			if o.stagedTotal == 0 || o.stagedInMap != o.stagedTotal {
				t.Fatalf("%d of %d staged scale slices are the mapping's bytes: the pager holds a copy (C-P01)", o.stagedInMap, o.stagedTotal)
			}
			eq(t, o.logits, "pread")
			t.Logf("slots=%s pread: %d tokens exact vs non-paged, %d stages by pread (%d with their scales); %d/%d expert scale arrays in the mapping",
				s, len(base.logits), o.preads, o.scalePreads, base.scalesInMap, base.scalesTotal)
		})
	}
}
