//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/prequant"
)

// TestG4ExpertBatch_matchesPerRow is D-P01's gate (docs/tasks/task-m26-mac-2026-10.md): the layer-major paged prefill
// with the batched expert phase 2 (g4ExpertBatchOn) against the same pass with the per-row phase 2, on the tiny Gemma 4
// MoE transcoded as M26's bundle is, paged at 2 and 3 slots of its 4 experts. Prompts of 5 and 12 rows from position
// 0 and 9 rows continuing a 3-token prefix. The last row's logits, every K/V element and 3 decode steps after it must be
// equal bit for bit. The batched arm must have run (g4BatchGroups), and an entry must have held moeBatchB pairs at each
// slot count, so the shared-weight path (one weight read for several pairs) is what was compared. (4 slots is every
// expert, which is not paged.)
func TestG4ExpertBatch_matchesPerRow(t *testing.T) {
	const ckpt = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
		t.Skipf("no fixture (%s): %v", ckpt, err)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	giw := filepath.Join(t.TempDir(), "gemma4-moe-tiny.metal.giw")
	if err := prequant.Transcode(context.Background(), ckpt, giw, "int4", false, decoder.GIWTargetMetal); err != nil {
		t.Fatalf("transcode: %v", err)
	}
	toks := []int{1, 7, 42, 100, 5, 200, 13, 88, 3, 64, 9, 150, 21, 77, 30, 11, 6, 25}
	defer func(on bool) { g4ExpertBatchOn = on }(g4ExpertBatchOn)
	for _, slots := range []int{2, 3} {
		t.Run("slots="+strconv.Itoa(slots), func(t *testing.T) {
			t.Setenv("GOINFER_METAL_MOE_SLOTS", strconv.Itoa(slots))
			m, err := decoder.Load(giw, decoder.Options{Backend: "metal", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			mr, ok := m.ResidentForwardForTest().(*metalResident)
			if !ok || mr.r.g4moe == nil || !mr.r.g4moe.paged {
				t.Fatalf("expected a paged Gemma 4 MoE build (%s)", m.ResidentDecline())
			}
			r := mr.r
			g4ExpertBatchOn = true
			if !r.g4ExpertBatchOK() {
				t.Fatalf("the batched phase 2 declines this fixture (H %d, moeInter %d)", r.H, r.g4moe.moeInter)
			}
			emb := func(i int) []float32 { return m.EmbedResidentForTest(toks[i]) }
			for _, c := range []struct{ prefix, n int }{{0, 5}, {0, 12}, {3, 9}} {
				total := c.prefix + c.n
				run := func(batched bool) (last []float32, kvs [][]uint16, dec [][]float32, groups int) {
					g4ExpertBatchOn = batched
					mr.Reset()
					for i := range c.prefix {
						if err := mr.ForwardNoLogits(emb(i), i); err != nil {
							t.Fatal(err)
						}
					}
					embs := make([][]float32, c.n)
					for i := range embs {
						embs[i] = emb(c.prefix + i)
					}
					g0 := r.g4BatchGroups
					last = append([]float32(nil), r.prefillG4Paged(embs, c.prefix, true)...)
					if err := r.takeExecErr(); err != nil {
						t.Fatal(err)
					}
					groups = r.g4BatchGroups - g0
					for l := range r.layers {
						d := r.layers[l].geom.kvDim
						o := r.kvHostOff(l, 2)
						kvs = append(kvs, append([]uint16(nil), r.kc[l].U16s()[o:o+total*d]...), append([]uint16(nil), r.vc[l].U16s()[o:o+total*d]...))
					}
					for s := range 3 {
						lg, err := mr.Forward(emb(total+s), total+s)
						if err != nil {
							t.Fatal(err)
						}
						dec = append(dec, append([]float32(nil), lg...))
					}
					return last, kvs, dec, groups
				}
				pLast, pKV, pDec, pGroups := run(false)
				bLast, bKV, bDec, bGroups := run(true)
				label := "prefix " + strconv.Itoa(c.prefix) + ", " + strconv.Itoa(c.n) + " rows"
				if pGroups != 0 || bGroups == 0 {
					t.Fatalf("%s: batched groups %d with the switch off, %d with it on", label, pGroups, bGroups)
				}
				same := func(what string, a, b []float32) {
					t.Helper()
					if len(a) != len(b) {
						t.Fatalf("%s: %s lengths %d / %d", label, what, len(a), len(b))
					}
					for j := range a {
						if math.Float32bits(a[j]) != math.Float32bits(b[j]) {
							t.Fatalf("%s: %s element %d: batched %v, per-row %v", label, what, j, b[j], a[j])
						}
					}
				}
				same("last-row logits", pLast, bLast)
				for i := range pKV {
					for j := range pKV[i] {
						if pKV[i][j] != bKV[i][j] {
							t.Fatalf("%s: K/V buffer %d element %d differs", label, i, j)
						}
					}
				}
				for s := range pDec {
					same("decode step "+strconv.Itoa(s), pDec[s], bDec[s])
				}
				t.Logf("%s: bit-identical; %d batched groups", label, bGroups)
			}
			t.Logf("slots %d: most pairs in one entry %d (of %d)", slots, r.g4BatchMaxCnt, moeBatchB)
			if r.g4BatchMaxCnt != moeBatchB {
				t.Fatalf("no entry filled (max %d pairs, want %d): the shared-weight path was not exercised", r.g4BatchMaxCnt, moeBatchB)
			}
		})
	}
}
