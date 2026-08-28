//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/prequant"
)

// TestMoEPagingPread_matchesByteCopy is TestMoEPaging_matchesNonPaged's twin for the PREAD staging
// path. The byte-identity test cannot cover pread: it loads the safetensors fixture, whose GiwPath
// is "" (srcPath is set only on the .giw load path), so stagePread is never wired and the byte-copy
// path is what it actually exercised. This transcodes the same fixture to a .giw, then drives three
// arms — non-paged, paged+byte-copy (GOINFER_MOE_PREAD=0), paged+pread — and requires EXACT logit
// equality across all three.
//
// It also asserts pool.preads > 0 on the pread arm. Without that the test is near-worthless: if the
// offset resolution silently fell back (a non-mmap-backed weight, a failed open), every arm would
// run the same byte-copy code and the comparison would pass while proving nothing about pread. That
// is the "prove the gate can go red" discipline — the byte-identity assertion alone cannot tell the
// fast path from its own fallback.
func TestMoEPagingPread_matchesByteCopy(t *testing.T) {
	const ckpt = "../decoder/testdata/qwen3_5_moe-tiny"
	if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
		t.Skipf("no qwen3_5_moe fixture at %s (run scripts/pin_qwen3_5_forward.py --moe): %v", ckpt, err)
	}
	giw := filepath.Join(t.TempDir(), "qwen3_5_moe-tiny.giw")
	if err := prequant.Transcode(context.Background(), ckpt, giw, "int4", false, false); err != nil {
		t.Fatalf("transcode fixture to .giw: %v", err)
	}

	// run drives ntok tokens and returns the logits plus the total pread count across paged layers.
	run := func(t *testing.T, slotsEnv, preadEnv string) ([][]float32, int) {
		if slotsEnv == "" {
			os.Unsetenv("GOINFER_METAL_MOE_SLOTS")
		} else {
			t.Setenv("GOINFER_METAL_MOE_SLOTS", slotsEnv)
		}
		if preadEnv == "" {
			os.Unsetenv("GOINFER_MOE_PREAD")
		} else {
			t.Setenv("GOINFER_MOE_PREAD", preadEnv)
		}
		mRes, err := decoder.Load(giw, decoder.Options{Backend: "metal", Quant: "int4"})
		if err != nil {
			t.Fatalf("load metal (slots=%q pread=%q): %v", slotsEnv, preadEnv, err)
		}
		defer mRes.Close()
		rf := mRes.ResidentForwardForTest()
		if rf == nil {
			t.Fatalf("not resident (slots=%q pread=%q): %s", slotsEnv, preadEnv, mRes.ResidentDecline())
		}
		_, _, _, _, _, _, vocab := mRes.Dims()
		rf.Reset()
		const ntok = 32
		out := make([][]float32, ntok)
		for i := range ntok {
			lr, err := rf.Forward(mRes.EmbedResidentForTest((i*131+7)%vocab), i)
			if err != nil {
				t.Fatalf("forward[%d] (slots=%q pread=%q): %v", i, slotsEnv, preadEnv, err)
			}
			out[i] = append([]float32(nil), lr...)
		}
		preads := 0
		mr, ok := rf.(*metalResident) // the adapter wrapping *resident (backend.go)
		if !ok {
			t.Fatalf("resident forward is %T, not *metal.metalResident — cannot read pool counters", rf)
		}
		for l := range mr.r.layers {
			if ml := mr.r.layers[l].moe; ml != nil && ml.pool != nil {
				preads += ml.pool.preads
			}
		}
		return out, preads
	}

	base, _ := run(t, "", "") // non-paged: all 4 experts stacked resident
	eq := func(t *testing.T, got [][]float32, label string) {
		t.Helper()
		for i := range base {
			if len(got[i]) != len(base[i]) {
				t.Fatalf("%s tok %d: length %d != %d", label, i, len(got[i]), len(base[i]))
			}
			for j := range base[i] {
				if got[i][j] != base[i][j] {
					t.Fatalf("%s tok %d elem %d: %v != non-paged %v — staging is observable in the "+
						"output, which a cache must never allow", label, i, j, got[i][j], base[i][j])
				}
			}
		}
	}
	for _, slots := range []string{"2", "3"} {
		t.Run("slots="+slots+"/bytecopy", func(t *testing.T) {
			got, preads := run(t, slots, "0")
			if preads != 0 {
				t.Fatalf("GOINFER_MOE_PREAD=0 still took the pread path (%d preads)", preads)
			}
			eq(t, got, "bytecopy")
			t.Logf("slots=%s byte-copy: %d tokens exact vs non-paged", slots, len(base))
		})
		t.Run("slots="+slots+"/pread", func(t *testing.T) {
			got, preads := run(t, slots, "")
			if preads == 0 {
				t.Fatal("pread path never ran (0 preads) — stagePread was not wired, so this test " +
					"compared the byte-copy path against itself and proved nothing about pread")
			}
			eq(t, got, "pread")
			t.Logf("slots=%s pread: %d tokens exact vs non-paged, %d expert stages served by pread",
				slots, len(base), preads)
		})
	}
}
