//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/prequant"
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
	const ckpt = "../../decoder/testdata/qwen3_5_moe-tiny"
	if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
		t.Skipf("no qwen3_5_moe fixture at %s (run scripts/pin_qwen3_5_forward.py --moe): %v", ckpt, err)
	}
	giw := filepath.Join(t.TempDir(), "qwen3_5_moe-tiny.giw")
	if err := prequant.Transcode(context.Background(), ckpt, giw, "int4", false, decoder.GIWTargetNone); err != nil {
		t.Fatalf("transcode fixture to .giw: %v", err)
	}

	// run drives ntok tokens and returns the logits plus the total pread count across paged layers.
	run := func(t *testing.T, slotsEnv, preadEnv string) ([][]float32, int, int64) {
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
		preads, scalePreads := 0, int64(0)
		mr, ok := rf.(*metalResident) // the adapter wrapping *resident (backend.go)
		if !ok {
			t.Fatalf("resident forward is %T, not *metal.metalResident — cannot read pool counters", rf)
		}
		for l := range mr.r.layers {
			if ml := mr.r.layers[l].moe; ml != nil && ml.pool != nil {
				preads += ml.pool.preads
				scalePreads += ml.pool.scalePreads.Load()
			}
		}
		return out, preads, scalePreads
	}

	base, _, _ := run(t, "", "") // non-paged: all 4 experts stacked resident
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
			got, preads, _ := run(t, slots, "0")
			if preads != 0 {
				t.Fatalf("GOINFER_MOE_PREAD=0 still took the pread path (%d preads)", preads)
			}
			eq(t, got, "bytecopy")
			t.Logf("slots=%s byte-copy: %d tokens exact vs non-paged", slots, len(base))
		})
		t.Run("slots="+slots+"/pread", func(t *testing.T) {
			got, preads, scalePreads := run(t, slots, "")
			if preads == 0 {
				t.Fatal("pread path never ran (0 preads) — stagePread was not wired, so this test " +
					"compared the byte-copy path against itself and proved nothing about pread")
			}
			if scalePreads != int64(preads) {
				t.Fatalf("%d of %d pread stages read their scales from the file: a v15 .giw holds every expert's (C-P01)", scalePreads, preads)
			}
			eq(t, got, "pread")
			t.Logf("slots=%s pread: %d tokens exact vs non-paged, %d expert stages served by pread, %d with their scales",
				slots, len(base), preads, scalePreads)
		})
	}
}

// TestFlintResidentParityMetal is the Metal resident and paged-MoE parity gate for Flint:
//   - testdata/flint-tiny (6 layers: 2 dense full/NoPE + 3 MoE sliding/RoPE + 1 MoE
//     full/NoPE, sliding_window=8, learned per-head attention sinks, QK-norm, top-2 of 8
//     sigmoid-normalized router + ungated shared expert) goes resident on Metal at int4;
//   - sequential decode across 12 positions (past the 8-token sliding window) agrees with
//     the CPU int4 forward;
//   - a .giw bundle loaded with MoECacheExperts=true, MoECacheSlots=3 (< 8 experts,
//     forcing LRU eviction) produces bit-identical decode logits to the all-resident run.
func TestFlintResidentParityMetal(t *testing.T) {
	const ckpt = "../../testdata/flint-tiny"

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	defer mRes.Close()

	if !mRes.HasAttnSinksResident() {
		t.Fatalf("flint-tiny HasAttnSinksResident() = false, want true")
	}
	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("flint-tiny did not go Metal-resident (decode path %q; decline: %s)",
			mRes.DecodePath(), mRes.ResidentDecline())
	}

	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()

	const ntok = 12 // > sliding_window=8
	_, _, _, _, _, _, vocab := mRes.Dims()
	prompt := make([]int, ntok)
	for i := range prompt {
		prompt[i] = (i*37 + 3) % vocab
	}

	rf.Reset()
	cache := mCPU.NewCache(ntok)
	worstCos := 1.0
	for i, tok := range prompt {
		lc, err := mCPU.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward[%d]: %v", i, err)
		}
		lr, err := rf.Forward(mRes.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("metal forward[%d]: %v", i, err)
		}
		for j, v := range lr {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("metal forward[%d]: logit[%d] = %v", i, j, v)
			}
		}
		worstCos = math.Min(worstCos, cosF(lr, lc))
	}
	t.Logf("flint-tiny Metal decode (%d positions across window=8) vs CPU int4: worst cosine %.6f", ntok, worstCos)
	if worstCos < 0.995 {
		t.Errorf("Metal decode vs CPU int4 worst cosine %.6f < 0.995", worstCos)
	}

	giwPath := filepath.Join(t.TempDir(), "flint-tiny.int4.metal.giw")
	if err := prequant.Transcode(t.Context(), ckpt, giwPath, "int4", false, decoder.GIWTargetMetal); err != nil {
		t.Fatalf("prequant.Transcode: %v", err)
	}

	runGIW := func(paged bool) [][]float32 {
		opts := decoder.Options{Backend: "metal", MoECacheExperts: paged}
		if paged {
			opts.MoECacheSlots = 3
		}
		m, err := decoder.Load(giwPath, opts)
		if err != nil {
			t.Fatalf("Load(.giw, paged=%v): %v", paged, err)
		}
		defer m.Close()
		r := m.ResidentForwardForTest()
		if r == nil {
			t.Fatalf(".giw (paged=%v) did not go resident: %s", paged, m.ResidentDecline())
		}
		r.Reset()
		out := make([][]float32, ntok)
		for i, tok := range prompt {
			lg, err := r.Forward(m.EmbedResidentForTest(tok), i)
			if err != nil {
				t.Fatalf(".giw forward[%d] (paged=%v): %v", i, paged, err)
			}
			out[i] = append([]float32(nil), lg...)
		}
		return out
	}

	unpaged, paged := runGIW(false), runGIW(true)
	for i := range unpaged {
		for j := range unpaged[i] {
			if unpaged[i][j] != paged[i][j] {
				t.Fatalf("step %d logit %d: paged %.9g != unpaged %.9g", i, j, paged[i][j], unpaged[i][j])
			}
		}
	}
	t.Logf("flint-tiny Metal .giw MoE pager (slots=3 vs all 8 experts): %d steps bit-identical", ntok)
}
