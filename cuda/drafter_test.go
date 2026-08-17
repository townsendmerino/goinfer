//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestResidentDrafter_fuseParity is the first gate on the resident block drafter: its `fc`
// fusion, on the device, against the f32 arithmetic the CPU trunk does.
//
// `fc` is the projection with NO counterpart in a normal decoder layer ([hidden, nTaps*hidden],
// here 2560x12800), so it is the one most likely to be mis-packed — and packWeight/upW have
// never been asked to handle a drafter's weights before. If the drafter's weights are packed or
// scaled wrongly, it surfaces here rather than eighteen kernels later with everything looking
// plausible.
//
// THE REFERENCE IS COMPUTED FROM THE INTERFACE'S OWN OUTPUTS — DrafterFC()'s f32 weights and
// DrafterHiddenNorm() — so this gates decoder.BlockDrafterWeights end to end at the same time:
// if the interface handed back the wrong matrix, both sides would be consistent only if the GPU
// were fed that same wrong matrix, and it is fed the device upload of it.
//
// COSINE, NOT EQUALITY, and deliberately: the device path quantizes activations to int8 exactly
// as the resident target does everywhere else, so it is not bit-identical to f32 CPU arithmetic
// and never will be. That is the same bar every GPU-int4/int8-vs-CPU-f32 parity test in this
// repo uses. (The one place a stricter bar applies is the verify's argmax, which must match
// exactly — see PrefillLastNArgmax.)
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_CUDA_MODEL=$HOME/models/qwen3-4b \
//	  go test -tags 'cuda goinfer_testhooks' -run TestResidentDrafter_fuseParity -v
func TestResidentDrafter_fuseParity(t *testing.T) {
	requireHeavyModel(t)
	tgt := os.Getenv("GOINFER_CUDA_MODEL")
	if tgt == "" {
		tgt = os.ExpandEnv("$HOME/models/qwen3-4b")
	}
	ddir := os.Getenv("GOINFER_DFLASH_F32")
	if ddir == "" {
		ddir = filepath.Join(os.Getenv("HOME"), "models", "qwen3-4b-dflash-f32")
	}
	if _, err := os.Stat(filepath.Join(ddir, "model.safetensors")); err != nil {
		t.Skipf("no drafter at %s", ddir)
	}
	if _, err := os.Stat(tgt); err != nil {
		t.Skipf("no target at %s", tgt)
	}
	mc, err := decoder.Load(tgt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load target: %v", err)
	}
	defer mc.Close()
	r, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatal("resident did not engage")
	}
	dr, err := decoder.LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("load drafter: %v", err)
	}
	defer dr.Close()

	var w decoder.BlockDrafterWeights = dr
	rd, err := r.AttachDrafter(w)
	if err != nil {
		t.Fatalf("AttachDrafter: %v", err)
	}
	geo := w.DrafterGeometry()
	fc := w.DrafterFC()
	K, N := fc.Cols(), fc.Rows()
	t.Logf("drafter attached: %d layers, hidden %d, fc %dx%d", geo.Layers, geo.Hidden, N, K)

	fcW, _ := fc.F32()
	if len(fcW) != N*K {
		t.Fatalf("fc f32 is %d, want %d — the drafter is not f32; this gate needs the f32 conversion", len(fcW), N*K)
	}
	hn := w.DrafterHiddenNorm()

	const rows = 6
	ctx := make([][]float32, rows)
	for i := range ctx {
		ctx[i] = make([]float32, K)
		for j := range ctx[i] {
			ctx[i][j] = float32(math.Sin(float64(i*K+j)*0.001)) * 0.4
		}
	}
	got, err := rd.FuseContext(ctx)
	if err != nil {
		t.Fatalf("FuseContext: %v", err)
	}
	if len(got) != rows {
		t.Fatalf("got %d rows, want %d", len(got), rows)
	}

	for i, row := range ctx {
		// f32 reference: fc matmul, then plain RMSNorm with the drafter's own eps.
		ref := make([]float32, N)
		for n := 0; n < N; n++ {
			var acc float64
			wr := fcW[n*K : (n+1)*K]
			for k, v := range row {
				acc += float64(wr[k]) * float64(v)
			}
			ref[n] = float32(acc)
		}
		var ss float64
		for _, v := range ref {
			ss += float64(v) * float64(v)
		}
		inv := 1 / math.Sqrt(ss/float64(N)+geo.NormEps)
		for n := range ref {
			ref[n] = float32(float64(ref[n]) * inv * float64(hn[n]))
		}
		var dot, na, nb float64
		for n := range ref {
			dot += float64(ref[n]) * float64(got[i][n])
			na += float64(ref[n]) * float64(ref[n])
			nb += float64(got[i][n]) * float64(got[i][n])
		}
		cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
		t.Logf("row %d: cosine %.6f", i, cos)
		if cos < 0.999 {
			t.Errorf("row %d: cosine %.6f < 0.999 — the drafter's fc is mis-packed or mis-scaled on device", i, cos)
		}
	}
}
