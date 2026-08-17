//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
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

// TestResidentDrafter_extendContext gates the drafter's context K/V on device.
//
// It checks the two things this path does that a decoder layer does NOT, because both are
// silent when wrong — the K/V would still be the right shape at the right positions:
//
//	INCREMENTAL: extending by 4 then 4 must land the same K/V as extending by 8 in one call.
//	That is the property the serving path depends on (rebuilding costs 2.4x at ctx=1024, per
//	TestDFlashDraftScaling), and an off-by-one in the write position breaks it while leaving
//	every buffer plausibly populated.
//
//	POSITION-DEPENDENT: rows written at different absolute positions must DIFFER even for
//	identical input, because RoPE rotates by position. If they matched, the rope call is being
//	handed the wrong start and every drafted token after the first block would be subtly wrong.
func TestResidentDrafter_extendContext(t *testing.T) {
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
	mc, err := decoder.Load(tgt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load target: %v", err)
	}
	defer mc.Close()
	r := mc.ResidentForwardForTest().(*cudaResident)
	dr, err := decoder.LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("load drafter: %v", err)
	}
	defer dr.Close()

	geo := dr.DrafterGeometry()
	kvDim := geo.NumKVHeads * geo.HeadDim

	mkfused := func(n, seed int) [][]float32 {
		out := make([][]float32, n)
		for i := range out {
			out[i] = make([]float32, geo.Hidden)
			for j := range out[i] {
				out[i][j] = float32(math.Sin(float64((i+seed)*geo.Hidden+j) * 0.002))
			}
		}
		return out
	}
	// Read layer 0's K cache back for the first n positions.
	readK := func(d *residentDrafter, n int) []float32 {
		host := make([]float32, n*kvDim)
		if e := d.r.do(func() error {
			if e := d.r.stream.Sync(); e != nil {
				return e
			}
			full := make([]float32, d.kvCap*kvDim)
			if e := gpu.Download(d.kc[0], full); e != nil {
				return e
			}
			copy(host, full[:n*kvDim])
			return nil
		}); e != nil {
			t.Fatalf("readK: %v", e)
		}
		return host
	}

	// --- one shot: 8 rows in a single call ---
	dA, err := r.AttachDrafter(dr)
	if err != nil {
		t.Fatalf("AttachDrafter: %v", err)
	}
	rows := mkfused(8, 0)
	// Make rows 0 and 4 IDENTICAL so the position check below isolates RoPE: same input at
	// different absolute positions must produce different K, or the rope start is wrong.
	copy(rows[4], rows[0])
	if err := dA.ExtendContext(rows); err != nil {
		t.Fatalf("one-shot ExtendContext: %v", err)
	}
	if dA.ContextLen() != 8 {
		t.Fatalf("ContextLen = %d, want 8", dA.ContextLen())
	}
	oneShot := readK(dA, 8)

	// --- incremental: 4 then 4, same rows ---
	dB, err := r.AttachDrafter(dr)
	if err != nil {
		t.Fatalf("AttachDrafter B: %v", err)
	}
	if err := dB.ExtendContext(rows[:4]); err != nil {
		t.Fatalf("incremental 1: %v", err)
	}
	if err := dB.ExtendContext(rows[4:]); err != nil {
		t.Fatalf("incremental 2: %v", err)
	}
	if dB.ContextLen() != 8 {
		t.Fatalf("incremental ContextLen = %d, want 8", dB.ContextLen())
	}
	incr := readK(dB, 8)

	diff := 0
	for i := range oneShot {
		if oneShot[i] != incr[i] {
			diff++
		}
	}
	if diff != 0 {
		t.Errorf("incremental != one-shot in %d/%d K values — the write position is wrong", diff, len(oneShot))
	} else {
		t.Logf("incremental (4+4) is BIT-IDENTICAL to one-shot (8) across %d K values", len(oneShot))
	}

	// --- position dependence: identical input at different positions must differ (RoPE) ---
	same := true
	for j := 0; j < kvDim; j++ {
		if oneShot[j] != oneShot[4*kvDim+j] {
			same = false
			break
		}
	}
	rowsEqual := true
	for j := range rows[0] {
		if rows[0][j] != rows[4][j] {
			rowsEqual = false
			break
		}
	}
	if !rowsEqual {
		t.Logf("position check: input rows 0 and 4 differ anyway, so RoPE is not isolated here")
	} else if same {
		t.Errorf("identical input rows produced identical K at positions 0 and 4 — RoPE is not being applied per position")
	}
}
