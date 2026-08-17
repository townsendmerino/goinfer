//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

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

// TestResidentDrafter_blockParity is the gate on the whole resident trunk: the drafter's block
// forward on device against the CPU implementation it was ported from.
//
// This is the one that matters. Every earlier gate covered one piece — fc fusion, context K/V,
// the attention mask — and each could be right while the assembly is wrong: a norm applied with
// the target's eps, the o-proj accumulating into the wrong buffer, the block written at the
// wrong absolute position. Those all produce output of exactly the right shape.
//
// COSINE, because the device path quantizes activations to int8 at every GEMV where the CPU
// trunk is f32 throughout — five layers of that, so the bar is looser than the single-projection
// fusion gate (0.999) and this is the accumulation of the same arithmetic, not a different one.
func TestResidentDrafter_blockParity(t *testing.T) {
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
	rd, err := r.AttachDrafter(dr)
	if err != nil {
		t.Fatalf("AttachDrafter: %v", err)
	}
	geo := dr.DrafterGeometry()

	const ctxN = 24
	M := dr.BlockSize()
	fused := make([][]float32, ctxN)
	for i := range fused {
		fused[i] = make([]float32, geo.Hidden)
		for j := range fused[i] {
			fused[i][j] = float32(math.Sin(float64(i*geo.Hidden+j) * 0.0017))
		}
	}
	block := make([][]float32, M)
	for i := range block {
		block[i] = make([]float32, geo.Hidden)
		for j := range block[i] {
			block[i][j] = float32(math.Cos(float64(i*geo.Hidden+j) * 0.0023))
		}
	}

	if err := rd.ExtendContext(fused); err != nil {
		t.Fatalf("ExtendContext: %v", err)
	}
	got, err := rd.DraftBlock(block)
	if err != nil {
		t.Fatalf("DraftBlock: %v", err)
	}
	want, err := decoder.DraftBlockCPUForTest(dr, fused, block)
	if err != nil {
		t.Fatalf("CPU reference: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("got %d rows, want %d", len(got), len(want))
	}
	worst := 1.0
	for i := range want {
		var dot, na, nb float64
		for j := range want[i] {
			dot += float64(want[i][j]) * float64(got[i][j])
			na += float64(want[i][j]) * float64(want[i][j])
			nb += float64(got[i][j]) * float64(got[i][j])
		}
		cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
		if cos < worst {
			worst = cos
		}
		t.Logf("row %2d: cosine %.6f", i, cos)
	}
	t.Logf("worst row cosine %.6f over %d rows, ctx=%d, %d layers", worst, len(got), ctxN, geo.Layers)
	if worst < 0.99 {
		t.Errorf("worst cosine %.6f < 0.99 — the resident trunk does not reproduce the CPU trunk", worst)
	}
}

// TestBatchedCapture_matchesPerToken gates the batched hidden-state seam against the per-token
// one it replaces.
//
// The drafter reads the target's residual at five tap layers for every token the verify commits.
// The existing seam does that with a sync and a download per tap PER TOKEN (0.465 ms/token
// measured); the batched one does one download per tap for the whole block. That is only a valid
// substitution if it records the SAME tensors — and a batched capture taken at the wrong point
// in the layer loop, or reading the residual before the MLP's residual add, would still be the
// right shape and the right magnitude.
//
// So: run M tokens sequentially with the per-token seam, run the same M as one batched call with
// the batched seam, and require BIT EQUALITY. Not cosine — both paths are the same kernels on
// the same weights, and the batched layer stack is already bit-identical to sequential
// (TestPrefillLast_e2e). Anything less than == here would mean the two seams disagree about
// which tensor they are recording.
func TestBatchedCapture_matchesPerToken(t *testing.T) {
	requireHeavyModel(t)
	tgt := os.Getenv("GOINFER_CUDA_MODEL")
	if tgt == "" {
		tgt = os.ExpandEnv("$HOME/models/qwen3-4b")
	}
	mc, err := decoder.Load(tgt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	r := mc.ResidentForwardForTest().(*cudaResident)
	_, _, _, _, _, _, vocab := mc.Dims()
	taps := []int{1, 9, 17, 25, 33}
	const M = 6
	rows := make([][]float32, M)
	ids := make([]int, M)
	for i := range rows {
		ids[i] = (i*7919 + 13) % (vocab - 1)
		rows[i] = mc.EmbedResidentForTest(ids[i])
	}
	// seed a context so the verify runs at a realistic depth
	seed := make([][]float32, 64)
	for i := range seed {
		seed[i] = mc.EmbedResidentForTest((i*2654435761 + 1) % (vocab - 1))
	}
	if _, e := r.PrefillLast(seed, 0); e != nil {
		t.Fatalf("seed: %v", e)
	}
	base := 64

	// --- per-token seam, one token at a time ---
	if e := r.SetHiddenCapture(taps); e != nil {
		t.Fatalf("SetHiddenCapture: %v", e)
	}
	perTok := make([][][]float32, M) // [m][tap][hidden]
	for m := 0; m < M; m++ {
		if e := r.do(func() error {
			if e := r.launchToken(rows[m], base+m, false); e != nil {
				return e
			}
			return r.stream.Sync()
		}); e != nil {
			t.Fatalf("launchToken %d: %v", m, e)
		}
		snap := make([][]float32, len(taps))
		for i, v := range r.HiddenCapture() {
			snap[i] = append([]float32(nil), v...)
		}
		perTok[m] = snap
	}
	_ = r.SetHiddenCapture(nil)

	// --- batched seam, one call ---
	if e := r.SetBatchedCapture(taps); e != nil {
		t.Fatalf("SetBatchedCapture: %v", e)
	}
	if _, e := r.PrefillLastNArgmax(rows, base); e != nil {
		t.Fatalf("PrefillLastNArgmax: %v", e)
	}
	got := r.BatchedCapture()
	_ = r.SetBatchedCapture(nil)

	hidden := len(perTok[0][0])
	if len(got) != len(taps) {
		t.Fatalf("batched capture has %d taps, want %d", len(got), len(taps))
	}
	bad := 0
	for ti := range taps {
		if len(got[ti]) != M*hidden {
			t.Fatalf("tap %d: batched row block is %d, want %d", taps[ti], len(got[ti]), M*hidden)
		}
		for m := 0; m < M; m++ {
			for j := 0; j < hidden; j++ {
				a, b := perTok[m][ti][j], got[ti][m*hidden+j]
				if a != b {
					if bad < 3 {
						t.Errorf("tap %d row %d dim %d: per-token %v, batched %v", taps[ti], m, j, a, b)
					}
					bad++
				}
			}
		}
	}
	if bad == 0 {
		t.Logf("batched capture is BIT-IDENTICAL to the per-token seam: %d taps x %d rows x %d dims",
			len(taps), M, hidden)
	} else {
		t.Errorf("%d values differ", bad)
	}

	// What it costs. The per-token seam measured 0.465 ms/token (5 taps), so ~2.3 ms per round
	// at four accepted -- a term the gate-3 projection carries. This is the batched replacement.
	best := func() float64 {
		b := 1e9
		for range 7 {
			t0 := time.Now()
			if _, e := r.PrefillLastNArgmax(rows, base); e != nil {
				t.Fatalf("timing: %v", e)
			}
			if d := float64(time.Since(t0).Microseconds()) / 1000; d < b {
				b = d
			}
		}
		return b
	}
	_ = r.SetBatchedCapture(nil)
	off := best()
	if e := r.SetBatchedCapture(taps); e != nil {
		t.Fatalf("re-arm: %v", e)
	}
	on := best()
	_ = r.SetBatchedCapture(nil)
	t.Logf("verify M=%d: capture OFF %.3f ms | ON %.3f ms | seam %+.3f ms for the WHOLE block",
		M, off, on, on-off)
	t.Logf("   per-token seam would be %.3f ms for %d committed tokens (0.465 ms each)", 0.465*float64(M), M)
}
